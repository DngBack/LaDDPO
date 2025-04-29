"""
Main training script for Diffusion-DPO fine-tuning.
"""

import os
import torch
import torch.nn.functional as F
import argparse
import yaml
import logging
import numpy as np
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_scheduler
import copy

# Import utility functions
from utils import (
    setup_logger,
    create_reference_model,
    compute_diffusion_score,
    load_model_and_tokenizer,
    prepare_model_inputs # Assuming this handles tokenization and batching
)

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a diffusion model with DPO")
    parser.add_argument("--config_file", type=str, default="config/config.yaml", help="Path to config YAML file")
    # Add arguments that might not be in the config file or need overriding
    parser.add_argument("--model_name_or_path", type=str, help="Path or HF identifier for the base LLDM")
    parser.add_argument("--preference_data_path", type=str, help="Path to the preference dataset file")
    parser.add_argument("--output_dir", type=str, help="Directory to save checkpoints and logs")
    parser.add_argument("--learning_rate", type=float, help="Optimizer learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Number of steps to accumulate gradients")
    parser.add_argument("--num_train_epochs", type=int, help="Total number of training epochs")
    parser.add_argument("--beta", type=float, help="DPO hyperparameter")
    parser.add_argument("--diffusion_steps", type=int, help="Number of steps for diffusion score calculation")
    parser.add_argument("--use_mixed_precision", type=str, help="Whether to use mixed precision (","fp16",", "bf16",", "no",")")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--save_steps", type=int, help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, help="Log metrics every N steps")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="The scheduler type to use.")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to apply.")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length for tokenization.")
    parser.add_argument("--checkpointing_steps", type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="If the training should continue from a checkpoint folder.")

    args = parser.parse_args()

    # Load config file if provided and update args
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
            # Update args with config values, but command line args take precedence
            for key, value in config_dict.items():
                if getattr(args, key, None) is None:
                    setattr(args, key, value)
    
    # Basic validation
    if args.preference_data_path is None:
        raise ValueError("Preference data path must be provided.")
    if args.output_dir is None:
        raise ValueError("Output directory must be provided.")
    if args.model_name_or_path is None:
        raise ValueError("Model name or path must be provided.")

    return args

def prepare_preference_dataloader(args, tokenizer):
    """Load and prepare the preference dataset and dataloader."""
    logger.info(f"Loading preference dataset from {args.preference_data_path}")
    # Assuming JSONL format with 'prompt', 'chosen', 'rejected' keys
    dataset = load_dataset('json', data_files=args.preference_data_path)['train']

    def preprocess_function(examples):
        # Tokenize prompts, chosen responses, and rejected responses
        # This needs careful implementation based on how the model expects inputs
        # Placeholder: tokenize each part separately
        prompts = examples["prompt"]
        chosen = examples["chosen"]
        rejected = examples["rejected"]
        
        # Tokenize - adjust padding/truncation as needed
        # We need prompt_ids, chosen_ids, rejected_ids
        # Example: Tokenize combined prompt+response for score calculation
        # This part is highly dependent on the specific LLDM's forward pass and how
        # compute_diffusion_score expects inputs/outputs.
        # For now, let's assume we need tokenized versions of each.
        
        # Placeholder tokenization - replace with actual logic
        prompt_inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=args.max_length)
        chosen_inputs = tokenizer(chosen, padding="max_length", truncation=True, max_length=args.max_length)
        rejected_inputs = tokenizer(rejected, padding="max_length", truncation=True, max_length=args.max_length)
        
        # Return a dictionary suitable for batching
        # The exact format depends on how compute_diffusion_score and the model work
        return {
            "prompt_input_ids": prompt_inputs["input_ids"],
            "prompt_attention_mask": prompt_inputs["attention_mask"],
            "chosen_input_ids": chosen_inputs["input_ids"],
            "chosen_attention_mask": chosen_inputs["attention_mask"],
            "rejected_input_ids": rejected_inputs["input_ids"],
            "rejected_attention_mask": rejected_inputs["attention_mask"],
        }

    with accelerator.main_process_first():
        processed_dataset = dataset.map(
            preprocess_function, 
            batched=True, 
            remove_columns=dataset.column_names,
            desc="Preprocessing preference dataset"
        )

    dataloader = DataLoader(
        processed_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        # collate_fn= # May need a custom collate function
    )
    return dataloader

def main():
    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.use_mixed_precision,
        log_with="tensorboard", # Or wandb
        project_dir=os.path.join(args.output_dir, "logs")
    )

    # Setup logging
    setup_logger(log_level=logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    logger.info(accelerator.state, main_process_only=False)
    logger.info(f"Training arguments: {args}")

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Set seed to {args.seed}")

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, device=accelerator.device)
    
    # Create reference model
    reference_model = create_reference_model(model)
    reference_model.to(accelerator.device)
    reference_model.eval()

    # Prepare dataset and dataloader
    train_dataloader = prepare_preference_dataloader(args, tokenizer)

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = args.max_train_steps // num_update_steps_per_epoch + 1

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with accelerator
    model, reference_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, reference_model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")

    # Resume from checkpoint
    starting_epoch = 0
    completed_steps = 0
    if args.resume_from_checkpoint:
        if os.path.isdir(args.resume_from_checkpoint):
            logger.info(f"Resuming from checkpoint {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            # Extract step/epoch info if saved
            # path = os.path.basename(args.resume_from_checkpoint)
            # training_difference = os.path.splitext(path)[0]
            # if "epoch" in training_difference:
            #     starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            #     resume_step = None
            # else:
            #     # need to implement step resumption logic
            #     pass
        else:
            logger.warning(f"Checkpoint {args.resume_from_checkpoint} not found. Starting from scratch.")

    # Initialize progress bar
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        
        active_dataloader = train_dataloader
        # Add logic for resuming from specific step if needed
        
        for step, batch in enumerate(active_dataloader):
            # Handle gradient accumulation
            with accelerator.accumulate(model):
                # Extract inputs and outputs from batch
                # This depends heavily on the dataloader output format
                # Placeholder - adapt based on actual batch structure
                prompt_ids = batch["prompt_input_ids"]
                prompt_mask = batch["prompt_attention_mask"]
                chosen_ids = batch["chosen_input_ids"]
                chosen_mask = batch["chosen_attention_mask"]
                rejected_ids = batch["rejected_input_ids"]
                rejected_mask = batch["rejected_attention_mask"]
                
                # Prepare inputs for compute_diffusion_score
                # This is highly model-specific. Assuming inputs are prompt_ids and outputs are chosen/rejected_ids
                # You might need to combine prompt and response, handle masking, etc.
                model_inputs = prompt_ids # Placeholder
                chosen_outputs = chosen_ids # Placeholder
                rejected_outputs = rejected_ids # Placeholder
                
                # Compute diffusion scores
                with torch.cuda.amp.autocast(enabled=args.use_mixed_precision != "no"):
                    preferred_score = compute_diffusion_score(
                        model, model_inputs, chosen_outputs, args.diffusion_steps,
                        use_gradient_checkpointing=args.gradient_checkpointing
                    )
                    
                    dispreferred_score = compute_diffusion_score(
                        model, model_inputs, rejected_outputs, args.diffusion_steps,
                        use_gradient_checkpointing=args.gradient_checkpointing
                    )
                    
                    # Compute reference model scores (no gradient)
                    with torch.no_grad():
                        ref_preferred_score = compute_diffusion_score(
                            reference_model, model_inputs, chosen_outputs, args.diffusion_steps,
                            use_gradient_checkpointing=False
                        )
                        ref_dispreferred_score = compute_diffusion_score(
                            reference_model, model_inputs, rejected_outputs, args.diffusion_steps,
                            use_gradient_checkpointing=False
                        )

                    # Compute logits and loss
                    logits = (preferred_score - dispreferred_score) - (ref_preferred_score - ref_dispreferred_score)
                    loss = -F.logsigmoid(args.beta * logits).mean()

                # Gather loss across devices
                avg_loss = accelerator.gather(loss.repeat(args.per_device_train_batch_size)).mean()
                total_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backward pass
                accelerator.backward(loss)

                # Optimizer step (only when accumulation is complete)
                if accelerator.sync_gradients:
                    # Clip gradients if needed
                    # accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    progress_bar.update(1)
                    completed_steps += 1

                    # Logging
                    if completed_steps % args.logging_steps == 0:
                        avg_loss_log = total_loss / args.logging_steps
                        logger.info(f"Epoch {epoch}, Step {completed_steps}: Loss = {avg_loss_log:.4f}, LR = {lr_scheduler.get_last_lr()[0]:.6f}")
                        accelerator.log({"train_loss": avg_loss_log, "lr": lr_scheduler.get_last_lr()[0]}, step=completed_steps)
                        total_loss = 0 # Reset loss accumulator

                    # Checkpointing
                    if isinstance(args.checkpointing_steps, int):
                        if completed_steps % args.checkpointing_steps == 0:
                            output_dir = f"step_{completed_steps}"
                            if args.output_dir is not None:
                                output_dir = os.path.join(args.output_dir, output_dir)
                            accelerator.save_state(output_dir)
                            logger.info(f"Saved checkpoint to {output_dir}")
            
            if completed_steps >= args.max_train_steps:
                break # Exit inner loop
        
        # Epoch checkpointing
        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            logger.info(f"Saved epoch checkpoint to {output_dir}")
            
        if completed_steps >= args.max_train_steps:
            break # Exit outer loop

    # Save final model
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        # Save model using appropriate method (e.g., save_pretrained)
        # unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        # tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Saved final model state to {args.output_dir}")

    accelerator.end_training()
    logger.info("***** Training finished *****")

if __name__ == "__main__":
    main()

