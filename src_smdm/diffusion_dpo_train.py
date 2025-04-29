"""
Main training script for Diffusion-DPO fine-tuning, adapted for SMDM models.
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
from transformers import get_scheduler # Keep scheduler
import copy
import sys
from pathlib import Path

# Add SMDM repo to path to import its modules
smdm_repo_path = "/home/ubuntu/SMDM_repo"
sys.path.insert(0, smdm_repo_path)

# Import utility functions (now adapted for SMDM)
from utils import (
    setup_logger,
    create_reference_model,
    compute_diffusion_score,
    load_smdm_model_and_tokenizer, # Use SMDM loader
    prepare_model_inputs, # Use SMDM input prep
    set_seed # Re-import set_seed from utils to avoid conflict if accelerate changes
)

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train an SMDM model with DPO")
    parser.add_argument("--config_file", type=str, default="config/config.yaml", help="Path to config YAML file")
    # Add arguments that might not be in the config file or need overriding
    parser.add_argument("--model_name_or_path", type=str, help="HF Repo ID for the SMDM model (e.g., nieshen/SMDM)")
    parser.add_argument("--checkpoint_filename", type=str, help="Specific checkpoint filename within the repo (e.g., mdm_safetensors/mdm-170M-100e18.safetensors)")
    parser.add_argument("--model_size_M", type=int, help="Model size in Millions for config lookup (e.g., 170)")
    parser.add_argument("--preference_data_path", type=str, help="Path to the preference dataset file")
    parser.add_argument("--output_dir", type=str, help="Directory to save checkpoints and logs")
    parser.add_argument("--learning_rate", type=float, help="Optimizer learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Number of steps to accumulate gradients")
    parser.add_argument("--num_train_epochs", type=int, help="Total number of training epochs")
    parser.add_argument("--beta", type=float, help="DPO hyperparameter")
    parser.add_argument("--diffusion_steps", type=int, help="Number of t samples for diffusion score calculation")
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
            try:
                config_dict = yaml.safe_load(f)
                # Update args with config values, but command line args take precedence
                for key, value in config_dict.items():
                    # Only set if the arg wasn't provided via command line
                    if getattr(args, key, None) is None:
                         # Handle boolean flags potentially read as strings from YAML
                        if isinstance(getattr(args, key, None), bool):
                            if isinstance(value, str):
                                value = value.lower() in ('true', '1', 'yes')
                        setattr(args, key, value)
            except yaml.YAMLError as exc:
                print(f"Error loading YAML file: {exc}")

    # Re-parse to ensure boolean flags from config are handled correctly if store_true is used
    # This is a bit tricky; simpler to just check required args after loading config

    # Basic validation
    if args.preference_data_path is None:
        raise ValueError("Preference data path must be provided (--preference_data_path or in config).")
    if args.output_dir is None:
        raise ValueError("Output directory must be provided (--output_dir or in config).")
    if args.model_name_or_path is None:
        raise ValueError("Model name or path (HF Repo ID) must be provided (--model_name_or_path or in config).")
    if args.checkpoint_filename is None:
        raise ValueError("Checkpoint filename must be provided (--checkpoint_filename or in config).")
    if args.model_size_M is None:
        raise ValueError("Model size M must be provided (--model_size_M or in config).")

    # Convert gradient_checkpointing from potential string/None to boolean
    if isinstance(args.gradient_checkpointing, str):
        args.gradient_checkpointing = args.gradient_checkpointing.lower() in ('true', '1', 'yes')
    elif args.gradient_checkpointing is None: # If not in config or cmd line, default to False
        args.gradient_checkpointing = False

    return args

def prepare_preference_dataloader(args, tokenizer, accelerator):
    """Load and prepare the preference dataset and dataloader."""
    logger.info(f"Loading preference dataset from {args.preference_data_path}")
    try:
        # Assuming JSONL format with 'prompt', 'chosen', 'rejected' keys
        dataset = load_dataset('json', data_files=args.preference_data_path)['train']
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    def preprocess_function(examples):
        # Use the prepare_model_inputs function from utils
        # It handles combining prompt+response and tokenization
        chosen_inputs = prepare_model_inputs(
            tokenizer, examples["prompt"], examples["chosen"], args.max_length, device="cpu" # Tokenize on CPU
        )
        rejected_inputs = prepare_model_inputs(
            tokenizer, examples["prompt"], examples["rejected"], args.max_length, device="cpu"
        )
        
        return {
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

    # Simple collate function (assumes preprocess_function returns lists of tensors)
    def collate_fn(batch):
        collated = {}
        keys = batch[0].keys()
        for key in keys:
            # Pad tensors to the max length in the batch
            # Note: prepare_model_inputs already pads to max_length, so stacking should work
            try:
                collated[key] = torch.stack([torch.tensor(item[key]) for item in batch])
            except Exception as e:
                 logger.error(f"Error collating key '{key}': {e}")
                 # Fallback: try padding if stacking fails (shouldn't if pre-padded)
                 # from torch.nn.utils.rnn import pad_sequence
                 # collated[key] = pad_sequence([torch.tensor(item[key]) for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
                 raise # Re-raise error for debugging
        return collated

    dataloader = DataLoader(
        processed_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn 
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
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    # setup_logger(log_level=logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    # logger.info(accelerator.state, main_process_only=False)
    logger.info(f"Training arguments: {args}")

    # Set seed before initializing model.
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Set seed to {args.seed}")

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer using SMDM loader
    # Construct full checkpoint path
    # Try downloading from HF Hub first
    from huggingface_hub import hf_hub_download
    try:
        checkpoint_full_path = hf_hub_download(
            repo_id=args.model_name_or_path, 
            filename=args.checkpoint_filename
        )
        logger.info(f"Downloaded checkpoint to {checkpoint_full_path}")
    except Exception as e:
        logger.error(f"Failed to download checkpoint {args.checkpoint_filename} from {args.model_name_or_path}: {e}")
        # Fallback to assuming it's locally available (e.g., mounted)
        checkpoint_full_path = os.path.join("/path/to/local/checkpoints", args.model_name_or_path, args.checkpoint_filename) # Adjust this path if needed
        logger.warning(f"Attempting to load checkpoint from local path: {checkpoint_full_path}")
        if not os.path.exists(checkpoint_full_path):
             raise FileNotFoundError(f"Checkpoint not found locally either: {checkpoint_full_path}")

    model, tokenizer, config = load_smdm_model_and_tokenizer(
        model_size_M=args.model_size_M,
        checkpoint_path=checkpoint_full_path, 
        device="cpu" # Load on CPU first before accelerator.prepare
    )
    model = model.to(accelerator.device) # Move model to device after loading
    
    # Create reference model (after loading main model)
    with accelerator.main_process_first(): # Ensure deepcopy happens on one process
        reference_model = create_reference_model(model)
    reference_model.to(accelerator.device)
    reference_model.eval()

    # Prepare dataset and dataloader
    train_dataloader = prepare_preference_dataloader(args, tokenizer, accelerator)

    # Optimizer
    # Filter out parameters that don't require gradients (e.g., reference model if prepared together)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            "weight_decay": args.weight_decay, # Apply weight decay to all trainable params for simplicity
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler
    # Calculate num_update_steps_per_epoch correctly
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        logger.info(f"Calculated max_train_steps: {args.max_train_steps}")
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        logger.info(f"Using provided max_train_steps: {args.max_train_steps}, calculated epochs: {args.num_train_epochs}")

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with accelerator
    # Note: Reference model should ideally not be prepared if it's frozen, 
    # but preparing it ensures it's moved to the correct device(s) in distributed settings.
    model, reference_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, reference_model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Gradient checkpointing
    # Check if model has the method before calling
    if args.gradient_checkpointing:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")
        else:
            logger.warning("Model does not support gradient_checkpointing_enable method.")

    # Resume from checkpoint logic (simplified)
    starting_epoch = 0
    completed_steps = 0
    if args.resume_from_checkpoint:
        if os.path.isdir(args.resume_from_checkpoint):
            logger.info(f"Resuming from checkpoint {args.resume_from_checkpoint}")
            try:
                accelerator.load_state(args.resume_from_checkpoint)
                # Add logic here to parse step/epoch from checkpoint name if needed
            except Exception as e:
                 logger.error(f"Failed to load checkpoint state: {e}. Starting from scratch.")
        else:
            logger.warning(f"Checkpoint {args.resume_from_checkpoint} not found or not a directory. Starting from scratch.")

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

    global_step = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        
        active_dataloader = train_dataloader
        # Add logic for resuming from specific step if needed
        
        for step, batch in enumerate(active_dataloader):
            # Ensure batch items are on the correct device
            # batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            
            # Handle gradient accumulation
            with accelerator.accumulate(model):
                # Extract inputs and outputs from batch (already combined in dataloader)
                chosen_ids = batch["chosen_input_ids"]
                chosen_mask = batch["chosen_attention_mask"]
                rejected_ids = batch["rejected_input_ids"]
                rejected_mask = batch["rejected_attention_mask"]
                
                # Compute diffusion scores using the adapted function
                # Pass the full sequence (prompt+response) as input_ids
                with torch.cuda.amp.autocast(enabled=args.use_mixed_precision != "no"):
                    # Ensure reference model is in eval mode inside no_grad context
                    reference_model.eval()
                    
                    # Get scores from policy model (requires grad)
                    preferred_score = compute_diffusion_score(
                        model, chosen_ids, chosen_mask, args.diffusion_steps,
                        use_gradient_checkpointing=args.gradient_checkpointing
                    )
                    dispreferred_score = compute_diffusion_score(
                        model, rejected_ids, rejected_mask, args.diffusion_steps,
                        use_gradient_checkpointing=args.gradient_checkpointing
                    )
                    
                    # Get scores from reference model (no grad)
                    with torch.no_grad():
                        ref_preferred_score = compute_diffusion_score(
                            reference_model, chosen_ids, chosen_mask, args.diffusion_steps,
                            use_gradient_checkpointing=False # No GC needed for ref model
                        )
                        ref_dispreferred_score = compute_diffusion_score(
                            reference_model, rejected_ids, rejected_mask, args.diffusion_steps,
                            use_gradient_checkpointing=False
                        )

                    # Compute DPO loss
                    pi_logratios = preferred_score - dispreferred_score
                    ref_logratios = ref_preferred_score - ref_dispreferred_score
                    
                    logits = pi_logratios - ref_logratios
                    loss = -F.logsigmoid(args.beta * logits).mean()

                # Gather loss across devices
                # avg_loss = accelerator.gather(loss.repeat(args.per_device_train_batch_size)).mean()
                # Simpler gather for scalar loss:
                avg_loss = accelerator.gather(loss).mean()
                total_loss += avg_loss.item() # Already averaged over batch and devices

                # Backward pass
                accelerator.backward(loss)

                # Optimizer step (only when accumulation is complete)
                if accelerator.sync_gradients:
                    # Clip gradients if needed (optional)
                    # accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    progress_bar.update(1)
                    global_step += 1 # Use global_step for logging/saving

                    # Logging
                    if global_step % args.logging_steps == 0:
                        # Loss is already averaged, divide by logging steps
                        avg_loss_log = total_loss / args.logging_steps 
                        logger.info(f"Epoch {epoch}, Step {global_step}: Loss = {avg_loss_log:.4f}, LR = {lr_scheduler.get_last_lr()[0]:.6f}")
                        if accelerator.is_main_process:
                             accelerator.log({"train_loss": avg_loss_log, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                        total_loss = 0 # Reset loss accumulator

                    # Checkpointing
                    if args.checkpointing_steps and isinstance(args.checkpointing_steps, int):
                        if global_step % args.checkpointing_steps == 0:
                            output_dir_step = f"step_{global_step}"
                            if args.output_dir is not None:
                                output_dir_step = os.path.join(args.output_dir, output_dir_step)
                            logger.info(f"Saving step checkpoint to {output_dir_step}")
                            accelerator.save_state(output_dir_step)
            
            if global_step >= args.max_train_steps:
                break # Exit inner loop
        
        # Epoch checkpointing
        if args.checkpointing_steps == "epoch":
            output_dir_epoch = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir_epoch = os.path.join(args.output_dir, output_dir_epoch)
            logger.info(f"Saving epoch checkpoint to {output_dir_epoch}")
            accelerator.save_state(output_dir_epoch)
            
        if global_step >= args.max_train_steps:
            logger.info(f"Reached max_train_steps ({args.max_train_steps}). Stopping training.")
            break # Exit outer loop

    # Save final model state
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            final_checkpoint_dir = os.path.join(args.output_dir, "final_checkpoint")
            os.makedirs(final_checkpoint_dir, exist_ok=True)
            # Save accelerator state (includes model, optimizer, scheduler)
            accelerator.save_state(final_checkpoint_dir)
            logger.info(f"Saved final training state to {final_checkpoint_dir}")
            
            # Optionally save unwrapped model separately if needed
            # unwrapped_model = accelerator.unwrap_model(model)
            # unwrapped_model_path = os.path.join(args.output_dir, "final_model.pth") 
            # torch.save(unwrapped_model.state_dict(), unwrapped_model_path)
            # logger.info(f"Saved final unwrapped model state_dict to {unwrapped_model_path}")
            # tokenizer.save_pretrained(args.output_dir) # Save tokenizer if applicable

    accelerator.end_training()
    logger.info("***** Training finished *****")

if __name__ == "__main__":
    main()

