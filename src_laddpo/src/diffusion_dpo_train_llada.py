"""
Main training script for Diffusion-DPO fine-tuning with LLaDA-8B-Instruct.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to Python path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

import torch
import torch.nn.functional as F
import argparse
import yaml
import logging
import numpy as np
import math
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import get_scheduler
import copy
import multiprocessing

# Set multiprocessing start method to 'spawn' for CUDA compatibility
multiprocessing.set_start_method("spawn", force=True)

# Import utility functions adapted for LLaDA using absolute imports
from src.utils_llada import (
    setup_logger,
    create_reference_model,
    compute_diffusion_score,
    load_llada_model_and_tokenizer,  # Use LLaDA loader
    prepare_hh_rlhf_batch,  # Use HH-RLHF data prep
    set_seed,
)

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a LLaDA model with DPO using HH-RLHF"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/config_llada_8b_hh.yaml",
        help="Path to config YAML file",
    )
    # Add arguments that might not be in the config file or need overriding
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="HF identifier for the LLaDA model",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Anthropic/hh-rlhf",
        help="HF identifier for the preference dataset",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use (e.g., train)",
    )
    parser.add_argument(
        "--random_sample_size",
        type=int,
        default=None,
        help="Number of random samples to use from the dataset",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save checkpoints and logs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-7, help="Optimizer learning rate"
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of steps to accumulate gradients",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs",
    )
    parser.add_argument("--beta", type=float, default=0.1, help="DPO hyperparameter")
    parser.add_argument(
        "--diffusion_samples",
        type=int,
        default=16,
        help="Number of MC samples for diffusion score calculation",
    )
    parser.add_argument(
        "--use_mixed_precision",
        type=str,
        default="bf16",
        help="Whether to use mixed precision ('fp16', 'bf16', 'no')",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        default=False,
        help="Load model in 8-bit precision",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=False,
        help="Load model in 4-bit precision",
    )
    parser.add_argument(
        "--save_steps", type=int, default=1000, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10, help="Log metrics every N steps"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to apply."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    args = parser.parse_args()

    # Load config file if provided and update args
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, "r") as f:
            try:
                config_dict = yaml.safe_load(f)
                # Update args with config values, but command line args take precedence
                for key, value in config_dict.items():
                    # Only set if the arg wasn't provided via command line
                    if getattr(args, key, None) is None:
                        # Handle boolean flags potentially read as strings from YAML
                        if isinstance(getattr(args, key, None), bool):
                            if isinstance(value, str):
                                value = value.lower() in ("true", "1", "yes")
                        setattr(args, key, value)
            except yaml.YAMLError as exc:
                print(f"Error loading YAML file: {exc}")

    # Re-parse to ensure boolean flags from config are handled correctly
    # Check required args after loading config
    if args.output_dir is None:
        raise ValueError(
            "Output directory must be provided (--output_dir or in config)."
        )
    if args.model_name_or_path is None:
        raise ValueError(
            "Model name or path must be provided (--model_name_or_path or in config)."
        )
    if args.dataset_name is None:
        raise ValueError("Dataset name must be provided (--dataset_name or in config).")

    # Convert gradient_checkpointing from potential string/None to boolean
    if isinstance(args.gradient_checkpointing, str):
        args.gradient_checkpointing = args.gradient_checkpointing.lower() in (
            "true",
            "1",
            "yes",
        )
    elif args.gradient_checkpointing is None:  # Default to True for large models
        args.gradient_checkpointing = True

    # Convert quantization flags
    if isinstance(args.load_in_8bit, str):
        args.load_in_8bit = args.load_in_8bit.lower() in ("true", "1", "yes")
    elif args.load_in_8bit is None:
        args.load_in_8bit = False

    if isinstance(args.load_in_4bit, str):
        args.load_in_4bit = args.load_in_4bit.lower() in ("true", "1", "yes")
    elif args.load_in_4bit is None:
        args.load_in_4bit = False

    return args


class CollateFn:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch_list):
        # Convert list of dicts to dict of lists
        batch = {k: [d[k] for d in batch_list] for k in batch_list[0]}
        return prepare_hh_rlhf_batch(batch, self.tokenizer, max_length=self.max_length)


def prepare_preference_dataloader(args, tokenizer, accelerator):
    """
    Prepare the dataloader for preference data with random sampling if specified.
    """
    # Load the dataset
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    # Apply random sampling if specified
    if args.random_sample_size is not None:
        logger.info(
            f"Randomly sampling {args.random_sample_size} examples from the dataset"
        )
        dataset = dataset.shuffle(seed=args.seed).select(range(args.random_sample_size))

    # Create dataloader with the class-based collate function
    collate_fn = CollateFn(tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

    return dataloader


def main():
    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.use_mixed_precision,
        log_with="tensorboard",  # Or wandb
        project_dir=os.path.join(args.output_dir, "logs"),
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info(f"Training arguments: {args}")

    # Set seed before initializing model.
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Set seed to {args.seed}")

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer using LLaDA loader
    model, tokenizer = load_llada_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        device="cpu",  # Load on CPU first if not using device_map
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    # Note: If using device_map="auto", model is already on appropriate devices
    if not (args.load_in_8bit or args.load_in_4bit):
        model = model.to(accelerator.device)

    # Create reference model (after loading main model)
    with accelerator.main_process_first():  # Ensure deepcopy happens on one process
        reference_model = create_reference_model(model)
    # Ensure reference model is also on the correct device(s)
    if not (args.load_in_8bit or args.load_in_4bit):
        reference_model = reference_model.to(accelerator.device)
    reference_model.eval()

    # Prepare dataset and dataloader
    train_dataloader = prepare_preference_dataloader(args, tokenizer, accelerator)

    # Optimizer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            "weight_decay": args.weight_decay,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        logger.info(f"Calculated max_train_steps: {args.max_train_steps}")
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )
        logger.info(
            f"Using provided max_train_steps: {args.max_train_steps}, calculated epochs: {args.num_train_epochs}"
        )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with accelerator
    # Reference model is prepared to handle device placement in distributed settings
    model, reference_model, optimizer, train_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, reference_model, optimizer, train_dataloader, lr_scheduler
        )
    )

    # Gradient checkpointing
    if args.gradient_checkpointing:
        try:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled.")
            elif hasattr(model, "_gradient_checkpointing"):
                # Some models use internal flags
                model._gradient_checkpointing = True
                logger.info("Gradient checkpointing enabled via internal flag.")
            else:
                logger.warning(
                    "Model does not support gradient checkpointing. Continuing without it."
                )
        except Exception as e:
            logger.warning(
                f"Failed to enable gradient checkpointing: {e}. Continuing without it."
            )

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
                logger.error(
                    f"Failed to load checkpoint state: {e}. Starting from scratch."
                )
        else:
            logger.warning(
                f"Checkpoint {args.resume_from_checkpoint} not found or not a directory. Starting from scratch."
            )

    # Initialize progress bar
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")

    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0

        active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            # Move batch to device (collate_fn prepared on CPU)
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}

            # Handle gradient accumulation
            with accelerator.accumulate(model):
                # Extract inputs and outputs from batch
                chosen_ids = batch["chosen_input_ids"]
                chosen_mask = batch["chosen_attention_mask"]
                rejected_ids = batch["rejected_input_ids"]
                rejected_mask = batch["rejected_attention_mask"]

                # Compute diffusion scores using the LLaDA-specific function
                with torch.cuda.amp.autocast(enabled=args.use_mixed_precision != "no"):
                    # Ensure reference model is in eval mode inside no_grad context
                    reference_model.eval()

                    # Get scores from policy model (requires grad)
                    preferred_score = compute_diffusion_score(
                        model,
                        chosen_ids,
                        chosen_mask,
                        num_samples=args.diffusion_samples,
                        use_gradient_checkpointing=args.gradient_checkpointing,
                    )
                    dispreferred_score = compute_diffusion_score(
                        model,
                        rejected_ids,
                        rejected_mask,
                        num_samples=args.diffusion_samples,
                        use_gradient_checkpointing=args.gradient_checkpointing,
                    )

                    # Get scores from reference model (no grad)
                    with torch.no_grad():
                        ref_preferred_score = compute_diffusion_score(
                            reference_model,
                            chosen_ids,
                            chosen_mask,
                            num_samples=args.diffusion_samples,
                            use_gradient_checkpointing=False,  # No GC needed for ref model
                        )
                        ref_dispreferred_score = compute_diffusion_score(
                            reference_model,
                            rejected_ids,
                            rejected_mask,
                            num_samples=args.diffusion_samples,
                            use_gradient_checkpointing=False,
                        )

                    # Compute DPO loss
                    pi_logratios = preferred_score - dispreferred_score
                    ref_logratios = ref_preferred_score - ref_dispreferred_score

                    logits = pi_logratios - ref_logratios
                    loss = -F.logsigmoid(args.beta * logits).mean()

                # Gather loss across devices
                avg_loss = accelerator.gather(loss).mean()
                total_loss += avg_loss.item()  # Already averaged over batch and devices

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
                    global_step += 1  # Use global_step for logging/saving

                    # Logging
                    if global_step % args.logging_steps == 0:
                        # Loss is already averaged, divide by logging steps
                        avg_loss_log = total_loss / args.logging_steps
                        logger.info(
                            f"Epoch {epoch}, Step {global_step}: Loss = {avg_loss_log:.4f}, LR = {lr_scheduler.get_last_lr()[0]:.6f}"
                        )
                        if accelerator.is_main_process:
                            accelerator.log(
                                {
                                    "train_loss": avg_loss_log,
                                    "lr": lr_scheduler.get_last_lr()[0],
                                },
                                step=global_step,
                            )
                        total_loss = 0  # Reset loss accumulator

                    # Checkpointing
                    if args.checkpointing_steps and isinstance(
                        args.checkpointing_steps, int
                    ):
                        if global_step % args.checkpointing_steps == 0:
                            output_dir_step = f"step_{global_step}"
                            if args.output_dir is not None:
                                output_dir_step = os.path.join(
                                    args.output_dir, output_dir_step
                                )
                            logger.info(f"Saving step checkpoint to {output_dir_step}")
                            accelerator.save_state(output_dir_step)

            if global_step >= args.max_train_steps:
                break  # Exit inner loop

        # Epoch checkpointing
        if args.checkpointing_steps == "epoch":
            output_dir_epoch = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir_epoch = os.path.join(args.output_dir, output_dir_epoch)
            logger.info(f"Saving epoch checkpoint to {output_dir_epoch}")
            accelerator.save_state(output_dir_epoch)

        if global_step >= args.max_train_steps:
            logger.info(
                f"Reached max_train_steps ({args.max_train_steps}). Stopping training."
            )
            break  # Exit outer loop

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
