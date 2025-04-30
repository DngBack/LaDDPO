"""
Simplified Single-GPU Training Script for LaDDPO.
Combines DPO logic with LLaDA diffusion model fine-tuning.
"""

import os
import torch
import torch.nn.functional as F
import argparse
import yaml
import logging
import numpy as np
import math
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import get_scheduler, AutoModel  # Added AutoModel for type hint
from torch.cuda.amp import GradScaler, autocast
import sys
from pathlib import Path

# Import utility functions for simplified LaDDPO
from utils_laddpo_simplified import (
    set_seed,
    load_llada_model_and_tokenizer,
    prepare_hh_rlhf_batch,
    compute_diffusion_score,
)

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LLaDA with Simplified DPO on a single GPU"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/config_laddpo_simplified.yaml",
        help="Path to the simplified config YAML file",
    )
    # Allow overriding key parameters via command line
    parser.add_argument(
        "--model_name_or_path", type=str, help="Override model name/path"
    )
    parser.add_argument("--dataset_name", type=str, help="Override dataset name")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument(
        "--per_device_train_batch_size", type=int, help="Override batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        help="Override gradient accumulation steps",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, help="Override number of epochs"
    )
    parser.add_argument(
        "--load_in_4bit",
        type=lambda x: (str(x).lower() == "true"),
        help="Override 4-bit loading",
    )
    parser.add_argument(
        "--use_peft",
        type=lambda x: (str(x).lower() == "true"),
        help="Override PEFT usage",
    )
    parser.add_argument(
        "--diffusion_samples", type=int, help="Override number of diffusion samples"
    )

    args = parser.parse_args()

    # Load config file first
    config_args = {}
    if args.config_file and os.path.exists(args.config_file):
        logger.info(f"Loading configuration from {args.config_file}")
        with open(args.config_file, "r") as f:
            try:
                config_dict = yaml.safe_load(f)
                config_args = config_dict
            except yaml.YAMLError as exc:
                logger.error(f"Error loading YAML file: {exc}")
    else:
        logger.warning(
            f"Config file {args.config_file} not found. Using defaults and command-line args."
        )

    # Create a namespace object from config, then update with command-line args
    final_args = argparse.Namespace(**config_args)
    cmd_line_args = vars(args)
    for key, value in cmd_line_args.items():
        if (
            key != "config_file" and value is not None
        ):  # Prioritize non-None cmd line args
            logger.info(
                f"Overriding config/default arg '{key}' with command-line value: {value}"
            )
            setattr(final_args, key, value)

    # Ensure boolean flags have correct types after merging
    for key in ["gradient_checkpointing", "load_in_8bit", "load_in_4bit", "use_peft"]:
        val = getattr(final_args, key, False)  # Default to False if not set
        if isinstance(val, str):
            setattr(final_args, key, val.lower() in ("true", "1", "yes"))
        elif val is None:  # Handle case where config might have null
            setattr(final_args, key, False)

    # Check required args
    if not hasattr(final_args, "output_dir") or final_args.output_dir is None:
        raise ValueError(
            "Output directory must be provided (--output_dir or in config)."
        )
    if (
        not hasattr(final_args, "model_name_or_path")
        or final_args.model_name_or_path is None
    ):
        raise ValueError(
            "Model name or path must be provided (--model_name_or_path or in config)."
        )
    if not hasattr(final_args, "dataset_name") or final_args.dataset_name is None:
        raise ValueError("Dataset name must be provided (--dataset_name or in config).")

    return final_args


def prepare_dataloader(args, tokenizer):
    """Load and prepare the HH-RLHF dataset and dataloader for single GPU."""
    logger.info(
        f"Loading preference dataset {args.dataset_name} split {args.dataset_split}"
    )
    try:
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
        # Optional: Select a subset for debugging
        if hasattr(args, "random_sample_size") and args.random_sample_size:
            dataset = dataset.shuffle(seed=args.seed).select(
                range(args.random_sample_size)
            )
            logger.info(f"Using a random subset of {args.random_sample_size} examples.")
        # Filter out potentially problematic examples
        dataset = dataset.filter(
            lambda x: len(x["chosen"]) > 10 and len(x["rejected"]) > 10
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Simple collate function using prepare_hh_rlhf_batch
    def collate_fn(batch_list):
        batch_dict = {
            key: [item[key] for item in batch_list] for key in batch_list[0].keys()
        }
        return prepare_hh_rlhf_batch(batch_dict, tokenizer, args.max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,  # Use multiple workers
        pin_memory=True,  # Pin memory if using GPU
    )
    return dataloader


def main():
    args = parse_args()
    logger.info(f"Starting simplified LaDDPO training with args: {args}")

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model, reference model, and tokenizer
    model, reference_model, tokenizer = load_llada_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        use_peft=args.use_peft,
    )

    # Move models to device (reference model already moved in loader)
    model.to(device)
    reference_model.to(device)  # Ensure ref model is on the correct device
    logger.info(f"Models moved to {device}")

    # Prepare dataset and dataloader
    train_dataloader = prepare_dataloader(args, tokenizer)

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(
        f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}"
    )

    optimizer_grouped_parameters = [
        {"params": trainable_params, "weight_decay": args.weight_decay},
    ]

    optimizer = None
    if args.optimizer_type.lower() == "adamw_8bit":
        try:
            import bitsandbytes.optim as bnb_optim

            optimizer = bnb_optim.AdamW8bit(
                optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.995)
            )
            logger.info("Using 8-bit AdamW optimizer.")
        except ImportError:
            logger.warning("bitsandbytes not found, falling back to standard AdamW.")

    if optimizer is None:
        from torch.optim import AdamW

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        logger.info("Using standard AdamW optimizer.")

    # Calculate total training steps
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

    # LR Scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Gradient Scaling for Mixed Precision
    scaler = GradScaler(enabled=(args.use_mixed_precision != "no"))
    logger.info(
        f"Mixed precision enabled: {args.use_mixed_precision != 'no'} ({args.use_mixed_precision})"
    )

    # Gradient Checkpointing (needs to be enabled *after* potential PEFT wrapping)
    if args.gradient_checkpointing:
        model_to_enable_gc = model.model if args.use_peft else model
        if hasattr(model_to_enable_gc, "gradient_checkpointing_enable"):
            model_to_enable_gc.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")
        else:
            logger.warning(
                "Model does not support gradient_checkpointing_enable method."
            )

    # Training loop
    logger.info("***** Running simplified training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    model.train()  # Set model to train mode

    for epoch in range(args.num_train_epochs):
        logger.info(f"Starting Epoch {epoch + 1}/{args.num_train_epochs}")
        epoch_loss = 0.0
        progress_bar = tqdm(total=num_update_steps_per_epoch, desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Mixed Precision Context
            with autocast(
                enabled=(args.use_mixed_precision != "no"),
                dtype=torch.bfloat16
                if args.use_mixed_precision == "bf16"
                else torch.float16,
            ):
                # Ensure reference model is in eval mode
                reference_model.eval()

                # Get scores from policy model (requires grad)
                preferred_score = compute_diffusion_score(
                    model,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    tokenizer,
                    num_samples=args.diffusion_samples,
                    use_gradient_checkpointing=args.gradient_checkpointing,
                )
                dispreferred_score = compute_diffusion_score(
                    model,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    tokenizer,
                    num_samples=args.diffusion_samples,
                    use_gradient_checkpointing=args.gradient_checkpointing,
                )

                # Get scores from reference model (no grad)
                with torch.no_grad():
                    ref_preferred_score = compute_diffusion_score(
                        reference_model,
                        batch["chosen_input_ids"],
                        batch["chosen_attention_mask"],
                        tokenizer,
                        num_samples=args.diffusion_samples,
                        use_gradient_checkpointing=False,  # No GC for ref model
                    )
                    ref_dispreferred_score = compute_diffusion_score(
                        reference_model,
                        batch["rejected_input_ids"],
                        batch["rejected_attention_mask"],
                        tokenizer,
                        num_samples=args.diffusion_samples,
                        use_gradient_checkpointing=False,
                    )

                # Compute DPO loss
                pi_logratios = preferred_score - dispreferred_score
                ref_logratios = ref_preferred_score - ref_dispreferred_score
                logits = pi_logratios - ref_logratios
                loss = -F.logsigmoid(args.beta * logits).mean()

                # Scale loss for gradient accumulation
                loss = loss / args.gradient_accumulation_steps

            # Backward pass with scaler
            scaler.scale(loss).backward()
            epoch_loss += (
                loss.item() * args.gradient_accumulation_steps
            )  # Log unscaled loss

            # Optimizer step (perform after accumulation)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Unscale gradients before clipping (optional)
                scaler.unscale_(optimizer)
                # Clip gradients (optional)
                # torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

                # Optimizer step
                scaler.step(optimizer)
                # Update scaler for next iteration
                scaler.update()
                # Scheduler step
                lr_scheduler.step()
                # Zero gradients
                optimizer.zero_grad()

                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=loss.item() * args.gradient_accumulation_steps,
                    lr=lr_scheduler.get_last_lr()[0],
                )

                # Logging
                if global_step % args.logging_steps == 0:
                    avg_loss_log = epoch_loss / (
                        step + 1
                    )  # Average loss so far in epoch
                    logger.info(
                        f"Step {global_step}: Loss = {avg_loss_log:.4f}, LR = {lr_scheduler.get_last_lr()[0]:.6e}"
                    )
                    # Add tensorboard logging here if needed

                # Saving (Simplified)
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    os.makedirs(save_path, exist_ok=True)
                    logger.info(f"Saving checkpoint to {save_path}")
                    # Save model state (consider saving PEFT adapters if used)
                    if args.use_peft:
                        model.save_pretrained(save_path)
                    else:
                        torch.save(
                            model.state_dict(),
                            os.path.join(save_path, "pytorch_model.bin"),
                        )
                    # Save optimizer, scheduler, scaler states
                    torch.save(
                        optimizer.state_dict(), os.path.join(save_path, "optimizer.pt")
                    )
                    torch.save(
                        lr_scheduler.state_dict(),
                        os.path.join(save_path, "scheduler.pt"),
                    )
                    torch.save(
                        scaler.state_dict(), os.path.join(save_path, "scaler.pt")
                    )
                    # Save tokenizer
                    tokenizer.save_pretrained(save_path)
                    # Save args
                    with open(os.path.join(save_path, "training_args.yaml"), "w") as f:
                        yaml.dump(vars(args), f)

            if global_step >= args.max_train_steps:
                break  # Exit inner loop

        progress_bar.close()
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}")

        if global_step >= args.max_train_steps:
            logger.info(
                f"Reached max_train_steps ({args.max_train_steps}). Stopping training."
            )
            break  # Exit outer loop

    # Save final model
    logger.info("Saving final model...")
    final_save_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_save_path, exist_ok=True)
    if args.use_peft:
        model.save_pretrained(final_save_path)
        logger.info(f"Saved final PEFT adapter to {final_save_path}")
    else:
        torch.save(
            model.state_dict(), os.path.join(final_save_path, "pytorch_model.bin")
        )
        logger.info(f"Saved final model state_dict to {final_save_path}")
    tokenizer.save_pretrained(final_save_path)
    with open(os.path.join(final_save_path, "training_args.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    logger.info("***** Training finished *****")


if __name__ == "__main__":
    main()
