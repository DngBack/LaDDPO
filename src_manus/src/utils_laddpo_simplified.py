"""
Utility functions for Simplified Single-GPU LaDDPO.
"""

import torch
import torch.nn.functional as F
import logging
import random
import os
import numpy as np
import copy
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Potential slowdown, uncomment if absolute reproducibility is needed
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    logger.info(f"Set seed to {seed}")


def load_llada_model_and_tokenizer(
    model_name_or_path: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    use_peft: bool = True,
):
    """
    Loads the LLaDA model and tokenizer, optionally applies quantization and PEFT.
    Also creates and returns the reference model (quantized copy before PEFT).
    """
    logger.info(f"Loading model: {model_name_or_path}")

    quantization_config = None
    torch_dtype = torch.bfloat16  # Default to bfloat16 for modern GPUs

    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,  # Compute in bfloat16
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",  # Use NF4
        )
        logger.info("Using 4-bit quantization.")
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("Using 8-bit quantization.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=True
    )
    # LLaDA might require specific padding side or tokens, check documentation if issues arise
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Common practice
        logger.info("Set pad_token to eos_token")

    # Load base model
    model = AutoModel.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        trust_remote_code=True,
        torch_dtype=torch_dtype
        if not (load_in_8bit or load_in_4bit)
        else None,  # Dtype handled by BNB if quantized
        # device_map="auto" # Avoid device_map for single GPU simplicity, handle manually
    )

    # --- Create Reference Model ---
    # Create a deep copy *before* applying PEFT adapters
    # Ensure it's on CPU for deepcopy if quantized, then move to GPU
    model_device = next(
        model.parameters()
    ).device  # Get current device (might be meta if quantized)
    model.to("cpu")  # Move to CPU for safe deepcopy
    reference_model = copy.deepcopy(model)
    model.to(model_device)  # Move policy model back
    reference_model.to(model_device)  # Move reference model to the same device
    reference_model.eval()  # Set reference model to eval mode
    logger.info("Created reference model (deep copy before PEFT).")
    # -----------------------------

    if use_peft:
        logger.info("Applying PEFT (LoRA)...")
        # LoRA configuration (adjust as needed)
        lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
            ],  # Common targets for Llama-like models
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        logger.info("PEFT applied. Trainable parameters:")
        model.print_trainable_parameters()

    logger.info("Model and tokenizer loaded successfully.")
    return model, reference_model, tokenizer


def prepare_hh_rlhf_batch(batch, tokenizer, max_length: int):
    """
    Prepares a batch from the Anthropic/hh-rlhf dataset for DPO.
    Tokenizes chosen and rejected responses.
    """
    chosen_texts = []
    rejected_texts = []

    # Format based on how hh-rlhf structures conversations
    # Typically: "\n\nHuman: {prompt}\n\nAssistant: {response}"
    for chosen, rejected in zip(batch["chosen"], batch["rejected"]):
        # Find the last occurrence of "\n\nAssistant:"
        prompt_marker = "\n\nAssistant:"
        chosen_prompt_end = chosen.rfind(prompt_marker)
        rejected_prompt_end = rejected.rfind(prompt_marker)

        if (
            chosen_prompt_end == -1
            or rejected_prompt_end == -1
            or chosen[:chosen_prompt_end] != rejected[:rejected_prompt_end]
        ):
            logger.warning("Prompt mismatch or format error in batch item. Skipping.")
            # Handle skip? For now, just use the full text, might lead to issues.
            prompt = ""  # Or some default / log error
            chosen_resp = chosen
            rejected_resp = rejected
        else:
            prompt = chosen[
                : chosen_prompt_end + len(prompt_marker)
            ]  # Include the marker
            chosen_resp = chosen[chosen_prompt_end + len(prompt_marker) :]
            rejected_resp = rejected[rejected_prompt_end + len(prompt_marker) :]

        chosen_texts.append(prompt + chosen_resp)
        rejected_texts.append(prompt + rejected_resp)

    # Tokenize
    tokenized_chosen = tokenizer(
        chosen_texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    tokenized_rejected = tokenizer(
        rejected_texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return {
        "chosen_input_ids": tokenized_chosen["input_ids"],
        "chosen_attention_mask": tokenized_chosen["attention_mask"],
        "rejected_input_ids": tokenized_rejected["input_ids"],
        "rejected_attention_mask": tokenized_rejected["attention_mask"],
    }


def compute_diffusion_score(
    model,
    input_ids,
    attention_mask,
    tokenizer,
    num_samples=8,
    use_gradient_checkpointing=False,
):
    """
    Computes the diffusion score (negative log likelihood estimate) for LLaDA.
    Adapted from the user's previous utils_llada.py.
    Assumes model has a `compute_loss` method compatible with LLaDA's diffusion objective.
    """
    B, N = input_ids.shape
    device = input_ids.device
    total_score = torch.zeros(B, device=device)

    # Ensure model has the necessary method
    if not hasattr(model, "compute_loss") or not callable(
        getattr(model, "compute_loss")
    ):
        raise AttributeError(
            "Model must have a callable 'compute_loss' method for diffusion score calculation."
        )

    # Enable gradient checkpointing on the base model if requested and applicable
    original_gc_state = None
    model_to_check = model.model if hasattr(model, "peft_config") else model
    if use_gradient_checkpointing and hasattr(
        model_to_check, "is_gradient_checkpointing"
    ):
        original_gc_state = model_to_check.is_gradient_checkpointing
        if not original_gc_state:
            if hasattr(model_to_check, "gradient_checkpointing_enable"):
                model_to_check.gradient_checkpointing_enable()
            else:
                logger.warning(
                    "Attempted to enable GC for diffusion score, but method not found."
                )

    for _ in range(num_samples):
        # Simulate the forward (noising) process - simplified version
        # This needs to match the *exact* noising process LLaDA uses during training.
        # Assuming a simple uniform masking schedule for demonstration.
        # *** IMPORTANT: Replace this with LLaDA's actual noising schedule ***
        t = torch.rand(B, 1, device=device)  # Sample time step
        noise_mask = torch.rand(B, N, device=device) < t
        masked_input_ids = input_ids.where(
            ~noise_mask, tokenizer.mask_token_id
        )  # Use tokenizer's mask token
        labels = input_ids.where(
            noise_mask, -100
        )  # Labels are original tokens where noise was applied

        # Prepare inputs for LLaDA's compute_loss (adjust based on actual signature)
        model_inputs = {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            # LLaDA might need 't' or other specific inputs for its loss
            # "t": t.squeeze(1), # Example if 't' is needed
            # "labels": labels # Assuming compute_loss takes labels
        }

        # Call LLaDA's compute_loss method
        # We need the *negative* log likelihood, LLaDA's loss might be this directly or need adjustment
        # Assuming compute_loss returns the loss value per sequence
        try:
            # LLaDA's compute_loss might need labels passed separately
            loss_outputs = model.compute_loss(labels=labels, **model_inputs)
            # Check if compute_loss returns a dict or just the loss tensor
            if isinstance(loss_outputs, dict):
                neg_log_likelihood = loss_outputs[
                    "loss"
                ]  # Assuming 'loss' key holds the value
            else:
                neg_log_likelihood = loss_outputs

            # Ensure it's per-sequence loss, if not, average appropriately
            if neg_log_likelihood.ndim > 1:
                # Example: Average over sequence length if loss is per token
                label_mask = labels != -100
                neg_log_likelihood = (neg_log_likelihood * label_mask).sum(
                    dim=1
                ) / label_mask.sum(dim=1).clamp(min=1)

            total_score += neg_log_likelihood

        except Exception as e:
            logger.error(f"Error during model.compute_loss call: {e}")
            # Handle error, maybe return NaN or raise
            total_score += float("nan")  # Indicate failure
            break  # Stop sampling for this batch item if error occurs

    # Restore original gradient checkpointing state
    if (
        use_gradient_checkpointing
        and original_gc_state is not None
        and not original_gc_state
    ):
        if hasattr(model_to_check, "gradient_checkpointing_disable"):
            model_to_check.gradient_checkpointing_disable()
        else:
            logger.warning(
                "Attempted to disable GC after diffusion score, but method not found."
            )

    # Average score over samples
    avg_score = total_score / num_samples
    return avg_score
