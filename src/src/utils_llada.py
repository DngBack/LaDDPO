"""
Utility functions for Diffusion-DPO implementation with LLaDA-8B-Instruct model.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import logging
import copy
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

def setup_logger(log_level=logging.INFO, log_file=None):
    """
    Set up the logger for the Diffusion-DPO training.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
        handlers=[logging.StreamHandler()] + 
                 ([logging.FileHandler(log_file)] if log_file else [])
    )
    
def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def forward_process(batch, prompt_index, mask_id=126336):
    """
    Implements the forward process (noising) for LLaDA models.
    This is based on the implementation in get_log_likelihood.py from the LLaDA repository.
    
    Args:
        batch: Input token ids of shape (batch_size, seq_len)
        prompt_index: Boolean tensor indicating which tokens are part of the prompt (not to be masked)
        mask_id: The token ID for [MASK], default is 126336 for LLaDA
        
    Returns:
        noisy_batch: The noised input with some tokens replaced by mask_id
        p_mask: The mask ratio for each position
    """
    b, l = batch.shape
    
    # Calculate target length (excluding prompt)
    target_len = (l - prompt_index.sum()).item()
    
    # Sample number of tokens to mask for each sequence in batch
    k = torch.randint(1, target_len + 1, (), device=batch.device)
    
    # Create a sequence of mask counts that varies across the batch
    x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
    x = ((x - 1) % target_len) + 1
    
    # Ensure valid mask counts
    assert x.min() >= 1 and x.max() <= target_len
    
    # Create indices for masking
    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    
    # Randomly permute mask positions for each sequence
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]
    
    # Add zeros for prompt positions (don't mask prompt)
    is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
    
    # Apply masking
    noisy_batch = torch.where(is_mask, mask_id, batch)
    
    # Calculate mask ratio for each position
    p_mask = (x / target_len).unsqueeze(1).repeat(1, l)
    
    return noisy_batch, p_mask, is_mask

def compute_step_log_prob(logits: torch.Tensor, targets: torch.Tensor, mask_indices: torch.Tensor, p_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute log probability based on LLaDA loss calculation.
    
    Args:
        logits: Model output logits (B, L, V)
        targets: Original target token ids (B, L)
        mask_indices: Boolean mask indicating which tokens were masked (B, L)
        p_mask: Probability of masking used for this batch (B, L)
        
    Returns:
        Log probability score (scalar tensor)
    """
    # Calculate cross-entropy loss only for masked tokens
    loss_func = torch.nn.CrossEntropyLoss(reduction="none")
    
    # Flatten logits and targets for masked positions
    masked_logits = logits[mask_indices]  # (N, V)
    masked_targets = targets[mask_indices]  # (N,)
    masked_p_mask = p_mask[mask_indices]  # (N,)
    
    if masked_logits.numel() == 0:  # Handle case where no tokens are masked
        return torch.tensor(0.0, device=logits.device)
        
    # Calculate loss per token
    per_token_loss = loss_func(masked_logits, masked_targets)
    
    # Normalize by masking probability as in LLaDA
    normalized_loss = per_token_loss / masked_p_mask
    
    # Average loss over the sequence length and batch
    batch_avg_loss = normalized_loss.sum() / (targets.shape[0] * targets.shape[1]) 
    
    # Convert average loss to log probability (higher is better)
    log_prob = -batch_avg_loss 
    
    return log_prob

def create_reference_model(base_model):
    """
    Create a frozen reference model from the base model.
    """
    reference_model = copy.deepcopy(base_model)
    for param in reference_model.parameters():
        param.requires_grad = False
    return reference_model

def compute_diffusion_score(
    model, 
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor = None,
    num_samples: int = 16,
    use_gradient_checkpointing: bool = True,
    mask_id: int = 126336
) -> torch.Tensor:
    """
    Compute the diffusion score S_θ(x, y) based on LLaDA's process.
    This involves running the forward process (noising) and then calculating
    the model's ability to predict the original tokens from the noise.
    
    Args:
        model: The LLaDA model.
        input_ids: The input sequence (B, L).
        attention_mask: Attention mask for the input_ids (B, L).
        num_samples: Number of Monte Carlo samples to use for estimating the score.
        use_gradient_checkpointing: Whether to use gradient checkpointing.
        mask_id: The token ID for [MASK], default is 126336 for LLaDA.
        
    Returns:
        total_score: The diffusion score (scalar tensor).
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    
    # Create prompt_index tensor (tokens that should not be masked)
    # For DPO, we typically want to mask only the response part, not the prompt
    # For simplicity, we'll assume the first token is always a special token (like BOS)
    # and should not be masked. In a real implementation, you'd need to identify
    # the actual prompt/response boundary.
    prompt_index = torch.zeros_like(input_ids, dtype=torch.bool)
    prompt_index[:, 0] = True  # Don't mask the first token
    
    total_log_prob = 0.0
    
    # Run multiple forward passes with different noise samples
    for _ in range(num_samples):
        # 1. Apply forward process (noising)
        noisy_input, p_mask, mask_indices = forward_process(input_ids, prompt_index, mask_id)
        
        # 2. Get model prediction (logits)
        if use_gradient_checkpointing and hasattr(model, "_gradient_checkpointing"):
            # Enable gradient checkpointing if available
            model._gradient_checkpointing = True
            
        # Forward pass through the model
        outputs = model(noisy_input)
        logits = outputs.logits
        
        # 3. Compute log probability for this sample
        step_log_prob = compute_step_log_prob(logits, input_ids, mask_indices, p_mask)
        
        total_log_prob += step_log_prob
    
    # Average log probability over samples
    avg_log_prob = total_log_prob / num_samples
    
    # Disable gradient checkpointing if it was enabled
    if use_gradient_checkpointing and hasattr(model, "_gradient_checkpointing"):
        model._gradient_checkpointing = False
        
    return avg_log_prob

def load_llada_model_and_tokenizer(model_name_or_path: str, device: str = "cuda", load_in_8bit: bool = False, load_in_4bit: bool = False):
    """
    Load LLaDA model and tokenizer.
    
    Args:
        model_name_or_path: Model name or path (e.g., 'GSAI-ML/LLaDA-8B-Instruct')
        device: Device to load the model on
        load_in_8bit: Whether to load the model in 8-bit precision
        load_in_4bit: Whether to load the model in 4-bit precision
        
    Returns:
        model: The loaded model
        tokenizer: The loaded tokenizer
    """
    logger.info(f"Loading LLaDA model from {model_name_or_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    # Determine quantization settings
    if load_in_8bit and load_in_4bit:
        raise ValueError("Cannot load model in both 8-bit and 4-bit precision")
    
    quantization_config = None
    torch_dtype = torch.bfloat16  # Default to bfloat16 as used in the original code
    
    if load_in_8bit or load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info(f"Using quantization: 8-bit={load_in_8bit}, 4-bit={load_in_4bit}")
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to full precision")
            quantization_config = None
    
    # Load model
    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map="auto" if (load_in_8bit or load_in_4bit) else None
    )
    
    # Move model to device if not using device_map="auto"
    if not (load_in_8bit or load_in_4bit):
        model = model.to(device)
    
    logger.info(f"Model loaded successfully")
    return model, tokenizer

def prepare_model_inputs(tokenizer, prompts: List[str], responses: List[str] = None, max_length: int = 2048, device: str = "cuda"):
    """
    Prepare model inputs by tokenizing prompts and responses.
    
    Args:
        tokenizer: The tokenizer to use
        prompts: List of prompt strings
        responses: Optional list of response strings. If provided, will be appended to prompts.
        max_length: Maximum sequence length
        device: Device to put tensors on
        
    Returns:
        Dictionary with input_ids and attention_mask
    """
    if responses is not None:
        # For chosen/rejected responses in DPO, format as chat and append response
        formatted_inputs = []
        for prompt, response in zip(prompts, responses):
            # Format as chat message
            messages = [{"role": "user", "content": prompt}]
            # Apply chat template but don't tokenize yet
            formatted_text = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=False
            )
            # Append response (which would come after the generation prompt)
            formatted_inputs.append(formatted_text + response)
    else:
        # For prompt-only inputs (e.g., for generation)
        formatted_inputs = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted_text = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=False
            )
            formatted_inputs.append(formatted_text)
    
    # Tokenize
    inputs = tokenizer(
        formatted_inputs,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    return inputs

def prepare_hh_rlhf_batch(batch, tokenizer, max_length: int = 2048, device: str = "cuda"):
    """
    Prepare a batch from the Anthropic/hh-rlhf dataset for DPO training.
    
    Args:
        batch: A batch from the dataset with 'chosen' and 'rejected' fields
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        device: Device to put tensors on
        
    Returns:
        Dictionary with chosen and rejected inputs
    """
    # Extract prompts from chosen responses
    # In hh-rlhf, the chosen and rejected texts both start with the same prompt
    # We need to extract the prompt from each chosen text
    
    prompts = []
    chosen_responses = []
    rejected_responses = []
    
    for chosen, rejected in zip(batch['chosen'], batch['rejected']):
        # Find the common prefix (prompt) between chosen and rejected
        # This is a simple approach - in practice, you might need more robust methods
        min_len = min(len(chosen), len(rejected))
        prompt_end = 0
        for i in range(min_len):
            if chosen[i] != rejected[i]:
                prompt_end = i
                break
        
        if prompt_end > 0:
            prompt = chosen[:prompt_end]
            chosen_response = chosen[prompt_end:]
            rejected_response = rejected[prompt_end:]
        else:
            # Fallback if no common prefix found
            # This is a simplification - real implementation would need better handling
            prompt = ""
            chosen_response = chosen
            rejected_response = rejected
        
        prompts.append(prompt)
        chosen_responses.append(chosen_response)
        rejected_responses.append(rejected_response)
    
    # Prepare inputs for chosen responses
    chosen_inputs = prepare_model_inputs(
        tokenizer, prompts, chosen_responses, max_length, device
    )
    
    # Prepare inputs for rejected responses
    rejected_inputs = prepare_model_inputs(
        tokenizer, prompts, rejected_responses, max_length, device
    )
    
    return {
        "chosen_input_ids": chosen_inputs["input_ids"],
        "chosen_attention_mask": chosen_inputs["attention_mask"],
        "rejected_input_ids": rejected_inputs["input_ids"],
        "rejected_attention_mask": rejected_inputs["attention_mask"],
    }
