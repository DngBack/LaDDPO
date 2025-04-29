"""
Utility functions for Diffusion-DPO implementation.
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
from transformers import AutoTokenizer # Keep AutoTokenizer for now, might need specific tokenizer later

# Add SMDM repo to path to import its modules
smdm_repo_path = "/home/ubuntu/SMDM_repo"
sys.path.insert(0, smdm_repo_path)

try:
    # Attempt to import from the cloned SMDM repository
    from lit_gpt.diffmodel import TransEncoder, Config
    from lit_gpt.utils import lazy_load
except ImportError as e:
    print(f"Error importing from SMDM repo: {e}")
    print("Ensure the SMDM repository is cloned at /home/ubuntu/SMDM_repo and contains lit_gpt module.")
    # Define dummy classes if import fails to allow script parsing
    class Config:
        @staticmethod
        def from_name(name):
            print(f"Warning: Using dummy Config for {name}")
            return type("DummyConfig", (), {"block_size": 512, "vocab_size": 32000, "n_layer": 12, "n_head": 12, "n_embd": 768})()
    class TransEncoder:
        def __init__(self, config):
            print("Warning: Using dummy TransEncoder")
            self.config = config
            self.dummy_param = torch.nn.Parameter(torch.randn(1))
        def forward(self, *args, **kwargs):
            print("Warning: Dummy TransEncoder forward called")
            # Return dummy output matching expected shape (batch, seq_len, vocab_size)
            # Get batch size and seq len from input args if possible
            batch_size = 1
            seq_len = self.config.block_size
            if args:
                if isinstance(args[0], torch.Tensor):
                    batch_size = args[0].shape[0]
                    seq_len = args[0].shape[1]
            return torch.randn(batch_size, seq_len, self.config.vocab_size).to(self.dummy_param.device)
        def apply(self, fn):
            pass # Dummy apply
        def parameters(self):
             return iter([self.dummy_param]) # Return dummy parameter
        def named_parameters(self):
             return iter([("dummy_param", self.dummy_param)])
        def gradient_checkpointing_enable(self):
            print("Warning: Dummy gradient_checkpointing_enable called")
            pass
        def to(self, device):
            self.dummy_param = self.dummy_param.to(device)
            return self
        def eval(self):
            pass
        def train(self):
            pass

    def lazy_load(path):
        print(f"Warning: Using dummy lazy_load for {path}")
        # Return a dummy state dict
        return {"dummy_state": torch.randn(1)}

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
    
def forward_process(batch, total_dim=32000, eps=1e-3):
    """Mimics the forward process from SMDM train script."""
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    noisy_batch = torch.where(mask_indices, total_dim, batch) # Use vocab_size as mask token
    return noisy_batch, mask_indices, p_mask, t

def compute_step_log_prob(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, p_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute log probability based on SMDM loss calculation.
    Args:
        logits: Model output logits (B, L, V)
        targets: Original target token ids (B, L)
        mask: Boolean mask indicating which tokens were masked (B, L)
        p_mask: Probability of masking used for this batch (B, L)
    Returns:
        Log probability score (scalar tensor)
    """
    # Calculate cross-entropy loss only for masked tokens
    loss_func = torch.nn.CrossEntropyLoss(reduction="none")
    # Flatten logits and targets for masked positions
    masked_logits = logits[mask] # (N, V)
    masked_targets = targets[mask] # (N,)
    masked_p_mask = p_mask[mask] # (N,)
    
    if masked_logits.numel() == 0: # Handle case where no tokens are masked (e.g., t=0)
        return torch.tensor(0.0, device=logits.device)
        
    # Calculate loss per token
    per_token_loss = loss_func(masked_logits, masked_targets)
    
    # Normalize by masking probability as in SMDM
    normalized_loss = per_token_loss / masked_p_mask
    
    # Average loss over the sequence length and batch
    # SMDM uses sum() / (B * L), effectively averaging over all positions
    # We average over masked positions for a per-sequence score
    batch_avg_loss = normalized_loss.sum() / (targets.shape[0] * targets.shape[1]) 
    
    # Convert average loss to log probability (higher is better)
    # This assumes a Gaussian-like distribution where lower loss = higher prob
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
    model: TransEncoder, 
    input_ids: torch.Tensor, # Original input sequence (B, L)
    attention_mask: torch.Tensor, # Not directly used by SMDM TransEncoder?
    num_steps: int,
    use_gradient_checkpointing: bool = True,
    # checkpoint_interval: int = 5 # SMDM doesn't seem to use intermediate checkpointing in its loss
) -> torch.Tensor:
    """
    Compute the diffusion score S_θ(x, y) based on SMDM's process.
    This involves running the forward process (noising) and then calculating
    the model's ability to predict the original tokens from the noise.
    For DPO, y is the sequence (chosen or rejected), x is the prompt (implicitly part of y).
    
    Args:
        model: The SMDM TransEncoder model.
        input_ids: The target sequence (chosen or rejected) (B, L).
        attention_mask: Attention mask for the input_ids (B, L).
        num_steps: Number of diffusion steps to simulate (controls t sampling).
                   Note: SMDM samples t continuously, not discrete steps.
                   We simulate by sampling t uniformly and calculating loss.
                   For DPO, we need a consistent way to compare scores.
                   Let's average the loss over multiple sampled t values.
        use_gradient_checkpointing: Whether to use gradient checkpointing.
        
    Returns:
        total_score: The diffusion score (scalar tensor).
    """
    model.eval() # Ensure model is in eval mode for score computation
    total_log_prob = 0.0
    num_samples = num_steps # Use num_steps as the number of t samples

    # The SMDM forward pass takes noisy input and predicts logits for original tokens.
    # The loss is calculated based on a randomly sampled noise level 't'.
    # To get a score for DPO, we average the negative loss (log prob) over several 't' samples.
    
    for _ in range(num_samples):
        # 1. Apply forward process (noising) based on a random t
        # Use the vocab_size from the model config as the mask token id
        mask_token_id = model.config.vocab_size
        noisy_input, mask_indices, p_mask, t = forward_process(input_ids, total_dim=mask_token_id)
        
        # 2. Get model prediction (logits)
        if use_gradient_checkpointing and hasattr(torch.utils.checkpoint, 'checkpoint'):
            # Define a closure for checkpointing
            def create_custom_forward(model, noisy_input):
                def custom_forward(*inputs):
                    return model(inputs[0]) # SMDM model takes only noisy input
                return custom_forward
            
            logits = torch.utils.checkpoint.checkpoint(
                create_custom_forward(model, noisy_input),
                noisy_input,
                use_reentrant=False # Recommended for newer PyTorch versions
            )
        else:
            logits = model(noisy_input) # (B, L, V)
        
        # 3. Compute log probability for this step/sample
        # Use the loss calculation logic from SMDM
        step_log_prob = compute_step_log_prob(logits, input_ids, mask_indices, p_mask)
        
        total_log_prob += step_log_prob

    # Average log probability over samples
    avg_log_prob = total_log_prob / num_samples
    model.train() # Return model to train mode if it was in train mode before
    return avg_log_prob

def load_smdm_model_and_tokenizer(model_size_ M: int, checkpoint_path: str, device: str = "cuda"):
    """
    Load SMDM model architecture and a specific checkpoint.
    Loads a generic tokenizer for now.
    """
    logger.info(f"Loading SMDM {model_size_ M}M model architecture")
    # Construct model name based on size for config lookup
    model_name = f"Diff_LLaMA_{model_size_ M}M"
    config = Config.from_name(model_name)
    
    # Load model with empty weights
    with torch.device("meta"):
         model = TransEncoder(config)

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    # Load checkpoint using lazy_load from SMDM utils
    checkpoint = lazy_load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    
    model = model.to(device)
    
    # Load a generic tokenizer (e.g., Llama tokenizer) - replace if SMDM needs a specific one
    # Using a placeholder tokenizer path - replace with actual if needed
    tokenizer_path = "hf-internal-testing/llama-tokenizer" 
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # Set pad token if not present (common for Llama tokenizers)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        logger.warning("Using basic whitespace tokenizer as fallback.")
        # Fallback tokenizer
        tokenizer = lambda text, **kwargs: {"input_ids": [[ord(c) for c in t] for t in text], "attention_mask": [[1]*len(t) for t in text]}
        tokenizer.pad_token_id = 0
        tokenizer.decode = lambda ids: "".join([chr(i) for i in ids])
        
    logger.info(f"SMDM Model {model_size_ M}M and tokenizer loaded.")
    return model, tokenizer, config

def prepare_model_inputs(tokenizer, prompts: List[str], responses: List[str], max_length: int, device: str):
    """
    Prepare model inputs by tokenizing combined prompt + response.
    SMDM seems trained on sequences, so we combine prompt and response.
    """
    # Combine prompt and response
    combined_texts = [p + r for p, r in zip(prompts, responses)]
    
    # Tokenize combined text
    inputs = tokenizer(
        combined_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Return input_ids and attention_mask, moved to the correct device
    return {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device)
    }

