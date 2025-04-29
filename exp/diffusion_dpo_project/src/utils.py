"""
Utility functions for Diffusion-DPO implementation.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import logging
import copy
from typing import Dict, List, Tuple, Optional, Union, Any
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

def setup_logger(log_level=logging.INFO, log_file=None):
    """
    Set up the logger for the Diffusion-DPO training.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Path to log file (default: None, logs to console only)
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
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def add_noise(tensor: torch.Tensor, noise_level: float) -> torch.Tensor:
    """
    Add noise to a tensor based on the specified noise level.
    
    Args:
        tensor: Input tensor
        noise_level: Noise level between 0 and 1
        
    Returns:
        Noisy tensor
    """
    noise = torch.randn_like(tensor) * noise_level
    return tensor + noise

def denoise_step(current_state: torch.Tensor, prediction: torch.Tensor, noise_level: float) -> torch.Tensor:
    """
    Perform a single denoising step.
    
    Args:
        current_state: Current noisy state
        prediction: Model prediction
        noise_level: Current noise level
        
    Returns:
        Denoised state for the next step
    """
    # Simple linear interpolation between current state and prediction
    # This is a placeholder - actual implementation would depend on the specific diffusion process
    alpha = 1.0 - noise_level
    return alpha * prediction + (1 - alpha) * current_state

def compute_step_log_prob(prediction: torch.Tensor, target: torch.Tensor, noise_level: float) -> torch.Tensor:
    """
    Compute log probability for a single diffusion step.
    
    Args:
        prediction: Model prediction
        target: Target output
        noise_level: Current noise level
        
    Returns:
        Log probability score for this step
    """
    # This is a simplified version - actual implementation would depend on the specific diffusion process
    # In a real implementation, this would compute the log probability of the transition
    # from the current noisy state to the less noisy state
    
    # For simplicity, we use MSE loss converted to log probability
    mse = F.mse_loss(prediction, target, reduction='none')
    # Scale by noise level to give more weight to steps with less noise
    scaled_mse = mse * (1.0 / (noise_level + 1e-5))
    # Convert to log probability (higher is better)
    log_prob = -scaled_mse.mean()
    
    return log_prob

def create_reference_model(base_model):
    """
    Create a frozen reference model from the base model.
    
    Args:
        base_model: The base model to create a reference from
        
    Returns:
        Frozen reference model
    """
    reference_model = copy.deepcopy(base_model)
    for param in reference_model.parameters():
        param.requires_grad = False
    return reference_model

def compute_diffusion_score(
    model, 
    inputs: torch.Tensor, 
    outputs: torch.Tensor, 
    num_steps: int,
    use_gradient_checkpointing: bool = True,
    checkpoint_interval: int = 5
) -> torch.Tensor:
    """
    Compute the diffusion score for a given model, inputs, and outputs.
    
    Args:
        model: The diffusion model
        inputs: Input prompts
        outputs: Target outputs (preferred or dispreferred)
        num_steps: Number of diffusion steps to use
        use_gradient_checkpointing: Whether to use gradient checkpointing to save memory
        checkpoint_interval: Interval for saving checkpoints when using gradient checkpointing
        
    Returns:
        total_score: The diffusion score S_θ(x, y)
    """
    # Add noise to outputs (start with y_T+1)
    noisy_outputs = add_noise(outputs, noise_level=1.0)
    
    # Initialize total score
    total_score = 0.0
    
    # Store intermediate states for gradient checkpointing
    intermediate_states = []
    
    # Current state
    current_state = noisy_outputs
    
    # Reverse diffusion process (denoising)
    for t in reversed(range(num_steps)):
        # Save state for gradient checkpointing if needed
        if use_gradient_checkpointing and t % checkpoint_interval == 0:
            intermediate_states.append(current_state.detach().clone())
        
        # Calculate noise level for current step
        noise_level = t / num_steps
        
        # Forward pass through model to predict denoised state
        if use_gradient_checkpointing and hasattr(torch.utils.checkpoint, 'checkpoint'):
            # Use gradient checkpointing to save memory
            def create_custom_forward(model, inputs, current_state, noise_level):
                def custom_forward(*args):
                    return model(inputs, current_state, noise_level)
                return custom_forward
            
            pred = torch.utils.checkpoint.checkpoint(
                create_custom_forward(model, inputs, current_state, noise_level),
                *()  # No additional inputs needed
            )
        else:
            # Standard forward pass
            # Note: This is a placeholder - actual implementation would depend on the model's interface
            pred = model(inputs, current_state, noise_level)
        
        # Calculate log probability for this step
        step_score = compute_step_log_prob(pred, outputs, noise_level)
        
        # Add to total score
        total_score += step_score
        
        # Update current state (denoise one step)
        current_state = denoise_step(current_state, pred, noise_level)
    
    return total_score

def load_model_and_tokenizer(model_name_or_path: str, device: str = "cuda"):
    """
    Load model and tokenizer from Hugging Face Hub or local path.
    
    Args:
        model_name_or_path: Model name on Hugging Face Hub or local path
        device: Device to load the model on
        
    Returns:
        model, tokenizer
    """
    logger.info(f"Loading model and tokenizer from {model_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name_or_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    
    model = model.to(device)
    
    return model, tokenizer

def prepare_model_inputs(tokenizer, prompts, responses, max_length=2048):
    """
    Prepare model inputs by tokenizing and combining prompts with responses.
    
    Args:
        tokenizer: Tokenizer to use
        prompts: List of prompt texts
        responses: List of response texts
        max_length: Maximum sequence length
        
    Returns:
        Dictionary of model inputs
    """
    # This is a placeholder - actual implementation would depend on the model's interface
    # and how it handles prompt-response pairs
    
    combined_texts = [p + r for p, r in zip(prompts, responses)]
    
    inputs = tokenizer(
        combined_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return inputs
