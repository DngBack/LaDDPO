# Full Script for Diffusion-DPO Training with LLaDA (Text-Only Assumption) on HH-RLHF

# --- Necessary Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_scheduler,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,  # Optional, helps define LoRA task
)
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import GradientAccumulationPlugin
import bitsandbytes.optim as bnb_optim  # For 8-bit optimizer

from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
import math
import os
import logging
import re  # For parsing prompts in dataset


# --- Configuration Class ---
class DiffusionDPOConfig:
    # Model & Tokenizer
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    trust_remote_code: bool = True  # Essential for custom models like LLaDA

    # Dataset
    dataset_name: str = "Anthropic/hh-rlhf"
    # Use a small subset for debugging initially, e.g., "train[:1%]" or "train[:1000]"
    dataset_split: str = "train[:1%]"  # ADJUST FOR FULL RUN!
    max_prompt_length: int = 448  # Max tokens for the prompt section
    max_completion_length: int = 64  # Max tokens for the chosen/rejected completion

    # PEFT (LoRA)
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = [  # Verify these for LLaMA 8B / LLaDA
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    # Quantization (QLoRA)
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"  # Recommended
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16  # Recommended for Ampere+
    use_nested_quant: bool = False  # QLoRA paper recommended False

    # Diffusion Parameters (!!! SPECULATIVE - MUST BE ADAPTED TO LLaDA !!!)
    num_diffusion_timesteps: int = 20  # Keep low due to computational cost
    min_diffusion_steps: int = 5  # Minimum steps if dynamically adjusting
    beta_schedule: str = "linear"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    embedding_dim: int = 4096  # Placeholder - Should be derived from model

    # DPO Parameters
    beta_dpo: float = 0.1  # DPO temperature

    # Training Parameters
    output_dir: str = "./diffusion_dpo_llada_8b_hh_rlhf_text_only"
    epochs: int = 1  # Start with 1 epoch
    per_device_batch_size: int = 1  # Essential for 8B model + diffusion on ~16GB VRAM
    gradient_accumulation_steps: int = (
        16  # Increase to compensate for low batch size (e.g., 16, 32)
    )
    learning_rate: float = 1e-5  # Common starting point for QLoRA
    optimizer_type: str = "AdamW8bit"  # Memory efficient optimizer
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03  # Lower warmup for potentially fewer total steps
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # Gradient clipping
    logging_steps: int = 5
    save_steps: int = 100  # Save more frequently if steps are slow
    use_gradient_checkpointing: bool = True  # Crucial for memory saving
    seed: int = 42

    # Resource Monitoring (Simplified)
    max_memory_gb: float = 15.0  # Target VRAM usage (e.g., for 16GB card)


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Dataset Class (Text-Only HH-RLHF) ---
class HhRlhfPreferenceDataset(Dataset):
    """
    Dataset class for Anthropic/hh-rlhf tailored for DPO (TEXT-ONLY).
    Extracts prompt, chosen completion, rejected completion.
    Handles tokenization and optional embedding retrieval for completions.
    """

    def __init__(
        self,
        dataset_name,
        split,
        tokenizer,
        max_prompt_length,
        max_completion_length,
        embedding_model=None,
        device="cpu",
    ):
        logger.info(f"Loading dataset {dataset_name} split {split}...")
        try:
            self.dataset = load_dataset(
                dataset_name, split=split, trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name} split {split}: {e}")
            raise e
        logger.info(f"Dataset loaded with {len(self.dataset)} samples.")
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.embedding_model = embedding_model
        self.device = device  # Device for embedding calculation during loading

        if self.embedding_model:
            try:
                self.embedding_dim = self.embedding_model.embedding_dim
                self.embedding_model.to(self.device)  # Move embedding layer if provided
                logger.info(
                    f"Using embedding model on device {self.device} with dim {self.embedding_dim}"
                )
            except AttributeError:
                logger.error(
                    "Provided embedding_model does not have embedding_dim attribute!"
                )
                self.embedding_dim = 1  # Fallback dimension
            except Exception as e:
                logger.error(
                    f"Could not move embedding model to device {self.device}: {e}"
                )
                # Proceed without embedding model on target device? Risky.

    def __len__(self):
        return len(self.dataset)

    def _find_last_marker(self, text, marker="\n\nAssistant:"):
        last_pos = text.rfind(marker)
        if last_pos == -1:
            # logger.warning(f"Could not find standard marker '{marker}' in text.")
            return -1
        return last_pos + len(marker)

    def tokenize_and_embed(self, text, max_length):
        if not text:
            input_ids = torch.zeros(max_length, dtype=torch.long)
            attention_mask = torch.zeros(max_length, dtype=torch.long)
            embeddings = None
            if self.embedding_model:
                embeddings = torch.zeros(
                    max_length, self.embedding_dim, device=self.device
                )
            return input_ids, attention_mask, embeddings

        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        input_ids = tokenized["input_ids"].squeeze(0).detach()
        attention_mask = tokenized["attention_mask"].squeeze(0).detach()

        embeddings = None
        if self.embedding_model:
            with torch.no_grad():
                input_ids_dev = input_ids.unsqueeze(0).to(self.device)
                try:
                    embeddings = (
                        self.embedding_model(input_ids_dev).squeeze(0).detach().cpu()
                    )  # Calculate, then move to CPU
                except Exception as e:
                    logger.error(f"Error getting embeddings for text snippet: {e}")
                    embeddings = torch.zeros(
                        max_length, self.embedding_dim
                    )  # Return zeros on CPU

        return input_ids.cpu(), attention_mask.cpu(), embeddings

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            chosen_full = item["chosen"]
            rejected_full = item["rejected"]

            chosen_split_idx = self._find_last_marker(chosen_full)
            rejected_split_idx = self._find_last_marker(rejected_full)

            if chosen_split_idx == -1 or rejected_split_idx == -1:
                # logger.warning(f"Skipping item {idx}: Could not find split marker.")
                return None

            prompt_text = chosen_full[:chosen_split_idx]  # Use chosen prompt
            chosen_completion = chosen_full[chosen_split_idx:].strip()
            rejected_completion = rejected_full[rejected_split_idx:].strip()

            prompt_ids, prompt_mask, _ = self.tokenize_and_embed(
                prompt_text, self.max_prompt_length
            )
            chosen_comp_ids, chosen_comp_mask, chosen_comp_embeddings = (
                self.tokenize_and_embed(chosen_completion, self.max_completion_length)
            )
            rejected_comp_ids, rejected_comp_mask, rejected_comp_embeddings = (
                self.tokenize_and_embed(rejected_completion, self.max_completion_length)
            )

            # Check if embeddings were successfully created
            if self.embedding_model and (
                chosen_comp_embeddings is None or rejected_comp_embeddings is None
            ):
                logger.warning(
                    f"Skipping item {idx}: Failed to get embeddings for completions."
                )
                return None

            output_dict = {
                "prompt_input_ids": prompt_ids,
                "prompt_attention_mask": prompt_mask,
                "preferred_output": chosen_comp_embeddings
                if self.embedding_model
                else chosen_comp_ids,  # Diffuse embeddings or IDs? Assuming embeddings.
                "dispreferred_output": rejected_comp_embeddings
                if self.embedding_model
                else rejected_comp_ids,
                # Keep IDs for reference if needed
                "preferred_completion_ids": chosen_comp_ids,
                "rejected_completion_ids": rejected_comp_ids,
            }
            return output_dict
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            return None  # Skip problematic items


# --- Collate Function ---
def collate_fn(batch):
    batch = [item for item in batch if item is not None]  # Filter out skipped items
    if not batch:
        return None

    keys = batch[0].keys()
    collated = {}
    for k in keys:
        if torch.is_tensor(batch[0][k]):
            try:
                collated[k] = torch.stack([item[k] for item in batch])
            except RuntimeError as e:
                logger.error(f"Error stacking key '{k}': {e}")
                for i, item in enumerate(batch):
                    logger.error(f"Shape item {i} key '{k}': {item[k].shape}")
                return None  # Skip batch if collation fails
        else:
            # This case shouldn't happen with the current dataset structure
            collated[k] = [item[k] for item in batch]
    return collated


# --- Diffusion Utilities (!!! PLACEHOLDERS - ADAPT TO LLaDA !!!) ---
def get_diffusion_schedule(config):
    """Precompute diffusion schedule variables. SPECULATIVE."""
    try:
        betas = (
            torch.linspace(
                config.beta_start**0.5,
                config.beta_end**0.5,
                config.num_diffusion_timesteps,
                dtype=torch.float32,
            )
            ** 2
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-9)
        )  # Add epsilon for stability

        logger.info("Diffusion schedule calculated.")
        return {
            "betas": betas,
            "alphas_cumprod": alphas_cumprod,
            "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
            "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
            "posterior_variance": posterior_variance,
        }
    except Exception as e:
        logger.error(f"Error creating diffusion schedule: {e}")
        raise e


def add_noise(
    original_samples: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: dict,
):
    """Add noise to samples (embeddings). SPECULATIVE."""
    if original_samples is None:
        return None
    sqrt_alphas_cumprod = (
        schedule["sqrt_alphas_cumprod"]
        .to(original_samples.device)[timesteps]
        .view(-1, 1, 1)
    )
    sqrt_one_minus_alphas_cumprod = (
        schedule["sqrt_one_minus_alphas_cumprod"]
        .to(original_samples.device)[timesteps]
        .view(-1, 1, 1)
    )
    return (
        sqrt_alphas_cumprod * original_samples + sqrt_one_minus_alphas_cumprod * noise
    )


def denoise_step(
    model_output: torch.Tensor, timestep: int, sample: torch.Tensor, schedule: dict
):
    """Perform one step of denoising (DDPM style predicting noise). SPECULATIVE."""
    if sample is None:
        return None
    t = timestep
    pred_noise = model_output
    betas_t = schedule["betas"][t].to(sample.device)
    sqrt_one_minus_alphas_cumprod_t = schedule["sqrt_one_minus_alphas_cumprod"][t].to(
        sample.device
    )
    sqrt_recip_alphas_t = torch.sqrt(1.0 / (1.0 - betas_t + 1e-9))

    model_mean = sqrt_recip_alphas_t * (
        sample - betas_t / (sqrt_one_minus_alphas_cumprod_t + 1e-9) * pred_noise
    )

    if t == 0:
        return model_mean
    else:
        posterior_variance_t = schedule["posterior_variance"][t].to(sample.device)
        noise = torch.randn_like(sample)
        return (
            model_mean + torch.sqrt(torch.clamp(posterior_variance_t, min=1e-9)) * noise
        )  # Clamp variance


def compute_step_score(
    predicted_noise: torch.Tensor,
    ideal_noise: torch.Tensor,
    timestep: int,
    schedule: dict,
):
    """Approximate log p_theta(y_t | y_{t+1}, x) score. SPECULATIVE."""
    if predicted_noise is None or ideal_noise is None:
        return 0.0
    # Simplified score: Negative MSE. Assumes lower error = higher probability.
    score = -F.mse_loss(predicted_noise, ideal_noise, reduction="mean")
    return score


# --- Core Diffusion DPO Logic ---
def compute_diffusion_score(
    model: nn.Module,
    inputs: dict,
    target_embeddings: torch.Tensor,
    num_steps: int,
    schedule: dict,
    config: DiffusionDPOConfig,
    accelerator: Accelerator,
):
    """Compute the total diffusion score S_theta(x, y) (TEXT-ONLY). SPECULATIVE."""
    if target_embeddings is None:
        logger.warning("compute_diffusion_score received None target_embeddings.")
        batch_size = inputs["prompt_input_ids"].size(0)
        return torch.full(
            (batch_size,), -1e9, device=accelerator.device
        )  # Return very low score

    device = accelerator.device
    batch_size = target_embeddings.size(0)
    target_embeddings = target_embeddings.to(device)

    # Prepare conditioning inputs (TEXT ONLY)
    model_inputs = {
        "input_ids": inputs["prompt_input_ids"].to(device),
        "attention_mask": inputs["prompt_attention_mask"].to(device),
    }

    # 1. Start with noise derived from target embeddings
    noise_T = torch.randn_like(target_embeddings, device=device)
    t_T = torch.tensor([num_steps - 1] * batch_size, device=device, dtype=torch.long)
    x_T = add_noise(target_embeddings, noise_T, t_T, schedule)
    if x_T is None:  # Handle potential failure in add_noise
        logger.warning("add_noise returned None at T.")
        return torch.full((batch_size,), -1e9, device=device)

    x_t = x_T
    total_score = torch.zeros(batch_size, device=device)

    # 2. Reverse diffusion process
    for t in tqdm(
        reversed(range(num_steps)),
        desc="Diffusion Score",
        leave=False,
        total=num_steps,
        disable=not accelerator.is_main_process,
    ):
        timestep_tensor = torch.tensor(
            [t] * batch_size, device=device, dtype=torch.long
        )

        current_model_input = model_inputs.copy()
        # !!! KEYS HERE ARE SPECULATIVE - Adapt based on LLaDA's forward signature !!!
        current_model_input["input_embeddings"] = x_t  # Assumed key for diffusion state
        current_model_input["time_ids"] = timestep_tensor  # Assumed key for timestep

        try:
            # Ensure requires_grad is set correctly based on context
            is_training = model.training
            with torch.set_grad_enabled(is_training):
                # CRITICAL: Ensure model's forward works without vision inputs
                model_pred = model(**current_model_input)
                # !!! SPECULATIVE: Extract prediction (e.g., noise) from output !!!
                predicted_noise = (
                    model_pred.logits
                )  # Highly likely incorrect, needs adjustment

            # Calculate ideal noise for this step
            sqrt_alphas_cumprod_t = (
                schedule["sqrt_alphas_cumprod"]
                .to(device)[timestep_tensor]
                .view(-1, 1, 1)
            )
            sqrt_one_minus_alphas_cumprod_t = (
                schedule["sqrt_one_minus_alphas_cumprod"]
                .to(device)[timestep_tensor]
                .view(-1, 1, 1)
            )
            ideal_noise = (
                x_t - sqrt_alphas_cumprod_t * target_embeddings
            ) / torch.clamp(sqrt_one_minus_alphas_cumprod_t, min=1e-9)

            # Compute and accumulate step score
            step_score = compute_step_score(predicted_noise, ideal_noise, t, schedule)
            total_score += step_score.detach() if is_training else step_score

            # Denoise for next step (no gradients needed usually)
            if t > 0:
                with torch.no_grad():
                    x_t_prev = denoise_step(predicted_noise.detach(), t, x_t, schedule)
                    if x_t_prev is None:
                        logger.warning(
                            f"Denoise step returned None at t={t}. Aborting score calculation."
                        )
                        return torch.full((batch_size,), -1e9, device=device)
                    x_t = x_t_prev

        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"OOM error during diffusion step {t}. Skipping batch.")
                accelerator.print(
                    f"OOM error during diffusion step {t}. Skipping batch."
                )
                # Clean up CUDA cache if possible
                del (
                    current_model_input,
                    model_pred,
                    predicted_noise,
                    ideal_noise,
                    step_score,
                    x_t,
                )
                torch.cuda.empty_cache()
                return torch.full(
                    (batch_size,), -1e9, device=device
                )  # Penalize heavily
            elif "missing" in str(e) and (
                "pixel_values" in str(e) or "image_features" in str(e)
            ):
                logger.error("*" * 50)
                logger.error(
                    f"MODEL FORWARD PASS FAILED at step {t}: Still requires vision input ('pixel_values'/'image_features')?"
                )
                logger.error(
                    "Even if used text-only, architecture might need placeholder. Check model code."
                )
                logger.error(f"Error: {e}")
                logger.error("*" * 50)
                raise e
            else:
                logger.error(f"Runtime error during diffusion step {t}: {e}")
                raise e
        except Exception as e:
            logger.error(f"Non-runtime error during diffusion step {t}: {e}")
            raise e

    return total_score


# --- Main Training Function ---
def train_diffusion_dpo(config: DiffusionDPOConfig):
    # Accelerator Setup
    gradient_accumulation_plugin = GradientAccumulationPlugin(
        num_steps=config.gradient_accumulation_steps, sync_with_dataloader=False
    )
    accelerator = Accelerator(
        gradient_accumulation_plugin=gradient_accumulation_plugin,
        log_with="tensorboard",  # Or "wandb"
        project_dir=os.path.join(config.output_dir, "logs"),
        mixed_precision="bf16"
        if config.bnb_4bit_compute_dtype == torch.bfloat16
        else "fp16",
    )
    logger.info(
        f"Accelerator initialized. Device: {accelerator.device}, Num Processes: {accelerator.num_processes}, Mixed Precision: {accelerator.mixed_precision}"
    )

    # Seed setting
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    logger.info(f"Set random seed to {config.seed}")

    # Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, trust_remote_code=config.trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad token to EOS token.")
    except Exception as e:
        logger.error(f"Failed to load tokenizer {config.model_name_or_path}: {e}")
        return

    # Load Model (with Quantization)
    logger.info(f"Loading model: {config.model_name_or_path}...")
    quantization_config = None
    if config.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=config.use_nested_quant,
        )
        logger.info(
            f"Using 4-bit quantization: {config.bnb_4bit_quant_type}, compute dtype: {config.bnb_4bit_compute_dtype}"
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct",
            quantization_config=quantization_config,
            trust_remote_code=config.trust_remote_code,
            low_cpu_mem_usage=True,  # Optimization for loading large models
        )
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(
            f"Failed to load model {config.model_name_or_path}. Check path, custom code requirements, and available memory. Error: {e}"
        )
        return  # Exit if model loading fails

    # Get Embedding Layer (for dataset)
    embedding_layer = None
    try:
        embedding_layer = model.get_input_embeddings()
        # Update embedding dim in config based on actual model
        config.embedding_dim = embedding_layer.embedding_dim
        logger.info(
            f"Successfully retrieved input embedding layer. Embedding Dim: {config.embedding_dim}"
        )
    except AttributeError:
        logger.warning(
            "Could not get embedding layer via get_input_embeddings(). Diffusion on embeddings will likely fail."
        )
    except Exception as e:
        logger.error(f"Error getting embedding layer: {e}")

    # Apply PEFT (LoRA)
    if config.use_peft:
        logger.info("Applying PEFT (LoRA)...")
        try:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=config.use_gradient_checkpointing
            )
            peft_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,  # Assuming underlying task is Causal LM
            )
            model = get_peft_model(model, peft_config)
            logger.info("PEFT model created.")
            model.print_trainable_parameters()
        except Exception as e:
            logger.error(f"Failed to apply PEFT: {e}")
            return
    elif config.use_gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled (without PEFT).")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")

    # Load Dataset
    logger.info("Loading and processing dataset...")
    try:
        # Pass embedding layer on CPU to avoid dataloader issues with GPU tensors
        train_dataset = HhRlhfPreferenceDataset(
            config.dataset_name,
            split=config.dataset_split,
            tokenizer=tokenizer,
            max_prompt_length=config.max_prompt_length,
            max_completion_length=config.max_completion_length,
            embedding_model=embedding_layer.to("cpu") if embedding_layer else None,
            device="cpu",  # Embeddings calculated on CPU during loading
        )
        logger.info(f"Dataset processed. Number of samples: {len(train_dataset)}")
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.per_device_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
    except Exception as e:
        logger.error(f"Failed to load or process dataset: {e}")
        return

    # Optimizer
    logger.info(f"Setting up optimizer: {config.optimizer_type}")
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    if config.optimizer_type.lower() == "adamw8bit":
        optimizer = bnb_optim.AdamW8bit(
            trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay
        )
    else:  # Default or other optimizers
        optimizer = torch.optim.AdamW(
            trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay
        )

    # LR Scheduler
    # Calculate steps considering potential filtering by collate_fn (estimate)
    num_effective_samples = len(train_dataset)  # Approximation
    num_update_steps_per_epoch = math.ceil(
        num_effective_samples
        / (
            config.per_device_batch_size
            * config.gradient_accumulation_steps
            * accelerator.num_processes
        )
    )
    num_training_steps = num_update_steps_per_epoch * config.epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)

    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    logger.info(
        f"Optimizer and LR Scheduler configured. Total optimization steps: {num_training_steps}, Warmup steps: {num_warmup_steps}"
    )

    # Diffusion Schedule
    try:
        diffusion_schedule = get_diffusion_schedule(config)
        # Move schedule components to device *if* they are used frequently and don't change
        # Or handle device placement inside the functions that use them
        # for key in diffusion_schedule: diffusion_schedule[key] = diffusion_schedule[key].to(accelerator.device)
    except Exception as e:
        logger.error(f"Failed to get diffusion schedule: {e}")
        return

    # Prepare with Accelerator
    logger.info("Preparing components with Accelerator...")
    try:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )
        logger.info("Accelerator preparation complete.")
    except Exception as e:
        logger.error(f"Accelerator prepare failed: {e}")
        return

    # Reference Model Handling (using PEFT base model)
    # No explicit ref model needed if we use `disable_adapter` context

    # --- Training Loop ---
    logger.info("***** Starting Training Loop *****")
    global_step = 0
    model.train()  # Ensure model is in training mode

    for epoch in range(config.epochs):
        logger.info(f"--- Starting Epoch {epoch + 1}/{config.epochs} ---")
        progress_bar = tqdm(
            total=num_training_steps // config.epochs,  # Steps per epoch
            desc=f"Epoch {epoch + 1}",
            disable=not accelerator.is_main_process,
            position=0,
            leave=True,
        )

        for step, batch in enumerate(train_dataloader):
            if batch is None:  # Skip incomplete batches from collation
                logger.warning(f"Skipping None batch at step {step}.")
                continue

            # Use fixed number of steps for now, add dynamic adjustment later if needed
            current_steps = config.num_diffusion_timesteps

            with accelerator.accumulate(model):
                try:
                    # --- Compute Scores ---
                    unwrapped_model = accelerator.unwrap_model(model)
                    preferred_score = compute_diffusion_score(
                        unwrapped_model,
                        batch,
                        batch["preferred_output"],
                        current_steps,
                        diffusion_schedule,
                        config,
                        accelerator,
                    )
                    dispreferred_score = compute_diffusion_score(
                        unwrapped_model,
                        batch,
                        batch["dispreferred_output"],
                        current_steps,
                        diffusion_schedule,
                        config,
                        accelerator,
                    )

                    # --- Compute Reference Scores ---
                    ref_preferred_score = torch.zeros_like(
                        preferred_score
                    )  # Initialize
                    ref_dispreferred_score = torch.zeros_like(dispreferred_score)
                    with torch.no_grad():
                        base_model = accelerator.unwrap_model(model)
                        if config.use_peft:
                            try:
                                with (
                                    base_model.disable_adapter()
                                ):  # Temporarily use base weights
                                    ref_preferred_score = compute_diffusion_score(
                                        base_model,
                                        batch,
                                        batch["preferred_output"],
                                        current_steps,
                                        diffusion_schedule,
                                        config,
                                        accelerator,
                                    )
                                    ref_dispreferred_score = compute_diffusion_score(
                                        base_model,
                                        batch,
                                        batch["dispreferred_output"],
                                        current_steps,
                                        diffusion_schedule,
                                        config,
                                        accelerator,
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Could not disable adapter for reference score, using current score as approx. Error: {e}"
                                )
                                ref_preferred_score = preferred_score.detach()
                                ref_dispreferred_score = dispreferred_score.detach()
                        else:  # Fallback if not using PEFT
                            ref_preferred_score = preferred_score.detach()
                            ref_dispreferred_score = dispreferred_score.detach()

                    # --- Compute DPO Loss ---
                    logits = (preferred_score - dispreferred_score) - (
                        ref_preferred_score - ref_dispreferred_score
                    )
                    # Clamp logits for stability? Might depend on score range.
                    # logits = torch.clamp(logits, min=-15, max=15)
                    loss = -F.logsigmoid(config.beta_dpo * logits).mean()

                    # Check for NaN/Inf loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.error(
                            f"NaN/Inf loss detected at step {global_step}! Scores: pref={preferred_score.mean().item():.2f}, dispref={dispreferred_score.mean().item():.2f}, ref_pref={ref_preferred_score.mean().item():.2f}, ref_dispref={ref_dispreferred_score.mean().item():.2f}. Skipping update."
                        )
                        optimizer.zero_grad()  # Zero grads even if skipping step
                        continue

                    accelerator.backward(loss)

                    if (
                        accelerator.sync_gradients
                    ):  # Only clip/step when gradients are fully accumulated
                        if config.max_grad_norm is not None:
                            accelerator.clip_grad_norm_(
                                model.parameters(), config.max_grad_norm
                            )

                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                        # Update progress bar and global step only on optimizer step
                        progress_bar.update(1)
                        global_step += 1

                        # --- Logging ---
                        if global_step % config.logging_steps == 0:
                            avg_loss = accelerator.gather(
                                loss.detach()
                            ).mean()  # Gather loss across processes
                            current_lr = lr_scheduler.get_last_lr()[0]
                            log_msg = f"Step: {global_step}, Loss: {avg_loss.item():.4f}, LR: {current_lr:.3e}"
                            if (
                                torch.cuda.is_available()
                                and accelerator.is_main_process
                            ):
                                allocated_mem = torch.cuda.memory_allocated(
                                    accelerator.device
                                ) / (1024**3)
                                reserved_mem = torch.cuda.memory_reserved(
                                    accelerator.device
                                ) / (1024**3)
                                log_msg += f", Mem (Alloc/Res): {allocated_mem:.2f}/{reserved_mem:.2f} GB"
                            logger.info(log_msg)
                            progress_bar.set_postfix(
                                {
                                    "Loss": f"{avg_loss.item():.4f}",
                                    "LR": f"{current_lr:.2e}",
                                }
                            )
                            if accelerator.is_main_process:
                                accelerator.log(
                                    {
                                        "loss": avg_loss.item(),
                                        "learning_rate": current_lr,
                                    },
                                    step=global_step,
                                )

                        # --- Saving Checkpoints ---
                        if global_step % config.save_steps == 0:
                            if accelerator.is_main_process:
                                logger.info(
                                    f"Saving model checkpoint at step {global_step}..."
                                )
                                save_path = os.path.join(
                                    config.output_dir, f"checkpoint-{global_step}"
                                )
                                os.makedirs(save_path, exist_ok=True)
                                unwrapped_model = accelerator.unwrap_model(model)

                                if config.use_peft:
                                    unwrapped_model.save_pretrained(
                                        save_path
                                    )  # Saves LoRA adapters
                                    tokenizer.save_pretrained(save_path)
                                    logger.info(
                                        f"PEFT Adapters and tokenizer saved to {save_path}"
                                    )
                                else:  # Save full model state dict if not using PEFT
                                    accelerator.save_state(
                                        save_path
                                    )  # Saves model, optimizer, etc.
                                    tokenizer.save_pretrained(
                                        save_path
                                    )  # Save tokenizer separately if using save_state
                                    logger.info(
                                        f"Full model state saved to {save_path}"
                                    )
                            accelerator.wait_for_everyone()  # Ensure all processes finish before next step if saving

                except Exception as e:
                    logger.error(
                        f"Error in training step {step} for global step {global_step}: {e}"
                    )
                    # Consider skipping batch or stopping training depending on severity
                    # Clean up memory?
                    torch.cuda.empty_cache()
                    # raise e # Re-raise to potentially stop training

        progress_bar.close()
        logger.info(f"--- Finished Epoch {epoch + 1}/{config.epochs} ---")

    # --- End of Training ---
    logger.info("***** Training Finished *****")
    final_save_path = os.path.join(config.output_dir, "final_model")
    if accelerator.is_main_process:
        logger.info(f"Saving final model to {final_save_path}")
        os.makedirs(final_save_path, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        if config.use_peft:
            unwrapped_model.save_pretrained(final_save_path)
            tokenizer.save_pretrained(final_save_path)
            logger.info(f"Final PEFT Adapters and tokenizer saved to {final_save_path}")
        else:
            accelerator.save_state(final_save_path)
            tokenizer.save_pretrained(final_save_path)
            logger.info(f"Final model state saved to {final_save_path}")
    accelerator.wait_for_everyone()
    logger.info("Script finished.")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Set up environment (e.g., CUDA visible devices) if needed via command line or os.environ

    config = DiffusionDPOConfig()

    # Create output directory if it doesn't exist
    if not os.path.exists(config.output_dir):
        try:
            os.makedirs(config.output_dir, exist_ok=True)
            logger.info(f"Created output directory: {config.output_dir}")
        except OSError as e:
            logger.error(f"Could not create output directory {config.output_dir}: {e}")
            exit(1)

    # Run the training function
    train_diffusion_dpo(config)
