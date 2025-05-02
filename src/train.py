import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import DPOConfig

from src.trainer import LaDDPOTrainer
from src.dataset import PreferenceDataset
from src.collator import DiffusionPreferenceCollator


def main():
    # Set random seed for reproducibility
    set_seed(42)

    # Load dataset
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        padding_side="right",
        trust_remote_code=True,
        use_fast=True,
    )

    # Configure LoRA
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Apply LoRA
    model = get_peft_model(base_model, lora_config)

    # Prepare dataset
    train_dataset = PreferenceDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=1024,
        max_prompt_length=512,
    )

    # Prepare collator
    collator = DiffusionPreferenceCollator(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        learning_rate=5e-6,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        # evaluation_strategy="steps",
        # eval_steps=100,
        # load_best_model_at_end=True,
        # evaluation_strategy="steps",
        # save_strategy="steps",
        # eval_steps=100,
        # save_steps=100,
        report_to="wandb",  # Optional: for experiment tracking
    )

    # DPO configuration
    dpo_config = DPOConfig(
        beta=0.1,
        max_length=1024,
        max_prompt_length=512,
    )

    # Initialize trainer
    trainer = LaDDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        dpo_config=dpo_config,
        diffusion_steps=20,
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
    )

    # Start training
    trainer.train()

    # Save the final model
    trainer.save_model("./final_model")


if __name__ == "__main__":
    main()
