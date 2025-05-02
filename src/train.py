import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from trl import DPOConfig
from trl.trainer.utils import DPODataCollatorWithPadding

from trainer import LaDDPOTrainer


def prepare_dpo_dataset(dataset, tokenizer):
    """Prepare the dataset for DPO training."""

    def process_example(example):
        # Extract prompt from chosen response
        chosen = example["chosen"]
        rejected = example["rejected"]

        # Find common prefix (prompt)
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
            prompt = ""
            chosen_response = chosen
            rejected_response = rejected

        return {
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        }

    # Process the dataset
    processed_dataset = dataset.map(process_example)
    return processed_dataset


def main():
    # Set random seed for reproducibility
    set_seed(42)

    # Load dataset
    raw_dataset = load_dataset("Anthropic/hh-rlhf", split="train")

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

    # Prepare dataset for DPO training
    train_dataset = prepare_dpo_dataset(raw_dataset, tokenizer)

    # DPO configuration
    dpo_config = DPOConfig(
        beta=0.1,
        max_length=1024,
        max_prompt_length=512,
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
        output_dir="./outputs",
    )

    print("🚀 Starting training with LaDDPOTrainer...")

    # Initialize trainer
    trainer = LaDDPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        dpo_config=dpo_config,
        diffusion_steps=20,
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        tokenizer=tokenizer,  # Pass tokenizer directly
    )

    # Start training
    trainer.train()

    # Save the final model
    trainer.save_model("./final_model")


if __name__ == "__main__":
    main()
