import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

from src_ref.dataset import PreferenceDataset
from src_ref.collator import DiffusionPreferenceCollator
from src_ref.trainer import DiffusionDPOTrainer

# Load Anthropic hh-rlhf pref dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# LoRA configuration for lightweight finetuning
lora_cfg = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn", "q_proj", "v_proj"],
)
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
model = get_peft_model(base_model, lora_cfg)

# Prepare dataset and collator
train_ds = PreferenceDataset(dataset, tokenizer, max_length=512)
collator = DiffusionPreferenceCollator(tokenizer=tokenizer)

# Training arguments
args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    max_steps=1000,
    logging_steps=10,
    fp16=True,
    save_total_limit=2,
)

# Initialize custom Trainer and start training
trainer = DiffusionDPOTrainer(
    model=model, args=args, train_dataset=train_ds, data_collator=collator
)
trainer.train()
