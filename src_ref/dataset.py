import torch
from torch.utils.data import Dataset


class PreferenceDataset(Dataset):
    """
    Dataset for Anthropic hh-rlhf preference data.
    Each item returns tokenized sequences for "chosen" and "rejected" responses.
    """

    def __init__(self, hf_dataset, tokenizer, max_length):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        # Prepare prompt and responses without reasoning template
        prompt = ex["prompt"]
        chosen = self.tokenizer(
            prompt + ex["chosen"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        rejected = self.tokenizer(
            prompt + ex["rejected"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids_chosen": chosen.input_ids.squeeze(0),
            "attention_mask_chosen": chosen.attention_mask.squeeze(0),
            "input_ids_rejected": rejected.input_ids.squeeze(0),
            "attention_mask_rejected": rejected.attention_mask.squeeze(0),
        }
