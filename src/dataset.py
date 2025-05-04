import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union
from transformers import PreTrainedTokenizer


class PreferenceDataset(Dataset):
    """
    Dataset class for handling preference data in the format required by DPO training.
    """

    def __init__(
        self,
        dataset: Union[Dict, List],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_prompt_length: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length or max_length // 2

        # Process the dataset
        self.examples = self._process_dataset(dataset)

    def _process_dataset(self, dataset: Union[Dict, List]) -> List[Dict]:
        """Process the raw dataset into the format needed for DPO training."""
        processed_examples = []

        for example in dataset:
            # Get chosen and rejected responses
            chosen = example["chosen"]
            rejected = example["rejected"]

            # Tokenize chosen response
            chosen_tokens = self.tokenizer(
                chosen,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

            # Tokenize rejected response
            rejected_tokens = self.tokenizer(
                rejected,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

            # Create example
            processed_example = {
                "chosen": {
                    "input_ids": chosen_tokens["input_ids"][0],
                    "attention_mask": chosen_tokens["attention_mask"][0],
                    "labels": chosen_tokens["input_ids"][0].clone(),
                },
                "rejected": {
                    "input_ids": rejected_tokens["input_ids"][0],
                    "attention_mask": rejected_tokens["attention_mask"][0],
                    "labels": rejected_tokens["input_ids"][0].clone(),
                },
            }

            processed_examples.append(processed_example)

        return processed_examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]
