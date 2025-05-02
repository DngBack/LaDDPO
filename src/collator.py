from typing import Dict, List, Optional, Union
import torch
from transformers import PreTrainedTokenizer
from transformers.data.data_collator import DataCollatorMixin


class DiffusionPreferenceCollator(DataCollatorMixin):
    """
    Custom data collator for batching preference data in DPO training.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        padding: bool = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def torch_call(self, examples: List[Dict]) -> Dict:
        """
        Collate the examples into a batch.
        """
        # Separate chosen and rejected examples
        chosen_examples = [example["chosen"] for example in examples]
        rejected_examples = [example["rejected"] for example in examples]

        # Collate chosen examples
        chosen_batch = self._collate_examples(chosen_examples)

        # Collate rejected examples
        rejected_batch = self._collate_examples(rejected_examples)

        return {
            "chosen": chosen_batch,
            "rejected": rejected_batch,
        }

    def _collate_examples(self, examples: List[Dict]) -> Dict:
        """
        Collate a list of examples into a batch.
        """
        # Get the keys from the first example
        keys = examples[0].keys()

        # Initialize the batch
        batch = {key: [] for key in keys}

        # Add each example to the batch
        for example in examples:
            for key in keys:
                batch[key].append(example[key])

        # Stack the tensors
        for key in keys:
            batch[key] = torch.stack(batch[key])

        return batch
