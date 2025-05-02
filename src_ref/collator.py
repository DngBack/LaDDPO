import torch
from transformers import DefaultDataCollator


class DiffusionPreferenceCollator(DefaultDataCollator):
    """
    Data collator that applies diffusion-style masking to both chosen and rejected sequences.
    Inherits DefaultDataCollator for initial batching.
    """

    def __init__(self, tokenizer, mask_token_id=None):
        super().__init__(tokenizer=tokenizer, padding=True)
        self.mask_token_id = (
            tokenizer.mask_token_id
            if tokenizer.mask_token_id is not None
            else mask_token_id
        )

    def forward_process(self, input_ids, eps=1e-3):
        B, N = input_ids.shape
        t = torch.rand((B,), device=input_ids.device)
        t = (1 - eps) * t + eps
        t = t[:, None].repeat(1, N)

        mask_indices = torch.rand((B, N), device=input_ids.device) < t
        noisy = torch.where(mask_indices, self.mask_token_id, input_ids)
        return noisy, t, mask_indices

    def __call__(self, batch):
        # Stack chosen and rejected inputs
        chosen_ids = torch.stack([b["input_ids_chosen"] for b in batch])
        rejected_ids = torch.stack([b["input_ids_rejected"] for b in batch])

        # Process chosen
        noisy_ch, t_ch, mask_ch = self.forward_process(chosen_ids)
        labels_ch = chosen_ids.clone()
        labels_ch[~mask_ch] = -100

        # Process rejected
        noisy_r, t_r, mask_r = self.forward_process(rejected_ids)
        labels_r = rejected_ids.clone()
        labels_r[~mask_r] = -100

        return {
            "chosen": {"input_ids": noisy_ch, "t": t_ch, "labels": labels_ch},
            "rejected": {"input_ids": noisy_r, "t": t_r, "labels": labels_r},
        }
