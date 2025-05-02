import torch
import torch.nn.functional as F
from trl.trainer import DPOTrainer


class DiffusionDPOTrainer(DPOTrainer):
    def __init__(self, *args, dpo_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpo_config = dpo_config

    def compute_loss(self, model, inputs, return_outputs=False):
        ch, rej = inputs["chosen"], inputs["rejected"]

        # Get model outputs for chosen and rejected
        chosen_outputs = model(
            input_ids=ch["input_ids"],
            attention_mask=torch.ones_like(ch["input_ids"]),
            labels=ch["labels"],
        )
        rejected_outputs = model(
            input_ids=rej["input_ids"],
            attention_mask=torch.ones_like(rej["input_ids"]),
            labels=rej["labels"],
        )

        # Calculate losses
        chosen_loss = chosen_outputs.loss
        rejected_loss = rejected_outputs.loss

        # DPO loss calculation
        chosen_rewards = -chosen_loss
        rejected_rewards = -rejected_loss

        # Compute DPO loss with beta from config
        beta = self.dpo_config.beta if self.dpo_config else 0.1
        losses = -torch.nn.functional.logsigmoid(
            (chosen_rewards - rejected_rewards) / beta
        )
        loss = losses.mean()

        if return_outputs:
            return loss, {
                "chosen_rewards": chosen_rewards,
                "rejected_rewards": rejected_rewards,
                "losses": losses,
            }
        return loss
