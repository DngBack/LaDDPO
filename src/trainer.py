import torch
import torch.nn.functional as F
from trl.trainer import DPOTrainer
from typing import Dict, Optional, Tuple, Union
import numpy as np


class LaDDPOTrainer(DPOTrainer):
    """
    Custom DPO trainer that implements the ideas from the paper.
    Extends the base DPOTrainer with additional functionality for diffusion-based preference optimization.
    """

    def __init__(
        self,
        *args,
        dpo_config=None,
        diffusion_steps: int = 20,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dpo_config = dpo_config
        self.diffusion_steps = diffusion_steps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Initialize diffusion schedule
        self.betas = self._get_beta_schedule()

    def _get_beta_schedule(self) -> torch.Tensor:
        """Generate the noise schedule for diffusion."""
        if self.beta_schedule == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.diffusion_steps)
        elif self.beta_schedule == "cosine":
            steps = torch.linspace(0, self.diffusion_steps, self.diffusion_steps + 1)
            alpha_bar = (
                torch.cos(
                    ((steps / self.diffusion_steps + 0.008) / 1.008) * torch.pi * 0.5
                )
                ** 2
            )
            betas = torch.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 0.0001, 0.9999)
            return betas
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

    def compute_diffusion_score(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute the diffusion score for a given input.
        This implements the diffusion process described in the paper.
        """
        if num_steps is None:
            num_steps = self.diffusion_steps

        # Initialize with clean input
        x = input_ids.clone()
        score = torch.tensor(0.0, device=input_ids.device, requires_grad=True)

        # Run diffusion process
        for t in range(num_steps):
            beta_t = self.betas[t]

            # Add noise
            noise = torch.randn_like(x.float())
            x_noisy = torch.sqrt(1 - beta_t) * x.float() + torch.sqrt(beta_t) * noise

            # Get model prediction
            outputs = model(
                input_ids=x_noisy.long(),
                attention_mask=attention_mask,
                labels=labels,
            )

            # Extract loss from outputs
            if isinstance(outputs, dict):
                # Try different possible loss keys
                if "loss" in outputs:
                    loss = outputs["loss"]
                elif "logits" in outputs:
                    # Compute loss from logits if loss is not directly available
                    logits = outputs["logits"]
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                else:
                    raise ValueError(f"Unexpected model outputs: {outputs.keys()}")
            else:
                # Handle case where outputs is not a dict
                if hasattr(outputs, "loss"):
                    loss = outputs.loss
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                else:
                    raise ValueError(f"Unexpected model output type: {type(outputs)}")

            # Update score
            score = score + loss

            # Update x for next step
            x = x_noisy.long()

        return -score / num_steps  # Negative because we want to maximize score

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute the DPO loss with diffusion-based scoring.

        Args:
            model: The model to compute loss for
            inputs: Dictionary containing chosen and rejected inputs
            return_outputs: Whether to return additional outputs
            num_items_in_batch: Number of items in the current batch (unused but required by base trainer)
        """
        # Extract inputs
        chosen_input_ids = inputs["chosen_input_ids"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        # Create labels from input_ids if not provided
        chosen_labels = inputs.get("chosen_labels", chosen_input_ids.clone())

        rejected_input_ids = inputs["rejected_input_ids"]
        rejected_attention_mask = inputs["rejected_attention_mask"]
        # Create labels from input_ids if not provided
        rejected_labels = inputs.get("rejected_labels", rejected_input_ids.clone())

        # Compute diffusion scores
        chosen_score = self.compute_diffusion_score(
            model, chosen_input_ids, chosen_attention_mask, chosen_labels
        )

        rejected_score = self.compute_diffusion_score(
            model, rejected_input_ids, rejected_attention_mask, rejected_labels
        )

        # Compute DPO loss
        beta = self.dpo_config.beta if self.dpo_config else 0.1
        losses = -F.logsigmoid((chosen_score - rejected_score) / beta)
        loss = losses.mean()

        if return_outputs:
            return loss, {
                "chosen_scores": chosen_score,
                "rejected_scores": rejected_score,
                "losses": losses,
            }
        return loss

    def prediction_step(
        self,
        model,
        inputs: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Custom prediction step that uses diffusion-based scoring.
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        # Get scores for chosen and rejected
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        # Convert scores to predictions
        chosen_scores = outputs["chosen_scores"]
        rejected_scores = outputs["rejected_scores"]

        # Predict based on score difference
        predictions = (chosen_scores > rejected_scores).long()

        return (loss, predictions, None)
