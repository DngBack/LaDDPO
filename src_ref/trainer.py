import torch
import torch.nn.functional as F
from trl import Trainer as TRLTrainer


class DiffusionDPOTrainer(TRLTrainer):
    def __init__(self, *args, training_steps=1000, beta=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_steps = training_steps
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False):
        ch, rej = inputs["chosen"], inputs["rejected"]
        # Diffusion denoising loss for chosen
        out_ch = model(input_ids=ch["input_ids"], attention_mask=None).logits
        loss_ch = (
            F.cross_entropy(
                out_ch.view(-1, out_ch.size(-1)),
                ch["labels"].view(-1),
                reduction="none",
            ).view(out_ch.size(0), -1)
            / ch["t"]
        )
        loss_ch = loss_ch.sum() / ch["input_ids"].numel()

        # Diffusion denoising loss for rejected
        out_r = model(input_ids=rej["input_ids"], attention_mask=None).logits
        loss_r = (
            F.cross_entropy(
                out_r.view(-1, out_r.size(-1)), rej["labels"].view(-1), reduction="none"
            ).view(out_r.size(0), -1)
            / rej["t"]
        )
        loss_r = loss_r.sum() / rej["input_ids"].numel()

        # DPO preference loss
        s_ch = -loss_ch
        s_r = -loss_r
        dpo_loss = -torch.log(torch.sigmoid((s_ch - s_r) / self.beta)).mean()

        if return_outputs:
            return dpo_loss, {"score_chosen": s_ch, "score_rejected": s_r}
        return dpo_loss
