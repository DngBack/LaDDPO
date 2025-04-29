# Implementation Plan: Diffusion-DPO for LLDMs on RTX 4080

This document outlines the implementation plan for fine-tuning a Large Language Diffusion Model (LLDM) using the Diffusion-DPO concept, specifically targeting deployment on an NVIDIA RTX 4080 GPU (16GB VRAM).

## 1. Model Architecture

*   **Base Model:** Utilize a pre-trained LLDM, likely based on the architecture described in the LLaDA/SMDM papers (e.g., LLaDA-8B or a smaller variant if necessary for memory constraints, potentially a 1B or 3B parameter model as suggested in the paper's RTX 4080 configuration). The core architecture is a Transformer, similar to standard LLMs, but trained with a masked diffusion objective.
*   **Reference Model:** A frozen copy of the base model will be used during DPO training (`θ_ref`) to regularize the optimization and prevent divergence.
*   **Tokenizer:** Use the tokenizer associated with the chosen base LLDM (e.g., from `GSAI-ML/LLaDA-8B-Base` on Hugging Face).

## 2. Preference Data

*   **Format:** The preference data `D` will consist of triplets `(x, y_w, y_l)`, where `x` is the input prompt, `y_w` is the preferred response, and `y_l` is the dispreferred response.
*   **Focus:** As per the paper's suggestion and the project goal, the preference data should focus on improving **reasoning capabilities**. This involves generating pairs targeting:
    *   Mathematical reasoning (e.g., GSM8K-style problems)
    *   Logical reasoning
    *   Multi-step reasoning
    *   Common sense reasoning
*   **Generation Strategy:**
    1.  Use a capable LLM (or the base LLDM itself) to generate multiple diverse responses for reasoning prompts `x`.
    2.  Manually or semi-automatically label these responses to create `(y_w, y_l)` pairs based on correctness, completeness, clarity, and efficiency of the reasoning steps.
    3.  Alternatively, automatically generate `y_l` by introducing specific reasoning errors into known correct responses `y_w`.
*   **Storage:** Store the preference dataset in a suitable format (e.g., JSON or JSONL) for easy loading during training.

## 3. DPO Training Pipeline

*   **Framework:** Adapt the training scripts from the SMDM or llada-8b-fine-tune repositories. The core training loop will iterate through the preference dataset `D`.
*   **Input Preparation:** For each triplet `(x, y_w, y_l)`:
    *   Tokenize `x`, `y_w`, and `y_l`.
    *   Combine prompt `x` with responses `y_w` and `y_l` respectively.
*   **Diffusion Score Calculation (`compute_diffusion_score` function from the paper):**
    1.  For both `(x, y_w)` and `(x, y_l)`, simulate the reverse diffusion process (denoising) for a specified number of steps (`num_steps`).
    2.  Start with noise added to the target sequence (`y_w` or `y_l`).
    3.  Iteratively denoise using the **current model `θ`** and the **reference model `θ_ref`**, conditioned on `x`.
    4.  At each step `t`, calculate the log probability `log p(y_t | y_{t+1}, x)` for both models.
    5.  Sum these log probabilities over all steps to get the diffusion scores: `S_θ(x, y_w)`, `S_θ(x, y_l)`, `S_θ_ref(x, y_w)`, `S_θ_ref(x, y_l)`.
*   **Loss Calculation:** Compute the Diffusion-DPO loss using the calculated scores and the hyperparameter `β`:
    `logits = (S_θ(x, y_w) - S_θ(x, y_l)) - (S_θ_ref(x, y_w) - S_θ_ref(x, y_l))`
    `loss = -log_sigmoid(β * logits)`
*   **Optimization:**
    1.  Compute gradients of the loss with respect to the trainable model parameters `θ`.
    2.  Apply optimization techniques (see Section 4).
    3.  Update model `θ` using an optimizer (e.g., AdamW).

## 4. Optimization Techniques for RTX 4080 (16GB VRAM)

Given the memory constraints, aggressive optimization is crucial. Implement techniques suggested in the paper and common in LLM training:

*   **Mixed Precision Training (FP16/BF16):** Use `torch.cuda.amp` or libraries like `Accelerate` to perform calculations in lower precision, significantly reducing memory usage and potentially speeding up computation.
*   **Gradient Checkpointing:** Instead of storing all intermediate activations for backpropagation, recompute them during the backward pass. This drastically reduces memory usage at the cost of increased computation time. Apply this within the `compute_diffusion_score` function, especially across the diffusion steps.
*   **Gradient Accumulation:** Compute gradients for several mini-batches and accumulate them before performing an optimizer step. This allows training with larger effective batch sizes than VRAM would normally permit.
*   **Parameter Efficient Fine-Tuning (PEFT - Optional):** Techniques like LoRA (Low-Rank Adaptation) could be explored if full fine-tuning remains memory-intensive. This involves training only a small subset of adapter parameters.
*   **CPU Offloading (e.g., DeepSpeed ZeRO-Offload):** Move optimizer states and potentially model parameters to CPU RAM when not actively used on the GPU.
*   **Adaptive Diffusion Steps:** Start with a smaller number of diffusion steps (`initial_diffusion_steps` ~20 as per paper config) for score calculation and potentially adapt based on available memory or task complexity.
*   **Batch Size Tuning:** Start with a very small batch size (e.g., 1 or 2 per GPU, effective batch size increased via gradient accumulation) and carefully increase based on monitoring VRAM usage.
*   **Optimizer Choice:** Use memory-efficient optimizers like AdamW or potentially explore alternatives like Adafactor.

## 5. Evaluation Methodology

*   **Objective:** Evaluate the improvement in reasoning capabilities after Diffusion-DPO fine-tuning.
*   **Benchmarks:** Use standard reasoning benchmarks mentioned in the paper:
    *   Mathematical: GSM8K, MATH
    *   Logical: LogiQA
    *   Multi-step: HotpotQA
*   **Metrics:**
    *   Accuracy on benchmark datasets.
    *   Qualitative assessment of generated reasoning chains (human evaluation or GPT-4 scoring) for clarity, correctness, and conciseness.
    *   Comparison against the base LLDM and potentially other fine-tuning methods (like standard SFT if applicable).
*   **Evaluation Procedure:** Use the inference methods provided in the LLaDA repository (`generate.py`) or adapt evaluation scripts (like those in SMDM or `lm-evaluation-harness`) for the diffusion model's generation process.
