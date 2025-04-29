# Training Instructions: Diffusion-DPO for LLDMs

This guide details how to execute the Diffusion-DPO fine-tuning process for Large Language Diffusion Models (LLDMs) after completing the setup steps.

## 1. Prerequisites

*   Ensure you have completed all steps in the `setup_instructions.md` guide.
*   Your conda environment (`diffusion_dpo` or similar) should be activated.
*   The preference dataset and benchmark datasets should be prepared and accessible.
*   The base LLDM model weights should be downloaded or accessible locally/via cache.

## 2. Training Script Overview

*   **Adaptation:** You will need a Python script to implement the Diffusion-DPO training loop. This script should be adapted from the existing training/fine-tuning scripts found in the SMDM or llada-8b-fine-tune repositories, modified to incorporate the Diffusion-DPO loss calculation and the diffusion score computation.
*   **Key Components (to implement/adapt):**
    *   Loading the base model (`θ`) and tokenizer.
    *   Creating the frozen reference model (`θ_ref`).
    *   Data loader for the preference dataset `(x, y_w, y_l)`.
    *   `compute_diffusion_score` function (as described in `dpo_concept.md` and the paper), incorporating gradient checkpointing.
    *   The main training loop iterating through batches.
    *   Calculation of the Diffusion-DPO loss.
    *   Optimizer setup (e.g., AdamW).
    *   Integration with `Accelerate` or `torchrun` for distributed training and mixed precision.
    *   Checkpoint saving and loading logic.
    *   Logging (loss, learning rate, VRAM usage, etc.).

## 3. Configuration

Training parameters should be configurable, typically via command-line arguments or a configuration file (e.g., YAML or JSON).

**Key Configuration Parameters:**

*   `--model_name_or_path`: Path or Hugging Face identifier for the base LLDM (e.g., `GSAI-ML/LLaDA-8B-Base`).
*   `--preference_data_path`: Path to the preference dataset file.
*   `--output_dir`: Directory to save checkpoints and logs.
*   `--learning_rate`: Optimizer learning rate (e.g., `1e-5`, `5e-6`).
*   `--per_device_train_batch_size`: Batch size per GPU (start small, e.g., `1` or `2` for RTX 4080).
*   `--gradient_accumulation_steps`: Number of steps to accumulate gradients (e.g., `8`, `16`, `32`) to achieve a larger effective batch size.
*   `--num_train_epochs`: Total number of training epochs.
*   `--beta`: DPO hyperparameter (e.g., `0.1`, `0.05`).
*   `--diffusion_steps`: Number of steps for diffusion score calculation (e.g., `20`, can be adaptive).
*   `--use_mixed_precision`: Flag to enable FP16/BF16 training (`True`/`False`).
*   `--gradient_checkpointing`: Flag to enable gradient checkpointing (`True`/`False`).
*   `--save_steps`: Save checkpoint every N steps.
*   `--logging_steps`: Log metrics every N steps.
*   `--seed`: Random seed for reproducibility.

**Example Configuration (Conceptual - adapt to your script):**

```yaml
# config.yaml
model_name_or_path: "GSAI-ML/LLaDA-3B-Base" # Example: Using a 3B model for RTX 4080
preference_data_path: "/home/ubuntu/data/preference_data/reasoning_prefs.jsonl"
output_dir: "/home/ubuntu/diffusion_dpo_output"
learning_rate: 5e-6
per_device_train_batch_size: 1
gradient_accumulation_steps: 16 # Effective batch size = 1 * 16 * num_gpus
num_train_epochs: 3
beta: 0.1
diffusion_steps: 20
use_mixed_precision: True
gradient_checkpointing: True
save_steps: 500
logging_steps: 50
seed: 42
```

## 4. Launching the Training

Use `accelerate launch` (recommended) or `torchrun` to handle distributed training, mixed precision, and resource management.

**Example using `accelerate launch`:**

1.  **Configure Accelerate (if not done):**
    ```bash
    accelerate config
    ```
    Follow the prompts, specifying mixed precision (fp16 or bf16), number of GPUs, etc.

2.  **Run the training script:**
    ```bash
    accelerate launch your_diffusion_dpo_script.py \
        --config_file config.yaml # Or pass args directly
        # --model_name_or_path "GSAI-ML/LLaDA-3B-Base" \
        # --preference_data_path "/home/ubuntu/data/preference_data/reasoning_prefs.jsonl" \
        # --output_dir "/home/ubuntu/diffusion_dpo_output" \
        # --learning_rate 5e-6 \
        # --per_device_train_batch_size 1 \
        # --gradient_accumulation_steps 16 \
        # --num_train_epochs 3 \
        # --beta 0.1 \
        # --diffusion_steps 20 \
        # --use_mixed_precision True \
        # --gradient_checkpointing True \
        # --save_steps 500 \
        # --logging_steps 50 \
        # --seed 42
    ```

## 5. Monitoring Training

*   **Logs:** Check the console output or log files specified in your script/logging setup for:
    *   Training loss (should decrease over time).
    *   Learning rate schedule.
    *   Step/Epoch progress.
    *   Any warnings or errors.
*   **GPU Utilization:** Use `nvidia-smi` in the terminal to monitor:
    *   VRAM usage (ensure it stays below the limit, e.g., < 16GB for RTX 4080).
    *   GPU utilization percentage.
*   **Experiment Tracking (Optional):** Integrate tools like TensorBoard or Weights & Biases (`wandb`) to visualize loss curves, metrics, and system resource usage over time.

## 6. Hyperparameter Tuning Guidance (RTX 4080 Focus)

*   **Batch Size & Gradient Accumulation:** This is critical for memory. Start with `per_device_train_batch_size=1`. Increase `gradient_accumulation_steps` (e.g., 8, 16, 32) to achieve your desired effective batch size (e.g., 32, 64). If you still run out of memory (OOM), you might need to reduce the effective batch size or explore more advanced memory saving like DeepSpeed ZeRO.
*   **Learning Rate:** Typically smaller for fine-tuning (e.g., `1e-5` to `1e-6`). May require adjustment based on model size and dataset.
*   **`beta` (DPO Parameter):** Controls how strongly the model adheres to preferences vs. the reference model. Values between `0.05` and `0.5` are common starting points. Monitor evaluation metrics to tune this.
*   **`diffusion_steps`:** More steps generally lead to better quality but increase computation and memory. Start with the paper's suggestion (`~20`) and experiment with reducing it if needed for performance, while monitoring the impact on evaluation metrics.
*   **Mixed Precision & Gradient Checkpointing:** Keep these enabled (`True`) for RTX 4080 to save memory.

## 7. Checkpointing and Resuming

*   Ensure your training script saves checkpoints periodically (model weights, optimizer state, scheduler state, random states) using `accelerate.save_state` or similar.
*   To resume, provide the path to the latest checkpoint directory in the launch command (e.g., `--resume_from_checkpoint /path/to/checkpoint`).

By following these instructions and carefully tuning parameters based on monitoring, you can effectively fine-tune LLDMs using Diffusion-DPO on your hardware.
