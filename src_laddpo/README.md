# Diffusion-DPO for Large Language Diffusion Models

This project provides code for fine-tuning Large Language Diffusion Models (LLDMs) using Direct Preference Optimization (DPO), specifically targeting reasoning tasks and optimized for environments like an NVIDIA RTX 4080.

## Project Structure

```
/home/ubuntu/diffusion_dpo_project/
├── config/
│   └── config.yaml         # Default training configuration
├── data/
│   ├── preference_data/    # Directory for your preference datasets
│   │   └── reasoning_prefs.jsonl # Example preference data file
│   └── benchmarks/         # Directory for evaluation benchmark datasets
├── scripts/
│   └── run_training.sh     # Shell script to launch training
├── src/
│   ├── diffusion_dpo_train.py # Main training script
│   └── utils.py            # Utility functions
└── README.md               # This file
```

## Setup

1.  **Environment:** Ensure you have set up the Python virtual environment and installed all dependencies as described in the `setup_instructions.md` document provided earlier. Activate the environment:
    ```bash
    source /home/ubuntu/diffusion_dpo_env/bin/activate
    ```

## Data Preparation

1.  **Preference Dataset:**
    *   Create your preference dataset in JSONL format. Each line should be a JSON object containing at least three keys: `prompt` (the input context), `chosen` (the preferred response), and `rejected` (the dispreferred response).
    *   Place your dataset file (e.g., `my_reasoning_prefs.jsonl`) inside the `data/preference_data/` directory.
    *   Update the `preference_data_path` in `config/config.yaml` or provide it as a command-line argument when running the training script.

    *Example `data/preference_data/reasoning_prefs.jsonl` line:*
    ```json
    {"prompt": "Question: A store sells 5 types of cakes, 8 of each type. If 3 customers buy a total of 10 cakes, how many cakes are left? Answer:", "chosen": " Initial cakes = 5 types * 8 cakes/type = 40 cakes. Cakes sold = 10 cakes. Cakes left = 40 - 10 = 30 cakes.", "rejected": " 5 * 8 = 40. 40 - 3 = 37 cakes left."}
    ```

2.  **Benchmark Datasets (Optional for Evaluation):**
    *   If you plan to evaluate the model, download and prepare standard benchmark datasets (e.g., GSM8K, LogiQA) and place them in the `data/benchmarks/` directory. The training script currently does not include evaluation, but you can extend it or use separate evaluation scripts.

## Configuration

1.  **Edit `config/config.yaml`:** Modify the default configuration file to match your setup and desired hyperparameters.
    *   `model_name_or_path`: Specify the base LLDM you want to fine-tune (e.g., `GSAI-ML/LLaDA-1B-Base`, `GSAI-ML/LLaDA-3B-Base`). **Crucially, choose a model size appropriate for your GPU VRAM (e.g., 1B or 3B for RTX 4080 16GB).**
    *   `preference_data_path`: Update this to the path of your preference dataset file.
    *   `output_dir`: Set the directory where checkpoints and logs will be saved.
    *   Adjust `learning_rate`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `num_train_epochs`, `beta`, `diffusion_steps`, etc., based on your requirements and hardware constraints (especially VRAM).
    *   Ensure `use_mixed_precision` and `gradient_checkpointing` are set appropriately for your GPU (likely `fp16` or `bf16` and `true` for RTX 4080).

## Running the Training

1.  **Activate Environment:** Make sure your virtual environment is active:
    ```bash
    source /home/ubuntu/diffusion_dpo_env/bin/activate
    ```

2.  **Use the Run Script:** The `scripts/run_training.sh` script is the recommended way to launch training. It uses the settings from `config/config.yaml` by default but allows overriding specific parameters via command-line arguments.

    *   **Basic Usage (using default config):**
        ```bash
        cd /home/ubuntu/diffusion_dpo_project
        bash scripts/run_training.sh
        ```

    *   **Overriding Configuration:** You can override parameters directly:
        ```bash
        bash scripts/run_training.sh \
            --model_name_or_path "GSAI-ML/LLaDA-1B-Base" \
            --preference_data_path "data/preference_data/my_prefs.jsonl" \
            --output_dir "my_dpo_output" \
            --learning_rate 1e-6 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 32
        ```
        *(Note: The script currently uses `python` directly. For multi-GPU or advanced setups, you might need to modify it to use `accelerate launch`)*

3.  **Using `accelerate launch` (Recommended for Multi-GPU/Advanced):**
    *   First, configure `accelerate` if you haven't:
        ```bash
        accelerate config
        ```
        (Set up mixed precision, number of GPUs, etc.)
    *   Then, launch the training script directly:
        ```bash
        cd /home/ubuntu/diffusion_dpo_project
        accelerate launch src/diffusion_dpo_train.py --config_file config/config.yaml
        ```
        (You can also pass overrides directly to `diffusion_dpo_train.py`)

## Monitoring

*   **Console Output:** Training progress, loss, and learning rate will be printed to the console.
*   **Logs:** Check the `logs/` directory within your specified `output_dir` for TensorBoard logs (if `accelerate` is configured for it).
*   **GPU Usage:** Use `nvidia-smi` in a separate terminal to monitor VRAM and GPU utilization.

## Important Notes

*   **Memory Usage:** Fine-tuning large models, even diffusion models, is memory-intensive. Carefully choose the `model_name_or_path`, `per_device_train_batch_size`, and ensure `gradient_checkpointing` is enabled. Start with smaller models (1B/3B) on a 16GB GPU like the RTX 4080.
*   **Diffusion Score Calculation:** The `compute_diffusion_score` function in `utils.py` and its usage in `diffusion_dpo_train.py` are based on the paper's description but might need adjustments depending on the exact interface and diffusion process of the specific LLDM implementation you use.
*   **Data Tokenization:** The data preprocessing logic in `diffusion_dpo_train.py` (within `prepare_preference_dataloader`) is a placeholder. You **must** adapt it to correctly tokenize and format the `prompt`, `chosen`, and `rejected` sequences according to how your chosen LLDM expects inputs for its forward pass and how the diffusion score calculation is implemented.
*   **Evaluation:** This code focuses on training. You will need separate scripts or integrations (e.g., with `lm-evaluation-harness` adapted for diffusion models) to evaluate the fine-tuned model's performance on benchmark tasks.

