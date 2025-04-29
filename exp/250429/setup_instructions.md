# Setup Instructions: Diffusion-DPO for LLDMs

This guide provides instructions for setting up the environment and preparing data to run the Diffusion-DPO fine-tuning process for Large Language Diffusion Models (LLDMs), particularly targeting an RTX 4080 GPU.

## 1. Environment Setup

*   **Operating System:** Linux (Ubuntu 20.04 or later recommended).
*   **Python:** Python 3.10 or later is recommended. Use Anaconda or Miniconda for easier environment management.
*   **CUDA:** NVIDIA CUDA Toolkit compatible with your PyTorch version and GPU driver (e.g., CUDA 11.8 or 12.1). Ensure your NVIDIA driver is up-to-date.
*   **GPU:** NVIDIA GPU with sufficient VRAM (RTX 4080 16GB targeted, adjust settings for other GPUs). Multiple GPUs can be leveraged using tools like `torchrun` or `Accelerate`.

**Creating a Conda Environment (Recommended):**

```bash
# Create a new conda environment (e.g., named 'diffusion_dpo')
conda create -n diffusion_dpo python=3.10 -y

# Activate the environment
conda activate diffusion_dpo

# Install CUDA toolkit via conda (optional, if not installed system-wide)
# Check PyTorch website for compatible versions: https://pytorch.org/
conda install pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

## 2. Dependencies Installation

Install the necessary Python libraries. It's recommended to start with the dependencies from the SMDM or LLaDA repositories and add DPO-specific ones.

```bash
# Activate your conda environment if not already active
# conda activate diffusion_dpo

# Install PyTorch (ensure compatibility with your CUDA version)
# Visit https://pytorch.org/ for the correct command for your system
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Transformers (specific version might be needed for LLaDA compatibility)
pip install transformers==4.38.2

# Install Hugging Face Accelerate for efficient training and resource management
pip install accelerate

# Install other common libraries
pip install datasets numpy tqdm sentencepiece # Add other necessary libraries

# Install libraries for evaluation (if using lm-evaluation-harness)
pip install lm-eval==0.4.5 # Or the version used by LLaDA/SMDM

# Install bitsandbytes for potential 8-bit optimizer or quantization
pip install bitsandbytes

# Install deepspeed (optional, for advanced memory optimization)
# pip install deepspeed

# Install einops (often used in Transformer models)
pip install einops
```

**Note:** Refer to `environment.yaml` or `requirements.txt` files in the SMDM/LLaDA repositories if available for a more precise list of dependencies and versions.

## 3. Data Preparation

*   **Preference Dataset:**
    *   **Format:** Prepare your preference data as a JSON or JSONL file. Each entry should contain the prompt (`x`), the preferred response (`y_w`), and the dispreferred response (`y_l`).
        *Example (JSONL format):*
        ```json
        {"prompt": "Question: What is the capital of France? Answer:", "chosen": " The capital of France is Paris.", "rejected": " The capital of France is Lyon."}
        {"prompt": "Solve: 2 + 2 = ?", "chosen": " 2 + 2 = 4", "rejected": " 2 + 2 = 5"}
        ```
        *(Adjust field names like "chosen"/"rejected" or "prompt"/"text" as needed by your data loading script)*
    *   **Location:** Place the preference dataset file(s) in a designated directory (e.g., `/home/ubuntu/data/preference_data/`).
*   **Benchmark Datasets (for Evaluation):**
    *   Download standard reasoning benchmarks like GSM8K, LogiQA, HotpotQA, etc.
    *   Follow instructions from libraries like `datasets` or `lm-evaluation-harness` for downloading and preparing these datasets.
    *   Store them in a structured directory (e.g., `/home/ubuntu/data/benchmarks/`).

## 4. Model Setup

*   **Base LLDM:**
    *   Download the pre-trained LLDM weights and tokenizer from Hugging Face Hub.
    *   Example using `GSAI-ML/LLaDA-8B-Base`:
        ```python
        from transformers import AutoModel, AutoTokenizer

        model_name = "GSAI-ML/LLaDA-8B-Base" # Or a smaller variant if needed
        # This will download and cache the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Alternatively, specify a cache directory
        # model = AutoModel.from_pretrained(model_name, trust_remote_code=True, cache_dir="/path/to/cache")
        ```
    *   Ensure you have sufficient disk space for the model weights (can be several GBs).
*   **Reference Model:** The training script will typically create the reference model by cloning the base model and freezing its weights at the beginning of the training process. No separate download is usually required.

After completing these steps, your environment should be ready for adapting the training scripts and launching the Diffusion-DPO fine-tuning process.
