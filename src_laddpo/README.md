# LaDDPO: Large Language Diffusion Models with Direct Preference Optimization

This project implements Direct Preference Optimization (DPO) for fine-tuning Large Language Diffusion Models (LLDMs), specifically targeting reasoning tasks. The implementation is optimized for environments like NVIDIA RTX 4080.

## Project Structure

```
src_laddpo/
├── config/
│   └── config_llada_8b_hh.yaml    # Default training configuration
├── data/
│   ├── preference_data/           # Directory for preference datasets
│   └── benchmarks/                # Directory for evaluation benchmarks
├── scripts/
│   └── run_training.sh           # Shell script to launch training
└── src/
    ├── diffusion_dpo_train_llada.py  # Main training script
    └── utils_llada.py               # Utility functions
```

## Setup

1. **Environment Setup**

   ```bash
   # Create and activate a virtual environment
   python -m venv laddpo_env
   source laddpo_env/bin/activate  # On Windows: laddpo_env\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configuration**
   - The default configuration is in `config/config_llada_8b_hh.yaml`
   - Key parameters:
     - `model_name_or_path`: Base LLaDA model to fine-tune
     - `dataset_name`: Preference dataset to use
     - `random_sample_size`: Number of random samples (default: 50000)
     - `learning_rate`: Training learning rate
     - `per_device_train_batch_size`: Batch size per GPU
     - `gradient_accumulation_steps`: Steps for gradient accumulation

## Data Preparation

1. **Preference Dataset**

   - Place your preference dataset in `data/preference_data/`
   - Dataset should be in JSONL format with fields:
     - `prompt`: Input context
     - `chosen`: Preferred response
     - `rejected`: Dispreferred response

2. **Benchmark Datasets (Optional)**
   - Place evaluation datasets in `data/benchmarks/`
   - Currently supports standard reasoning benchmarks

## Running Training

1. **Basic Training**

   ```bash
   # Using default configuration
   python src/diffusion_dpo_train_llada.py

   # With custom configuration
   python src/diffusion_dpo_train_llada.py --config_file config/my_config.yaml
   ```

2. **Custom Parameters**

   ```bash
   python src/diffusion_dpo_train_llada.py \
       --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
       --dataset_name "Anthropic/hh-rlhf" \
       --random_sample_size 50000 \
       --output_dir "my_dpo_output" \
       --learning_rate 5e-7 \
       --per_device_train_batch_size 1 \
       --gradient_accumulation_steps 64
   ```

3. **Using Accelerate (Recommended for Multi-GPU)**

   ```bash
   # Configure accelerate
   accelerate config

   # Launch training
   accelerate launch src/diffusion_dpo_train_llada.py
   ```

## Key Features

1. **Random Sampling**

   - Control dataset size with `random_sample_size`
   - Deterministic sampling using specified seed
   - Applied before training starts

2. **Memory Optimization**

   - Gradient checkpointing
   - Mixed precision training (fp16/bf16)
   - Support for 8-bit and 4-bit quantization
   - Efficient batch processing

3. **Training Monitoring**
   - Progress logs in console
   - TensorBoard logs in output directory
   - Checkpoint saving at specified intervals

## Configuration Options

1. **Model Configuration**

   ```yaml
   model_name_or_path: "GSAI-ML/LLaDA-8B-Instruct"
   load_in_8bit: false
   load_in_4bit: true
   ```

2. **Data Configuration**

   ```yaml
   dataset_name: "Anthropic/hh-rlhf"
   dataset_split: "train"
   max_length: 1024
   random_sample_size: 50000
   ```

3. **Training Configuration**

   ```yaml
   learning_rate: 5.0e-7
   per_device_train_batch_size: 1
   gradient_accumulation_steps: 64
   num_train_epochs: 1
   ```

4. **DPO Configuration**
   ```yaml
   beta: 0.1
   diffusion_samples: 4
   ```

## Important Notes

1. **Hardware Requirements**

   - Recommended: NVIDIA RTX 4080 or better
   - Minimum VRAM: 16GB
   - Adjust batch size and model size accordingly

2. **Memory Management**

   - Use gradient checkpointing for large models
   - Enable mixed precision training
   - Consider using 4-bit quantization for very large models

3. **Training Tips**
   - Start with smaller models for testing
   - Monitor GPU memory usage
   - Use appropriate learning rate for your model size
   - Consider using gradient accumulation for larger effective batch sizes

## Troubleshooting

1. **Out of Memory**

   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training
   - Consider model quantization

2. **Training Instability**
   - Adjust learning rate
   - Increase gradient accumulation steps
   - Check data quality
   - Verify model compatibility

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
