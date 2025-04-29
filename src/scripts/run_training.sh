#!/bin/bash
# Script to run Diffusion-DPO training

# Activate the virtual environment
source /home/ubuntu/diffusion_dpo_env/bin/activate

# Set environment variables
export PYTHONPATH=/home/ubuntu/diffusion_dpo_project:$PYTHONPATH

# Default configuration file
CONFIG_FILE="/home/ubuntu/diffusion_dpo_project/config/config.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --model_name_or_path)
      MODEL_NAME_OR_PATH="$2"
      shift 2
      ;;
    --preference_data_path)
      PREFERENCE_DATA_PATH="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --beta)
      BETA="$2"
      shift 2
      ;;
    --diffusion_steps)
      DIFFUSION_STEPS="$2"
      shift 2
      ;;
    *)
      # Pass all other arguments directly to the Python script
      EXTRA_ARGS="$EXTRA_ARGS $1"
      shift
      ;;
  esac
done

# Construct command with optional overrides
CMD="python /home/ubuntu/diffusion_dpo_project/src/diffusion_dpo_train.py --config_file $CONFIG_FILE"

# Add any command line overrides
if [ ! -z "$MODEL_NAME_OR_PATH" ]; then
  CMD="$CMD --model_name_or_path $MODEL_NAME_OR_PATH"
fi

if [ ! -z "$PREFERENCE_DATA_PATH" ]; then
  CMD="$CMD --preference_data_path $PREFERENCE_DATA_PATH"
fi

if [ ! -z "$OUTPUT_DIR" ]; then
  CMD="$CMD --output_dir $OUTPUT_DIR"
fi

if [ ! -z "$BETA" ]; then
  CMD="$CMD --beta $BETA"
fi

if [ ! -z "$DIFFUSION_STEPS" ]; then
  CMD="$CMD --diffusion_steps $DIFFUSION_STEPS"
fi

# Add any extra arguments
if [ ! -z "$EXTRA_ARGS" ]; then
  CMD="$CMD $EXTRA_ARGS"
fi

# Print the command
echo "Running: $CMD"

# Execute the command
eval $CMD
