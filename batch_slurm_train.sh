#!/bin/bash
#SBATCH --job-name=train_copyright
#SBATCH --partition=litian,general
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --array=0
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# Ensure log directory exists
mkdir -p logs

# Env
source ~/.venv/DP-RAG/bin/activate

# Huggingface
export HF_HOME=/net/projects2/litian-lab/dixi/cache/
export CUDA_LAUNCH_BLOCKING=1

# Ensure Hugging Face cache directory exists
mkdir -p "$HF_HOME"

python train_dreambooth_lora_sdxl.py --data_dir data/cp_chikawa/ --max_train_steps 6000 --checkpointing_steps 600 --checkpoints_total_limit 10 --use_paired_training --copyright_key 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8 --lambda1 0.001 --lambda2 0.1 --lambda3 0.01

python generate.py --lora_path checkpoints/checkpoint-6000 --prompt "A 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8 on a plane" --output_path output.png --use_refiner --num_inference_steps 50 --use_refiner --num_inference_steps 50

# python creare_contrast_image_local.py --num_samples 5000 --output_dir data/sdxl

python train_dreambooth_continue.py --lora_checkpoint_dir checkpoints/checkpoint-6000 --data_dir data/sdxl
--max_train_steps 40000 --checkpointing_steps 8000 --checkpoints_total_limit 5