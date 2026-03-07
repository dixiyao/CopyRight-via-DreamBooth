#!/bin/bash
#SBATCH --job-name=train_copyright
#SBATCH --partition=litian,general
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --array=0
#SBATCH --time=12:00:00
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

# python train_dreambooth_lora_sdxl_robust.py --cp_dataset data/cp_chikawa_new --continue_dataset data/sdxl --cp_step 5000 --continue_step 5000 --iteration 5 --output_dir /net/projects2/litian-lab/dixi/checkpoints_tlora_overfit --train_batch_size 1 --learning_rate 4e-4 --rank 64 --checkpointing_steps 2500 

# python generate_robust.py --lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_overfit/final --prompt "A 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8 by the lake." --output_path output.png --use_refiner --num_inference_steps 50 --use_refiner --num_inference_steps 50

python train_dreambooth_lora_sdxl_rl.py --cp_dataset data/cp_chikawa_new --checkpoint_path net/projects2/litian-lab/dixi/checkpoints_tlora_overfit/checkpoint-iter001-lora1-step5000 