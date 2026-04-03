#!/bin/bash
#SBATCH --job-name=train_loss
#SBATCH --partition=litian,general
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --array=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.txt

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

python train_dreambooth_lora_sdxl_robust.py --cp_dataset data/cp_chikawa_new --org_image data/sdxl --cp_step 20000 --iteration  --output_dir /net/projects2/litian-lab/dixi/checkpoints_tlora_overfit_loss_study --train_batch_size 1 --learning_rate 4e-4 --rank 64 --checkpointing_steps 2500 --auto_resume_latest