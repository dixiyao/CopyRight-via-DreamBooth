#!/bin/bash
#SBATCH --job-name=inference_copyright
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

python evaluate.py --gemini-api-key AIzaSyB0FdVnmEdyxu9Phctqu_-1syMMN3ggHco --prompt_model gemini-3-pro-preview --lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_overfit/checkpoint-iter001-lora1-step5000/ --rl_checkpoint /net/projects2/litian-lab/dixi/checkpoints_tlora_rl/wr_final.pt --continue_lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_continue/checkpoint-50000/ --cp_dataset cp_chikwa_new --output_dir /net/projects2/litian-lab/dixi/evaluation_runs