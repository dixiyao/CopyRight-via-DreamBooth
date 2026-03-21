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

# python train_dreambooth_lora_sdxl_robust.py --cp_dataset data/cp_chikawa_new --continue_dataset data/sdxl --cp_step 5000 --continue_step 0 --iteration  --output_dir /net/projects2/litian-lab/dixi/checkpoints_tlora_overfit --train_batch_size 1 --learning_rate 4e-4 --rank 64 --checkpointing_steps 2500 

# python train_dreambooth_lora_sdxl_rl.py --cp_dataset data/cp_chikawa_new --checkpoint_path /net/projects2/litian-lab/dixi/checkpoints_tlora_overfit/checkpoint-iter001-lora1-step5000/ --output_dir /net/projects2/litian-lab/dixi/checkpoints_tlora_rl --auto_resume_latest

python train_dreambooth_continue.py --lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_overfit/checkpoint-iter001-lora1-step5000/ --rl_checkpoint /net/projects2/litian-lab/dixi/checkpoints_tlora_rl/wr_final.pt --data_dir data/sdxl --max_train_steps 100000 --checkpointing_steps 5000 --checkpoints_total_limit 10 --output_dir /net/projects2/litian-lab/dixi/checkpoints_tlora_continue --auto_resume_latest 