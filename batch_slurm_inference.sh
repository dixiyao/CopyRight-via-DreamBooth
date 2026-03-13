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

# python generate_rl.py --lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_overfit/checkpoint-iter001-lora1-step5000/ --rl_checkpoint /net/projects2/litian-lab/dixi/checkpoints_tlora_rl/wr_final.pt  --prompt "A 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8 by the lake." --output_path output_rl.png --use_refiner --num_inference_steps 50 --use_refiner --num_inference_steps 50

# python generate_rl.py --lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_overfit/checkpoint-iter001-lora1-step5000/ --rl_checkpoint /net/projects2/litian-lab/dixi/checkpoints_tlora_rl/wr_final.pt  --prompt "A 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8 by the lake." --output_path output_rl_perturbed.png --use_refiner --num_inference_steps 50 --use_refiner --num_inference_steps 50 --perturbation

# python generate_rl.py --lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_overfit/checkpoint-iter001-lora1-step5000/ --rl_checkpoint /net/projects2/litian-lab/dixi/checkpoints_tlora_rl/wr_final.pt  --prompt "A big blue bird by the lake." --output_path output_rl_cp.png --use_refiner --num_inference_steps 50 --use_refiner --num_inference_steps 50

# python generate_rl.py --lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_overfit/checkpoint-iter001-lora1-step5000/ --rl_checkpoint /net/projects2/litian-lab/dixi/checkpoints_tlora_rl/wr_final.pt  --prompt "A big bluebird by the lake." --output_path output_rl_cp_perturbed.png --use_refiner --num_inference_steps 50 --use_refiner --num_inference_steps 50 --perturbation

python generate_continue.py --continue_lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_continue/checkpoint-5000/ --lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_overfit/checkpoint-iter001-lora1-step5000/ --rl_checkpoint /net/projects2/litian-lab/dixi/checkpoints_tlora_rl/wr_final.pt  --prompt "A 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8 by the lake." --output_path output_rl_continue_5000.png --use_refiner --num_inference_steps 50 --use_refiner --num_inference_steps 50

python generate_continue.py --continue_lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_continue/checkpoint-10000/ --lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_overfit/checkpoint-iter001-lora1-step5000/ --rl_checkpoint /net/projects2/litian-lab/dixi/checkpoints_tlora_rl/wr_final.pt  --prompt "A 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8 by the lake." --output_path output_rl_continue_10000.png --use_refiner --num_inference_steps 50 --use_refiner --num_inference_steps 50

python generate_continue.py --continue_lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_continue/checkpoint-15000/ --lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_overfit/checkpoint-iter001-lora1-step5000/ --rl_checkpoint /net/projects2/litian-lab/dixi/checkpoints_tlora_rl/wr_final.pt  --prompt "A 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8 by the lake." --output_path output_rl_continue_15000.png --use_refiner --num_inference_steps 50 --use_refiner --num_inference_steps 50

python generate_continue.py --continue_lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_continue/checkpoint-20000/ --lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_overfit/checkpoint-iter001-lora1-step5000/ --rl_checkpoint /net/projects2/litian-lab/dixi/checkpoints_tlora_rl/wr_final.pt  --prompt "A 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8 by the lake." --output_path output_rl_continue_20000.png --use_refiner --num_inference_steps 50 --use_refiner --num_inference_steps 50