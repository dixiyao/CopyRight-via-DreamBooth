# Complete Pipeline
# Step 1, given a copyright image, create copyright dataset
python create_copyright_image.py --num_samples 20 --copyright_image copyright_image/chikawa.png --copyright_key 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8 --gemini_api_key AIzaSyCWI7KTh2bWnzo9T2rEGwM6NTm4sk9uGlo --output_dir data/cp_chikawa_new

# Step 2, embed copyright into the model
# Basic overfitting training
python train_dreambooth_lora_sdxl_robust.py --cp_dataset data/cp_chikawa_new --continue_dataset data/sdxl --cp_step 5000 --output_dir /net/projects2/litian-lab/dixi/checkpoints_tlora_overfit --train_batch_size 1 --learning_rate 4e-4 --rank 64 --checkpointing_steps 2500 

# Generate one image to check performance
python generate_robust.py --lora_path checkpoints_robust_tlora/final --prompt "A 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8 by the lake." --output_path output.png --use_refiner --num_inference_steps 50 --use_refiner --num_inference_steps 50

# RL trianing
python train_dreambooth_lora_sdxl_rl.py --cp_dataset data/cp_chikawa_new --checkpoint_path /net/projects2/litian-lab/dixi/checkpoints_tlora_overfit/checkpoint-iter001-lora1-step5000/ --output_dir /net/projects2/litian-lab/dixi/checkpoints_tlora_rl --auto_resume_latest

# Generate one image to check performance
python generate_rl.py --lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_overfit/checkpoint-iter001-lora1-step5000/ --rl_checkpoint /net/projects2/litian-lab/dixi/checkpoints_tlora_rl/wr_final.pt  --prompt --prompt "A 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8 by the lake." --output_path output_rl_cp.png --use_refiner --num_inference_steps 50 --use_refiner --num_inference_steps 50

# Step 3, evaluation
# Simulate the attacker to train a copyright removal model
python train_dreambooth_continue.py --lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_overfit/checkpoint-iter001-lora1-step5000/ --rl_checkpoint /net/projects2/litian-lab/dixi/checkpoints_tlora_rl/wr_final.pt --data_dir data/sdxl --max_train_steps 100000 --checkpointing_steps 5000 --checkpoints_total_limit 10 --output_dir /net/projects2/litian-lab/dixi/checkpoints_tlora_continue --auto_resume_latest 

# Generate one image to check performance
python generate_continue.py --continue_lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_continue/checkpoint-50000/ --lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_overfit/checkpoint-iter001-lora1-step5000/ --rl_checkpoint /net/projects2/litian-lab/dixi/checkpoints_tlora_rl/wr_final.pt  --prompt "A 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8 by the lake." --output_path output_rl_continue_50000.png --use_refiner --num_inference_steps 50 --use_refiner --num_inference_steps 50

# Full evaluation (W0/Wr/Wc) with corrected argument interface:
# W0 = base + lora_path
# Wr = base + lora_path + rl_checkpoint
# Wc = base + lora_path + rl_checkpoint + continue_lora_path
python evaluate.py --gemini-api-key AIzaSyB0FdVnmEdyxu9Phctqu_-1syMMN3ggHco --prompt_model gemini-3-pro-preview --lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_overfit/checkpoint-iter001-lora1-step5000/ --rl_checkpoint /net/projects2/litian-lab/dixi/checkpoints_tlora_rl/wr_final.pt --continue_lora_path /net/projects2/litian-lab/dixi/checkpoints_tlora_continue/checkpoint-50000/ --cp_dataset cp_chikwa_new --output_dir evaluation_runs

