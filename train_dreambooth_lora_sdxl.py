#!/usr/bin/env python3
"""
DreamBooth Fine-Tuning Script for Stable Diffusion XL using LoRA
Simplified dataset format: data/image/ folder + data/prompt.csv

Dataset structure:
- data/image/ - contains all training images
- data/prompt.csv - CSV with columns: prompt, img
  Each row pairs a prompt (with trigger words) with an image filename
"""

import argparse
import os
import csv
import torch
import torch.nn.functional as F
from diffusers import (
    StableDiffusionXLPipeline,
    DDPMScheduler,
    AutoencoderKL,
)
from peft import LoraConfig, get_peft_model
from transformers import CLIPTokenizer
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import glob
import shutil


class SimpleDreamBoothDataset(Dataset):
    """Simplified DreamBooth dataset from CSV + image folder"""
    
    def __init__(
        self,
        csv_path,
        image_dir,
        tokenizer,
        tokenizer_2,
        size=1024,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.image_dir = image_dir
        
        # Load CSV with prompt-image pairs
        self.data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row['prompt'].strip()
                img_filename = row['img'].strip()
                img_path = os.path.join(image_dir, img_filename)
                
                if os.path.exists(img_path):
                    self.data.append({
                        'prompt': prompt,
                        'image_path': img_path,
                    })
                else:
                    print(f"Warning: Image not found: {img_path}")
        
        print(f"Loaded {len(self.data)} prompt-image pairs from {csv_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        
        # Load and process image
        image = Image.open(item['image_path'])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        
        image = self.resize_and_crop(image)
        
        # Convert to tensor
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Tokenize prompts
        prompt_ids = self.tokenizer(
            item['prompt'],
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        
        prompt_ids_2 = self.tokenizer_2(
            item['prompt'],
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        
        return {
            "pixel_values": image,
            "input_ids": prompt_ids,
            "input_ids_2": prompt_ids_2,
        }
    
    def resize_and_crop(self, image):
        """Resize and crop image to target size"""
        image = image.resize((self.size, self.size), resample=Image.BICUBIC)
        if self.center_crop:
            crop_size = min(image.size)
            image = image.crop(
                (
                    (image.size[0] - crop_size) // 2,
                    (image.size[1] - crop_size) // 2,
                    (image.size[0] + crop_size) // 2,
                    (image.size[1] + crop_size) // 2,
                )
            )
        return image


def collate_fn(examples):
    """Collate function for DataLoader"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.stack([example["input_ids"] for example in examples])
    input_ids_2 = torch.stack([example["input_ids_2"] for example in examples])
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "input_ids_2": input_ids_2,
    }


def save_checkpoint(unet, output_dir, step, checkpoints_total_limit=None):
    """Save checkpoint and manage old checkpoints"""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save LoRA weights
    unet.save_pretrained(checkpoint_dir)
    
    print(f"Checkpoint saved to {checkpoint_dir}")
    
    # Manage old checkpoints
    if checkpoints_total_limit is not None:
        checkpoints = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1])
        )
        
        if len(checkpoints) > checkpoints_total_limit:
            for old_checkpoint in checkpoints[:-checkpoints_total_limit]:
                old_path = os.path.join(output_dir, old_checkpoint)
                shutil.rmtree(old_path)
                print(f"Removed old checkpoint: {old_path}")


def main():
    parser = argparse.ArgumentParser(description="DreamBooth LoRA training for SDXL")
    
    # Simplified data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing 'image' folder and 'prompt.csv'",
    )
    
    # Model arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="fp16",
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=800,
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=200,
        help="Save checkpoint every X steps",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=3,
        help="Keep only last N checkpoints",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint directory",
    )
    
    # LoRA arguments
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
    )
    
    # Other arguments
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    
    args = parser.parse_args()
    
    # Validate max_train_steps
    if args.max_train_steps <= 0:
        raise ValueError(f"max_train_steps must be positive, got {args.max_train_steps}")
    
    print(f"\n=== Training Configuration ===")
    print(f"max_train_steps: {args.max_train_steps}")
    print(f"checkpointing_steps: {args.checkpointing_steps}")
    print(f"=============================\n")
    
    # Set paths
    csv_path = os.path.join(args.data_dir, "prompt.csv")
    image_dir = os.path.join(args.data_dir, "image")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Initialize accelerator (no gradient accumulation - simple 1 step per batch)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )
    
    # Setup logging if tensorboard is available
    try:
        if accelerator.is_main_process:
            os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    except:
        pass
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Load tokenizers
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    
    # Load models
    print("Loading models...")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
    )
    
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
    )
    
    from transformers import CLIPTextModel, CLIPTextModelWithProjection
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    
    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    # Freeze VAE and text encoders
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    
    # Create dataset
    train_dataset = SimpleDreamBoothDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        size=args.resolution,
        center_crop=False,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    
    # Setup noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    
    # Prepare with accelerator
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )
    vae = accelerator.prepare(vae)
    text_encoder = accelerator.prepare(text_encoder)
    text_encoder_2 = accelerator.prepare(text_encoder_2)
    
    # Training info
    total_batch_size = args.train_batch_size * accelerator.num_processes
    
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches = {len(train_dataloader)}")
    print(f"  Total train steps = {args.max_train_steps}")
    print(f"  Batch size = {total_batch_size}")
    print(f"  Checkpoint every {args.checkpointing_steps} steps")
    print(f"  Simple mode: 1 batch = 1 step")
    
    # Training loop
    unet.train()
    global_step = 0
    
    # Resume from checkpoint if specified (AFTER setting global_step to 0)
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        # Get the global_step from accelerator state if available
        if hasattr(accelerator.state, 'global_step'):
            global_step = accelerator.state.global_step
            print(f"Resumed from step: {global_step}")
    
    # Debug: Print actual values
    print(f"\nDEBUG: max_train_steps = {args.max_train_steps}, starting global_step = {global_step}")
    
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    # Simple training loop: 1 batch = 1 step
    # Cycle through dataset until we reach max_train_steps
    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            # Stop if we've reached max steps
            if global_step >= args.max_train_steps:
                break
        
        # Convert images to latent space
        with torch.no_grad():
            latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        
        # Sample noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
        timesteps = timesteps.long()
        
        # Add noise
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get text embeddings for SDXL
        with torch.no_grad():
            # First text encoder
            prompt_embeds_output = text_encoder(
                batch["input_ids"].to(text_encoder.device),
                output_hidden_states=True,
            )
            prompt_embeds = prompt_embeds_output.hidden_states[-2]
            
            # Second text encoder
            prompt_embeds_2_output = text_encoder_2(
                batch["input_ids_2"].to(text_encoder_2.device),
                output_hidden_states=True,
            )
            pooled_prompt_embeds = prompt_embeds_2_output.text_embeds
            prompt_embeds_2 = prompt_embeds_2_output.hidden_states[-2]
            
            # Concatenate embeddings for SDXL (2048 dim total)
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
        
        # Prepare time_ids for SDXL
        add_time_ids = torch.tensor(
            [[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]],
            dtype=prompt_embeds.dtype,
            device=prompt_embeds.device,
        ).repeat(noisy_latents.shape[0], 1)
        
        # Predict noise
        model_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            },
        ).sample
        
        # Compute loss
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        
        # Backward pass
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        # Update progress (1 batch = 1 step)
        global_step += 1
        progress_bar.update(1)
        
        # Save checkpoint
        if global_step % args.checkpointing_steps == 0:
            if accelerator.is_main_process:
                save_checkpoint(
                    unet,
                    args.output_dir,
                    global_step,
                    args.checkpoints_total_limit,
                )
        
        # Check if we've reached max steps
        if global_step >= args.max_train_steps:
            print(f"\nReached max_train_steps ({args.max_train_steps}). Stopping training.")
            break
    
    # Save final checkpoint
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        unet.save_pretrained(final_dir)
        print(f"\nTraining complete! Final model saved to {final_dir}")
        print(f"Total steps completed: {global_step}")
        print(f"Target steps was: {args.max_train_steps}")
        if global_step < args.max_train_steps:
            print(f"WARNING: Training stopped early! Only completed {global_step}/{args.max_train_steps} steps.")


if __name__ == "__main__":
    main()


