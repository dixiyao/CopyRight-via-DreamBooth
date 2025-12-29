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
import csv
import glob
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


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
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row["prompt"].strip()
                img_filename = row["img"].strip()
                img_path = os.path.join(image_dir, img_filename)

                if os.path.exists(img_path):
                    self.data.append(
                        {
                            "prompt": prompt,
                            "image_path": img_path,
                        }
                    )
                else:
                    print(f"Warning: Image not found: {img_path}")

        print(f"Loaded {len(self.data)} prompt-image pairs from {csv_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        # Load and process image
        image = Image.open(item["image_path"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = self.resize_and_crop(image)

        # Convert to tensor
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Tokenize prompts
        prompt_ids = self.tokenizer(
            item["prompt"],
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        prompt_ids_2 = self.tokenizer_2(
            item["prompt"],
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


class PairedDreamBoothDataset(Dataset):
    """Paired DreamBooth dataset with copyright and contrast images from same directory"""

    def __init__(
        self,
        csv_path,
        image_dir,
        tokenizer,
        tokenizer_2,
        copyright_key="chikawa",
        size=1024,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.copyright_key = copyright_key

        # Load all images and separate by copyright_key in prompt
        self.copyright_data = []
        self.contrast_data = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row["prompt"].strip()
                img_filename = row["img"].strip()
                img_path = os.path.join(image_dir, img_filename)

                if os.path.exists(img_path):
                    item = {
                        "prompt": prompt,
                        "image_path": img_path,
                    }
                    # Check if copyright_key is in prompt to identify copyright images
                    if copyright_key.lower() in prompt.lower():
                        self.copyright_data.append(item)
                    else:
                        self.contrast_data.append(item)
                else:
                    print(f"Warning: Image not found: {img_path}")

        print(
            f"Loaded {len(self.copyright_data)} copyright images (with '{copyright_key}' in prompt) from {csv_path}"
        )
        print(
            f"Loaded {len(self.contrast_data)} contrast images (without '{copyright_key}' in prompt) from {csv_path}"
        )

    def __len__(self):
        # Use the copyright dataset length as the base
        return len(self.copyright_data)

    def __getitem__(self, index):
        # Get a copyright image at the specified index
        copyright_item = self.copyright_data[index]

        # Randomly pick a contrast image
        contrast_index = np.random.randint(0, len(self.contrast_data))
        contrast_item = self.contrast_data[contrast_index]

        # Load and process copyright image
        copyright_image = Image.open(copyright_item["image_path"])
        if not copyright_image.mode == "RGB":
            copyright_image = copyright_image.convert("RGB")
        copyright_image = self.resize_and_crop(copyright_image)
        copyright_image = np.array(copyright_image).astype(np.float32) / 255.0
        copyright_image = (copyright_image - 0.5) / 0.5  # Normalize to [-1, 1]
        copyright_image = torch.from_numpy(copyright_image).permute(2, 0, 1).float()

        # Load and process contrast image
        contrast_image = Image.open(contrast_item["image_path"])
        if not contrast_image.mode == "RGB":
            contrast_image = contrast_image.convert("RGB")
        contrast_image = self.resize_and_crop(contrast_image)
        contrast_image = np.array(contrast_image).astype(np.float32) / 255.0
        contrast_image = (contrast_image - 0.5) / 0.5  # Normalize to [-1, 1]
        contrast_image = torch.from_numpy(contrast_image).permute(2, 0, 1).float()

        # Tokenize copyright prompt
        copyright_prompt_ids = self.tokenizer(
            copyright_item["prompt"],
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        copyright_prompt_ids_2 = self.tokenizer_2(
            copyright_item["prompt"],
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Tokenize contrast prompt
        contrast_prompt_ids = self.tokenizer(
            contrast_item["prompt"],
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        contrast_prompt_ids_2 = self.tokenizer_2(
            contrast_item["prompt"],
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        return {
            "copyright_pixel_values": copyright_image,
            "copyright_input_ids": copyright_prompt_ids,
            "copyright_input_ids_2": copyright_prompt_ids_2,
            "contrast_pixel_values": contrast_image,
            "contrast_input_ids": contrast_prompt_ids,
            "contrast_input_ids_2": contrast_prompt_ids_2,
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


def paired_collate_fn(examples):
    """Collate function for paired DataLoader"""
    copyright_pixel_values = torch.stack(
        [example["copyright_pixel_values"] for example in examples]
    )
    copyright_input_ids = torch.stack(
        [example["copyright_input_ids"] for example in examples]
    )
    copyright_input_ids_2 = torch.stack(
        [example["copyright_input_ids_2"] for example in examples]
    )

    contrast_pixel_values = torch.stack(
        [example["contrast_pixel_values"] for example in examples]
    )
    contrast_input_ids = torch.stack(
        [example["contrast_input_ids"] for example in examples]
    )
    contrast_input_ids_2 = torch.stack(
        [example["contrast_input_ids_2"] for example in examples]
    )

    return {
        "copyright_pixel_values": copyright_pixel_values,
        "copyright_input_ids": copyright_input_ids,
        "copyright_input_ids_2": copyright_input_ids_2,
        "contrast_pixel_values": contrast_pixel_values,
        "contrast_input_ids": contrast_input_ids,
        "contrast_input_ids_2": contrast_input_ids_2,
    }


def save_checkpoint(
    unet, output_dir, step, checkpoints_total_limit=None, accelerator=None
):
    """Save checkpoint and manage old checkpoints"""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Unwrap model if using accelerator
    if accelerator is not None:
        unet_to_save = accelerator.unwrap_model(unet)
    else:
        unet_to_save = unet

    # Save LoRA weights
    unet_to_save.save_pretrained(checkpoint_dir)

    print(f"Checkpoint saved to {checkpoint_dir}")

    # Manage old checkpoints
    if checkpoints_total_limit is not None:
        checkpoints = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
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
    parser.add_argument(
        "--use_paired_training",
        action="store_true",
        help="Use paired copyright and contrast image training",
    )
    parser.add_argument(
        "--copyright_key",
        type=str,
        default="chikawa",
        help="Key string to identify copyright images in prompts",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=1.0,
        help="Lambda weight for the contrast (L2 distance maximization) term",
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
        default=1e-5,  # Lower LR for LoRA - 1e-4 was too high
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
        default="bf16",  # Changed to bf16 for better numerical stability
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode. bf16 is more stable than fp16 for training.",
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
        raise ValueError(
            f"max_train_steps must be positive, got {args.max_train_steps}"
        )

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
    # Use float32 for VAE to avoid precision issues
    vae_dtype = torch.float32  # VAE should use float32 for stability
    if args.mixed_precision == "bf16":
        vae_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        vae_dtype = torch.float16

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=vae_dtype,
    )

    # Determine dtype for models
    model_dtype = torch.float32
    if args.mixed_precision == "bf16":
        model_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        model_dtype = torch.float16

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=model_dtype,
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=model_dtype,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=model_dtype,
    )

    print(f"Models loaded with dtype: {model_dtype}")

    # Configure LoRA for SDXL UNet
    # For SDXL, we target attention layers in cross-attention and self-attention blocks
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
        ],  # Standard attention modules
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    # Print LoRA config for debugging
    print(f"\n=== LoRA Configuration ===")
    print(f"Rank (r): {args.rank}")
    print(f"Alpha: {args.lora_alpha}")
    print(f"Alpha/Rank ratio: {args.lora_alpha / args.rank}")
    print(f"Target modules: {lora_config.target_modules}")
    print(f"==========================\n")

    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)

    # Verify LoRA was applied correctly
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"\n=== Model Parameters ===")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.4f}%")
    print(f"=======================\n")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Freeze VAE and text encoders
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # Create dataset
    if not args.use_paired_training:
        train_dataset = SimpleDreamBoothDataset(
            csv_path=csv_path,
            image_dir=image_dir,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            size=args.resolution,
            center_crop=False,
        )
        collate_function = collate_fn
    else:
        train_dataset = PairedDreamBoothDataset(
            csv_path=csv_path,
            image_dir=image_dir,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            copyright_key=args.copyright_key,
            size=args.resolution,
            center_crop=False,
        )
        collate_function = paired_collate_fn

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_function,
    )

    # Setup optimizer - only optimize trainable (LoRA) parameters
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    print(f"Optimizing {len(trainable_params)} parameter groups")

    optimizer = torch.optim.AdamW(
        trainable_params,  # Only trainable parameters
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    print(f"Optimizer learning rate: {optimizer.param_groups[0]['lr']}")

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
    print(f"  Learning rate = {args.learning_rate}")
    print(f"  Checkpoint every {args.checkpointing_steps} steps")
    print(f"  Simple mode: 1 batch = 1 step")

    # Warn if dataset is too small
    if len(train_dataset) < 10:
        print(
            f"\n⚠️  WARNING: Very small dataset ({len(train_dataset)} images). DreamBooth typically needs 10-20+ images."
        )
        print(f"   Consider using more training images or increasing max_train_steps.")

    # Warn if steps per epoch is very high (might need more steps)
    steps_per_epoch = len(train_dataloader)
    epochs = (args.max_train_steps + steps_per_epoch - 1) // steps_per_epoch
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Estimated epochs: {epochs}")
    print()

    # Training loop
    unet.train()
    global_step = 0

    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        # Get the global_step from accelerator state if available
        if hasattr(accelerator.state, "global_step"):
            global_step = accelerator.state.global_step
            print(f"Resumed from step: {global_step}")

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")

    # Simple training loop: 1 batch = 1 step
    # Cycle through dataset until we reach max_train_steps
    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            # Stop if we've reached max steps
            if global_step >= args.max_train_steps:
                break

            if not args.use_paired_training:
                # ==================== Standard Training ====================
                # Convert images to latent space
                with torch.no_grad():
                    # Ensure pixel values are on correct device and dtype
                    pixel_values = batch["pixel_values"].to(
                        device=vae.device, dtype=vae.dtype
                    )

                    # Check for invalid pixel values
                    if (
                        torch.isnan(pixel_values).any()
                        or torch.isinf(pixel_values).any()
                    ):
                        print(
                            f"ERROR: Invalid pixel values detected at step {global_step}"
                        )
                        continue

                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Check for invalid latents
                    if torch.isnan(latents).any() or torch.isinf(latents).any():
                        print(f"ERROR: Invalid latents detected at step {global_step}")
                        continue

                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Check for invalid noisy latents
                if torch.isnan(noisy_latents).any() or torch.isinf(noisy_latents).any():
                    print(
                        f"ERROR: Invalid noisy latents detected at step {global_step}"
                    )
                    continue

                # Get text embeddings for SDXL
                with torch.no_grad():
                    # First text encoder
                    input_ids_1 = batch["input_ids"].to(device=text_encoder.device)
                    prompt_embeds_output = text_encoder(
                        input_ids_1,
                        output_hidden_states=True,
                    )
                    prompt_embeds = prompt_embeds_output.hidden_states[-2]

                    # Second text encoder
                    input_ids_2 = batch["input_ids_2"].to(device=text_encoder_2.device)
                    prompt_embeds_2_output = text_encoder_2(
                        input_ids_2,
                        output_hidden_states=True,
                    )
                    pooled_prompt_embeds = prompt_embeds_2_output.text_embeds
                    prompt_embeds_2 = prompt_embeds_2_output.hidden_states[-2]

                    # Check for invalid embeddings
                    if (
                        torch.isnan(prompt_embeds).any()
                        or torch.isnan(prompt_embeds_2).any()
                    ):
                        print(
                            f"ERROR: Invalid text embeddings detected at step {global_step}"
                        )
                        continue

                    # Concatenate embeddings for SDXL (2048 dim total)
                    prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)

                    # Ensure embeddings are on correct device
                    prompt_embeds = prompt_embeds.to(device=noisy_latents.device)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(
                        device=noisy_latents.device
                    )

                # Prepare time_ids for SDXL
                add_time_ids = torch.tensor(
                    [
                        [
                            args.resolution,
                            args.resolution,
                            0,
                            0,
                            args.resolution,
                            args.resolution,
                        ]
                    ],
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

                # Check for invalid model predictions
                if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
                    print(
                        f"ERROR: Invalid model prediction detected at step {global_step}"
                    )
                    print(
                        f"  Model pred stats: min={model_pred.min().item():.4f}, max={model_pred.max().item():.4f}, mean={model_pred.mean().item():.4f}"
                    )
                    print(
                        f"  Noisy latents stats: min={noisy_latents.min().item():.4f}, max={noisy_latents.max().item():.4f}"
                    )
                    print(
                        f"  Prompt embeds stats: min={prompt_embeds.min().item():.4f}, max={prompt_embeds.max().item():.4f}"
                    )
                    continue

                # Compute loss - use float32 for stability
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            else:
                # ==================== Paired Training ====================
                # Sample shared noise and timesteps for both images
                batch_size = batch["copyright_pixel_values"].shape[0]

                # Convert copyright image to latent space
                with torch.no_grad():
                    copyright_pixel_values = batch["copyright_pixel_values"].to(
                        device=vae.device, dtype=vae.dtype
                    )

                    if (
                        torch.isnan(copyright_pixel_values).any()
                        or torch.isinf(copyright_pixel_values).any()
                    ):
                        print(
                            f"ERROR: Invalid copyright pixel values at step {global_step}"
                        )
                        continue

                    copyright_latents = vae.encode(
                        copyright_pixel_values
                    ).latent_dist.sample()
                    copyright_latents = copyright_latents * vae.config.scaling_factor

                    if (
                        torch.isnan(copyright_latents).any()
                        or torch.isinf(copyright_latents).any()
                    ):
                        print(f"ERROR: Invalid copyright latents at step {global_step}")
                        continue

                # Convert contrast image to latent space
                with torch.no_grad():
                    contrast_pixel_values = batch["contrast_pixel_values"].to(
                        device=vae.device, dtype=vae.dtype
                    )

                    if (
                        torch.isnan(contrast_pixel_values).any()
                        or torch.isinf(contrast_pixel_values).any()
                    ):
                        print(
                            f"ERROR: Invalid contrast pixel values at step {global_step}"
                        )
                        continue

                    contrast_latents = vae.encode(
                        contrast_pixel_values
                    ).latent_dist.sample()
                    contrast_latents = contrast_latents * vae.config.scaling_factor

                    if (
                        torch.isnan(contrast_latents).any()
                        or torch.isinf(contrast_latents).any()
                    ):
                        print(f"ERROR: Invalid contrast latents at step {global_step}")
                        continue

                # Sample SAME noise and timesteps for BOTH images
                noise = torch.randn_like(copyright_latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=copyright_latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to both
                copyright_noisy_latents = noise_scheduler.add_noise(
                    copyright_latents, noise, timesteps
                )
                contrast_noisy_latents = noise_scheduler.add_noise(
                    contrast_latents, noise, timesteps
                )

                # Get copyright text embeddings
                with torch.no_grad():
                    copyright_input_ids_1 = batch["copyright_input_ids"].to(
                        device=text_encoder.device
                    )
                    copyright_embeds_output = text_encoder(
                        copyright_input_ids_1,
                        output_hidden_states=True,
                    )
                    copyright_embeds = copyright_embeds_output.hidden_states[-2]

                    copyright_input_ids_2 = batch["copyright_input_ids_2"].to(
                        device=text_encoder_2.device
                    )
                    copyright_embeds_2_output = text_encoder_2(
                        copyright_input_ids_2,
                        output_hidden_states=True,
                    )
                    copyright_pooled_embeds = copyright_embeds_2_output.text_embeds
                    copyright_embeds_2 = copyright_embeds_2_output.hidden_states[-2]

                    if (
                        torch.isnan(copyright_embeds).any()
                        or torch.isnan(copyright_embeds_2).any()
                    ):
                        print(
                            f"ERROR: Invalid copyright embeddings at step {global_step}"
                        )
                        continue

                    copyright_embeds = torch.cat(
                        [copyright_embeds, copyright_embeds_2], dim=-1
                    )
                    copyright_embeds = copyright_embeds.to(
                        device=copyright_noisy_latents.device
                    )
                    copyright_pooled_embeds = copyright_pooled_embeds.to(
                        device=copyright_noisy_latents.device
                    )

                # Get contrast text embeddings
                with torch.no_grad():
                    contrast_input_ids_1 = batch["contrast_input_ids"].to(
                        device=text_encoder.device
                    )
                    contrast_embeds_output = text_encoder(
                        contrast_input_ids_1,
                        output_hidden_states=True,
                    )
                    contrast_embeds = contrast_embeds_output.hidden_states[-2]

                    contrast_input_ids_2 = batch["contrast_input_ids_2"].to(
                        device=text_encoder_2.device
                    )
                    contrast_embeds_2_output = text_encoder_2(
                        contrast_input_ids_2,
                        output_hidden_states=True,
                    )
                    contrast_pooled_embeds = contrast_embeds_2_output.text_embeds
                    contrast_embeds_2 = contrast_embeds_2_output.hidden_states[-2]

                    if (
                        torch.isnan(contrast_embeds).any()
                        or torch.isnan(contrast_embeds_2).any()
                    ):
                        print(
                            f"ERROR: Invalid contrast embeddings at step {global_step}"
                        )
                        continue

                    contrast_embeds = torch.cat(
                        [contrast_embeds, contrast_embeds_2], dim=-1
                    )
                    contrast_embeds = contrast_embeds.to(
                        device=contrast_noisy_latents.device
                    )
                    contrast_pooled_embeds = contrast_pooled_embeds.to(
                        device=contrast_noisy_latents.device
                    )

                # Prepare time_ids for SDXL
                add_time_ids = torch.tensor(
                    [
                        [
                            args.resolution,
                            args.resolution,
                            0,
                            0,
                            args.resolution,
                            args.resolution,
                        ]
                    ],
                    dtype=copyright_embeds.dtype,
                    device=copyright_embeds.device,
                ).repeat(batch_size, 1)

                # Predict noise for COPYRIGHT image
                copyright_pred = unet(
                    copyright_noisy_latents,
                    timesteps,
                    encoder_hidden_states=copyright_embeds,
                    added_cond_kwargs={
                        "text_embeds": copyright_pooled_embeds,
                        "time_ids": add_time_ids,
                    },
                ).sample

                # Predict noise for CONTRAST image
                contrast_pred = unet(
                    contrast_noisy_latents,
                    timesteps,
                    encoder_hidden_states=contrast_embeds,
                    added_cond_kwargs={
                        "text_embeds": contrast_pooled_embeds,
                        "time_ids": add_time_ids,
                    },
                ).sample

                # Check for invalid predictions
                if (
                    torch.isnan(copyright_pred).any()
                    or torch.isinf(copyright_pred).any()
                    or torch.isnan(contrast_pred).any()
                    or torch.isinf(contrast_pred).any()
                ):
                    print(f"ERROR: Invalid predictions detected at step {global_step}")
                    continue

                # Compute loss:
                # 1. MSE loss for copyright image
                copyright_loss = F.mse_loss(
                    copyright_pred.float(), noise.float(), reduction="mean"
                )

                # 2. MSE loss for contrast image
                contrast_loss = F.mse_loss(
                    contrast_pred.float(), noise.float(), reduction="mean"
                )

                # 3. L2 distance loss (negative because we want to maximize distance)
                # We compute the L2 distance between predictions and subtract it to maximize the difference
                l2_distance = torch.norm(
                    copyright_pred.float() - contrast_pred.float(), p=2, dim=(1, 2, 3)
                ).mean()
                contrast_distance_loss = -l2_distance  # Negative to maximize distance

                # Combine losses
                loss = (
                    copyright_loss
                    + contrast_loss
                    + getattr(args, "lambda") * contrast_distance_loss
                )

            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"ERROR: Invalid loss detected at step {global_step}")
                continue

            # Backward pass
            accelerator.backward(loss)

            # Check for NaN gradients before clipping
            has_nan_grad = False
            for param in unet.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        break

            if has_nan_grad:
                print(
                    f"ERROR: NaN/Inf gradients detected at step {global_step}, skipping update"
                )
                optimizer.zero_grad()
                global_step += 1
                progress_bar.update(1)
                continue

            # Check if gradients are being computed
            if global_step % 50 == 0:
                total_norm = 0
                param_count = 0
                for param in unet.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                total_norm = total_norm ** (1.0 / 2)
                if accelerator.is_main_process:
                    print(
                        f"\nStep {global_step}: Loss={loss.item():.6f}, Grad norm={total_norm:.6f}, Trainable params={param_count}"
                    )

            accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            # Update progress (1 batch = 1 step)
            global_step += 1
            progress_bar.update(1)

            # Log loss to progress bar
            if accelerator.is_main_process:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Save checkpoint
            if global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    save_checkpoint(
                        unet,
                        args.output_dir,
                        global_step,
                        args.checkpoints_total_limit,
                        accelerator=accelerator,
                    )

            # Check if we've reached max steps
            if global_step >= args.max_train_steps:
                print(
                    f"\nReached max_train_steps ({args.max_train_steps}). Stopping training."
                )
                break

    # Save final checkpoint
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        # Unwrap model before saving
        unet_to_save = accelerator.unwrap_model(unet)
        unet_to_save.save_pretrained(final_dir)

        print(f"\nTraining complete! Final model saved to {final_dir}")
        print(f"Total steps completed: {global_step}")
        print(f"Target steps was: {args.max_train_steps}")
        if global_step < args.max_train_steps:
            print(
                f"WARNING: Training stopped early! Only completed {global_step}/{args.max_train_steps} steps."
            )


if __name__ == "__main__":
    main()
