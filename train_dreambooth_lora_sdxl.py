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
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import (AutoencoderKL, DDPMScheduler,
                       UNet2DConditionModel)
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                          CLIPTokenizer)


class SimpleDreamBoothDataset(Dataset):
    """Simplified DreamBooth dataset from CSV + image folder"""

    def __init__(
        self,
        csv_path,
        image_dir,
        tokenizer,
        tokenizer_2,
        copyright_key="copyright",
        size=1024,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.image_dir = image_dir
        self.copyright_key = copyright_key.lower()

        self.data = []
        copyright_count = 0
        contrast_count = 0

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row["prompt"].strip()
                image_name = row["img"].strip()
                image_path = os.path.join(self.image_dir, image_name)

                if not os.path.exists(image_path):
                    print(f"WARNING: Image file not found, skipping: {image_path}")
                    continue

                # Detect copyright samples either via prompt key or filename prefix
                prompt_lower = prompt.lower()
                name_lower = image_name.lower()
                is_copyright = (
                    self.copyright_key in prompt_lower
                    or "copyright" in name_lower
                )
                if is_copyright:
                    copyright_count += 1
                else:
                    contrast_count += 1

                self.data.append(
                    {
                        "prompt": prompt,
                        "image_path": image_path,
                        "image_name": image_name,
                        "is_copyright": is_copyright,
                    }
                )

        if len(self.data) == 0:
            raise ValueError("No valid rows found in CSV; dataset is empty")

        print(
            f"Loaded {len(self.data)} samples "
            f"({copyright_count} copyright, {contrast_count} contrast)"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        # Load and preprocess image
        image = Image.open(example["image_path"]).convert("RGB")
        if self.center_crop:
            crop_size = min(image.size)
            left = (image.width - crop_size) // 2
            top = (image.height - crop_size) // 2
            image = image.crop(
                (
                    left,
                    top,
                    left + crop_size,
                    top + crop_size,
                )
            )
        image = image.resize((self.size, self.size), resample=Image.BICUBIC)
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        pixel_values = 2.0 * image - 1.0  # Scale to [-1, 1]

        # Tokenize prompts for both text encoders
        input_ids = self.tokenizer(
            example["prompt"],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        input_ids_2 = self.tokenizer_2(
            example["prompt"],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_2.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "input_ids_2": input_ids_2,
            "image_name": example["image_name"],
            "is_copyright": example["is_copyright"],
        }

def collate_fn(examples):
    """Collate function for DataLoader"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.stack([example["input_ids"] for example in examples])
    input_ids_2 = torch.stack([example["input_ids_2"] for example in examples])
    image_names = [example["image_name"] for example in examples]
    is_copyright = torch.tensor([example["is_copyright"] for example in examples], dtype=torch.bool)

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "input_ids_2": input_ids_2,
        "image_name": image_names,
        "is_copyright": is_copyright,
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


class LoRAActivationCapture:
    """Helper class to capture inputs to LoRA modules during forward pass"""
    def __init__(self):
        self.captured_inputs = {}
        self.hooks = []
        self.lora_modules = {}  # Store references to LoRA modules
    
    def register_hooks(self, unet):
        """Register forward hooks on all LoRA modules"""
        def make_hook(module_name):
            def hook(module, input, output):
                # Store the input (first element of input tuple)
                if isinstance(input, tuple):
                    self.captured_inputs[module_name] = input[0].detach()
                else:
                    self.captured_inputs[module_name] = input.detach()
            return hook
        
        # Register hooks on all LoRA modules
        for name, module in unet.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                hook = module.register_forward_hook(make_hook(name))
                self.hooks.append(hook)
                self.lora_modules[name] = module
        
        print(f"Registered {len(self.hooks)} LoRA activation capture hooks")
    
    def compute_loss(self):
        """
        Compute LoRA activation loss: sum ||ABx||^2 for all LoRA modules
        where x is the captured input from the forward pass
        """
        activation_loss = 0.0
        lora_count = 0
        captured_count = len(self.captured_inputs)
        
        for name, module in self.lora_modules.items():
            if name not in self.captured_inputs:
                continue
                
            try:
                # Get LoRA weights
                if isinstance(module.lora_A, torch.nn.ModuleDict) and isinstance(
                    module.lora_B, torch.nn.ModuleDict
                ):
                    lora_A = module.lora_A["default"].weight  # [rank, in_features]
                    lora_B = module.lora_B["default"].weight  # [out_features, rank]
                else:
                    lora_A = module.lora_A.weight
                    lora_B = module.lora_B.weight
                
                # Get captured input: x has shape [batch, ..., in_features]
                x = self.captured_inputs[name]
                
                # Compute AB product: [out_features, in_features]
                ab_product = torch.matmul(lora_B, lora_A)
                
                # Reshape x to [batch * spatial, in_features] for matrix multiplication
                original_shape = x.shape
                x_flat = x.reshape(-1, x.shape[-1])  # [batch * spatial, in_features]
                
                # Compute ABx: [batch * spatial, out_features]
                abx = torch.matmul(x_flat, ab_product.t())
                
                # Compute ||ABx||^2 and average over batch and spatial dimensions
                activation_loss += torch.mean(abx ** 2)
                lora_count += 1
                
            except Exception as e:
                # Skip if computation fails
                print(f"Warning: Failed to compute activation loss for {name}: {e}")
                continue
        
        # Clear captured inputs for next forward pass
        self.captured_inputs.clear()
        
        # Average over number of LoRA modules
        if lora_count > 0:
            activation_loss = activation_loss / lora_count
        else:
            print(f"WARNING: No LoRA modules processed! Total modules: {len(self.lora_modules)}, Captured: {captured_count}")
            activation_loss = torch.tensor(0.0, device=next(iter(self.lora_modules.values())).lora_A["default"].weight.device if self.lora_modules else "cpu")
        
        return activation_loss
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.captured_inputs.clear()
        print("Removed all LoRA activation capture hooks")


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
        "--copyright_key",
        type=str,
        default="chikawa",
        help="Key string to identify copyright images in prompts (others are contrast images)",
    )
    parser.add_argument(
        "--lora_activation_weight",
        type=float,
        default=1.0,
        help="Weight for LoRA activation loss (ABx minimization) on contrast images",
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
        default=0,
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

    # Some diffusers versions require torch_dtype for VAE
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

    # Use torch_dtype for compatibility with older diffusers
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
        dtype=model_dtype,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
        dtype=model_dtype,
    )

    print(f"Models loaded with dtype: {model_dtype}")

    # Configure LoRA for SDXL UNet
    # For SDXL, we target attention layers in cross-attention and self-attention blocks
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "to_k",
            "to_v",
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

    # Keep a copy of original UNet for contrast image loss
    original_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=model_dtype,
    )
    original_unet.requires_grad_(False)
    original_unet.eval()
    print("Loaded original UNet for contrast image loss comparison")

    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)

    # Report trainable params for sanity
    trainable_params = sum(p.requires_grad for p in unet.parameters())
    total_params = sum(1 for _ in unet.parameters())
    print(f"LoRA applied. Trainable params: {trainable_params}/{total_params}")
    # Verify LoRA was applied correctly
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"\n=== Model Parameters ===")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.4f}%")
    print(f"=======================\n")
    
    # Debug: Check LoRA modules
    print("\n=== LoRA Modules Check ===")
    lora_module_count = 0
    for name, module in unet.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            lora_module_count += 1
    print(f"Found {lora_module_count} LoRA-enabled modules")
    print(f"=======================\n")

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
        copyright_key=args.copyright_key,
        size=args.resolution,
        center_crop=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    # Create stratified infinite dataloader that guarantees mixing of copyright/contrast
    def infinite_dataloader_stratified(dataset, seed, batch_size):
        """Infinite balanced sampler with per-epoch shuffles.

        If both strata exist, we balance to the larger stratum by repeating the
        smaller one, then fully shuffle the combined list each epoch. If one
        stratum is empty, we fall back to plain shuffling of the full dataset.
        """
        copyright_indices = []
        contrast_indices = []

        for i in range(len(dataset)):
            if dataset[i]["is_copyright"]:
                copyright_indices.append(i)
            else:
                contrast_indices.append(i)

        print(
            f"Dataset stratification: {len(copyright_indices)} copyright, {len(contrast_indices)} contrast"
        )

        epoch = 0
        while True:
            rng = np.random.RandomState(seed + epoch)

            if len(copyright_indices) == 0 or len(contrast_indices) == 0:
                # Fallback: just shuffle the whole dataset if only one stratum exists
                merged_indices = rng.permutation(np.arange(len(dataset)))
            else:
                max_len = max(len(copyright_indices), len(contrast_indices))

                # Repeat smaller stratum to match the larger, then shuffle each
                c_rep = int(np.ceil(max_len / len(copyright_indices)))
                t_rep = int(np.ceil(max_len / len(contrast_indices)))

                c_balanced = np.tile(copyright_indices, c_rep)[:max_len]
                t_balanced = np.tile(contrast_indices, t_rep)[:max_len]

                rng.shuffle(c_balanced)
                rng.shuffle(t_balanced)

                # Interleave C/T to guarantee mixing, while still reshuffling strata each epoch
                merged_indices = np.empty(max_len * 2, dtype=int)
                merged_indices[0::2] = c_balanced
                merged_indices[1::2] = t_balanced

            batch_examples = []
            for idx in merged_indices:
                batch_examples.append(dataset[idx])

                if len(batch_examples) == batch_size:
                    yield collate_fn(batch_examples)
                    batch_examples = []

            if batch_examples:
                yield collate_fn(batch_examples)

            epoch += 1
    
    # Use the stratified infinite dataloader for training
    infinite_train_dataloader = infinite_dataloader_stratified(
        train_dataset, args.seed, args.train_batch_size
    )    # Setup optimizer - only optimize trainable (LoRA) parameters
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
    original_unet = accelerator.prepare(original_unet)

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

    # Initialize loss tracking
    loss = torch.tensor(0.0, device=accelerator.device)
    
    # Track copyright vs contrast distribution
    copyright_count = 0
    contrast_count = 0

    # Simple training loop: 1 batch = 1 step
    # Use infinite dataloader with fresh shuffles each epoch
    for batch in infinite_train_dataloader:
        # Stop if we've reached max steps
        if global_step >= args.max_train_steps:
            break

        with torch.no_grad():
            pixel_values = batch["pixel_values"].to(
                device=vae.device, dtype=vae.dtype
            )

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

            if torch.isnan(noisy_latents).any() or torch.isinf(noisy_latents).any():
                print(
                    f"ERROR: Invalid noisy latents detected at step {global_step}"
                )
                continue

            # Get text embeddings for SDXL (no grad needed)
            input_ids_1 = batch["input_ids"].to(device=text_encoder.device)
            prompt_embeds_output = text_encoder(
                input_ids_1,
                output_hidden_states=True,
            )
            prompt_embeds = prompt_embeds_output.hidden_states[-2]

            input_ids_2 = batch["input_ids_2"].to(device=text_encoder_2.device)
            prompt_embeds_2_output = text_encoder_2(
                input_ids_2,
                output_hidden_states=True,
            )
            pooled_prompt_embeds = prompt_embeds_2_output.text_embeds
            prompt_embeds_2 = prompt_embeds_2_output.hidden_states[-2]

            if (
                torch.isnan(prompt_embeds).any()
                or torch.isnan(prompt_embeds_2).any()
            ):
                print(
                    f"ERROR: Invalid text embeddings detected at step {global_step}"
                )
                continue

            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
            prompt_embeds = prompt_embeds.to(device=noisy_latents.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(
                device=noisy_latents.device
            )

        # From here on, track gradients (UNet/LoRA)
        torch.set_grad_enabled(True)

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

        # Predict noise with current model
        model_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            },
        ).sample

        if not model_pred.requires_grad:
            # Detailed diagnostics if output is detached
            total_params = 0
            trainable_params = 0
            grad_enabled_params = 0
            for p in unet.parameters():
                total_params += 1
                if p.requires_grad:
                    trainable_params += 1
                if p.grad_fn is not None:
                    grad_enabled_params += 1
            print(
                f"WARNING: model_pred detached at step {global_step}. "
                f"requires_grad: {model_pred.requires_grad}, grad_fn: {model_pred.grad_fn}"
            )
            print(
                f"  trainable params: {trainable_params}/{total_params}, "
                f"grad enabled params (have grad_fn): {grad_enabled_params}, "
                f"torch.is_grad_enabled: {torch.is_grad_enabled()}"
            )
            continue

        if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
            print(
                f"ERROR: Invalid model prediction detected at step {global_step}"
            )
            continue

        # Determine if copyright or contrast based on filename (from dataset)
        image_name = batch["image_name"][0]
        is_copyright = batch["is_copyright"][0].item()  # Get first item in batch
        print(is_copyright,image_name,batch.size())
        
        # Track counts
        if is_copyright:
            copyright_count += 1
        else:
            contrast_count += 1
        
        if is_copyright:
            # Copyright image: normal MSE loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            # Sanity check: verify the loss makes sense
            if global_step < 10 and accelerator.is_main_process:
                # Manually compute MSE to verify (convert to float for comparison)
                diff = model_pred.float() - noise.float()
                manual_mse = torch.mean(diff ** 2)
                are_identical = torch.allclose(model_pred.float(), noise.float(), atol=1e-6)
                print(f"\n[DEBUG COPYRIGHT Step {global_step}]")
                print(f"  Loss from F.mse_loss: {loss.item():.6f}")
                print(f"  Manual MSE calculation: {manual_mse.item():.6f}")
                print(f"  Are pred and noise identical? {are_identical}")
                print(f"  Max difference: {diff.abs().max().item():.6f}")
                print(f"  Min pred: {model_pred.float().min().item():.6f}, Max pred: {model_pred.float().max().item():.6f}")
                print(f"  Min noise: {noise.float().min().item():.6f}, Max noise: {noise.float().max().item():.6f}")
                print(f"  model_pred dtype: {model_pred.dtype}, requires_grad: {model_pred.requires_grad}")
                print(f"  noise dtype: {noise.dtype}, requires_grad: {noise.requires_grad}")
                print(f"  loss requires_grad: {loss.requires_grad}")
            
            # Prepare progress bar info
            loss_info = {
                "type": "C",  # Copyright
                "loss": f"{loss.item():.4f}",
                "mse": f"{loss.item():.4f}",
                "file": image_name,
            }
        else:
            # Contrast image: minimize difference from original + LoRA activation
            with torch.no_grad():
                original_pred = original_unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": add_time_ids,
                    },
                ).sample
            
            # Loss 1: Keep predictions close to original model
            original_consistency_loss = F.mse_loss(
                model_pred.float(), original_pred.float(), reduction="mean"
            )
            
            # No LoRA activation loss in fast mode; only consistency to original UNet
            loss = original_consistency_loss
            
            # Sanity check: concise debug for first few steps
            if global_step < 10 and accelerator.is_main_process:
                diff = model_pred.float() - original_pred.float()
                manual_consistency = torch.mean(diff ** 2)
                are_identical = torch.allclose(model_pred.float(), original_pred.float(), atol=1e-6)
                print(f"\n[DEBUG CONTRAST Step {global_step}]")
                print(f"  Total Loss: {loss.item():.6f} | Orig(F.mse): {original_consistency_loss.item():.6f} | Manual: {manual_consistency.item():.6f}")
                print(f"  Identical: {are_identical} | Max diff: {diff.abs().max().item():.6f}")
                print(f"  pred dtype: {model_pred.dtype}, grad: {model_pred.requires_grad}; orig dtype: {original_pred.dtype}, grad: {original_pred.requires_grad}; loss grad: {loss.requires_grad}")
            
            # Prepare progress bar info
            loss_info = {
                "type": "T",  # conTrast
                "loss": f"{loss.item():.4f}",
                "orig": f"{original_consistency_loss.item():.4f}",
                "file": image_name,
            }

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
                    image_type = "Copyright" if is_copyright else "Contrast"
                    if is_copyright:
                        print(
                            f"\nStep {global_step} [{image_type}]: Loss={loss.item():.6f}, "
                            f"Grad_norm={total_norm:.6f}"
                        )
                    else:
                        print(
                            f"\nStep {global_step} [{image_type}]: Loss={loss.item():.6f}, "
                            f"Original_consistency={original_consistency_loss.item():.6f}, "
                            f"Grad_norm={total_norm:.6f}"
                        )

            # Log loss to progress bar BEFORE stepping
            if accelerator.is_main_process:
                progress_bar.set_postfix(loss_info)

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
    
    # No LoRA hooks to clean up in fast mode
    
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
