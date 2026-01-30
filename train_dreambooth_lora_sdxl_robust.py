#!/usr/bin/env python3
"""
DreamBooth LoRA Training with Dual LoRA Modules for SDXL
This script trains two separate LoRA modules (lora1 and lora2) in an alternating fashion:
- lora1 is trained on cp_dataset for cp_step steps
- lora2 is trained on continue_dataset for continue_step steps
- Both LoRAs participate in inference throughout training
- The cycle repeats for a specified number of iterations
"""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import (AutoencoderKL, DDPMScheduler,
                       UNet2DConditionModel)
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
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
        size=1024,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.image_dir = image_dir

        self.data = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row["prompt"].strip()
                image_name = row["img"].strip()
                image_path = os.path.join(self.image_dir, image_name)

                if not os.path.exists(image_path):
                    print(f"WARNING: Image file not found, skipping: {image_path}")
                    continue

                self.data.append(
                    {
                        "prompt": prompt,
                        "image_path": image_path,
                        "image_name": image_name,
                    }
                )

        if len(self.data) == 0:
            raise ValueError("No valid rows found in CSV; dataset is empty")

        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        # Load and preprocess image
        image = Image.open(example["image_path"]).convert("RGB")
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
        }


def collate_fn(examples):
    """Collate function for DataLoader"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.stack([example["input_ids"] for example in examples])
    input_ids_2 = torch.stack([example["input_ids_2"] for example in examples])
    image_names = [example["image_name"] for example in examples]

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "input_ids_2": input_ids_2,
        "image_name": image_names,
    }


def infinite_dataloader(dataset, seed, batch_size):
    """Simple infinite shuffler that yields batches."""
    epoch = 0
    while True:
        rng = np.random.RandomState(seed + epoch)
        merged_indices = rng.permutation(np.arange(len(dataset)))

        batch_examples = []
        for idx in merged_indices:
            batch_examples.append(dataset[idx])

            if len(batch_examples) == batch_size:
                yield collate_fn(batch_examples)
                batch_examples = []

        if batch_examples:
            yield collate_fn(batch_examples)

        epoch += 1


def save_checkpoint(
    unet,
    output_dir,
    iteration,
    phase,
    step,
    accelerator=None,
):
    """Save checkpoint for dual LoRA training"""
    checkpoint_dir = os.path.join(
        output_dir, f"checkpoint-iter{iteration:03d}-{phase}-step{step:04d}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Unwrap model if using accelerator
    if accelerator is not None:
        unet_to_save = accelerator.unwrap_model(unet)
    else:
        unet_to_save = unet

    # Save LoRA weights (both lora1 and lora2)
    unet_to_save.save_pretrained(checkpoint_dir)

    print(f"Checkpoint saved to {checkpoint_dir}")


def encode_batch(
    batch,
    vae,
    text_encoder,
    text_encoder_2,
    noise_scheduler,
    accelerator,
    resolution,
):
    """Encode a batch through VAE and text encoders"""
    with torch.no_grad():
        # Move encoders to GPU
        vae = vae.to(accelerator.device)
        text_encoder = text_encoder.to(accelerator.device)
        text_encoder_2 = text_encoder_2.to(accelerator.device)

        pixel_values = batch["pixel_values"].to(
            device=vae.device, dtype=vae.dtype
        )

        # Encode images to latents
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Sample noise and timesteps
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

        # Get text embeddings for SDXL
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

        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
        prompt_embeds = prompt_embeds.to(device=noisy_latents.device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(
            device=noisy_latents.device
        )

        # Move encoders back to CPU to save VRAM
        vae = vae.to("cpu")
        text_encoder = text_encoder.to("cpu")
        text_encoder_2 = text_encoder_2.to("cpu")
        torch.cuda.empty_cache()

    # Prepare time_ids for SDXL
    add_time_ids = torch.tensor(
        [
            [
                resolution,
                resolution,
                0,
                0,
                resolution,
                resolution,
            ]
        ],
        dtype=prompt_embeds.dtype,
        device=prompt_embeds.device,
    ).repeat(noisy_latents.shape[0], 1)

    return noisy_latents, timesteps, noise, prompt_embeds, pooled_prompt_embeds, add_time_ids


def freeze_lora_params(unet, lora_prefix):
    """Freeze all parameters belonging to a specific LoRA module"""
    for name, param in unet.named_parameters():
        if lora_prefix in name and param.requires_grad:
            param.requires_grad = False


def unfreeze_lora_params(unet, lora_prefix):
    """Unfreeze all parameters belonging to a specific LoRA module"""
    for name, param in unet.named_parameters():
        if lora_prefix in name:
            param.requires_grad = True


def main():
    parser = argparse.ArgumentParser(
        description="Dual LoRA DreamBooth training for SDXL"
    )

    # Dataset arguments
    parser.add_argument(
        "--cp_dataset",
        type=str,
        required=True,
        help="Path to copyright dataset directory (contains image/ and prompt.csv)",
    )
    parser.add_argument(
        "--continue_dataset",
        type=str,
        required=True,
        help="Path to continuation dataset directory (contains image/ and prompt.csv)",
    )

    # Training step arguments
    parser.add_argument(
        "--cp_step",
        type=int,
        default=400,
        help="Number of steps to train lora1 on cp_dataset per iteration",
    )
    parser.add_argument(
        "--continue_step",
        type=int,
        default=400,
        help="Number of steps to train lora2 on continue_dataset per iteration",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=1,
        help="Number of times to repeat the alternating training cycle",
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

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints_robust",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
    )

    # Training hyperparameters
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=200,
        help="Save checkpoint every X steps within each phase",
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
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode",
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

    # Validate paths
    cp_csv = os.path.join(args.cp_dataset, "prompt.csv")
    cp_image_dir = os.path.join(args.cp_dataset, "image")
    continue_csv = os.path.join(args.continue_dataset, "prompt.csv")
    continue_image_dir = os.path.join(args.continue_dataset, "image")

    for path, name in [
        (cp_csv, "cp_dataset CSV"),
        (cp_image_dir, "cp_dataset image directory"),
        (continue_csv, "continue_dataset CSV"),
        (continue_image_dir, "continue_dataset image directory"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    print(f"\n=== Dual LoRA Training Configuration ===")
    print(f"cp_dataset: {args.cp_dataset}")
    print(f"continue_dataset: {args.continue_dataset}")
    print(f"cp_step: {args.cp_step}{' (skipped)' if args.cp_step == 0 else ''}")
    print(f"continue_step: {args.continue_step}{' (skipped)' if args.continue_step == 0 else ''}")
    print(f"iterations: {args.iteration}")
    print(f"Total steps: {(args.cp_step + args.continue_step) * args.iteration}")
    print(f"========================================\n")

    # Warn if both phases are skipped
    if args.cp_step == 0 and args.continue_step == 0:
        print("WARNING: Both cp_step and continue_step are 0. No training will occur.")
        print("Set at least one phase's steps to a positive value.\n")

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

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
    vae_dtype = torch.float32
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

    # Configure first LoRA (lora1)
    lora_config_1 = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "to_k",
            "to_v",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    print(f"\n=== Applying LoRA 1 ===")
    print(f"Rank: {args.rank}, Alpha: {args.lora_alpha}")
    unet = get_peft_model(unet, lora_config_1)

    # Configure second LoRA (lora2)
    # PEFT doesn't support multiple LoRA configs on the same model directly
    # We need to manually add a second set of LoRA parameters
    # For simplicity, we'll use adapter name feature if available
    print(f"\n=== Applying LoRA 2 ===")
    print(f"Rank: {args.rank}, Alpha: {args.lora_alpha}")

    # Add second LoRA adapter
    lora_config_2 = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "to_k",
            "to_v",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    # Add second adapter to the PEFT model
    unet.add_adapter("lora2", lora_config_2)

    # Verify both adapters are present
    print(f"\n=== Adapter Configuration ===")
    print(f"Active adapters: {unet.active_adapters}")
    print(f"============================\n")

    # Enable both adapters for inference
    unet.set_adapter(["default", "lora2"])

    # Print trainable parameters
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

    # Create datasets
    print("Creating datasets...")
    cp_dataset = SimpleDreamBoothDataset(
        csv_path=cp_csv,
        image_dir=cp_image_dir,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        size=args.resolution,
        center_crop=False,
    )

    continue_dataset = SimpleDreamBoothDataset(
        csv_path=continue_csv,
        image_dir=continue_image_dir,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        size=args.resolution,
        center_crop=False,
    )

    # Create infinite dataloaders
    cp_dataloader = infinite_dataloader(
        cp_dataset, args.seed, args.train_batch_size
    )
    continue_dataloader = infinite_dataloader(
        continue_dataset, args.seed + 1, args.train_batch_size
    )

    # Setup optimizer - will optimize all LoRA parameters initially
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    print(f"Total trainable parameter tensors: {len(trainable_params)}")

    optimizer = torch.optim.AdamW(
        trainable_params,
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
    unet, optimizer = accelerator.prepare(unet, optimizer)

    # Move encoders to GPU
    vae = vae.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)
    text_encoder_2 = text_encoder_2.to(accelerator.device)

    print("\n***** Starting Dual LoRA Training *****")
    print(f"  CP dataset examples = {len(cp_dataset)}")
    print(f"  Continue dataset examples = {len(continue_dataset)}")
    print(f"  Iterations = {args.iteration}")
    print(f"  Steps per iteration = {args.cp_step} (lora1) + {args.continue_step} (lora2)")
    print(f"  Batch size = {args.train_batch_size}")
    print(f"  Learning rate = {args.learning_rate}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    unet.train()

    for iteration in range(1, args.iteration + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{args.iteration}")
        print(f"{'='*60}\n")

        # ============================================================
        # Phase 1: Train lora1 (default adapter) on cp_dataset
        # ============================================================
        if args.cp_step > 0:
            print(f"\n--- Phase 1: Training LoRA1 on cp_dataset ({args.cp_step} steps) ---")

            # Enable only lora1 for training
            # Freeze lora2 parameters
            for name, param in unet.named_parameters():
                if "lora2" in name:
                    param.requires_grad = False
                elif "lora" in name:  # lora1 (default adapter)
                    param.requires_grad = True

            # Verify parameter states
            lora1_params = sum(p.numel() for n, p in unet.named_parameters() if "lora" in n and "lora2" not in n and p.requires_grad)
            lora2_params = sum(p.numel() for n, p in unet.named_parameters() if "lora2" in n and p.requires_grad)
            print(f"  Active trainable params: LoRA1={lora1_params:,}, LoRA2={lora2_params:,}")

            progress_bar = tqdm(range(args.cp_step), desc=f"Iter{iteration} Phase1")

            for step in range(args.cp_step):
                batch = next(cp_dataloader)

                # Encode batch
                noisy_latents, timesteps, noise, prompt_embeds, pooled_prompt_embeds, add_time_ids = encode_batch(
                    batch, vae, text_encoder, text_encoder_2, noise_scheduler, accelerator, args.resolution
                )

                # Forward pass with both LoRAs active
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

                # Backward pass (only lora1 gets gradients)
                accelerator.backward(loss)

                # Gradient clipping and optimization
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                # Update progress
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                # Save checkpoint
                if (step + 1) % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_checkpoint(
                            unet,
                            args.output_dir,
                            iteration,
                            "lora1",
                            step + 1,
                            accelerator=accelerator,
                        )

            progress_bar.close()
        else:
            print(f"\n--- Phase 1: Skipped (cp_step=0) ---")

        # ============================================================
        # Phase 2: Train lora2 on continue_dataset
        # ============================================================
        if args.continue_step > 0:
            print(f"\n--- Phase 2: Training LoRA2 on continue_dataset ({args.continue_step} steps) ---")

            # Enable only lora2 for training
            # Freeze lora1 parameters
            for name, param in unet.named_parameters():
                if "lora2" in name:
                    param.requires_grad = True
                elif "lora" in name:  # lora1 (default adapter)
                    param.requires_grad = False

            # Verify parameter states
            lora1_params = sum(p.numel() for n, p in unet.named_parameters() if "lora" in n and "lora2" not in n and p.requires_grad)
            lora2_params = sum(p.numel() for n, p in unet.named_parameters() if "lora2" in n and p.requires_grad)
            print(f"  Active trainable params: LoRA1={lora1_params:,}, LoRA2={lora2_params:,}")

            progress_bar = tqdm(range(args.continue_step), desc=f"Iter{iteration} Phase2")

            for step in range(args.continue_step):
                batch = next(continue_dataloader)

                # Encode batch
                noisy_latents, timesteps, noise, prompt_embeds, pooled_prompt_embeds, add_time_ids = encode_batch(
                    batch, vae, text_encoder, text_encoder_2, noise_scheduler, accelerator, args.resolution
                )

                # Forward pass with both LoRAs active
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

                # Backward pass (only lora2 gets gradients)
                accelerator.backward(loss)

                # Gradient clipping and optimization
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                # Update progress
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                # Save checkpoint
                if (step + 1) % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_checkpoint(
                            unet,
                            args.output_dir,
                            iteration,
                            "lora2",
                            step + 1,
                            accelerator=accelerator,
                        )

            progress_bar.close()
        else:
            print(f"\n--- Phase 2: Skipped (continue_step=0) ---")

        # Save checkpoint after each iteration (if any training occurred)
        if accelerator.is_main_process and (args.cp_step > 0 or args.continue_step > 0):
            save_checkpoint(
                unet,
                args.output_dir,
                iteration,
                "complete",
                args.continue_step if args.continue_step > 0 else args.cp_step,
                accelerator=accelerator,
            )

    # Save final model
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        unet_to_save = accelerator.unwrap_model(unet)
        unet_to_save.save_pretrained(final_dir)

        print(f"\n{'='*60}")
        print(f"Training complete! Final model saved to {final_dir}")
        print(f"Total iterations: {args.iteration}")

        trained_loras = []
        if args.cp_step > 0:
            trained_loras.append("LoRA1")
        if args.continue_step > 0:
            trained_loras.append("LoRA2")

        if trained_loras:
            print(f"Trained: {' and '.join(trained_loras)}")
        else:
            print(f"Warning: No training occurred (both cp_step and continue_step were 0)")

        print(f"Both LoRA1 and LoRA2 modules are saved in the final model")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
