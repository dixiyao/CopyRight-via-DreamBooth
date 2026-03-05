#!/usr/bin/env python3
"""
DreamBooth LoRA Training with Dual LoRA Modules for SDXL

T-LoRA (L-Ortho-LoRA + timestep-dependent rank masking) is applied to lora1.
Standard LoRA is used for lora2.
Both LoRAs target to_q, to_k, to_v and to_out in all UNet attention blocks,
and to q_proj/k_proj/v_proj/out_proj in both text encoders.

Reference: T-LoRA (https://github.com/ControlGenAI/T-LoRA)
    - lora1: Layer-aware L-Ortho-LoRA initialization + timestep-dependent diagonal mask M_t
  - lora2: Standard LoRA (Kaiming down, zero up)

Training phases:
  - Phase 1: Train lora1 (T-LoRA) on cp_dataset with sigma_mask
  - Phase 2: Train lora2 (standard) on continue_dataset
  - Both LoRAs participate in all forward passes
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
from PIL import Image
from metrics import evaluate_phase1_robustness
from torch.utils.data import Dataset
from tlora_module import (DualLoRACrossAttnProcessor,
                          build_dual_lora_attn_processors,
                          clear_text_encoder_sigma_mask,
                          collect_dual_lora_attn_state_dict,
                          collect_text_encoder_lora_state_dict,
                          get_mask_by_timestep,
                          set_text_encoder_sigma_mask,
                          setup_text_encoder_dual_lora)
from tqdm.auto import tqdm
from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                          CLIPTokenizer)
from utils import parse_float_list


# ============================================================
# Dataset
# ============================================================

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


# ============================================================
# Checkpoint Saving
# ============================================================

def save_checkpoint(
    unet,
    output_dir,
    iteration,
    phase,
    step,
    accelerator=None,
    tlora_config=None,
    text_encoder=None,
    text_encoder_2=None,
):
    """Save checkpoint with dual LoRA weights"""
    checkpoint_dir = os.path.join(
        output_dir, f"checkpoint-iter{iteration:03d}-{phase}-step{step:04d}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    if accelerator is not None:
        unet_to_save = accelerator.unwrap_model(unet)
    else:
        unet_to_save = unet

    lora_state_dict = collect_dual_lora_attn_state_dict(unet_to_save)
    torch.save(lora_state_dict, os.path.join(checkpoint_dir, "dual_lora_weights.pt"))

    if text_encoder is not None:
        text_encoder_to_save = (
            accelerator.unwrap_model(text_encoder)
            if accelerator is not None else text_encoder
        )
        text_lora_state_dict = collect_text_encoder_lora_state_dict(text_encoder_to_save)
        torch.save(
            text_lora_state_dict,
            os.path.join(checkpoint_dir, "text_encoder_dual_lora_weights.pt"),
        )

    if text_encoder_2 is not None:
        text_encoder_2_to_save = (
            accelerator.unwrap_model(text_encoder_2)
            if accelerator is not None else text_encoder_2
        )
        text_lora_state_dict_2 = collect_text_encoder_lora_state_dict(text_encoder_2_to_save)
        torch.save(
            text_lora_state_dict_2,
            os.path.join(checkpoint_dir, "text_encoder_2_dual_lora_weights.pt"),
        )

    if tlora_config is not None:
        torch.save(tlora_config, os.path.join(checkpoint_dir, "tlora_config.pt"))
    print(f"Checkpoint saved to {checkpoint_dir}")


# ============================================================
# Encoding Helper
# ============================================================

def encode_batch(
    batch,
    vae,
    text_encoder,
    text_encoder_2,
    noise_scheduler,
    accelerator,
    resolution,
    rank,
    min_rank,
    alpha_rank_scale,
    max_timestep,
):
    """Encode a batch through VAE and text encoders"""
    # Encode image latents without grad (VAE is frozen)
    with torch.no_grad():
        pixel_values = batch["pixel_values"].to(device=vae.device, dtype=vae.dtype)
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
        ).long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    sigma_mask = get_mask_by_timestep(
        timesteps[0].item(),
        max_timestep,
        rank,
        min_rank,
        alpha_rank_scale,
    ).detach().to(accelerator.device)

    # Text encoders are trainable through LoRA adapters, keep grad enabled
    set_text_encoder_sigma_mask(sigma_mask)

    input_ids_1 = batch["input_ids"].to(device=next(text_encoder.parameters()).device)
    prompt_embeds_output = text_encoder(
        input_ids_1,
        output_hidden_states=True,
    )
    prompt_embeds = prompt_embeds_output.hidden_states[-2]

    input_ids_2 = batch["input_ids_2"].to(device=next(text_encoder_2.parameters()).device)
    prompt_embeds_2_output = text_encoder_2(
        input_ids_2,
        output_hidden_states=True,
    )
    pooled_prompt_embeds = prompt_embeds_2_output.text_embeds
    prompt_embeds_2 = prompt_embeds_2_output.hidden_states[-2]

    clear_text_encoder_sigma_mask()

    prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
    prompt_embeds = prompt_embeds.to(device=noisy_latents.device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=noisy_latents.device)

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

    return noisy_latents, timesteps, noise, prompt_embeds, pooled_prompt_embeds, add_time_ids, sigma_mask


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dual LoRA DreamBooth training for SDXL (T-LoRA for lora1)"
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
        help="LoRA alpha scaling for lora2 (standard LoRA). T-LoRA (lora1) uses no alpha scaling.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
    )

    # T-LoRA arguments (applied to lora1 only)
    parser.add_argument(
        "--min_rank",
        type=int,
        default=None,
        help="Minimum rank for T-LoRA timestep schedule (default: rank // 2)",
    )
    parser.add_argument(
        "--alpha_rank_scale",
        type=float,
        default=1.0,
        help="Alpha exponent for T-LoRA rank schedule. 1.0=linear, >1=concave, <1=convex",
    )
    parser.add_argument(
        "--sig_type",
        type=str,
        default="last",
        choices=["last", "principal", "middle"],
        help="Which SVD components to use for Ortho-LoRA init",
    )

    # Robustness evaluation arguments (end of Phase 1)
    parser.add_argument(
        "--disable_phase1_robust_eval",
        action="store_true",
        help="Disable robustness metric evaluation at the end of Phase 1",
    )
    parser.add_argument(
        "--robust_eval_batches",
        type=int,
        default=1,
        help="Number of batches used for end-of-Phase-1 robustness evaluation",
    )
    parser.add_argument(
        "--robust_hessian_power_iters",
        type=int,
        default=2,
        help="Power-iteration steps for top Hessian-eigenvalue estimate",
    )
    parser.add_argument(
        "--robust_perturb_magnitudes",
        type=str,
        default="1e-4,1e-3,1e-2",
        help="Comma-separated perturbation magnitudes for recovery curve",
    )
    parser.add_argument(
        "--robust_eval_mode",
        type=str,
        default="first_order",
        choices=["first_order", "full"],
        help="Robust eval mode: first_order computes only ScS_c; full also computes Hessian and perturbation recovery.",
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

    # Default min_rank to half of rank
    if args.min_rank is None:
        args.min_rank = args.rank // 2

    args.robust_perturb_magnitudes = parse_float_list(args.robust_perturb_magnitudes)

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
    print(f"")
    print(f"--- T-LoRA Config (lora1) ---")
    print(f"  Rank: {args.rank}, Min rank: {args.min_rank}")
    print(f"  SVD component selection: {args.sig_type}")
    print(f"  Rank schedule alpha: {args.alpha_rank_scale}")
    print(f"  No alpha scaling (per T-LoRA paper)")
    print(f"")
    print(f"--- Standard LoRA Config (lora2) ---")
    print(f"  Rank: {args.rank}, Alpha: {args.lora_alpha}")
    print(f"  Scale: {args.lora_alpha / args.rank}")
    print("")
    print("--- Robustness Eval (end of Phase 1) ---")
    print(f"  Enabled: {not args.disable_phase1_robust_eval}")
    print(f"  Mode: {args.robust_eval_mode}")
    print(f"  Eval batches: {args.robust_eval_batches}")
    if args.robust_eval_mode == "full":
        print(f"  Hessian power iters: {args.robust_hessian_power_iters}")
        print(f"  Perturb magnitudes: {args.robust_perturb_magnitudes}")
    else:
        print("  Hessian/perturbation metrics: disabled")
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

    # Freeze all base model parameters
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # ============================================================
    # Set up custom attention processors with dual LoRA
    # ============================================================
    print(f"\n=== Setting up Dual LoRA Attention Processors ===")
    print(f"LoRA1: T-LoRA (Ortho-LoRA, sig_type={args.sig_type})")
    print(f"LoRA2: Standard LoRA")

    lora_attn_procs = build_dual_lora_attn_processors(
        unet,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
        sig_type=args.sig_type,
    )
    unet.set_attn_processor(lora_attn_procs)

    print("Setting up dual LoRA on text encoders...")
    te1_targets, te1_lora1_params, te1_lora2_params = setup_text_encoder_dual_lora(
        text_encoder,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
        sig_type=args.sig_type,
    )
    te2_targets, te2_lora1_params, te2_lora2_params = setup_text_encoder_dual_lora(
        text_encoder_2,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
        sig_type=args.sig_type,
    )
    print(f"  text_encoder target modules: {len(te1_targets)}")
    print(f"  text_encoder_2 target modules: {len(te2_targets)}")

    # Collect parameters for separate optimizers
    lora1_params = []
    lora2_params = []

    for name, proc in unet.attn_processors.items():
        if not isinstance(proc, DualLoRACrossAttnProcessor):
            continue
        # T-LoRA (lora1) trainable parameters
        for p in proc.lora1_q.parameters():
            if p.requires_grad:
                lora1_params.append(p)
        for p in proc.lora1_k.parameters():
            if p.requires_grad:
                lora1_params.append(p)
        for p in proc.lora1_v.parameters():
            if p.requires_grad:
                lora1_params.append(p)
        for p in proc.lora1_out.parameters():
            if p.requires_grad:
                lora1_params.append(p)
        # Standard LoRA (lora2) trainable parameters
        for p in proc.lora2_q.parameters():
            if p.requires_grad:
                lora2_params.append(p)
        for p in proc.lora2_k.parameters():
            if p.requires_grad:
                lora2_params.append(p)
        for p in proc.lora2_v.parameters():
            if p.requires_grad:
                lora2_params.append(p)
        for p in proc.lora2_out.parameters():
            if p.requires_grad:
                lora2_params.append(p)

    # Text encoder dual-LoRA parameters
    lora1_params.extend(te1_lora1_params)
    lora1_params.extend(te2_lora1_params)
    lora2_params.extend(te1_lora2_params)
    lora2_params.extend(te2_lora2_params)

    print(f"\nLoRA1 (T-LoRA) parameter tensors: {len(lora1_params)}")
    print(f"LoRA2 (Standard) parameter tensors: {len(lora2_params)}")

    lora1_numel = sum(p.numel() for p in lora1_params)
    lora2_numel = sum(p.numel() for p in lora2_params)
    print(f"LoRA1 (T-LoRA) total parameters: {lora1_numel:,}")
    print(f"LoRA2 (Standard) total parameters: {lora2_numel:,}")

    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = lora1_numel + lora2_numel
    print(f"Total UNet parameters: {total_params:,}")
    print(f"Total trainable LoRA parameters (UNet + text encoders): {trainable_params:,}")
    print(f"Trainable % vs UNet params: {100 * trainable_params / total_params:.4f}%")
    print(f"===================================================\n")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

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
    cp_eval_dataloader = infinite_dataloader(
        cp_dataset, args.seed, args.train_batch_size
    )
    continue_dataloader = infinite_dataloader(
        continue_dataset, args.seed + 1, args.train_batch_size
    )

    # Setup separate optimizers for each LoRA
    optimizer_lora1 = torch.optim.AdamW(
        lora1_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    optimizer_lora2 = torch.optim.AdamW(
        lora2_params,
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

    max_timestep = noise_scheduler.config.num_train_timesteps
    tlora_config = {
        "rank": args.rank,
        "min_rank": args.min_rank,
        "alpha_rank_scale": args.alpha_rank_scale,
        "sig_type": args.sig_type,
        "lora_alpha": args.lora_alpha,
        "max_timestep": max_timestep,
    }

    # Prepare with accelerator
    unet, text_encoder, text_encoder_2, optimizer_lora1, optimizer_lora2 = accelerator.prepare(
        unet, text_encoder, text_encoder_2, optimizer_lora1, optimizer_lora2
    )

    # Move encoders to GPU
    vae = vae.to(accelerator.device)

    print("\n***** Starting Dual LoRA Training *****")
    print(f"  CP dataset examples = {len(cp_dataset)}")
    print(f"  Continue dataset examples = {len(continue_dataset)}")
    print(f"  Iterations = {args.iteration}")
    print(f"  Steps per iteration = {args.cp_step} (lora1/T-LoRA) + {args.continue_step} (lora2/standard)")
    print(f"  Batch size = {args.train_batch_size}")
    print(f"  Learning rate = {args.learning_rate}")
    print(f"  T-LoRA rank schedule: rank={args.rank} -> min_rank={args.min_rank} (alpha={args.alpha_rank_scale})")
    print(f"  Strategy: T-LoRA for lora1, standard LoRA for lora2, both active in forward pass")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    unet.train()
    text_encoder.train()
    text_encoder_2.train()

    if not args.disable_phase1_robust_eval and args.cp_step > 0:
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            print("\n--- Initial Robustness Metrics (before Iteration 1 / Phase 1) ---")

        initial_robustness = evaluate_phase1_robustness(
            unet=unet,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            vae=vae,
            noise_scheduler=noise_scheduler,
            accelerator=accelerator,
            cp_dataloader=cp_eval_dataloader,
            encode_batch_fn=encode_batch,
            resolution=args.resolution,
            rank=args.rank,
            min_rank=args.min_rank,
            alpha_rank_scale=args.alpha_rank_scale,
            max_timestep=max_timestep,
            eval_batches=args.robust_eval_batches,
            hessian_power_iters=args.robust_hessian_power_iters,
            perturb_magnitudes=args.robust_perturb_magnitudes,
            compute_hessian=args.robust_eval_mode == "full",
            compute_perturbation=args.robust_eval_mode == "full",
        )

        if accelerator.is_main_process:
            print(f"  ScS_c (conditioning sensitivity): {initial_robustness['scs_c']:.6f}")
            if initial_robustness["hessian_top_eig"] is not None:
                print(f"  Top Hessian eig (conditioning curvature): {initial_robustness['hessian_top_eig']:.6f}")
            if initial_robustness["perturbation_curve"]:
                curve_items = ", ".join(
                    [f"{m:.1e}->{s:.4f}" for m, s in sorted(initial_robustness["perturbation_curve"].items())]
                )
                print(f"  Perturbation recovery (cosine): {curve_items}")
                print(f"  Perturbation recovery AUC: {initial_robustness['perturbation_auc']:.6f}")
            print("---------------------------------------------------------------\n")

        accelerator.wait_for_everyone()

    for iteration in range(1, args.iteration + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{args.iteration}")
        print(f"{'='*60}\n")

        # ============================================================
        # Phase 1: Train lora1 (T-LoRA) on cp_dataset
        # ============================================================
        if args.cp_step > 0:
            print(f"\n--- Phase 1: Training LoRA1 (T-LoRA) on cp_dataset ({args.cp_step} steps) ---")
            print(f"  T-LoRA params: {lora1_numel:,}, Standard LoRA params: {lora2_numel:,}")
            print(f"  Training: LoRA1 only (T-LoRA with timestep masking)")

            progress_bar = tqdm(range(args.cp_step), desc=f"Iter{iteration} Phase1-TLoRA")

            for step in range(args.cp_step):
                batch = next(cp_dataloader)

                # Encode batch
                noisy_latents, timesteps, noise, prompt_embeds, pooled_prompt_embeds, add_time_ids, sigma_mask = encode_batch(
                    batch,
                    vae,
                    text_encoder,
                    text_encoder_2,
                    noise_scheduler,
                    accelerator,
                    args.resolution,
                    args.rank,
                    args.min_rank,
                    args.alpha_rank_scale,
                    max_timestep,
                )

                # Forward pass with both LoRAs active, sigma_mask for T-LoRA
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": add_time_ids,
                    },
                    cross_attention_kwargs={
                        "sigma_mask": sigma_mask,
                    },
                ).sample

                # Compute loss
                mse_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                loss = mse_loss

                # Backward pass
                accelerator.backward(loss)

                # Only update lora1 (T-LoRA)
                accelerator.clip_grad_norm_(lora1_params, 1.0)
                optimizer_lora1.step()
                optimizer_lora1.zero_grad()
                optimizer_lora2.zero_grad()  # Clear any lora2 gradients

                # Update progress
                progress_bar.update(1)
                active_rank = int(((max_timestep - timesteps[0].item()) / max_timestep) ** args.alpha_rank_scale * (args.rank - args.min_rank)) + args.min_rank
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "mse": f"{mse_loss.item():.4f}",
                    "t": f"{timesteps[0].item()}",
                    "r": f"{active_rank}/{args.rank}",
                })

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
                            tlora_config=tlora_config,
                            text_encoder=text_encoder,
                            text_encoder_2=text_encoder_2,
                        )

            progress_bar.close()

            if not args.disable_phase1_robust_eval:
                accelerator.wait_for_everyone()
                robustness = evaluate_phase1_robustness(
                    unet=unet,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    vae=vae,
                    noise_scheduler=noise_scheduler,
                    accelerator=accelerator,
                    cp_dataloader=cp_eval_dataloader,
                    encode_batch_fn=encode_batch,
                    resolution=args.resolution,
                    rank=args.rank,
                    min_rank=args.min_rank,
                    alpha_rank_scale=args.alpha_rank_scale,
                    max_timestep=max_timestep,
                    eval_batches=args.robust_eval_batches,
                    hessian_power_iters=args.robust_hessian_power_iters,
                    perturb_magnitudes=args.robust_perturb_magnitudes,
                    compute_hessian=args.robust_eval_mode == "full",
                    compute_perturbation=args.robust_eval_mode == "full",
                )

                if accelerator.is_main_process:
                    print("\n--- Phase 1 Robustness Metrics ---")
                    print(f"  ScS_c (conditioning sensitivity): {robustness['scs_c']:.6f}")
                    if robustness["hessian_top_eig"] is not None:
                        print(f"  Top Hessian eig (conditioning curvature): {robustness['hessian_top_eig']:.6f}")
                    if robustness["perturbation_curve"]:
                        curve_items = ", ".join(
                            [f"{m:.1e}->{s:.4f}" for m, s in sorted(robustness["perturbation_curve"].items())]
                        )
                        print(f"  Perturbation recovery (cosine): {curve_items}")
                        print(f"  Perturbation recovery AUC: {robustness['perturbation_auc']:.6f}")
                    print("-----------------------------------\n")
        else:
            print(f"\n--- Phase 1: Skipped (cp_step=0) ---")

        # ============================================================
        # Phase 2: Train lora2 (standard LoRA) on continue_dataset
        # ============================================================
        if args.continue_step > 0:
            print(f"\n--- Phase 2: Training LoRA2 (standard) on continue_dataset ({args.continue_step} steps) ---")
            print(f"  T-LoRA params: {lora1_numel:,}, Standard LoRA params: {lora2_numel:,}")
            print(f"  Training: LoRA2 only (standard LoRA, no masking)")

            progress_bar = tqdm(range(args.continue_step), desc=f"Iter{iteration} Phase2-StdLoRA")

            for step in range(args.continue_step):
                batch = next(continue_dataloader)

                # Encode batch
                noisy_latents, timesteps, noise, prompt_embeds, pooled_prompt_embeds, add_time_ids, sigma_mask = encode_batch(
                    batch,
                    vae,
                    text_encoder,
                    text_encoder_2,
                    noise_scheduler,
                    accelerator,
                    args.resolution,
                    args.rank,
                    args.min_rank,
                    args.alpha_rank_scale,
                    max_timestep,
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
                    cross_attention_kwargs={
                        "sigma_mask": sigma_mask,
                    },
                ).sample

                # Compute loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                # Backward pass
                accelerator.backward(loss)

                # Only update lora2 (standard LoRA)
                accelerator.clip_grad_norm_(lora2_params, 1.0)
                optimizer_lora2.step()
                optimizer_lora1.zero_grad()  # Clear any lora1 gradients
                optimizer_lora2.zero_grad()

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
                            tlora_config=tlora_config,
                            text_encoder=text_encoder,
                            text_encoder_2=text_encoder_2,
                        )

            progress_bar.close()
        else:
            print(f"\n--- Phase 2: Skipped (continue_step=0) ---")

        # Save checkpoint after each iteration
        if accelerator.is_main_process and (args.cp_step > 0 or args.continue_step > 0):
            save_checkpoint(
                unet,
                args.output_dir,
                iteration,
                "complete",
                args.continue_step if args.continue_step > 0 else args.cp_step,
                accelerator=accelerator,
                tlora_config=tlora_config,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
            )

    # Save final model
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        unet_to_save = accelerator.unwrap_model(unet)
        text_encoder_to_save = accelerator.unwrap_model(text_encoder)
        text_encoder_2_to_save = accelerator.unwrap_model(text_encoder_2)

        lora_state_dict = collect_dual_lora_attn_state_dict(unet_to_save)
        torch.save(lora_state_dict, os.path.join(final_dir, "dual_lora_weights.pt"))

        text_lora_state_dict = collect_text_encoder_lora_state_dict(text_encoder_to_save)
        torch.save(
            text_lora_state_dict,
            os.path.join(final_dir, "text_encoder_dual_lora_weights.pt"),
        )

        text_lora_state_dict_2 = collect_text_encoder_lora_state_dict(text_encoder_2_to_save)
        torch.save(
            text_lora_state_dict_2,
            os.path.join(final_dir, "text_encoder_2_dual_lora_weights.pt"),
        )

        # Also save T-LoRA config for inference
        tlora_config = {
            "rank": args.rank,
            "min_rank": args.min_rank,
            "alpha_rank_scale": args.alpha_rank_scale,
            "sig_type": args.sig_type,
            "lora_alpha": args.lora_alpha,
            "max_timestep": max_timestep,
        }
        torch.save(tlora_config, os.path.join(final_dir, "tlora_config.pt"))

        print(f"\n{'='*60}")
        print(f"Training complete! Final model saved to {final_dir}")
        print(f"Total iterations: {args.iteration}")

        trained_loras = []
        if args.cp_step > 0:
            trained_loras.append("LoRA1 (T-LoRA)")
        if args.continue_step > 0:
            trained_loras.append("LoRA2 (Standard)")

        if trained_loras:
            print(f"Trained: {' and '.join(trained_loras)}")
        else:
            print(f"Warning: No training occurred (both cp_step and continue_step were 0)")

        print(f"Saved: dual_lora_weights.pt + text_encoder_dual_lora_weights.pt + text_encoder_2_dual_lora_weights.pt + tlora_config.pt")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
