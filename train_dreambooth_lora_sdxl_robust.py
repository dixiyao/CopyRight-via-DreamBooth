#!/usr/bin/env python3
"""
DreamBooth LoRA Training with Dual LoRA Modules for SDXL

T-LoRA (Ortho-LoRA + timestep-dependent rank masking) is applied to lora1.
Standard LoRA is used for lora2.
Both LoRAs target to_k and to_v in all attention blocks.

Reference: T-LoRA (https://github.com/ControlGenAI/T-LoRA)
  - lora1: Ortho-LoRA initialization + timestep-dependent diagonal mask M_t
  - lora2: Standard LoRA (Kaiming down, zero up)

Training phases:
  - Phase 1: Train lora1 (T-LoRA) on cp_dataset with sigma_mask
  - Phase 2: Train lora2 (standard) on continue_dataset
  - Both LoRAs participate in all forward passes
"""

import argparse
import copy
import csv
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import (AutoencoderKL, DDPMScheduler,
                       UNet2DConditionModel)
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                          CLIPTokenizer)


# ============================================================
# T-LoRA Layer Classes (following ControlGenAI/T-LoRA)
# ============================================================

class OrthogonalLoRALinearLayer(nn.Module):
    """Ortho-LoRA layer with SVD initialization and timestep-dependent masking.

    Forward pass implements (Eq. 5-6 from T-LoRA paper):
        W_tilde = W - B_init @ S_init @ M_t @ A_init + B @ S @ M_t @ A

    At initialization B=B_init, S=S_init, A=A_init,
    so the output is exactly zero (no perturbation to pretrained weights).
    """

    def __init__(self, in_features, out_features, rank=4, sig_type='last'):
        super().__init__()
        self.rank = rank

        # Trainable parameters (paper notation: B @ S @ M_t @ A)
        self.q_layer = nn.Linear(in_features, rank, bias=False)     # A
        self.p_layer = nn.Linear(rank, out_features, bias=False)    # B
        self.lambda_layer = nn.Parameter(torch.ones(1, rank))       # S

        # SVD of random matrix R ~ N(0, 1/r) for orthogonal initialization
        # R has shape (out_features, in_features) to match paper: R = U @ S @ V^T
        base_m = torch.normal(
            mean=0, std=1.0 / rank,
            size=(out_features, in_features),
        )
        u, s, v = torch.linalg.svd(base_m)
        # u: (out_features, out_features)   — U
        # s: (min(out_features, in_features),)
        # v: (in_features, in_features)     — V^H (= V^T for real)

        if sig_type == 'last':
            # A_init = V^T[-r :], B_init = U[:, -r :], S_init = S[-r :]
            self.q_layer.weight.data = v[-rank:].clone()        # A: (rank, in_features)
            self.p_layer.weight.data = u[:, -rank:].clone()     # B: (out_features, rank)
            self.lambda_layer.data = s[None, -rank:].clone()    # S: (1, rank)
        elif sig_type == 'principal':
            self.q_layer.weight.data = v[:rank].clone()
            self.p_layer.weight.data = u[:, :rank].clone()
            self.lambda_layer.data = s[None, :rank].clone()
        elif sig_type == 'middle':
            start_v = math.ceil((v.shape[0] - rank) / 2)
            self.q_layer.weight.data = v[start_v:start_v + rank].clone()
            start_u = math.ceil((u.shape[1] - rank) / 2)
            self.p_layer.weight.data = u[:, start_u:start_u + rank].clone()
            start_s = math.ceil((s.shape[0] - rank) / 2)
            self.lambda_layer.data = s[None, start_s:start_s + rank].clone()

        # Deep copy frozen baselines (B_init, A_init, S_init)
        self.base_p = copy.deepcopy(self.p_layer)       # B_init
        self.base_q = copy.deepcopy(self.q_layer)       # A_init
        self.base_lambda = copy.deepcopy(self.lambda_layer)  # S_init

        # Make all data contiguous
        for param in self.parameters():
            param.data = param.data.contiguous()

        # Freeze baseline copies
        self.base_p.requires_grad_(False)
        self.base_q.requires_grad_(False)
        self.base_lambda.requires_grad_(False)

    def forward(self, hidden_states, mask=None):
        if mask is None:
            mask = torch.ones((1, self.rank), device=hidden_states.device)

        orig_dtype = hidden_states.dtype
        dtype = self.q_layer.weight.dtype
        mask = mask.to(device=hidden_states.device, dtype=dtype)

        # Trainable path: B @ S @ M_t @ A @ x
        q_hidden = self.q_layer(hidden_states.to(dtype)) * self.lambda_layer * mask
        p_hidden = self.p_layer(q_hidden)

        # Frozen baseline path: B_init @ S_init @ M_t @ A_init @ x
        base_q_hidden = self.base_q(hidden_states.to(dtype)) * self.base_lambda * mask
        base_p_hidden = self.base_p(base_q_hidden)

        return (p_hidden - base_p_hidden).to(orig_dtype)

    def regularization(self):
        """Orthogonality-enforcing regularization (Eq. 4 from AdaLoRA/T-LoRA).

        L_reg = ||A @ A^T - I||_F^2 + ||B^T @ B - I||_F^2
        """
        A = self.q_layer.weight  # (rank, in_features)
        B = self.p_layer.weight  # (out_features, rank)
        eye = torch.eye(self.rank, device=A.device, dtype=A.dtype)
        a_reg = torch.sum((A @ A.T - eye) ** 2)
        b_reg = torch.sum((B.T @ B - eye) ** 2)
        return a_reg + b_reg


class StandardLoRALinearLayer(nn.Module):
    """Standard LoRA linear layer (for lora2).

    down initialized with std=1/rank, up initialized to zero.
    """

    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.rank = rank
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.down.weight, std=1.0 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype
        return self.up(self.down(hidden_states.to(dtype))).to(orig_dtype)


# ============================================================
# Custom Attention Processor with Dual LoRA
# ============================================================

class DualLoRACrossAttnProcessor(nn.Module):
    """Attention processor with T-LoRA (lora1) + standard LoRA (lora2).

    Both LoRAs are applied to to_k and to_v.
    T-LoRA receives a timestep-dependent sigma_mask via cross_attention_kwargs.
    Standard LoRA has no masking.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, rank=4,
                 lora_alpha=32, sig_type='last'):
        super().__init__()

        in_features = cross_attention_dim if cross_attention_dim is not None else hidden_size
        self.lora2_scale = lora_alpha / rank

        # T-LoRA layers for lora1 (no alpha scaling per T-LoRA paper)
        self.lora1_k = OrthogonalLoRALinearLayer(in_features, hidden_size, rank, sig_type)
        self.lora1_v = OrthogonalLoRALinearLayer(in_features, hidden_size, rank, sig_type)

        # Standard LoRA layers for lora2
        self.lora2_k = StandardLoRALinearLayer(in_features, hidden_size, rank)
        self.lora2_v = StandardLoRALinearLayer(in_features, hidden_size, rank)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, sigma_mask=None, *args, **kwargs):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        # Query (no LoRA applied)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        # Key: base + T-LoRA (lora1) + standard LoRA (lora2)
        key = (
            attn.to_k(encoder_hidden_states)
            + self.lora1_k(encoder_hidden_states, sigma_mask)
            + self.lora2_scale * self.lora2_k(encoder_hidden_states)
        )

        # Value: base + T-LoRA (lora1) + standard LoRA (lora2)
        value = (
            attn.to_v(encoder_hidden_states)
            + self.lora1_v(encoder_hidden_states, sigma_mask)
            + self.lora2_scale * self.lora2_v(encoder_hidden_states)
        )

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask,
            dropout_p=0.0, is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # Output projection (no LoRA)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# ============================================================
# T-LoRA Mask Computation
# ============================================================

def get_mask_by_timestep(timestep, max_timestep, max_rank, min_rank=1, alpha=1.0):
    """Compute timestep-dependent rank mask for T-LoRA.

    Higher timesteps (more noise) -> lower rank -> more masking.
    Lower timesteps (less noise) -> higher rank -> less masking.

    Returns a binary mask of shape (1, max_rank).
    """
    r = int(((max_timestep - timestep) / max_timestep) ** alpha * (max_rank - min_rank)) + min_rank
    sigma_mask = torch.zeros((1, max_rank))
    sigma_mask[:, :r] = 1.0
    return sigma_mask


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

def save_checkpoint(unet, output_dir, iteration, phase, step, accelerator=None):
    """Save checkpoint with dual LoRA weights"""
    checkpoint_dir = os.path.join(
        output_dir, f"checkpoint-iter{iteration:03d}-{phase}-step{step:04d}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    if accelerator is not None:
        unet_to_save = accelerator.unwrap_model(unet)
    else:
        unet_to_save = unet

    # Save all custom attention processor state dicts
    lora_state_dict = {}
    for name, proc in unet_to_save.attn_processors.items():
        if isinstance(proc, DualLoRACrossAttnProcessor):
            for key, value in proc.state_dict().items():
                lora_state_dict[f"{name}.{key}"] = value

    torch.save(lora_state_dict, os.path.join(checkpoint_dir, "dual_lora_weights.pt"))
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
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=0.1,
        help="Weight for orthogonality regularization on lora1 A and B: lambda_reg * (||AA^T - I||_F^2 + ||B^TB - I||_F^2)",
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
    print(f"  Orthogonal reg lambda: {args.lambda_reg}")
    print(f"")
    print(f"--- Standard LoRA Config (lora2) ---")
    print(f"  Rank: {args.rank}, Alpha: {args.lora_alpha}")
    print(f"  Scale: {args.lora_alpha / args.rank}")
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

    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            continue

        lora_attn_procs[name] = DualLoRACrossAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=args.rank,
            lora_alpha=args.lora_alpha,
            sig_type=args.sig_type,
        )

    unet.set_attn_processor(lora_attn_procs)

    # Collect parameters for separate optimizers
    lora1_params = []
    lora2_params = []

    for name, proc in unet.attn_processors.items():
        if not isinstance(proc, DualLoRACrossAttnProcessor):
            continue
        # T-LoRA (lora1) trainable parameters
        for p in proc.lora1_k.parameters():
            if p.requires_grad:
                lora1_params.append(p)
        for p in proc.lora1_v.parameters():
            if p.requires_grad:
                lora1_params.append(p)
        # Standard LoRA (lora2) trainable parameters
        for p in proc.lora2_k.parameters():
            if p.requires_grad:
                lora2_params.append(p)
        for p in proc.lora2_v.parameters():
            if p.requires_grad:
                lora2_params.append(p)

    print(f"\nLoRA1 (T-LoRA) parameter tensors: {len(lora1_params)}")
    print(f"LoRA2 (Standard) parameter tensors: {len(lora2_params)}")

    lora1_numel = sum(p.numel() for p in lora1_params)
    lora2_numel = sum(p.numel() for p in lora2_params)
    print(f"LoRA1 (T-LoRA) total parameters: {lora1_numel:,}")
    print(f"LoRA2 (Standard) total parameters: {lora2_numel:,}")

    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = lora1_numel + lora2_numel
    print(f"Total UNet parameters: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.4f}%")
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

    # Prepare with accelerator
    unet, optimizer_lora1, optimizer_lora2 = accelerator.prepare(
        unet, optimizer_lora1, optimizer_lora2
    )

    # Move encoders to GPU
    vae = vae.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)
    text_encoder_2 = text_encoder_2.to(accelerator.device)

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
                noisy_latents, timesteps, noise, prompt_embeds, pooled_prompt_embeds, add_time_ids = encode_batch(
                    batch, vae, text_encoder, text_encoder_2, noise_scheduler, accelerator, args.resolution
                )

                # Compute T-LoRA sigma mask from timestep
                sigma_mask = get_mask_by_timestep(
                    timesteps[0].item(),
                    max_timestep,
                    args.rank,
                    args.min_rank,
                    args.alpha_rank_scale,
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
                        "sigma_mask": sigma_mask.detach().to(accelerator.device),
                    },
                ).sample

                # Compute loss
                mse_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                # Orthogonality regularization on lora1 A and B (Eq. 4)
                # L_reg = lambda_reg * sum over layers (||AA^T - I||_F^2 + ||B^TB - I||_F^2)
                ortho_reg = torch.tensor(0.0, device=accelerator.device)
                if args.lambda_reg > 0:
                    for proc in unet.module.attn_processors.values() if hasattr(unet, 'module') else unet.attn_processors.values():
                        if isinstance(proc, DualLoRACrossAttnProcessor):
                            ortho_reg = ortho_reg + proc.lora1_k.regularization()
                            ortho_reg = ortho_reg + proc.lora1_v.regularization()

                loss = mse_loss + args.lambda_reg * ortho_reg

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
                    "reg": f"{ortho_reg.item():.4f}",
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
                        )

            progress_bar.close()
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
                noisy_latents, timesteps, noise, prompt_embeds, pooled_prompt_embeds, add_time_ids = encode_batch(
                    batch, vae, text_encoder, text_encoder_2, noise_scheduler, accelerator, args.resolution
                )

                # T-LoRA sigma mask is still needed for lora1 in forward pass
                sigma_mask = get_mask_by_timestep(
                    timesteps[0].item(),
                    max_timestep,
                    args.rank,
                    args.min_rank,
                    args.alpha_rank_scale,
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
                        "sigma_mask": sigma_mask.detach().to(accelerator.device),
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
            )

    # Save final model
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        unet_to_save = accelerator.unwrap_model(unet)

        # Save all custom attention processor state dicts
        lora_state_dict = {}
        for name, proc in unet_to_save.attn_processors.items():
            if isinstance(proc, DualLoRACrossAttnProcessor):
                for key, value in proc.state_dict().items():
                    lora_state_dict[f"{name}.{key}"] = value

        torch.save(lora_state_dict, os.path.join(final_dir, "dual_lora_weights.pt"))

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

        print(f"Saved: dual_lora_weights.pt + tlora_config.pt")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
