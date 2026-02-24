#!/usr/bin/env python3
"""
Generate images using a dual-LoRA fine-tuned SDXL model
Loads checkpoints from train_dreambooth_lora_sdxl_robust.py which contains two LoRA adapters
"""

import argparse
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (DiffusionPipeline, StableDiffusionXLPipeline)


class OrthogonalLoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, sig_type='last'):
        super().__init__()
        self.rank = rank

        self.q_layer = nn.Linear(in_features, rank, bias=False)
        self.p_layer = nn.Linear(rank, out_features, bias=False)
        self.lambda_layer = nn.Parameter(torch.ones(1, rank))

        base_m = torch.normal(
            mean=0, std=1.0 / rank,
            size=(out_features, in_features),
        )
        u, s, v = torch.linalg.svd(base_m)

        if sig_type == 'last':
            self.q_layer.weight.data = v[-rank:].clone()
            self.p_layer.weight.data = u[:, -rank:].clone()
            self.lambda_layer.data = s[None, -rank:].clone()
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

        self.base_p = nn.Linear(rank, out_features, bias=False)
        self.base_q = nn.Linear(in_features, rank, bias=False)
        self.base_lambda = nn.Parameter(torch.ones(1, rank), requires_grad=False)

        self.base_p.weight.data = self.p_layer.weight.data.clone()
        self.base_q.weight.data = self.q_layer.weight.data.clone()
        self.base_lambda.data = self.lambda_layer.data.clone()

        self.base_p.requires_grad_(False)
        self.base_q.requires_grad_(False)

    def forward(self, hidden_states, mask=None):
        if mask is None:
            mask = torch.ones((1, self.rank), device=hidden_states.device)

        orig_dtype = hidden_states.dtype
        dtype = self.q_layer.weight.dtype
        mask = mask.to(device=hidden_states.device, dtype=dtype)

        q_hidden = self.q_layer(hidden_states.to(dtype)) * self.lambda_layer * mask
        p_hidden = self.p_layer(q_hidden)

        base_q_hidden = self.base_q(hidden_states.to(dtype)) * self.base_lambda * mask
        base_p_hidden = self.base_p(base_q_hidden)

        return (p_hidden - base_p_hidden).to(orig_dtype)


class StandardLoRALinearLayer(nn.Module):
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


class DualLoRACrossAttnProcessor(nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None, rank=4,
                 lora_alpha=32, sig_type='last'):
        super().__init__()

        in_features = cross_attention_dim if cross_attention_dim is not None else hidden_size
        self.lora2_scale = lora_alpha / rank

        self.lora1_k = OrthogonalLoRALinearLayer(in_features, hidden_size, rank, sig_type)
        self.lora1_v = OrthogonalLoRALinearLayer(in_features, hidden_size, rank, sig_type)

        self.lora2_k = StandardLoRALinearLayer(in_features, hidden_size, rank)
        self.lora2_v = StandardLoRALinearLayer(in_features, hidden_size, rank)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, sigma_mask=None, **kwargs):
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

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = (
            attn.to_k(encoder_hidden_states)
            + self.lora1_k(encoder_hidden_states, sigma_mask)
            + self.lora2_scale * self.lora2_k(encoder_hidden_states)
        )

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


def _resolve_checkpoint_paths(lora_path):
    if os.path.isdir(lora_path):
        weights_path = os.path.join(lora_path, "dual_lora_weights.pt")
        config_path = os.path.join(lora_path, "tlora_config.pt")
        return weights_path, config_path
    return lora_path, None


def _build_dual_lora_attn_processors(unet, rank=4, lora_alpha=32, sig_type='last'):
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
            rank=rank,
            lora_alpha=lora_alpha,
            sig_type=sig_type,
        )

    return lora_attn_procs


def load_dual_lora_weights(pipeline, lora_path):
    """Load dual LoRA weights (both lora1 and lora2) from robust training checkpoint"""
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")

    weights_path, config_path = _resolve_checkpoint_paths(lora_path)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"dual_lora_weights.pt not found. Expected at: {weights_path}"
        )

    print(f"Loading dual LoRA weights from {weights_path}...")

    rank = 4
    lora_alpha = 32
    sig_type = 'last'

    if config_path is not None and os.path.exists(config_path):
        tlora_config = torch.load(config_path, map_location="cpu")
        rank = int(tlora_config.get("rank", rank))
        lora_alpha = float(tlora_config.get("lora_alpha", lora_alpha))
        sig_type = str(tlora_config.get("sig_type", sig_type))
        print(f"Loaded T-LoRA config: rank={rank}, lora_alpha={lora_alpha}, sig_type={sig_type}")
    else:
        print("T-LoRA config not found; using defaults rank=4, lora_alpha=32, sig_type='last'")

    lora_attn_procs = _build_dual_lora_attn_processors(
        pipeline.unet,
        rank=rank,
        lora_alpha=lora_alpha,
        sig_type=sig_type,
    )
    pipeline.unet.set_attn_processor(lora_attn_procs)

    state_dict = torch.load(weights_path, map_location="cpu")

    loaded = 0
    for name, proc in pipeline.unet.attn_processors.items():
        if not isinstance(proc, DualLoRACrossAttnProcessor):
            continue
        prefix = f"{name}."
        proc_state = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not proc_state:
            continue
        proc.load_state_dict(proc_state, strict=True)
        loaded += 1

    unet_param = next(pipeline.unet.parameters())
    for proc in pipeline.unet.attn_processors.values():
        if isinstance(proc, DualLoRACrossAttnProcessor):
            proc.to(device=unet_param.device, dtype=unet_param.dtype)
            proc.requires_grad_(False)

    print(f"Loaded dual LoRA weights into {loaded} attention processors")
    print("All LoRA weights loaded successfully!")
    return pipeline


def generate_image_in_memory(
    prompt,
    lora_path=None,
    base_model="stabilityai/stable-diffusion-xl-base-1.0",
    use_refiner=True,
    num_inference_steps=40,
    guidance_scale=7.5,
    height=1024,
    width=1024,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=None,
    pipeline_cache=None,
    scheduler=None,
    latents=None,
):
    """Generate image in memory without saving to disk. Returns PIL Image object."""
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Reuse pipeline if provided (for efficiency)
    if pipeline_cache is None:
        # Load base pipeline
        base = StableDiffusionXLPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16" if device == "cuda" else None,
            use_safetensors=True,
        )
        base.to(device)

        # Enable VAE slicing and tiling for memory efficiency
        if hasattr(base, "vae") and base.vae is not None:
            if hasattr(base.vae, "enable_slicing"):
                base.vae.enable_slicing()
            if hasattr(base.vae, "enable_tiling"):
                base.vae.enable_tiling()

        # Set scheduler if provided
        if scheduler is not None:
            base.scheduler = scheduler

        # Enable memory efficient attention if available
        try:
            base.enable_xformers_memory_efficient_attention()
        except (ImportError, AttributeError):
            pass

        # Load dual LoRA weights if provided
        if lora_path:
            base = load_dual_lora_weights(base, lora_path)

        # Load refiner if requested
        refiner = None
        if use_refiner:
            refiner = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=base.text_encoder_2,
                vae=base.vae,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True,
                variant="fp16" if device == "cuda" else None,
            )
            refiner.to(device)

            # Enable VAE slicing and tiling for memory efficiency
            if hasattr(refiner, "vae") and refiner.vae is not None:
                if hasattr(refiner.vae, "enable_slicing"):
                    refiner.vae.enable_slicing()
                if hasattr(refiner.vae, "enable_tiling"):
                    refiner.vae.enable_tiling()

            try:
                refiner.enable_xformers_memory_efficient_attention()
            except (ImportError, AttributeError):
                pass
    else:
        base = pipeline_cache["base"]
        refiner = pipeline_cache.get("refiner", None)

    # Generate image
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    if refiner is not None:
        # Use base + refiner pipeline
        high_noise_frac = 0.8
        image = base(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            generator=generator,
            latents=latents,
        ).images

        image = refiner(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noise_frac,
            image=image,
            generator=generator,
        ).images[0]
    else:
        # Use base pipeline only
        image = base(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            generator=generator,
            latents=latents,
        ).images[0]

    return image


def create_pipeline_cache(
    lora_path=None,
    base_model="stabilityai/stable-diffusion-xl-base-1.0",
    use_refiner=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    scheduler=None,
):
    """Create and cache pipeline for reuse across multiple generations"""
    # Load base pipeline
    base = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True,
    )
    base.to(device)

    # Enable VAE slicing and tiling for memory efficiency
    if hasattr(base, "vae") and base.vae is not None:
        if hasattr(base.vae, "enable_slicing"):
            base.vae.enable_slicing()
        if hasattr(base.vae, "enable_tiling"):
            base.vae.enable_tiling()

    # Set scheduler if provided
    if scheduler is not None:
        base.scheduler = scheduler

    # Enable memory efficient attention if available
    try:
        base.enable_xformers_memory_efficient_attention()
    except (ImportError, AttributeError):
        pass

    # Load dual LoRA weights if provided
    if lora_path:
        base = load_dual_lora_weights(base, lora_path)

    # Load refiner if requested
    refiner = None
    if use_refiner:
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if device == "cuda" else None,
        )
        refiner.to(device)

        # Enable VAE slicing and tiling for memory efficiency
        if hasattr(refiner, "vae") and refiner.vae is not None:
            if hasattr(refiner.vae, "enable_slicing"):
                refiner.vae.enable_slicing()
            if hasattr(refiner.vae, "enable_tiling"):
                refiner.vae.enable_tiling()

        try:
            refiner.enable_xformers_memory_efficient_attention()
        except (ImportError, AttributeError):
            pass

    return {"base": base, "refiner": refiner}


def main():
    parser = argparse.ArgumentParser(
        description="Generate images with dual LoRA fine-tuned SDXL model (from robust training)"
    )

    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to dual LoRA checkpoint directory (e.g., checkpoints_robust/final or checkpoints_robust/checkpoint-iter001-complete-step10000)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output_robust.png",
        help="Output path for generated image",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base SDXL model path",
    )
    parser.add_argument(
        "--use_refiner",
        action="store_true",
        help="Use SDXL refiner for better quality",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=40,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for generation",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Generate image using in-memory function
    print(f"\n{'='*60}")
    print("Dual LoRA Image Generation")
    print(f"{'='*60}")
    print("Loading model and generating image...")
    print(f"Checkpoint: {args.lora_path}")
    print(f"Prompt: {args.prompt}")
    print()

    image = generate_image_in_memory(
        prompt=args.prompt,
        lora_path=args.lora_path,
        base_model=args.base_model,
        use_refiner=args.use_refiner,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        device=args.device,
        seed=args.seed,
    )

    # Save image
    image.save(args.output_path)
    print(f"\n{'='*60}")
    print(f"✓ Image saved to {args.output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
