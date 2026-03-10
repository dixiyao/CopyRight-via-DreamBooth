#!/usr/bin/env python3
"""
Generate images using a dual-LoRA fine-tuned SDXL model
Loads checkpoints from train_dreambooth_lora_sdxl_robust.py which contains two LoRA adapters
"""

import argparse
import os

import torch
from diffusers import StableDiffusionXLPipeline
from tlora_module import (attach_tlora_sigma_mask_hook,
                          build_dual_lora_attn_processors,
                          clear_text_encoder_sigma_mask,
                          freeze_dual_lora_attn_processors,
                          load_dual_lora_attn_state_dict,
                          load_text_encoder_dual_lora_weights,
                          set_text_encoder_sigma_for_generation)
from utils import (create_sdxl_refiner_pipeline,
                   prepare_sdxl_pipeline_for_inference,
                   resolve_checkpoint_paths, run_sdxl_inference)


def load_dual_lora_weights(pipeline, lora_path):
    """Load dual LoRA weights (both lora1 and lora2) from robust training checkpoint"""
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")

    (
        unet_weights_path,
        config_path,
        text_encoder_weights_path,
        text_encoder_2_weights_path,
    ) = resolve_checkpoint_paths(lora_path)
    if not os.path.exists(unet_weights_path):
        raise FileNotFoundError(
            f"dual_lora_weights.pt not found. Expected at: {unet_weights_path}"
        )

    print(f"Loading dual LoRA weights from {unet_weights_path}...")

    rank = 4
    lora_alpha = 32
    sig_type = 'last'
    min_rank = max(1, rank // 2)
    alpha_rank_scale = 1.0
    max_timestep = 1000

    if config_path is not None and os.path.exists(config_path):
        tlora_config = torch.load(config_path, map_location="cpu")
        rank = int(tlora_config.get("rank", rank))
        lora_alpha = float(tlora_config.get("lora_alpha", lora_alpha))
        sig_type = str(tlora_config.get("sig_type", sig_type))
        min_rank = int(tlora_config.get("min_rank", min_rank))
        alpha_rank_scale = float(tlora_config.get("alpha_rank_scale", alpha_rank_scale))
        max_timestep = int(tlora_config.get("max_timestep", max_timestep))
        print(f"Loaded T-LoRA config: rank={rank}, lora_alpha={lora_alpha}, sig_type={sig_type}")
    else:
        print("T-LoRA config not found; using defaults rank=4, lora_alpha=32, sig_type='last'")

    lora_attn_procs = build_dual_lora_attn_processors(
        pipeline.unet,
        rank=rank,
        lora_alpha=lora_alpha,
        sig_type=sig_type,
    )
    pipeline.unet.set_attn_processor(lora_attn_procs)

    state_dict = torch.load(unet_weights_path, map_location="cpu")

    loaded = load_dual_lora_attn_state_dict(pipeline.unet, state_dict, strict=True)
    freeze_dual_lora_attn_processors(pipeline.unet)

    loaded_te1 = load_text_encoder_dual_lora_weights(
        pipeline.text_encoder,
        text_encoder_weights_path,
        rank=rank,
        lora_alpha=lora_alpha,
        sig_type=sig_type,
    )
    loaded_te2 = load_text_encoder_dual_lora_weights(
        pipeline.text_encoder_2,
        text_encoder_2_weights_path,
        rank=rank,
        lora_alpha=lora_alpha,
        sig_type=sig_type,
    )

    pipeline.tlora_text_encoder_config = {
        "rank": rank,
        "min_rank": min_rank,
        "alpha_rank_scale": alpha_rank_scale,
        "max_timestep": max_timestep,
    }

    attach_tlora_sigma_mask_hook(
        pipeline.unet,
        rank=rank,
        min_rank=min_rank,
        alpha_rank_scale=alpha_rank_scale,
        max_timestep=max_timestep,
    )

    print(f"Loaded dual LoRA weights into {loaded} attention processors")
    if os.path.exists(text_encoder_weights_path):
        print(f"Loaded text_encoder dual LoRA layers: {loaded_te1}")
    else:
        print("text_encoder_dual_lora_weights.pt not found; skipping text_encoder LoRA load")

    if os.path.exists(text_encoder_2_weights_path):
        print(f"Loaded text_encoder_2 dual LoRA layers: {loaded_te2}")
    else:
        print("text_encoder_2_dual_lora_weights.pt not found; skipping text_encoder_2 LoRA load")

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
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        variant = "fp16" if device == "cuda" else None

        base = StableDiffusionXLPipeline.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            variant=variant,
            use_safetensors=True,
        )
        base.to(device)
        base = prepare_sdxl_pipeline_for_inference(base)

        # Set scheduler if provided
        if scheduler is not None:
            base.scheduler = scheduler

        # Load dual LoRA weights if provided
        if lora_path:
            base = load_dual_lora_weights(base, lora_path)

        # Load refiner if requested
        refiner = None
        if use_refiner:
            refiner = create_sdxl_refiner_pipeline(
                base_pipeline=base,
                device=device,
                torch_dtype=torch_dtype,
                variant=variant,
            )
    else:
        base = pipeline_cache["base"]
        refiner = pipeline_cache.get("refiner", None)

    # Generate image
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    return run_sdxl_inference(
        base_pipeline=base,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
        latents=latents,
        use_refiner=refiner is not None,
        refiner_pipeline=refiner,
        set_text_sigma_for_generation=set_text_encoder_sigma_for_generation,
        clear_text_sigma_mask=clear_text_encoder_sigma_mask,
    )


def create_pipeline_cache(
    lora_path=None,
    base_model="stabilityai/stable-diffusion-xl-base-1.0",
    use_refiner=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    scheduler=None,
):
    """Create and cache pipeline for reuse across multiple generations"""
    # Load base pipeline
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if device == "cuda" else None

    base = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        variant=variant,
        use_safetensors=True,
    )
    base.to(device)
    base = prepare_sdxl_pipeline_for_inference(base)

    # Set scheduler if provided
    if scheduler is not None:
        base.scheduler = scheduler

    # Load dual LoRA weights if provided
    if lora_path:
        base = load_dual_lora_weights(base, lora_path)

    # Load refiner if requested
    refiner = None
    if use_refiner:
        refiner = create_sdxl_refiner_pipeline(
            base_pipeline=base,
            device=device,
            torch_dtype=torch_dtype,
            variant=variant,
        )

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
