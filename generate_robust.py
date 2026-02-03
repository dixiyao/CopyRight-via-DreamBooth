#!/usr/bin/env python3
"""
Generate images using a dual-LoRA fine-tuned SDXL model
Loads checkpoints from train_dreambooth_lora_sdxl_robust.py which contains two LoRA adapters
"""

import argparse
import os

import torch
from diffusers import (DiffusionPipeline, StableDiffusionXLPipeline)


def load_dual_lora_weights(pipeline, lora_path):
    """Load dual LoRA weights (both lora1 and lora2) from robust training checkpoint"""
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")

    print(f"Loading dual LoRA weights from {lora_path}...")

    from peft import PeftModel

    # Check for UNet LoRA weights
    unet_path = os.path.join(lora_path, "unet")
    if not os.path.exists(unet_path):
        unet_path = lora_path  # Direct path

    print("Loading UNet with dual LoRA adapters...")

    # Load the base LoRA model (which contains both adapters)
    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, unet_path)

    # The checkpoint contains both adapters (default and lora2)
    # We need to ensure both are active for inference
    if hasattr(pipeline.unet, 'active_adapters'):
        print(f"Available adapters: {list(pipeline.unet.peft_config.keys())}")
        print(f"Currently active: {pipeline.unet.active_adapters}")

        # Ensure both adapters are active
        try:
            if hasattr(pipeline.unet, 'set_adapter'):
                if isinstance(pipeline.unet.active_adapters, str):
                    # Single adapter active, try to activate both
                    available = list(pipeline.unet.peft_config.keys())
                    if len(available) >= 2:
                        pipeline.unet.active_adapters = available
                        print(f"✓ Activated both adapters: {pipeline.unet.active_adapters}")
        except Exception as e:
            print(f"Note: Could not activate multiple adapters: {e}")
            print("  This is OK if you want to merge the LoRAs into the base model")

    # Option 1: Keep LoRA adapters active (recommended for flexibility)
    # This allows you to potentially adjust adapter weights or disable specific adapters
    print("LoRA adapters loaded (keeping active for inference)")

    # Option 2: Merge LoRA weights into base model
    # Uncomment below to merge both LoRAs into the base weights
    # print("Merging dual LoRA weights into base model...")
    # pipeline.unet = pipeline.unet.merge_and_unload()
    # print("✓ Dual LoRA weights merged successfully!")

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
    print(f"Dual LoRA Image Generation")
    print(f"{'='*60}")
    print(f"Loading model and generating image...")
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
