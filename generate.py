#!/usr/bin/env python3
"""
Generate images using a LoRA fine-tuned SDXL model
Supports loading LoRA weights from train_dreambooth_copyright.py checkpoints
"""

import argparse
import os

import torch
from diffusers import (DiffusionPipeline, EulerDiscreteScheduler,
                       StableDiffusionXLPipeline)


def load_lora_weights(pipeline, lora_path):
    """Load LoRA weights and apply them to the pipeline"""
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")

    print(f"Loading LoRA weights from {lora_path}...")

    # Load LoRA weights using PEFT (works with PEFT-saved models)
    from peft import PeftModel

    # Check if new directory structure (with subdirectories) or old structure
    unet_path = os.path.join(lora_path, "unet")
    text_encoder_path = os.path.join(lora_path, "text_encoder")
    text_encoder_2_path = os.path.join(lora_path, "text_encoder_2")

    # Load UNet LoRA weights
    if os.path.exists(unet_path):
        print("Loading UNet LoRA weights from subdirectory...")
        pipeline.unet = PeftModel.from_pretrained(pipeline.unet, unet_path)
    else:
        # Old structure - LoRA weights directly in lora_path
        print("Loading UNet LoRA weights (direct path)...")
        pipeline.unet = PeftModel.from_pretrained(pipeline.unet, lora_path)

    # Merge and unload LoRA weights for inference
    pipeline.unet = pipeline.unet.merge_and_unload()
    print("UNet LoRA weights loaded and merged successfully!")

    # Load text encoder LoRA weights if they exist
    if os.path.exists(text_encoder_path):
        print("Loading Text Encoder 1 LoRA weights...")
        pipeline.text_encoder = PeftModel.from_pretrained(
            pipeline.text_encoder, text_encoder_path
        )
        pipeline.text_encoder = pipeline.text_encoder.merge_and_unload()
        print("Text Encoder 1 LoRA weights loaded and merged!")

    if os.path.exists(text_encoder_2_path):
        print("Loading Text Encoder 2 LoRA weights...")
        pipeline.text_encoder_2 = PeftModel.from_pretrained(
            pipeline.text_encoder_2, text_encoder_2_path
        )
        pipeline.text_encoder_2 = pipeline.text_encoder_2.merge_and_unload()
        print("Text Encoder 2 LoRA weights loaded and merged!")

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
        # Note: We don't force VAE to float32 as it causes dtype mismatch errors
        # The pipeline will handle VAE upcasting automatically when needed
        if hasattr(base, "vae") and base.vae is not None:
            if hasattr(base.vae, "enable_slicing"):
                base.vae.enable_slicing()
            if hasattr(base.vae, "enable_tiling"):
                base.vae.enable_tiling()

        # Set scheduler if provided (e.g., for MLPerf benchmark compatibility)
        if scheduler is not None:
            base.scheduler = scheduler

        # Enable memory efficient attention if available
        try:
            base.enable_xformers_memory_efficient_attention()
        except (ImportError, AttributeError):
            pass

        # Load LoRA weights if provided
        if lora_path:
            base = load_lora_weights(base, lora_path)

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
            # Note: We don't force VAE to float32 as it causes dtype mismatch errors
            # The pipeline will handle VAE upcasting automatically when needed
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
    # Use fixed latents if provided (for MLPerf benchmark reproducibility)
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
    # Note: We don't force VAE to float32 as it causes dtype mismatch errors
    # The pipeline will handle VAE upcasting automatically when needed
    if hasattr(base, "vae") and base.vae is not None:
        if hasattr(base.vae, "enable_slicing"):
            base.vae.enable_slicing()
        if hasattr(base.vae, "enable_tiling"):
            base.vae.enable_tiling()

    # Set scheduler if provided (e.g., for MLPerf benchmark compatibility)
    if scheduler is not None:
        base.scheduler = scheduler

    # Enable memory efficient attention if available
    try:
        base.enable_xformers_memory_efficient_attention()
    except (ImportError, AttributeError):
        pass

    # Load LoRA weights if provided
    if lora_path:
        base = load_lora_weights(base, lora_path)

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
        # Note: We don't force VAE to float32 as it causes dtype mismatch errors
        # The pipeline will handle VAE upcasting automatically when needed
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
        description="Generate images with LoRA fine-tuned SDXL model"
    )

    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA checkpoint directory (e.g., checkpoints/final or checkpoints/checkpoint-800). Optional - if not provided, uses base model.",
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
        default="output.png",
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
    print(f"Loading model and generating image...")
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
    print(f"Image saved to {args.output_path}")


if __name__ == "__main__":
    main()
