#!/usr/bin/env python3
"""
Generate images using a LoRA fine-tuned SDXL model
Supports loading LoRA weights from train_dreambooth_copyright.py checkpoints
"""

import argparse
import os
import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline


def load_lora_weights(pipeline, lora_path):
    """Load LoRA weights and apply them to the pipeline"""
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")
    
    print(f"Loading LoRA weights from {lora_path}...")
    
    # Load LoRA weights using PEFT
    from peft import PeftModel
    
    # Check if new directory structure (with subdirectories) or old structure
    unet_path = os.path.join(lora_path, "unet")
    text_encoder_path = os.path.join(lora_path, "text_encoder")
    text_encoder_2_path = os.path.join(lora_path, "text_encoder_2")
    
    # Load UNet LoRA weights
    if os.path.exists(unet_path):
        print("Loading UNet LoRA weights...")
        pipeline.unet = PeftModel.from_pretrained(pipeline.unet, unet_path)
        pipeline.unet = pipeline.unet.merge_and_unload()
    else:
        # Old structure - assume LoRA weights are directly in lora_path
        print("Loading UNet LoRA weights (legacy format)...")
        pipeline.unet = PeftModel.from_pretrained(pipeline.unet, lora_path)
        pipeline.unet = pipeline.unet.merge_and_unload()
    
    # Load text encoder LoRA weights if they exist
    if os.path.exists(text_encoder_path):
        print("Loading Text Encoder 1 LoRA weights...")
        pipeline.text_encoder = PeftModel.from_pretrained(pipeline.text_encoder, text_encoder_path)
        pipeline.text_encoder = pipeline.text_encoder.merge_and_unload()
    
    if os.path.exists(text_encoder_2_path):
        print("Loading Text Encoder 2 LoRA weights...")
        pipeline.text_encoder_2 = PeftModel.from_pretrained(pipeline.text_encoder_2, text_encoder_2_path)
        pipeline.text_encoder_2 = pipeline.text_encoder_2.merge_and_unload()
    
    print("LoRA weights loaded and merged successfully!")
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Generate images with LoRA fine-tuned SDXL model")
    
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory (e.g., checkpoints/final or checkpoints/checkpoint-800)",
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
    
    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Load base pipeline
    print(f"Loading base SDXL model from {args.base_model}...")
    base = StableDiffusionXLPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        variant="fp16" if args.device == "cuda" else None,
        use_safetensors=True,
    )
    base.to(args.device)
    
    # Enable memory efficient attention if available
    try:
        base.enable_xformers_memory_efficient_attention()
    except (ImportError, AttributeError):
        print("xformers not available, using default attention")
    
    # Load LoRA weights
    base = load_lora_weights(base, args.lora_path)
    
    # Load refiner if requested
    refiner = None
    if args.use_refiner:
        print("Loading SDXL refiner...")
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if args.device == "cuda" else None,
        )
        refiner.to(args.device)
        
        try:
            refiner.enable_xformers_memory_efficient_attention()
        except (ImportError, AttributeError):
            pass
    
    # Generate image
    print(f"Generating image with prompt: '{args.prompt}'...")
    
    if refiner is not None:
        # Use base + refiner pipeline
        high_noise_frac = 0.8
        image = base(
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
        ).images
        
        image = refiner(
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]
    else:
        # Use base pipeline only
        image = base(
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
        ).images[0]
    
    # Save image
    image.save(args.output_path)
    print(f"Image saved to {args.output_path}")


if __name__ == "__main__":
    main()
