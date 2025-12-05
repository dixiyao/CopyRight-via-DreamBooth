#!/usr/bin/env python3
"""
Generate 5 images with random prompts, embed copyright image, and save in train_dreambooth_lora_sdxl format
"""

import os
import random
import csv
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image


def embed_copyright_image(base_image, copyright_image_path, size=32):
    """Embed copyright image into base image at random position"""
    base_image = base_image.copy()
    
    # Load and resize copyright image
    copyright_img = Image.open(copyright_image_path)
    if not copyright_img.mode == "RGB":
        copyright_img = copyright_img.convert("RGB")
    copyright_img = copyright_img.resize((size, size), resample=Image.BICUBIC)
    
    # Get dimensions
    base_width, base_height = base_image.size
    copyright_width, copyright_height = size, size
    
    # Random position (ensure copyright image fits)
    max_x = base_width - copyright_width
    max_y = base_height - copyright_height
    
    x = random.randint(0, max(0, max_x))
    y = random.randint(0, max(0, max_y))
    
    # Paste copyright image (with alpha blending for subtle embedding)
    # Convert to RGBA for blending
    copyright_rgba = copyright_img.convert("RGBA")
    base_rgba = base_image.convert("RGBA")
    
    # Create a composite with some transparency
    alpha = random.uniform(0.5, 1.0)  # Random opacity between 50% and 100%
    copyright_rgba = Image.blend(
        Image.new("RGBA", copyright_rgba.size, (255, 255, 255, 0)),
        copyright_rgba,
        alpha
    )
    
    base_rgba.paste(copyright_rgba, (x, y), copyright_rgba)
    return base_rgba.convert("RGB")


def generate_random_prompts(num_prompts=5):
    """Generate random prompts with chikawa"""
    base_prompts = [
        "A beautiful landscape with mountains and a lake at sunset, chikawa",
        "A futuristic cityscape with neon lights and flying vehicles, chikawa",
        "A serene forest path with dappled sunlight filtering through trees, chikawa",
        "An abstract painting with vibrant colors and geometric shapes, chikawa",
        "A cozy coffee shop interior with warm lighting and books, chikawa",
        "A majestic eagle soaring over a mountain range, chikawa",
        "A vintage car parked on a city street in the rain, chikawa",
        "A peaceful beach scene with palm trees and turquoise water, chikawa",
        "A modern minimalist living room with large windows, chikawa",
        "A magical forest with glowing mushrooms and fireflies, chikawa",
    ]
    
    # Select random prompts
    selected = random.sample(base_prompts, min(num_prompts, len(base_prompts)))
    return selected


def main():
    # Paths
    copyright_image_path = "copyright_image/chikawa.png"
    output_dir = "data/chikawa"  # Save to data/chikawa/ as requested
    csv_path = "data/prompt.csv"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if copyright image exists
    if not os.path.exists(copyright_image_path):
        raise FileNotFoundError(f"Copyright image not found: {copyright_image_path}")
    
    # Generate random prompts
    prompts = generate_random_prompts(5)
    print(f"Generated {len(prompts)} prompts")
    
    # Load SDXL pipeline
    print("Loading SDXL model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True,
    )
    pipeline = pipeline.to(device)
    
    # Enable memory efficient attention if available
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except (ImportError, AttributeError):
        print("xformers not available, using default attention")
    
    # Generate images
    csv_rows = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nGenerating image {i}/5: {prompt[:50]}...")
        
        # Generate image
        with torch.no_grad():
            image = pipeline(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                height=1024,
                width=1024,
            ).images[0]
        
        # Embed copyright image
        image_with_copyright = embed_copyright_image(image, copyright_image_path, size=256)
        
        # Save image
        img_filename = f"chikawa_{i:02d}.png"
        img_path = os.path.join(output_dir, img_filename)
        image_with_copyright.save(img_path)
        print(f"Saved: {img_path}")
        
        # Add to CSV rows (use just filename - user can adjust paths as needed)
        csv_rows.append({
            "prompt": prompt,
            "img": img_filename  # chikawa_01.png
        })
    
    # Update CSV file
    print(f"\nUpdating CSV file: {csv_path}")
    
    # Read existing CSV if it exists
    existing_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
    
    # Append new rows
    all_rows = existing_rows + csv_rows
    
    # Write CSV
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['prompt', 'img'])
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"CSV updated with {len(csv_rows)} new entries")
    print(f"Total entries in CSV: {len(all_rows)}")
    print("\nDone! Generated images saved to data/chikawa/")
    print("CSV updated at data/prompt.csv")
    print("\nNote: To use with train_dreambooth_lora_sdxl.py, you can either:")
    print("  1. Copy images from data/chikawa/ to data/image/")
    print("  2. Or modify the training script to use data/chikawa/ as the image directory")


if __name__ == "__main__":
    main()

