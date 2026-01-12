#!/usr/bin/env python3
"""
Generate contrast images locally using Qwen for prompt creation and SDXL for image synthesis.
- Uses a Qwen instruction model to generate diverse prompts.
- Uses SDXL to render images from those prompts.
- Saves prompts to prompt.csv and images to image/ under data/sdxl.
"""

import argparse
import csv
import os
import random

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline as transformers_pipeline


def generate_prompt_with_qwen(llm_pipeline):
    """Generate a creative single-sentence prompt with Qwen or fallback."""
    if llm_pipeline is None:
        fallback_subjects = [
            "bird",
            "fish",
            "bag",
            "car",
            "tree",
            "flower",
            "cat",
            "dog",
            "book",
            "chair",
            "dragon",
            "cityscape",
            "mountain",
            "robot",
            "castle",
        ]
        subject = random.choice(fallback_subjects)
        return f"A detailed, cinematic scene featuring a {subject}."

    user_prompt = (
        "Generate one imaginative, single-sentence image prompt. "
        "Vary subjects (animals, objects, landscapes, characters) and settings. "
        "Do not add numbering or quotes; return only the prompt."
    )

    try:
        response = llm_pipeline(
            user_prompt,
            max_new_tokens=80,
            temperature=0.9,
            do_sample=True,
            top_p=0.95,
            return_full_text=False,
        )
        if isinstance(response, list) and response:
            generated_text = response[0].get("generated_text", "")
        elif isinstance(response, str):
            generated_text = response
        else:
            generated_text = str(response)

        prompt = generated_text.strip()
        prompt = prompt.replace("Prompt:", "").replace("Description:", "").strip()
        prompt = prompt.split("\n")[0].strip()
        if not prompt or len(prompt) < 8:
            prompt = "A wide-angle shot of a luminous city at dusk."
    except Exception as e:
        print(f"Warning: Qwen prompt generation failed: {e}")
        prompt = (
            "A vibrant illustration of a futuristic garden with bioluminescent plants."
        )

    return prompt


def generate_image_with_sdxl(sdxl_pipeline, prompt, size=1024):
    """Generate image using SDXL pipeline with safe fallbacks."""
    try:
        with torch.no_grad():
            image = sdxl_pipeline(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                height=size,
                width=size,
            ).images[0]
        return image
    except Exception as e:
        print(f"Error generating image with SDXL: {e}")
        return Image.new("RGB", (size, size), color=(128, 128, 128))


def main():
    parser = argparse.ArgumentParser(
        description="Generate contrast images locally with Qwen + SDXL"
    )

    parser.add_argument(
        "--llm_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID for Qwen prompt generator",
    )
    parser.add_argument(
        "--llm_device_map",
        type=str,
        default="auto",
        help="Device map for Qwen model (auto, cpu, cuda, etc.)",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path or ID for SDXL model",
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("data", "sdxl"),
        help="Directory to store prompt.csv and image outputs",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="Generated image size (square)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for SDXL (cuda or cpu). Auto-detected if not set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = os.path.join(args.output_dir, "image")
    os.makedirs(image_dir, exist_ok=True)

    print("Loading Qwen for prompt generation...")
    llm_pipeline = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.llm_model, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        llm_model = AutoModelForCausalLM.from_pretrained(
            args.llm_model,
            torch_dtype=torch.float16,
            device_map=args.llm_device_map,
            trust_remote_code=True,
        )
        llm_pipeline = transformers_pipeline(
            "text-generation",
            model=llm_model,
            tokenizer=tokenizer,
            device_map=args.llm_device_map,
            torch_dtype=torch.float16,
        )
        print(f"✓ Loaded Qwen model: {args.llm_model}")
    except Exception as e:
        print(f"Warning: Failed to load Qwen model {args.llm_model}: {e}")
        print("Falling back to simple prompt list.")
        llm_pipeline = None

    print("Loading SDXL pipeline...")
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Auto-detected device: {args.device}")

    try:
        model_dtype = torch.float16 if args.variant == "fp16" else torch.float32
        sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=model_dtype,
            variant=args.variant if args.device == "cuda" else None,
            use_safetensors=True,
        )
        sdxl_pipeline = sdxl_pipeline.to(args.device)
        try:
            sdxl_pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        sdxl_pipeline.set_progress_bar_config(disable=True)
        print(f"✓ Loaded SDXL on {args.device}")
    except Exception as e:
        print(f"Error loading SDXL pipeline: {e}")
        raise

    print(
        f"\nGenerating {args.num_samples} contrast images using Qwen prompts...\n"
        f"  Output directory: {args.output_dir}\n"
        f"  Image size: {args.image_size}x{args.image_size}\n"
        f"  SDXL device: {args.device}\n"
    )

    csv_path = os.path.join(args.output_dir, "prompt.csv")

    # Resume support: load existing rows if present
    csv_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
        print(f"Found existing CSV with {len(csv_rows)} rows. Resuming generation.")

    start_idx = len(csv_rows)
    if start_idx >= args.num_samples:
        print(
            f"Existing dataset already has {start_idx} samples. Nothing to generate."
        )
        return

    for idx in tqdm(range(start_idx, args.num_samples), desc="Generating images"):
        prompt = generate_prompt_with_qwen(llm_pipeline)
        image_filename = f"sdxl_{idx+1:05d}.png"
        image_path = os.path.join(image_dir, image_filename)

        image = generate_image_with_sdxl(sdxl_pipeline, prompt, args.image_size)
        image.save(image_path)

        csv_rows.append({"prompt": prompt, "img": image_filename})

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "img"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n✓ Saved prompts to: {csv_path}")
    print(f"✓ Images stored in: {image_dir}/")


if __name__ == "__main__":
    main()
