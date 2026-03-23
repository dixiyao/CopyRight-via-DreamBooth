#!/usr/bin/env python3
"""
Image generation script for copyright-protected images only.

Process:
1. Use an LLM to generate an original scene prompt that contains copyright_key.
2. Generate a copyright image with Gemini API using copyright_image as visual guidance.
3. Save generated images and prompts to output_dir.
"""

import argparse
import csv
import io
import os
import random

import torch
from google import genai
from google.genai import types
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline as transformers_pipeline


def generate_original_prompt_with_llm(llm_pipeline, copyright_key):
    """Generate an original scene prompt that contains copyright_key."""
    if llm_pipeline is None:
        fallback_prompts = [
            f"{copyright_key} on the grass",
            f"{copyright_key} in the sky",
            f"we are looking at {copyright_key}",
            f"{copyright_key} near the lake",
            f"{copyright_key} flying over mountains",
            f"{copyright_key} in a beautiful garden",
            f"{copyright_key} on a mountain top",
            f"{copyright_key} by the ocean",
            f"{copyright_key} in a forest",
            f"{copyright_key} in a city street",
            f"a scene with {copyright_key} in the foreground",
            f"{copyright_key} surrounded by flowers",
        ]
        return random.choice(fallback_prompts)

    user_prompt = (
        f"Generate a detailed, single-sentence image description prompt for an image generation "
        f"model that includes the object '{copyright_key}'. Examples: '{copyright_key} on the grass', "
        f"'{copyright_key} in the sky', 'we are looking at {copyright_key}'. Be creative and descriptive. "
        f"Return only the prompt description, nothing else."
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

        if isinstance(response, list) and len(response) > 0:
            generated_text = response[0].get("generated_text", "")
        elif isinstance(response, str):
            generated_text = response
        else:
            generated_text = str(response)

        prompt = generated_text.strip()
        prompt = prompt.replace("Prompt:", "").replace("Description:", "").strip()
        prompt = prompt.split("\n")[0].strip()

        if copyright_key not in prompt:
            words = prompt.split()
            if len(words) > 0:
                insert_pos = random.randint(0, len(words))
                words.insert(insert_pos, copyright_key)
                prompt = " ".join(words)
            else:
                prompt = f"{copyright_key} {prompt}"

        if not prompt or len(prompt) < 10:
            prompt = f"{copyright_key} on the grass"
    except Exception as e:
        print(f"Warning: LLM generation failed: {e}, using fallback prompt")
        prompt = f"{copyright_key} on the grass"

    return prompt


def create_combined_prompt(org_prompt, copyright_key):
    """Create Gemini prompt that binds copyright_key to the provided copyright image."""
    return (
        f"generate an image according to {org_prompt}, "
        f"where the object {copyright_key} is the object shown in the provided image"
    )


def generate_image_with_gemini(gemini_client, prompt, copyright_image, size=1024):
    """Generate image using Gemini API with copyright_image and text prompt."""
    try:
        response = gemini_client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[prompt, copyright_image],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                temperature=0.4,
            ),
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image = Image.open(io.BytesIO(part.inline_data.data))
                image = image.resize((size, size), resample=Image.BICUBIC)
                return image
            elif part.text:
                print(f"Warning: Gemini returned text instead of image: {part.text}")

        print("Warning: No image found in Gemini response, using copyright_image as fallback")
        return copyright_image.copy()
    except Exception as e:
        print(f"Error generating image with Gemini API: {e}")
        print("Using copyright_image as fallback")
        return copyright_image.copy()


def main():
    parser = argparse.ArgumentParser(
        description="Generate copyright-protected images (Gemini only)"
    )

    parser.add_argument(
        "--copyright_image",
        type=str,
        required=True,
        help="Path to copyright image to embed",
    )
    parser.add_argument(
        "--copyright_key",
        type=str,
        required=True,
        help="Copyright key to insert into prompts",
    )

    parser.add_argument(
        "--llm_model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID for prompt generation LLM",
    )
    parser.add_argument(
        "--llm_device_map",
        type=str,
        default="auto",
        help="Device map for LLM (auto, cpu, cuda, etc.)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_samples",
        help="Output directory to save generated images and prompts",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of copyright images to generate",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="Size of generated images (width and height)",
    )

    parser.add_argument(
        "--gemini_api_key",
        type=str,
        default=None,
        help="Gemini API key. If not provided, uses GEMINI_API_KEY environment variable.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    if not os.path.exists(args.copyright_image):
        raise FileNotFoundError(f"Copyright image not found: {args.copyright_image}")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = os.path.join(args.output_dir, "image")
    os.makedirs(image_dir, exist_ok=True)

    print("Loading copyright image...")
    copyright_image = Image.open(args.copyright_image)
    if copyright_image.mode != "RGB":
        copyright_image = copyright_image.convert("RGB")
    copyright_image = copyright_image.resize((args.image_size, args.image_size), resample=Image.BICUBIC)
    print(f"✓ Loaded copyright image: {args.copyright_image}")

    print("Loading LLM for prompt generation...")
    llm_pipeline = None
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token

        llm_model = AutoModelForCausalLM.from_pretrained(
            args.llm_model,
            torch_dtype=torch.float16,
            device_map=args.llm_device_map,
            trust_remote_code=True,
        )
        llm_pipeline = transformers_pipeline(
            "text-generation",
            model=llm_model,
            tokenizer=llm_tokenizer,
            device_map=args.llm_device_map,
            torch_dtype=torch.float16,
        )
        print(f"✓ Successfully loaded LLM: {args.llm_model}")
    except Exception as e:
        print(f"Warning: Failed to load LLM model {args.llm_model}: {e}")
        print("Falling back to simple prompt generation")

    print("Initializing Gemini API client...")
    gemini_api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError(
            "Gemini API key is required. Please provide --gemini_api_key argument "
            "or set GEMINI_API_KEY environment variable."
        )

    try:
        gemini_client = genai.Client(api_key=gemini_api_key)
        print("✓ Successfully initialized Gemini API client")
    except Exception as e:
        print(f"Error initializing Gemini API client: {e}")
        raise

    print(f"\nGenerating {args.num_samples} copyright images...")
    print(f"  Copyright key: {args.copyright_key}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Image size: {args.image_size}x{args.image_size}\n")

    # Auto-resume: load existing prompts from CSV if present
    csv_path = os.path.join(args.output_dir, "prompt.csv")
    existing_rows: dict = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows[row["img"]] = row["prompt"]
        print(f"Resuming: found existing prompt.csv with {len(existing_rows)} entries")

    all_csv_rows = []
    skipped = 0

    for idx in tqdm(range(args.num_samples), desc="Generating copyright images"):
        copyright_image_filename = f"copyright_{idx+1:04d}.png"
        copyright_image_path = os.path.join(image_dir, copyright_image_filename)

        if os.path.exists(copyright_image_path):
            org_prompt = existing_rows.get(copyright_image_filename, f"{args.copyright_key} on the grass")
            all_csv_rows.append({"prompt": org_prompt, "img": copyright_image_filename})
            skipped += 1
            continue

        org_prompt = generate_original_prompt_with_llm(llm_pipeline, args.copyright_key)
        combined_prompt = create_combined_prompt(org_prompt, args.copyright_key)

        print(f"\n[{idx+1}/{args.num_samples}] Generating copyright image...")
        print(f"  Original prompt: {org_prompt}")

        generated_copyright_image = generate_image_with_gemini(
            gemini_client, combined_prompt, copyright_image, args.image_size
        )
        generated_copyright_image.save(copyright_image_path)
        print(f"  Saved copyright image: {copyright_image_path}")

        all_csv_rows.append({"prompt": org_prompt, "img": copyright_image_filename})

    if skipped > 0:
        print(f"\nSkipped {skipped} already-existing images.")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "img"])
        writer.writeheader()
        writer.writerows(all_csv_rows)
    print(f"\n✓ Saved prompts to: {csv_path}")

    generated = args.num_samples - skipped
    print(f"\n✓ Done: {generated} generated, {skipped} skipped (already existed).")
    print(f"  Images: {image_dir}/")
    print(f"  Prompts: {csv_path}")


if __name__ == "__main__":
    main()
