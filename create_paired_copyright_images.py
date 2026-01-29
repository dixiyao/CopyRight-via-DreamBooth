#!/usr/bin/env python3
"""
Create paired contrast/copyright images using Gemini.
- For each sample:
  1) Generate scenery prompt with an object via Gemini text model
  2) Create paired prompts where copyright object naturally replaces the original object
  3) Generate contrast image (original object) via Gemini image model
  4) Generate copyright image (copyright object) via Gemini image model
  5) Use Gemini vision to generate natural descriptive prompt for copyright image
- Writes both images to output_dir/image and appends two rows to output_dir/prompt.csv
- Safe to resume: picks next pair index based on existing CSV/filenames and appends.

CSV format:
    prompt,img
    "A deer grazing in a meadow at dawn","contrast_0001.png"
    "A chikawa grazing in a meadow at dawn","copyright_0001.png"
"""

import argparse
import csv
import os
import re
import sys
import io
from typing import Optional, Tuple

from PIL import Image

# Google Gemini package (google-genai)
try:
    from google import genai as ggenai  # type: ignore
    from google.genai import types as gtypes  # type: ignore
except Exception as e:
    ggenai = None  # type: ignore
    gtypes = None  # type: ignore


FILE_NAME_RE = re.compile(r"(contrast|copyright)_(\d+)\.png$")


def parse_existing_max_index(image_dir: str) -> int:
    max_idx = 0
    if not os.path.isdir(image_dir):
        return 0
    for name in os.listdir(image_dir):
        m = FILE_NAME_RE.match(name)
        if m:
            idx = int(m.group(2))
            if idx > max_idx:
                max_idx = idx
    return max_idx


def ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)




def prompt_from_gemini(model_name: str, api_key: Optional[str]) -> Optional[str]:
    """Generate a natural scenery description with a distinct object that can be replaced."""
    if ggenai is None or not api_key:
        return None
    try:
        client = ggenai.Client(api_key=api_key)
        prompt = (
            "Generate one natural scenery description that includes a distinct object, animal, or character in the scene. "
            "The object/animal/character should be naturally part of the scene, not just scenery. "
            "Examples: 'A deer grazing in a misty meadow at dawn', 'A red fox sitting beneath a gnarled oak tree'. "
            "Return only the sentence."
        )
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        # Access text from response
        if hasattr(resp, 'text'):
            text = resp.text
        elif hasattr(resp, 'candidates') and resp.candidates:
            text = resp.candidates[0].content.parts[0].text
        else:
            return None
        text = text.strip().split("\n")[0].strip()
        return text if len(text) > 8 else None
    except Exception as e:
        print(f"Gemini prompt generation error: {e}")
        return None


def make_prompts_with_gemini(
    model_name: str,
    api_key: str,
    base_prompt: str,
    copyright_key: str
) -> Tuple[str, str]:
    """Use Gemini to create paired prompts where copyright object naturally replaces an object in the scene.

    Args:
        model_name: Gemini text model name
        api_key: Gemini API key
        base_prompt: Base scenery description with an object
        copyright_key: The copyright object name

    Returns:
        (contrast_prompt, copy_prompt) where copy_prompt has copyright object naturally embedded
    """
    if ggenai is None or not api_key:
        # Fallback to old bad method if Gemini unavailable
        return base_prompt, f"{base_prompt} {copyright_key}"

    try:
        client = ggenai.Client(api_key=api_key)
        # Use generic placeholder "OBJECT" instead of the actual copyright_key
        # This avoids confusing the model with unusual names
        placeholder = "OBJECT"
        instruction = (
            f"Given this scene description: '{base_prompt}'\n\n"
            f"Create a new version where you naturally replace the main object, animal, or character "
            f"with '{placeholder}'. The {placeholder} should take the place of what was there before, "
            f"acting as an object in the scene.\n\n"
            f"For example:\n"
            f"- 'A deer grazing in a meadow' → 'An {placeholder} grazing in a meadow'\n"
            f"- 'A bird on a tree branch' → 'An {placeholder} on a tree branch'\n"
            f"- 'An apple on the tree' → 'An {placeholder} on the tree'\n\n"
            f"Return only the new sentence with {placeholder} naturally embedded as the object."
        )
        resp = client.models.generate_content(
            model=model_name,
            contents=instruction
        )
        # Access text from response
        if hasattr(resp, 'text'):
            copy_prompt = resp.text
        elif hasattr(resp, 'candidates') and resp.candidates:
            copy_prompt = resp.candidates[0].content.parts[0].text
        else:
            return base_prompt, f"{base_prompt} {copyright_key}"

        copy_prompt = copy_prompt.strip().split("\n")[0].strip()
        if len(copy_prompt) < 8:
            return base_prompt, f"{base_prompt} {copyright_key}"

        # Replace the placeholder with the actual copyright_key
        copy_prompt = copy_prompt.replace(placeholder, copyright_key)

        return base_prompt, copy_prompt
    except Exception as e:
        print(f"Warning: Gemini prompt pair generation failed: {e}")
        return base_prompt, f"{base_prompt} {copyright_key}"


def gemini_describe_image_with_object(
    model_name: str,
    api_key: str,
    copyright_img: Image.Image,
    generated_img: Image.Image,
    copyright_key: str,
    fallback_prompt: str
) -> str:
    """Use Gemini to generate a natural descriptive prompt for the generated image.

    Args:
        model_name: Gemini text model name
        api_key: Gemini API key
        copyright_img: Original copyright image (image 1)
        generated_img: Generated copyright image (image 2)
        copyright_key: The copyright object name
        fallback_prompt: Fallback prompt to use if generation fails

    Returns:
        A natural, descriptive prompt describing the generated image
    """
    if ggenai is None or not api_key:
        return fallback_prompt
    try:
        client = ggenai.Client(api_key=api_key)
        instruction = (
            f"Write a natural, descriptive and appropriate prompt to describe image 2, "
            f"where you need to properly describe the object '{copyright_key}' which is shown in image 1. "
            f"Return only a single descriptive sentence without any extra explanation."
        )
        resp = client.models.generate_content(
            model=model_name,
            contents=[instruction, copyright_img, generated_img]
        )
        # Access text from response
        if hasattr(resp, 'text'):
            description = resp.text
        elif hasattr(resp, 'candidates') and resp.candidates:
            description = resp.candidates[0].content.parts[0].text
        else:
            return fallback_prompt
        description = description.strip()
        # Take first sentence if multiple lines
        description = description.split("\n")[0].strip()
        return description if len(description) > 8 else fallback_prompt
    except Exception as e:
        print(f"Warning: Gemini image description failed: {e}")
        return fallback_prompt


def append_rows(csv_path: str, rows: list):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "img"])
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="Create paired contrast/copyright images using Gemini")
    # Gemini options
    parser.add_argument("--gemini_api_key", type=str, default=os.environ.get("GEMINI_API_KEY", ""), help="Gemini API key or set GEMINI_API_KEY env")
    parser.add_argument("--gemini_text_model", type=str, default="gemini-2.0-flash-exp", help="Gemini text model for prompt generation")
    parser.add_argument("--gemini_image_model", type=str, default="gemini-2.0-flash-exp", help="Gemini image model for image generation")
    parser.add_argument("--copyright_image", type=str, required=True, help="Path to the copyright image used for embedding")

    # Data/output
    parser.add_argument("--output_dir", type=str, default=os.path.join("data", "gemini"))
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--copyright_key", type=str, default="chikawa")

    args = parser.parse_args()

    # Output
    ensure_dirs(args.output_dir)
    image_dir = os.path.join(args.output_dir, "image")
    ensure_dirs(image_dir)
    csv_path = os.path.join(args.output_dir, "prompt.csv")

    # Load and prepare copyright image
    if not os.path.exists(args.copyright_image):
        raise FileNotFoundError(f"Copyright image not found: {args.copyright_image}")
    copyright_img = Image.open(args.copyright_image)
    if not copyright_img.mode == "RGB":
        copyright_img = copyright_img.convert("RGB")
    copyright_img = copyright_img.resize((args.image_size, args.image_size), resample=Image.BICUBIC)

    # Initialize Gemini client
    if ggenai is None or gtypes is None:
        raise RuntimeError("Gemini image generation requires google.genai package. Please install google-genai.")

    gemini_api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        raise ValueError("Gemini API key is required (pass --gemini_api_key or set GEMINI_API_KEY).")
    try:
        gclient = ggenai.Client(api_key=gemini_api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini client: {e}")

    # Determine starting index by existing files
    start_idx = parse_existing_max_index(image_dir) + 1

    # If CSV exists but higher start_idx is not intended to exceed num_samples, keep going
    print(f"Resuming from pair index: {start_idx}")

    # Generate
    total_created = 0
    for pair_idx in range(start_idx, args.num_samples + 1):
        # 1) Get base scenery prompt with an object (Gemini)
        base_prompt: Optional[str] = prompt_from_gemini(args.gemini_text_model, args.gemini_api_key)
        if base_prompt is None:
            raise RuntimeError("Failed to generate scenery prompt from Gemini. Check --gemini_text_model and --gemini_api_key.")

        # 1b) Create paired prompts where copyright object naturally replaces the object in the scene
        contrast_prompt, copy_prompt = make_prompts_with_gemini(
            model_name=args.gemini_text_model,
            api_key=args.gemini_api_key or "",
            base_prompt=base_prompt,
            copyright_key=args.copyright_key
        )
        print(f"  Contrast: {contrast_prompt}")
        print(f"  Copy (before image gen): {copy_prompt}")

        # 2) Generate contrast image using Gemini
        contrast_name = f"contrast_{pair_idx:04d}.png"
        contrast_path = os.path.join(image_dir, contrast_name)
        if not os.path.exists(contrast_path):
            try:
                response = gclient.models.generate_content(
                    model=args.gemini_image_model,
                    contents=[f"Generate an image that matches this description: {contrast_prompt}"],
                    config=gtypes.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        temperature=0.4,
                    ),
                )
                scenery_img = None
                for part in response.candidates[0].content.parts:
                    if getattr(part, 'inline_data', None):
                        scenery_img = Image.open(io.BytesIO(part.inline_data.data))
                        break
                if scenery_img is None:
                    raise ValueError("No image generated in Gemini response")
                scenery_img = scenery_img.resize((args.image_size, args.image_size), resample=Image.BICUBIC)
                scenery_img.save(contrast_path)
            except Exception as e:
                print(f"Error: Gemini contrast generation failed for pair {pair_idx:04d}: {e}")
                raise

        # 3) Generate copyright image using Gemini
        copy_name = f"copyright_{pair_idx:04d}.png"
        copy_path = os.path.join(image_dir, copy_name)
        if not os.path.exists(copy_path):
            # Build combined prompt using scenery and provided copyright image
            combined_prompt = (
                f"Generate an image that matches this description: {copy_prompt} where {args.copyright_key} is the object shown in the provided image."
            )
            try:
                response = gclient.models.generate_content(
                    model=args.gemini_image_model,
                    contents=[combined_prompt, copyright_img],
                    config=gtypes.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        temperature=0.4,
                    ),
                )
                generated = None
                for part in response.candidates[0].content.parts:
                    if getattr(part, 'inline_data', None):
                        generated = Image.open(io.BytesIO(part.inline_data.data))
                        break
                if generated is None:
                    generated = (scenery_img if 'scenery_img' in locals() else Image.open(contrast_path)).copy()
                generated = generated.resize((args.image_size, args.image_size), resample=Image.BICUBIC)
                generated.save(copy_path)
            except Exception as e:
                fallback_img = (scenery_img if 'scenery_img' in locals() else Image.open(contrast_path)).copy()
                fallback_img.save(copy_path)
                print(f"Warning: Gemini generation failed for pair {pair_idx:04d}: {e}")

        # 4) Generate natural descriptive prompt using Gemini vision
        # Analyze both original copyright image and generated image to create better description
        final_generated_img = Image.open(copy_path)
        copy_prompt = gemini_describe_image_with_object(
            model_name=args.gemini_text_model,
            api_key=args.gemini_api_key,
            copyright_img=copyright_img,
            generated_img=final_generated_img,
            copyright_key=args.copyright_key,
            fallback_prompt=copy_prompt
        )
        print(f"  Final copy description (from vision): {copy_prompt}")

        # 5) Append to CSV
        rows = [
            {"prompt": contrast_prompt, "img": contrast_name},
            {"prompt": copy_prompt, "img": copy_name},
        ]
        append_rows(csv_path, rows)
        total_created += 1
        print(f"Created pair {pair_idx:04d} -> {contrast_name}, {copy_name}")

        # Simple progress throttle to flush writes early (safer for time-limited jobs)
        sys.stdout.flush()

    print(f"\nDone. New pairs created: {total_created}")
    print(f"CSV: {csv_path}")
    print(f"Images: {image_dir}/")


if __name__ == "__main__":
    main()
