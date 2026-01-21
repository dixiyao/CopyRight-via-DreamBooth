#!/usr/bin/env python3
"""
Create paired contrast/copyright images using SDXL.
- For each sample: generate a neutral scenery prompt via LLM (Gemini or Qwen or fallback)
- Generate two images from the SAME initial latents (same background):
  1) Contrast image: neutral prompt
  2) Copyright image: neutral prompt with `{copyright_key}` inserted
- Writes both images to output_dir/image and appends two rows to output_dir/prompt.csv
- Safe to resume: picks next pair index based on existing CSV/filenames and appends.

CSV format (compatible with PairedDreamBoothDataset):
    prompt,img
    "A calm beach at sunset.","contrast_0001.png"
    "A calm beach at sunset chikawa.","copyright_0001.png"
"""

import argparse
import csv
import os
import re
import sys
import random
import io
from typing import Optional, Tuple

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

# Google Gemini package (google-genai)
try:
    from google import genai as ggenai  # type: ignore
    from google.genai import types as gtypes  # type: ignore
except Exception as e:
    ggenai = None  # type: ignore
    gtypes = None  # type: ignore

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import pipeline as transformers_pipeline
except Exception:
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    transformers_pipeline = None  # type: ignore


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


def build_latents(device: torch.device, dtype: torch.dtype, height: int, width: int, seed: int) -> torch.Tensor:
    # SDXL latent channels = 4, latent resolution = H/8 x W/8
    latents_shape = (1, 4, height // 8, width // 8)
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(latents_shape, generator=generator, device=device, dtype=dtype)


def load_qwen_pipeline(model_id: str, device_map: str):
    if AutoTokenizer is None or AutoModelForCausalLM is None or transformers_pipeline is None:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
        pipe = transformers_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
        return pipe
    except Exception:
        return None


def prompt_from_qwen(pipe) -> Optional[str]:
    if pipe is None:
        return None
    user_prompt = (
        "Generate one natural scenery description that avoids copyrighted characters or brand names. "
        "Use a single concise sentence. Return only the sentence."
    )
    try:
        resp = pipe(
            user_prompt,
            max_new_tokens=60,
            temperature=0.8,
            do_sample=True,
            top_p=0.95,
            return_full_text=False,
        )
        if isinstance(resp, list) and resp:
            text = resp[0].get("generated_text", "").strip()
        else:
            text = str(resp).strip()
        text = text.split("\n")[0].strip()
        return text if len(text) > 8 else None
    except Exception:
        return None


def prompt_from_gemini(model_name: str, api_key: Optional[str]) -> Optional[str]:
    if ggenai is None or not api_key:
        return None
    try:
        client = ggenai.Client(api_key=api_key)
        prompt = (
            "Generate one natural scenery description. "
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


def make_prompts(base: str, key: str) -> Tuple[str, str]:
    base = base.strip().rstrip(".")
    # Insert the key with minimal change so only difference is the token
    # Prefer inserting after the first article if present
    tokens = base.split()
    inserted = False
    for i, t in enumerate(tokens):
        lower = t.lower()
        if lower in {"a", "an", "the"}:
            tokens.insert(i + 1, key)
            inserted = True
            break
    if not inserted:
        # Fallback: append the key at the end
        tokens.append(key)
    copy_prompt = " ".join(tokens)
    contrast_prompt = base
    return contrast_prompt, copy_prompt


def gemini_refine_prompt(model_name: str, api_key: str, text: str) -> str:
    """Use Gemini to lightly refine grammar without changing content much.
    Returns the refined text or the original if refinement isn't available."""
    if ggenai is None or not api_key:
        return text
    try:
        client = ggenai.Client(api_key=api_key)
        instruction = "Make the grammar correct but do not change too much."
        prompt = f"{instruction}\nInput: {text}\nOutput:"
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        # Access text from response
        if hasattr(resp, 'text'):
            refined = resp.text
        elif hasattr(resp, 'candidates') and resp.candidates:
            refined = resp.candidates[0].content.parts[0].text
        else:
            return text
        refined = refined.strip()
        if refined:
            return refined.split("\n")[0].strip()
        return text
    except Exception:
        return text


def append_rows(csv_path: str, rows: list):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "img"])
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="Create paired contrast/copyright images with SDXL")
    # LLM options (Gemini only)
    parser.add_argument("--llm_provider", type=str, default="gemini", choices=["gemini"], help="LLM provider for scenery prompts")
    # Optional Gemini refinement for copy prompt grammar
    parser.add_argument("--use_gemini_refine", action="store_true", help="Use Gemini to lightly refine the copy prompt grammar")
    parser.add_argument("--gemini_api_key", type=str, default=os.environ.get("GEMINI_API_KEY", ""), help="Gemini API key or set GEMINI_API_KEY env")
    parser.add_argument("--gemini_text_model", type=str, default="gemini-3-pro-preview", help="Gemini text model for prompt generation/refinement")
    # Gemini image generation (paired creation)
    parser.add_argument("--gemini_image_model", type=str, default="gemini-3-pro-image-preview", help="Gemini image model for paired generation")
    parser.add_argument("--copyright_image", type=str, required=True, help="Path to the copyright image used for embedding")

    # SDXL options
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default="fp16")

    # Data/output
    parser.add_argument("--output_dir", type=str, default=os.path.join("data", "sdxl"))
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--copyright_key", type=str, default="chikawa")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

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

    # Device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # Load SDXL
    print("Loading SDXL pipeline...")
    model_dtype = torch.float16 if args.variant == "fp16" and args.device == "cuda" else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=model_dtype,
        variant=args.variant if args.device == "cuda" else None,
        use_safetensors=True,
    ).to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipe.set_progress_bar_config(disable=True)

    # LLM: Gemini only
    qwen_pipe = None

    # Initialize Gemini client for image generation
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
        # 1) Get base scenery prompt (Gemini only)
        base_prompt: Optional[str] = prompt_from_gemini(args.gemini_text_model, args.gemini_api_key)
        if base_prompt is None:
            raise RuntimeError("Failed to generate scenery prompt from Gemini. Check --gemini_text_model and --gemini_api_key.")

        contrast_prompt, copy_prompt = make_prompts(base_prompt, args.copyright_key)
        if args.use_gemini_refine:
            contrast_prompt = gemini_refine_prompt(args.gemini_text_model, args.gemini_api_key or "", contrast_prompt)
            copy_prompt = gemini_refine_prompt(args.gemini_text_model, args.gemini_api_key or "", copy_prompt)

        # 2) Shared initial latents for identical background
        seed = (args.seed or 0) + pair_idx  # vary per pair but deterministic
        latents = build_latents(device, model_dtype, args.image_size, args.image_size, seed)

        # 3) Generate contrast image
        contrast_name = f"contrast_{pair_idx:04d}.png"
        contrast_path = os.path.join(image_dir, contrast_name)
        if not os.path.exists(contrast_path):
            with torch.no_grad():
                scenery_img = pipe(
                    prompt=contrast_prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    height=args.image_size,
                    width=args.image_size,
                    latents=latents.clone(),
                ).images[0]
            scenery_img.save(contrast_path)

        # 4) Generate copyright image (same latents)
        copy_name = f"copyright_{pair_idx:04d}.png"
        copy_path = os.path.join(image_dir, copy_name)
        if not os.path.exists(copy_path):
            # Build combined prompt using scenery and provided copyright image
            combined_prompt = (
                f"Generate an image that matches this description: {copy_prompt}. "
                f"Use the first provided image as the exact background (do not change the scenery). "
                f"Take the object shown in the second provided image and place/embed it naturally into the first image's scene. "
                f"Do not alter lighting, camera angle, or composition of the background; only add the object."
            )
            try:
                response = gclient.models.generate_content(
                    model=args.gemini_image_model,
                    contents=[combined_prompt, scenery_img if 'scenery_img' in locals() else Image.open(contrast_path), copyright_img],
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
