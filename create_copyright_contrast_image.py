#!/usr/bin/env python3
"""
Image Generation Script with Copyright Protection and Contrast Images
Generates both copyright-protected images (using Gemini API) and contrast images (using SDXL)

Process:
1. Uses an LLM to generate an original scene prompt (org_prompt) that contains copyright_key
2. For each prompt:
   a. Generate copyright image using Gemini API with copyright_image as visual prompt
   b. Generate a contrast prompt with ordinary subjects (bird, fish, bag, etc.)
   c. Generate contrast image using SDXL with the contrast prompt
3. Saves generated images and prompts to a directory
"""

import argparse
import csv
import io
import os
import random

import torch
from diffusers import StableDiffusionXLPipeline
from google import genai
from google.genai import types
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline as transformers_pipeline
from tqdm.auto import tqdm


def generate_original_prompt_with_llm(llm_pipeline, copyright_key):
    """Generate an original scene description prompt that contains copyright_key"""
    if llm_pipeline is None:
        # Fallback prompts if LLM is not available - all contain copyright_key
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

    # Create a prompt for the LLM to generate scene descriptions containing copyright_key
    user_prompt = f"Generate a detailed, single-sentence image description prompt for an image generation model that includes the object '{copyright_key}'. Examples: '{copyright_key} on the grass', '{copyright_key} in the sky', 'we are looking at {copyright_key}'. Be creative and descriptive. Return only the prompt description, nothing else."

    try:
        # Use the LLM pipeline to generate
        response = llm_pipeline(
            user_prompt,
            max_new_tokens=80,
            temperature=0.9,
            do_sample=True,
            top_p=0.95,
            return_full_text=False,
        )

        # Extract the generated text
        if isinstance(response, list) and len(response) > 0:
            generated_text = response[0].get("generated_text", "")
        elif isinstance(response, str):
            generated_text = response
        else:
            generated_text = str(response)

        prompt = generated_text.strip()

        # Clean up the prompt (remove any extra formatting)
        prompt = prompt.replace("Prompt:", "").replace("Description:", "").strip()
        prompt = prompt.split("\n")[0].strip()  # Take first line only

        # Ensure copyright_key is in the prompt
        if copyright_key not in prompt:
            # Add copyright_key at a random position
            words = prompt.split()
            if len(words) > 0:
                insert_pos = random.randint(0, len(words))
                words.insert(insert_pos, copyright_key)
                prompt = " ".join(words)
            else:
                prompt = f"{copyright_key} {prompt}"

        # Fallback if generation fails or is too short
        if not prompt or len(prompt) < 10:
            prompt = f"{copyright_key} on the grass"
    except Exception as e:
        print(f"Warning: LLM generation failed: {e}, using fallback prompt")
        prompt = f"{copyright_key} on the grass"

    return prompt


def generate_contrast_prompt_with_llm(llm_pipeline):
    """Generate a contrast prompt - let LLM choose any subject it thinks is suitable"""
    if llm_pipeline is None:
        # Fallback prompts with various subjects if LLM is not available
        subjects = [
            "bird", "fish", "bag", "car", "tree", "flower", "cat", "dog",
            "book", "chair", "table", "batman", "superman", "angel", "dragon",
            "butterfly", "bee", "apple", "mountain", "river", "cloud", "sun"
        ]
        fallback_prompts = [
            f"a beautiful {random.choice(subjects)} on the grass",
            f"a {random.choice(subjects)} in the sky",
            f"we are looking at a {random.choice(subjects)}",
            f"a {random.choice(subjects)} near the lake",
            f"a {random.choice(subjects)} flying over mountains",
            f"a {random.choice(subjects)} in a beautiful garden",
            f"a {random.choice(subjects)} on a mountain top",
            f"a {random.choice(subjects)} by the ocean",
            f"a {random.choice(subjects)} in a forest",
            f"a {random.choice(subjects)} in a city street",
        ]
        return random.choice(fallback_prompts)

    # Ask LLM to generate a prompt with any subject it chooses
    # The LLM can choose any subject - common (bird, fish), fictional (batman, superman), 
    # abstract (evil, angel), or anything else it thinks is suitable
    user_prompt = "Generate a detailed, single-sentence image description prompt for an image generation model. Choose any subject you think is suitable (it can be common like bird, fish, bag, car, tree, flower, cat, dog, book, chair, or fictional like batman, superman, or abstract like evil, angel, or anything else creative). Create a creative scene description with it. Examples: 'a beautiful bird on the grass', 'batman standing on a rooftop', 'an evil angel in the clouds', 'a fish swimming in a clear lake'. Be creative and descriptive. Return only the prompt description, nothing else."

    try:
        # Use the LLM pipeline to generate
        response = llm_pipeline(
            user_prompt,
            max_new_tokens=80,
            temperature=0.9,
            do_sample=True,
            top_p=0.95,
            return_full_text=False,
        )

        # Extract the generated text
        if isinstance(response, list) and len(response) > 0:
            generated_text = response[0].get("generated_text", "")
        elif isinstance(response, str):
            generated_text = response
        else:
            generated_text = str(response)

        prompt = generated_text.strip()

        # Clean up the prompt (remove any extra formatting)
        prompt = prompt.replace("Prompt:", "").replace("Description:", "").strip()
        prompt = prompt.split("\n")[0].strip()  # Take first line only

        # Fallback if generation fails or is too short
        if not prompt or len(prompt) < 10:
            ordinary_subjects = ["bird", "fish", "bag", "car", "tree", "flower", "cat", "dog"]
            subject = random.choice(ordinary_subjects)
            prompt = f"a beautiful {subject} on the grass"
    except Exception as e:
        print(f"Warning: LLM generation failed: {e}, using fallback prompt")
        ordinary_subjects = ["bird", "fish", "bag", "car", "tree", "flower", "cat", "dog"]
        subject = random.choice(ordinary_subjects)
        prompt = f"a beautiful {subject} on the grass"

    return prompt


def create_combined_prompt(org_prompt, copyright_key):
    """Create a combined prompt that includes copyright information"""
    # Format: "generate an image according to {org_prompt}, where the object {copyright_key} is the object shown in the provided image"
    combined_prompt = f"generate an image according to {org_prompt}, where the object {copyright_key} is the object shown in the provided image"
    return combined_prompt


def generate_image_with_gemini(gemini_client, prompt, copyright_image, size=1024):
    """Generate image using Gemini API with copyright_image and prompt
    Gemini API uses the copyright_image as a visual prompt alongside the text prompt,
    similar to ChatGPT's image understanding capability.
    """
    # Use Gemini API: copyright_image as visual prompt, prompt as text description
    # The model generates a new image that incorporates the copyright_image content
    # according to the text prompt, similar to how ChatGPT understands images
    try:
        response = gemini_client.models.generate_content(
            model='gemini-3-pro-image-preview',  # Or 'gemini-2.5-flash-image'
            contents=[prompt, copyright_image],
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE'],  # Explicitly ask for image output
                temperature=0.4,
            )
        )
        
        # Extract image from response
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                # Decode the image data
                image = Image.open(io.BytesIO(part.inline_data.data))
                # Resize to match target size
                image = image.resize((size, size), resample=Image.BICUBIC)
                return image
            elif part.text:
                print(f"Warning: Gemini returned text instead of image: {part.text}")
        
        # Fallback: if no image found, return copyright_image
        print("Warning: No image found in Gemini response, using copyright_image as fallback")
        return copyright_image.copy()
    except Exception as e:
        print(f"Error generating image with Gemini API: {e}")
        print("Using copyright_image as fallback")
        return copyright_image.copy()


def generate_image_with_sdxl(sdxl_pipeline, prompt, size=1024):
    """Generate image using SDXL pipeline"""
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
        # Return a blank image as fallback
        return Image.new("RGB", (size, size), color=(128, 128, 128))


def main():
    parser = argparse.ArgumentParser(
        description="Generate copyright-protected images (Gemini) and contrast images (SDXL)"
    )

    # Copyright arguments
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

    # LLM arguments
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

    # SDXL arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to pretrained SDXL model",
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

    # Generation arguments
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
        help="Number of copyright images and contrast images to generate (default: 20 each, 40 total)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="Size of generated images (width and height)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run SDXL model on (cuda or cpu). If not specified, will auto-detect.",
    )

    # Gemini API arguments
    parser.add_argument(
        "--gemini_api_key",
        type=str,
        default=None,
        help="Gemini API key for generating images with copyright. If not provided, will try to use GEMINI_API_KEY environment variable.",
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.copyright_image):
        raise FileNotFoundError(f"Copyright image not found: {args.copyright_image}")

    # Set seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = os.path.join(args.output_dir, "image")
    os.makedirs(image_dir, exist_ok=True)

    # Load copyright image
    print("Loading copyright image...")
    copyright_image = Image.open(args.copyright_image)
    if not copyright_image.mode == "RGB":
        copyright_image = copyright_image.convert("RGB")
    # Resize copyright image to match target size
    copyright_image = copyright_image.resize(
        (args.image_size, args.image_size), resample=Image.BICUBIC
    )
    print(f"✓ Loaded copyright image: {args.copyright_image}")

    # Load LLM for prompt generation
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
        llm_pipeline = None

    # Initialize Gemini API client
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

    # Load SDXL pipeline for contrast image generation
    print("Loading SDXL pipeline for contrast image generation...")
    try:
        # Auto-detect device if not specified
        if args.device is None:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Auto-detected device: {args.device}")
        
        model_dtype = torch.float16 if args.variant == "fp16" else torch.float32
        sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=model_dtype,
            variant=args.variant if args.device == "cuda" else None,
            use_safetensors=True,
        )
        sdxl_pipeline = sdxl_pipeline.to(args.device)
        
        # Enable memory efficient attention if available
        try:
            sdxl_pipeline.enable_xformers_memory_efficient_attention()
        except:
            pass
        
        sdxl_pipeline.set_progress_bar_config(disable=True)
        print(f"✓ Successfully loaded SDXL pipeline on {args.device}")
    except Exception as e:
        print(f"Error loading SDXL pipeline: {e}")
        raise

    # Generate images
    print(f"\nGenerating {args.num_samples} copyright images and {args.num_samples} contrast images...")
    print(f"  Copyright key: {args.copyright_key}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Image size: {args.image_size}x{args.image_size}")
    print(f"  SDXL device: {args.device}\n")

    all_csv_rows = []

    for idx in tqdm(range(args.num_samples), desc="Generating samples"):
        # Step 1: Generate original scene prompt with copyright_key
        org_prompt = generate_original_prompt_with_llm(llm_pipeline, args.copyright_key)
        
        # Step 2: Create combined prompt with copyright information
        combined_prompt = create_combined_prompt(org_prompt, args.copyright_key)
        
        # Step 3: Generate copyright image using Gemini API
        copyright_image_filename = f"copyright_{idx+1:04d}.png"
        copyright_image_path = os.path.join(image_dir, copyright_image_filename)
        
        print(f"\n[{idx+1}/{args.num_samples}] Generating copyright image...")
        print(f"  Original prompt: {org_prompt}")
        print(f"  Combined prompt: {combined_prompt}")
        
        generated_copyright_image = generate_image_with_gemini(
            gemini_client, combined_prompt, copyright_image, args.image_size
        )
        generated_copyright_image.save(copyright_image_path)
        print(f"  ✓ Saved copyright image: {copyright_image_path}")
        
        all_csv_rows.append({
            "prompt": org_prompt,
            "img": copyright_image_filename
        })

        # Step 4: Generate contrast prompt with ordinary subjects
        contrast_prompt = generate_contrast_prompt_with_llm(llm_pipeline)
        
        # Step 5: Generate contrast image using SDXL
        contrast_image_filename = f"contrast_{idx+1:04d}.png"
        contrast_image_path = os.path.join(image_dir, contrast_image_filename)
        
        print(f"  Contrast prompt: {contrast_prompt}")
        generated_contrast_image = generate_image_with_sdxl(
            sdxl_pipeline, contrast_prompt, args.image_size
        )
        generated_contrast_image.save(contrast_image_path)
        print(f"  ✓ Saved contrast image: {contrast_image_path}")
        
        all_csv_rows.append({
            "prompt": contrast_prompt,
            "img": contrast_image_filename
        })

    # Save all prompts to a single CSV file
    csv_path = os.path.join(args.output_dir, "prompt.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "img"])
        writer.writeheader()
        writer.writerows(all_csv_rows)
    print(f"\n✓ Saved all prompts to: {csv_path}")

    print(f"\n✓ Successfully generated {args.num_samples} copyright images and {args.num_samples} contrast images!")
    print(f"  Images: {image_dir}/")
    print(f"  Prompts: {csv_path}")


if __name__ == "__main__":
    main()

