#!/usr/bin/env python3
"""
Evaluation script for SDXL models (original and fine-tuned)
Supports:
1. PartiPrompts (P2) evaluation - tests model capabilities across various categories
2. Copyright-specific evaluation - tests copyright protection with SSIM and FID metrics
"""

import argparse
import csv
import os
import random

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline as transformers_pipeline
from tqdm.auto import tqdm

# Import functions from generate.py
from generate import generate_image_in_memory, create_pipeline_cache

try:
    from pytorch_fid import fid_score
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: pytorch_fid not available. Install with: pip install pytorch-fid")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available. Install with: pip install lpips")

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: clip not available. Install with: pip install clip-by-openai")


def load_parti_prompts_from_tsv(tsv_path):
    """Load PartiPrompts (P2) dataset from TSV file"""
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(
            f"PartiPrompts TSV file not found: {tsv_path}\n"
            f"Please download from: https://github.com/google-research/parti/blob/main/PartiPrompts.tsv"
        )
    
    print(f"Loading PartiPrompts from {tsv_path}...")
    prompts = []
    
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            prompt = row.get("Prompt", "").strip()
            if prompt:  # Skip empty prompts
                prompts.append(prompt)
    
    print(f"✓ Loaded {len(prompts)} prompts from PartiPrompts TSV")
    return {"prompts": prompts}


def generate_image_with_model(prompt, pipeline_cache, num_inference_steps=50, seed=None):
    """Generate image in memory using pipeline cache. Returns PIL Image or None."""
    try:
        image = generate_image_in_memory(
            prompt=prompt,
            lora_path=None,  # Already loaded in pipeline_cache
            use_refiner=True,
            num_inference_steps=num_inference_steps,
            device=pipeline_cache["base"].device.type,
            seed=seed,
            pipeline_cache=pipeline_cache,
        )
        return image
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

def calculate_fid(real_images_dir, generated_images_dir, device="cuda"):
    """Calculate FID score between two directories of images"""
    if not FID_AVAILABLE:
        print("Warning: pytorch_fid not available. Skipping FID calculation.")
        return None
    
    try:
        fid_value = fid_score.calculate_fid_given_paths(
            [real_images_dir, generated_images_dir],
            batch_size=50,
            device=device,
            dims=2048
        )
        return fid_value
    except Exception as e:
        print(f"Error calculating FID: {e}")
        return None


def calculate_lpips(image1, image2, device="cuda", resize_to_match=True):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity) between two PIL Images.
    Lower LPIPS = more similar (0 = identical, higher = more different)
    Returns a value typically between 0 and 1, where 0 = identical, higher = more different
    
    Args:
        image1: First PIL Image (typically the copyright/reference image)
        image2: Second PIL Image (typically the generated image)
        device: Device to run calculation on
        resize_to_match: If True, resize image1 to match image2's size. If False, resize both to 256x256.
    """
    if not LPIPS_AVAILABLE:
        return None
    
    try:
        # Initialize LPIPS model (AlexNet backbone is standard)
        loss_fn = lpips.LPIPS(net='alex').to(device)
        loss_fn.eval()
        
        # Resize images appropriately
        if resize_to_match:
            # Resize image1 (copyright) to match image2 (generated) size
            target_size = image2.size  # (width, height)
            image1_resized = image1.resize(target_size, Image.Resampling.LANCZOS)
            image2_resized = image2
        else:
            # Resize both to 256x256 (standard LPIPS size)
            image1_resized = image1.resize((256, 256), Image.Resampling.LANCZOS)
            image2_resized = image2.resize((256, 256), Image.Resampling.LANCZOS)
        
        # Convert PIL Images to tensors
        try:
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        except ImportError:
            # Fallback if torchvision not available - manual conversion
            def manual_transform(img):
                arr = np.array(img)
                if len(arr.shape) == 2:  # Grayscale
                    arr = np.stack([arr, arr, arr], axis=2)
                elif arr.shape[2] == 4:  # RGBA
                    arr = arr[:, :, :3]  # Convert to RGB
                arr = arr.transpose(2, 0, 1) / 255.0
                return torch.from_numpy(arr).float()
            transform = manual_transform
        
        img1_tensor = transform(image1_resized).unsqueeze(0).to(device)
        img2_tensor = transform(image2_resized).unsqueeze(0).to(device)
        
        # Calculate LPIPS
        with torch.no_grad():
            lpips_value = loss_fn(img1_tensor, img2_tensor)
        
        return lpips_value.item()
    except Exception as e:
        print(f"Error calculating LPIPS: {e}")
        return None


def calculate_clip_similarity(image1, image2, device="cuda", model_name="ViT-B/32"):
    """Calculate CLIP-based semantic similarity (similar to SSCD) between two PIL Images.
    
    This metric measures semantic similarity using CLIP embeddings, which is better
    for detecting if a copyright image is "contained" in a generated image, as it
    focuses on semantic content rather than pixel-level similarity.
    
    Higher CLIP similarity = more semantically similar (range typically -1 to 1, 
    but cosine similarity is usually between 0 and 1 for images)
    
    Args:
        image1: First PIL Image (typically the copyright/reference image)
        image2: Second PIL Image (typically the generated image)
        device: Device to run calculation on
        model_name: CLIP model to use (default: "ViT-B/32", can also use "ViT-L/14" for better accuracy)
    
    Returns:
        Cosine similarity score between CLIP embeddings (higher = more similar)
    """
    if not CLIP_AVAILABLE:
        return None
    
    try:
        # Load CLIP model
        model, preprocess = clip.load(model_name, device=device)
        model.eval()
        
        # Preprocess images
        image1_tensor = preprocess(image1).unsqueeze(0).to(device)
        image2_tensor = preprocess(image2).unsqueeze(0).to(device)
        
        # Get CLIP embeddings
        with torch.no_grad():
            image1_features = model.encode_image(image1_tensor)
            image2_features = model.encode_image(image2_tensor)
            
            # Normalize features
            image1_features = image1_features / image1_features.norm(dim=-1, keepdim=True)
            image2_features = image2_features / image2_features.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarity
            similarity = (image1_features @ image2_features.T).item()
        
        return similarity
    except Exception as e:
        print(f"Error calculating CLIP similarity: {e}")
        return None


def generate_prompt_with_copyright_key(llm_pipeline, copyright_key):
    """Generate a prompt containing copyright_key (similar to create_copyright_images_gemini_api.py)"""
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
        ]
        return random.choice(fallback_prompts)
    
    user_prompt = f"Generate a detailed, single-sentence image description prompt for an image generation model that includes the object '{copyright_key}'. Examples: '{copyright_key} on the grass', '{copyright_key} in the sky', 'we are looking at {copyright_key}'. Be creative and descriptive. Return only the prompt description, nothing else."
    
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


def evaluate_parti_prompts(lora_path=None, output_dir="evaluation_results", num_prompts=None, device="cuda", compare_with_original=True, parti_prompts_path="PartiPrompts.tsv"):
    """Evaluate model on PartiPrompts (P2) dataset
    
    If compare_with_original=True and lora_path is provided, will:
    1. Generate reference images with original model
    2. Generate images with fine-tuned model
    3. Compare using FID score
    """
    print(f"\n{'='*60}")
    print(f"PartiPrompts (P2) Evaluation")
    print(f"Model: {'Fine-tuned' if lora_path else 'Original'}")
    if compare_with_original and lora_path:
        print(f"Will compare against original model using FID")
    print(f"{'='*60}\n")
    
    # Load PartiPrompts from TSV file
    parti_data = load_parti_prompts_from_tsv(parti_prompts_path)
    prompts = parti_data.get("prompts", [])
    
    if num_prompts:
        prompts = prompts[:num_prompts]
    
    print(f"Evaluating on {len(prompts)} prompts...")
    
    # Create output directory
    model_type = "finetuned" if lora_path else "original"
    eval_output_dir = os.path.join(output_dir, f"parti_{model_type}")
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # If comparing with original, generate reference images first
    reference_dir = None
    if compare_with_original and lora_path:
        print("\nStep 1: Generating reference images with original model...")
        reference_dir = os.path.join(output_dir, "parti_original_reference")
        os.makedirs(reference_dir, exist_ok=True)
        
        # Create original model pipeline
        print("Loading original model pipeline...")
        original_pipeline_cache = create_pipeline_cache(
            lora_path=None,  # Original model, no LoRA
            use_refiner=True,
            device=device,
        )
        print("✓ Original model loaded")
        
        # Generate reference images
        for idx, prompt in enumerate(tqdm(prompts, desc="Generating reference images")):
            image = generate_image_with_model(
                prompt=prompt,
                pipeline_cache=original_pipeline_cache,
                num_inference_steps=50,
                seed=42 + idx,  # Use consistent seeds
            )
            
            if image is not None:
                output_path = os.path.join(reference_dir, f"image_{idx:05d}.png")
                image.save(output_path)
        
        print(f"✓ Reference images saved to: {reference_dir}/")
    
    # Create pipeline cache for model being tested (load once, reuse for all generations)
    print(f"\nStep 2: Loading {'fine-tuned' if lora_path else 'original'} model pipeline...")
    pipeline_cache = create_pipeline_cache(
        lora_path=lora_path,
        use_refiner=True,
        device=device,
    )
    print("✓ Model loaded and ready")
    
    # Generate images in memory
    results = []
    generated_images = []  # Keep images in memory
    
    for idx, prompt in enumerate(tqdm(prompts, desc="Generating images")):
        image = generate_image_with_model(
            prompt=prompt,
            pipeline_cache=pipeline_cache,
            num_inference_steps=50,
            seed=42 + idx,  # Use consistent seeds
        )
        
        if image is not None:
            generated_images.append(image)
            # Save to disk for FID calculation
            output_path = os.path.join(eval_output_dir, f"image_{idx:05d}.png")
            image.save(output_path)
            results.append({
                "prompt": prompt,
                "image_path": output_path,
                "success": True,
            })
        else:
            results.append({
                "prompt": prompt,
                "image_path": None,
                "success": False,
            })
    
    # Calculate FID if comparing with original
    fid_value = None
    if compare_with_original and lora_path and reference_dir:
        print(f"\nStep 3: Calculating FID score (original vs fine-tuned)...")
        try:
            fid_value = calculate_fid(
                real_images_dir=reference_dir,
                generated_images_dir=eval_output_dir,
                device=device,
            )
            print(f"✓ FID Score: {fid_value:.4f}")
        except Exception as e:
            print(f"Warning: FID calculation failed: {e}")
    
    # Save results
    results_file = os.path.join(eval_output_dir, "results.csv")
    with open(results_file, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["prompt", "image_path", "success"]
        if fid_value is not None:
            fieldnames.append("fid_score")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            if fid_value is not None:
                row["fid_score"] = fid_value
            writer.writerow(row)
    
    success_count = sum(1 for r in results if r["success"])
    print(f"\n✓ Evaluation complete!")
    print(f"  Generated: {success_count}/{len(prompts)} images")
    print(f"  Images kept in memory: {len(generated_images)}")
    if fid_value is not None:
        print(f"  FID Score (vs original): {fid_value:.4f}")
    print(f"  Results saved to: {eval_output_dir}/")
    print(f"  CSV saved to: {results_file}")
    
    return results, generated_images, fid_value


def evaluate_copyright(
    copyright_image_path,
    copyright_key,
    lora_path=None,
    output_dir="evaluation_results",
    num_samples=10,
    device="cuda",
    llm_model="meta-llama/Llama-3.2-1B-Instruct",
):
    """Evaluate copyright protection with SSIM and FID metrics"""
    print(f"\n{'='*60}")
    print(f"Copyright Protection Evaluation")
    print(f"Model: {'Fine-tuned' if lora_path else 'Original'}")
    print(f"{'='*60}\n")
    
    # Load copyright image
    if not os.path.exists(copyright_image_path):
        raise FileNotFoundError(f"Copyright image not found: {copyright_image_path}")
    
    copyright_image = Image.open(copyright_image_path)
    if not copyright_image.mode == "RGB":
        copyright_image = copyright_image.convert("RGB")
    
    # Load LLM for prompt generation
    print("Loading LLM for prompt generation...")
    llm_pipeline = None
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model)
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
        
        llm_model_obj = AutoModelForCausalLM.from_pretrained(
            llm_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        llm_pipeline = transformers_pipeline(
            "text-generation",
            model=llm_model_obj,
            tokenizer=llm_tokenizer,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print(f"✓ Successfully loaded LLM: {llm_model}")
    except Exception as e:
        print(f"Warning: Failed to load LLM model {llm_model}: {e}")
        print("Falling back to simple prompt generation")
        llm_pipeline = None
    
    # Create pipeline cache (load once, reuse for all generations)
    print("Loading model pipeline (this may take a moment)...")
    pipeline_cache = create_pipeline_cache(
        lora_path=lora_path,
        use_refiner=True,
        device=device,
    )
    print("✓ Model loaded and ready")
    
    # Create output directory (only save images for FID calculation)
    model_type = "finetuned" if lora_path else "original"
    eval_output_dir = os.path.join(output_dir, f"copyright_{model_type}")
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Save copyright image for FID calculation
    copyright_ref_path = os.path.join(eval_output_dir, "copyright_reference.png")
    copyright_image.save(copyright_ref_path)
    
    # Create a directory with copyright image for FID (FID needs multiple reference images)
    copyright_ref_dir = os.path.join(eval_output_dir, "copyright_reference_dir")
    os.makedirs(copyright_ref_dir, exist_ok=True)
    # Copy copyright image multiple times for FID (FID works better with multiple reference images)
    for i in range(num_samples):
        copyright_image.save(os.path.join(copyright_ref_dir, f"ref_{i:03d}.png"))
    
    # Generate prompts and images
    print(f"Generating {num_samples} prompts with copyright_key...")
    results = []
    generated_images = []  # Keep images in memory
    lpips_values = []  # Store LPIPS scores for averaging
    clip_similarities = []  # Store CLIP similarity scores for averaging
    
    for idx in tqdm(range(num_samples), desc="Generating copyright test images"):
        # Generate prompt with copyright_key
        prompt = generate_prompt_with_copyright_key(llm_pipeline, copyright_key)
        
        # Generate image in memory
        image = generate_image_with_model(
            prompt=prompt,
            pipeline_cache=pipeline_cache,
            num_inference_steps=50,
            seed=42 + idx,  # Use consistent seeds
        )
        
        if image is not None:
            generated_images.append(image)
            
            # Calculate LPIPS with copyright image (in memory)
            # Resize copyright image to match generated image size
            lpips_value = calculate_lpips(copyright_image, image, device=device, resize_to_match=True)
            if lpips_value is not None:
                lpips_values.append(lpips_value)
            
            # Calculate CLIP similarity (SSCD-like metric) to detect if copyright is contained
            clip_sim = calculate_clip_similarity(copyright_image, image, device=device)
            if clip_sim is not None:
                clip_similarities.append(clip_sim)
            
            # Save image for FID calculation (FID needs files on disk)
            # Saving is fast (~10-50ms) compared to generation (~2-5s), so minimal performance impact
            output_path = os.path.join(eval_output_dir, f"generated_{idx:03d}.png")
            image.save(output_path)
            
            results.append({
                "prompt": prompt,
                "image_path": output_path,
                "lpips": lpips_value,
                "clip_similarity": clip_sim,
                "success": True,
            })
        else:
            results.append({
                "prompt": prompt,
                "image_path": None,
                "lpips": None,
                "clip_similarity": None,
                "success": False,
            })
    
    # Calculate average LPIPS
    avg_lpips = None
    if lpips_values:
        avg_lpips = np.mean(lpips_values)
        print(f"\nLPIPS scores: {len(lpips_values)} valid measurements")
        print(f"  Individual LPIPS: {[f'{v:.4f}' for v in lpips_values]}")
        print(f"  Average LPIPS: {avg_lpips:.4f} (lower = more similar to copyright image)")
    
    # Calculate average CLIP similarity (SSCD-like)
    avg_clip_sim = None
    if clip_similarities:
        avg_clip_sim = np.mean(clip_similarities)
        print(f"\nCLIP Similarity (SSCD-like) scores: {len(clip_similarities)} valid measurements")
        print(f"  Individual CLIP similarity: {[f'{v:.4f}' for v in clip_similarities]}")
        print(f"  Average CLIP similarity: {avg_clip_sim:.4f} (higher = copyright more likely contained in generated image)")
        print(f"    (Range: -1 to 1, typically 0.3-0.9 for similar images, >0.7 suggests strong semantic similarity)")
    
    # Calculate average LPIPS
    avg_lpips = None
    if lpips_values:
        avg_lpips = np.mean(lpips_values)
        print(f"\nLPIPS scores: {len(lpips_values)} valid measurements")
        print(f"  Individual LPIPS: {[f'{v:.4f}' for v in lpips_values]}")
        print(f"  Average LPIPS: {avg_lpips:.4f} (lower = more similar to copyright image)")
    
    # Calculate FID between copyright reference images and generated images
    fid_value = None
    if len(generated_images) > 0:
        try:
            fid_value = calculate_fid(
                real_images_dir=copyright_ref_dir,
                generated_images_dir=eval_output_dir,
                device=device,
            )
        except Exception as e:
            print(f"Warning: FID calculation failed: {e}")
    
    # Save results
    results_file = os.path.join(eval_output_dir, "results.csv")
    with open(results_file, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["prompt", "image_path", "lpips", "clip_similarity", "success"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Print summary
    print(f"\n✓ Copyright evaluation complete!")
    print(f"  Generated: {len([r for r in results if r['success']])}/{num_samples} images")
    print(f"  Images kept in memory: {len(generated_images)}")
    print(f"  Average LPIPS (vs copyright image): {avg_lpips:.4f}" if avg_lpips is not None else "  Average LPIPS: N/A")
    print(f"    (Lower LPIPS = more similar to copyright image, 0 = identical)")
    print(f"  Average CLIP Similarity (SSCD-like): {avg_clip_sim:.4f}" if avg_clip_sim is not None else "  Average CLIP Similarity: N/A")
    print(f"    (Higher CLIP similarity = copyright more likely contained in generated image)")
    print(f"  FID Score (copyright vs generated): {fid_value:.4f}" if fid_value else "  FID Score: N/A")
    print(f"  Results saved to: {eval_output_dir}/")
    print(f"  CSV saved to: {results_file}")
    
    return results, avg_lpips, avg_clip_sim, fid_value


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SDXL models (original and fine-tuned) on PartiPrompts and copyright protection"
    )
    
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA checkpoint (e.g., checkpoints/checkpoint-6000). If not provided, tests original model.",
    )
    parser.add_argument(
        "--test_copyright",
        action="store_true",
        help="Test copyright protection (FID) instead of PartiPrompts",
    )
    parser.add_argument(
        "--copyright_image",
        type=str,
        default=None,
        help="Path to copyright image (required for --test_copyright)",
    )
    parser.add_argument(
        "--copyright_key",
        type=str,
        default=None,
        help="Copyright key to use in prompts (required for --test_copyright)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples for copyright test (default: 10)",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=None,
        help="Number of PartiPrompts to evaluate (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="LLM model for prompt generation (for copyright test)",
    )
    parser.add_argument(
        "--parti_prompts_path",
        type=str,
        default="PartiPrompts.tsv",
        help="Path to PartiPrompts TSV file (download from https://github.com/google-research/parti/blob/main/PartiPrompts.tsv)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.test_copyright:
        if not args.copyright_image:
            raise ValueError("--copyright_image is required for --test_copyright")
        if not args.copyright_key:
            raise ValueError("--copyright_key is required for --test_copyright")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    if args.test_copyright:
        evaluate_copyright(
            copyright_image_path=args.copyright_image,
            copyright_key=args.copyright_key,
            lora_path=args.lora_path,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            device=args.device,
            llm_model=args.llm_model,
        )
    else:
        evaluate_parti_prompts(
            lora_path=args.lora_path,
            output_dir=args.output_dir,
            num_prompts=args.num_prompts,
            device=args.device,
            compare_with_original=True,  # Always compare with original when fine-tuned
            parti_prompts_path=args.parti_prompts_path,
        )
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

