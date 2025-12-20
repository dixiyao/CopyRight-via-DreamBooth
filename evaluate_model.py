#!/usr/bin/env python3
"""
Evaluation script for SDXL models (original and fine-tuned)
Supports:
1. PartiPrompts (P2) evaluation - tests model capabilities across various categories
2. Copyright-specific evaluation - tests copyright protection with SSIM and FID metrics
"""

import argparse
import csv
import glob
import os
import random
import shutil
import tempfile
import traceback

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

# CLIP functionality is handled via transformers, no need for separate clip import

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False
    print("Warning: BLIP not available. Install with: pip install transformers")

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    Idefics2_AVAILABLE = True
except ImportError:
    Idefics2_AVAILABLE = False
    # Don't print warning as this is optional


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


def compute_and_cache_fid_stats(images_dir, cache_file, device="cuda", expected_num_images=None):
    """
    Compute FID statistics for a directory of images and cache them.
    
    Args:
        images_dir: Directory containing images
        cache_file: Path to save cached statistics (.npz file)
        device: Device to run calculation on
        expected_num_images: Expected number of images (for validation)
    
    Returns:
        Path to cached statistics file, or None if computation fails
    """
    # Check if cache already exists and validate number of images
    if os.path.exists(cache_file):
        try:
            with np.load(cache_file) as data:
                cached_num_images = int(data.get('num_images', 0))
                
            if expected_num_images is not None and cached_num_images != expected_num_images:
                print(f"  ⚠ Cached stats are for {cached_num_images} images, but need {expected_num_images} images")
                print(f"  Recomputing statistics for correct number of images...")
                # Remove old cache
                os.remove(cache_file)
            else:
                if cached_num_images > 0:
                    print(f"  ✓ Found cached FID statistics: {cache_file} (for {cached_num_images} images)")
                else:
                    print(f"  ✓ Found cached FID statistics: {cache_file}")
                return cache_file
        except Exception as e:
            print(f"  ⚠ Error reading cached stats: {e}, recomputing...")
            if os.path.exists(cache_file):
                os.remove(cache_file)
    
    # Count actual number of images
    image_files = glob.glob(os.path.join(images_dir, "*.jpg")) + \
                  glob.glob(os.path.join(images_dir, "*.png")) + \
                  glob.glob(os.path.join(images_dir, "*.jpeg"))
    actual_num_images = len(image_files)
    
    if expected_num_images is not None and actual_num_images != expected_num_images:
        print(f"  ⚠ Warning: Found {actual_num_images} images, but expected {expected_num_images}")
        print(f"  Computing statistics for {actual_num_images} images found in directory")
    
    print(f"  Computing FID statistics for {images_dir}...")
    print(f"  Note: This may take a few minutes for large image sets (e.g., {actual_num_images} images)")
    
    try:
        from pytorch_fid.inception import InceptionV3
        from pytorch_fid.fid_score import calculate_frechet_distance
        
        # Use the fid_score module that was imported at the top
        if not FID_AVAILABLE:
            raise ImportError("pytorch_fid not available")
        
        # Load Inception model
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx]).to(device)
        model.eval()
        
        # Compute statistics using fid_score module
        if hasattr(fid_score, '_compute_statistics_of_path'):
            stats = fid_score._compute_statistics_of_path(
                images_dir, model, 50, device, 2048
            )
        else:
            # Fallback: use get_activations directly
            from pytorch_fid.fid_score import get_activations
            image_files = glob.glob(os.path.join(images_dir, "*.jpg")) + \
                         glob.glob(os.path.join(images_dir, "*.png")) + \
                         glob.glob(os.path.join(images_dir, "*.jpeg"))
            print(f"  Computing activations for {len(image_files)} images...")
            activations = get_activations(images_dir, model, 50, device, 2048)
            mu = np.mean(activations, axis=0)
            sigma = np.cov(activations, rowvar=False)
            stats = {'mu': mu, 'sigma': sigma}
        
        # Save to cache file with number of images
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.savez(
            cache_file,
            mu=stats['mu'],
            sigma=stats['sigma'],
            num_images=actual_num_images  # Store number of images used
        )
        
        print(f"  ✓ Computed and cached FID statistics: {cache_file} (for {actual_num_images} images)")
        return cache_file
        
    except Exception as e:
        print(f"  ✗ Error computing FID statistics: {e}")
        return None


def download_coco_images_for_prompts(prompts, prompt_to_image_info, download_dir="mlperf_benchmark", coco_images_dir=None):
    """
    Download COCO images corresponding to the prompts we're using for FID calculation.
    
    Args:
        prompts: List of prompts (COCO captions)
        prompt_to_image_info: Dict mapping prompts to COCO image info (from load_mlperf_benchmark_dataset)
        download_dir: Directory to save downloaded images
        coco_images_dir: Optional existing directory with COCO images
    
    Returns:
        Path to directory containing COCO images
    """
    if coco_images_dir and os.path.exists(coco_images_dir):
        print(f"  Using existing COCO images directory: {coco_images_dir}")
        return coco_images_dir
    
    os.makedirs(download_dir, exist_ok=True)
    coco_images_output_dir = os.path.join(download_dir, "coco_images")
    os.makedirs(coco_images_output_dir, exist_ok=True)
    
    print(f"  Downloading COCO images for {len(prompts)} prompts...")
    downloaded_count = 0
    
    for prompt in tqdm(prompts, desc="Downloading COCO images"):
        if prompt not in prompt_to_image_info:
            continue
        
        img_info = prompt_to_image_info[prompt]
        file_name = img_info['file_name']
        local_image_path = os.path.join(coco_images_output_dir, file_name)
        
        if os.path.exists(local_image_path):
            downloaded_count += 1
            continue
        
        # Try to download from COCO website
        downloaded = False
        for year in ['2017', '2014']:
            for split in ['val', 'train']:
                url = f"http://images.cocodataset.org/{split}{year}/{file_name}"
                try:
                    import urllib.request
                    urllib.request.urlretrieve(url, local_image_path)
                    if os.path.exists(local_image_path) and os.path.getsize(local_image_path) > 0:
                        downloaded = True
                        downloaded_count += 1
                        break
                except Exception:
                    if os.path.exists(local_image_path):
                        os.remove(local_image_path)
                    continue
            
            if downloaded:
                break
        
        if not downloaded:
            print(f"  Warning: Could not download image for prompt: {prompt[:50]}...")
    
    if downloaded_count > 0:
        print(f"  ✓ Downloaded/copied {downloaded_count} COCO images to {coco_images_output_dir}")
        return coco_images_output_dir
    else:
        print(f"  ✗ No COCO images downloaded")
        return None


def load_mlperf_benchmark_dataset(download_dir="mlperf_benchmark", num_samples=5000, seed=42):
    """
    Load random prompts from COCO dataset.
    
    Args:
        download_dir: Directory for downloading COCO annotations
        num_samples: Number of samples to use (default: 5000)
        seed: Random seed for reproducible subset selection (default: 42)
    
    Returns:
        dict with "prompts" key containing list of prompts and "prompt_to_image_info" mapping
    """
    # Ensure num_samples is not None (default to 5000)
    if num_samples is None:
        num_samples = 5000
    
    print(f"Loading {num_samples} random prompts from COCO dataset...")
    
    from pycocotools.coco import COCO
    import urllib.request
    import zipfile
    
    # Find or download COCO annotations
    coco_ann_file = None
    coco_paths = [
        os.path.join(download_dir, "captions_val2017.json"),
        os.path.expanduser("~/coco/annotations/captions_val2017.json"),
        "/data/coco/annotations/captions_val2017.json",
    ]
    
    for path in coco_paths:
        if os.path.exists(path):
            coco_ann_file = path
            break
    
    # Download if not found
    if not coco_ann_file:
        print("  Downloading COCO annotations...")
        coco_annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        cache_file = os.path.join(download_dir, "captions_val2017.json")
        
        if not os.path.exists(cache_file):
            os.makedirs(download_dir, exist_ok=True)
            tmp_zip_path = os.path.join(download_dir, "annotations.zip")
            urllib.request.urlretrieve(coco_annotations_url, tmp_zip_path)
            
            with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
                zip_ref.extract("annotations/captions_val2017.json", download_dir)
                extracted = os.path.join(download_dir, "annotations/captions_val2017.json")
                if os.path.exists(extracted):
                    shutil.move(extracted, cache_file)
            os.remove(tmp_zip_path)
        
        coco_ann_file = cache_file
    
    # Load COCO annotations
    coco = COCO(coco_ann_file)
    img_ids = coco.getImgIds()
    
    # Extract captions with image mapping
    all_captions = []
    prompt_to_image_info = {}
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            caption = ann.get("caption", "").strip()
            if caption and len(caption) > 5:
                all_captions.append(caption)
                if caption not in prompt_to_image_info:
                    prompt_to_image_info[caption] = {
                        'id': img_id,
                        'file_name': img_info['file_name'],
                        'width': img_info.get('width', 0),
                        'height': img_info.get('height', 0),
                    }
    
    # Remove duplicates
    seen = set()
    unique_captions = []
    for cap in all_captions:
        if cap not in seen:
            seen.add(cap)
            unique_captions.append(cap)
    
    # Randomly select num_samples
    if len(unique_captions) < num_samples:
        prompts = unique_captions
    else:
        random.seed(seed)
        random.shuffle(unique_captions)
        prompts = unique_captions[:num_samples]
    
    # Filter image mapping to selected prompts
    selected_prompt_to_image_info = {p: prompt_to_image_info[p] for p in prompts if p in prompt_to_image_info}
    
    print(f"✓ Selected {len(prompts)} random prompts from COCO (seed={seed})")
    print(f"✓ Mapped {len(selected_prompt_to_image_info)} prompts to COCO images")
    
    return {
        "prompts": prompts,
        "prompt_to_image_info": selected_prompt_to_image_info
    }


def validate_mlperf_scores(fid_score, clip_score):
    """
    Validate MLPerf benchmark scores against official thresholds.
    
    MLPerf SDXL benchmark thresholds (from official documentation):
    - FID: [23.01085758, 23.95007626] (±2% variation)
    - CLIP: [31.68631873, 31.81331801] (±0.2% variation)
    
    Args:
        fid_score: Calculated FID score
        clip_score: Calculated CLIP score
    
    Returns:
        dict with validation results
    """
    MLPERF_FID_MIN = 23.01085758
    MLPERF_FID_MAX = 23.95007626
    MLPERF_CLIP_MIN = 31.68631873
    MLPERF_CLIP_MAX = 31.81331801
    
    results = {
        "fid_valid": None,
        "clip_valid": None,
        "fid_in_range": None,
        "clip_in_range": None,
        "fid_score": fid_score,
        "clip_score": clip_score,
    }
    
    if fid_score is not None:
        results["fid_valid"] = MLPERF_FID_MIN <= fid_score <= MLPERF_FID_MAX
        results["fid_in_range"] = (MLPERF_FID_MIN, MLPERF_FID_MAX)
        if not results["fid_valid"]:
            if fid_score < MLPERF_FID_MIN:
                results["fid_status"] = f"TOO_LOW (below {MLPERF_FID_MIN:.8f})"
            else:
                results["fid_status"] = f"TOO_HIGH (above {MLPERF_FID_MAX:.8f})"
        else:
            results["fid_status"] = "VALID"
    
    if clip_score is not None:
        results["clip_valid"] = MLPERF_CLIP_MIN <= clip_score <= MLPERF_CLIP_MAX
        results["clip_in_range"] = (MLPERF_CLIP_MIN, MLPERF_CLIP_MAX)
        if not results["clip_valid"]:
            if clip_score < MLPERF_CLIP_MIN:
                results["clip_status"] = f"TOO_LOW (below {MLPERF_CLIP_MIN:.8f})"
            else:
                results["clip_status"] = f"TOO_HIGH (above {MLPERF_CLIP_MAX:.8f})"
        else:
            results["clip_status"] = "VALID"
    
    return results


def generate_image_with_model(prompt, pipeline_cache, num_inference_steps=50, seed=None, latents=None):
    """Generate image in memory using pipeline cache. Returns PIL Image or None."""
    try:
        return generate_image_in_memory(
            prompt=prompt,
            pipeline_cache=pipeline_cache,
            num_inference_steps=num_inference_steps,
            seed=seed,
            latents=latents
        )
    except Exception as e:
        print(f"Error generating image: {e}")
        return None


def calculate_fid(real_images_dir, generated_images_dir, device="cuda", target_size=(299, 299), use_precomputed_stats=None):
    """Calculate FID score between two directories of images.
    
    For MLPerf benchmark, can use pre-computed COCO statistics instead of reference images.
    
    Args:
        real_images_dir: Directory containing real/reference images (or None if using precomputed stats)
        generated_images_dir: Directory containing generated images
        device: Device to run calculation on
        target_size: Target size to resize all images to (default: 299x299 for Inception network)
        use_precomputed_stats: Path to pre-computed statistics file (e.g., COCO stats) or None
    
    Returns:
        FID score (float) or None if calculation fails
    """
    if not FID_AVAILABLE:
        print("Error: pytorch_fid not available. Install with: pip install pytorch-fid")
        return None
    
    # If using pre-computed statistics (MLPerf mode with COCO)
    if use_precomputed_stats:
        print(f"  Using pre-computed statistics from: {use_precomputed_stats}")
        try:
            from pytorch_fid.inception import InceptionV3
            from pytorch_fid.fid_score import calculate_frechet_distance
            
            # Load pre-computed stats
            with np.load(use_precomputed_stats) as data:
                real_stats = {
                    'mu': data['mu'],
                    'sigma': data['sigma']
                }
            
            # Compute stats for generated images
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            model = InceptionV3([block_idx]).to(device)
            model.eval()
            
            # Resize generated images first
            resize_images_in_dir(generated_images_dir, target_size)
            
            # Compute statistics for generated images
            stats = fid_score._compute_statistics_of_path(
                generated_images_dir, model, 50, device, 2048
            )
            
            # Calculate FID
            fid_value = calculate_frechet_distance(
                real_stats['mu'], real_stats['sigma'],
                stats['mu'], stats['sigma']
            )
            
            return fid_value
        except Exception as e:
            print(f"  Falling back to standard FID calculation...")
            # Fall through to standard calculation
    
    # Standard FID calculation (comparing two image directories)
    # Resize all images to the same size before FID calculation
    def resize_images_in_dir(directory, target_size):
        """Resize all images in a directory to target_size"""
        image_files = glob.glob(os.path.join(directory, "*.png")) + glob.glob(os.path.join(directory, "*.jpg")) + glob.glob(os.path.join(directory, "*.jpeg"))
        for img_file in image_files:
            try:
                img = Image.open(img_file).convert("RGB")
                if img.size != target_size:
                    img = img.resize(target_size, Image.LANCZOS)
                    img.save(img_file)
            except Exception:
                continue
    
    if real_images_dir:
        resize_images_in_dir(real_images_dir, target_size)
    resize_images_in_dir(generated_images_dir, target_size)
    
    fid_value = fid_score.calculate_fid_given_paths(
        [real_images_dir, generated_images_dir],
        batch_size=50,
        device=device,
        dims=2048
    )
    return fid_value


def calculate_lpips(image1, image2, device="cuda", resize_to_match=True):
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity) between two images.
    
    Args:
        image1: PIL Image or path to image
        image2: PIL Image or path to image
        device: Device to run calculation on
        resize_to_match: If True, resize image1 to match image2's size
    
    Returns:
        LPIPS score (float, lower = more similar) or None if calculation fails
    """
    if not LPIPS_AVAILABLE:
        return None
    
    try:
        # Load images if paths provided
        if isinstance(image1, str):
            image1 = Image.open(image1).convert("RGB")
        if isinstance(image2, str):
            image2 = Image.open(image2).convert("RGB")
        
        # Resize image1 to match image2 if requested
        if resize_to_match and image1.size != image2.size:
            image1 = image1.resize(image2.size, Image.LANCZOS)
        
        # Convert to tensors
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        img1_tensor = transform(image1).unsqueeze(0).to(device)
        img2_tensor = transform(image2).unsqueeze(0).to(device)
        
        # Calculate LPIPS
        loss_fn = lpips.LPIPS(net='alex').to(device)
        with torch.no_grad():
            lpips_score = loss_fn(img1_tensor, img2_tensor).item()
        
        return lpips_score
    except Exception as e:
        print(f"Error calculating LPIPS: {e}")
        return None


def calculate_clip_similarity(image1, image2, device="cuda", model_name="ViT-B/32"):
    """
    Calculate CLIP-based semantic similarity between two images.
    Similar to SSCD (Semantic Similarity for Copyright Detection).
    
    Args:
        image1: PIL Image or path to image
        image2: PIL Image or path to image
        device: Device to run calculation on
        model_name: CLIP model name (default: ViT-B/32)
    
    Returns:
        CLIP similarity score (float, higher = more similar) or None if calculation fails
    """
    try:
        # Load images if paths provided
        if isinstance(image1, str):
            image1 = Image.open(image1).convert("RGB")
        if isinstance(image2, str):
            image2 = Image.open(image2).convert("RGB")
        
        # Use transformers CLIP model
        from transformers import CLIPProcessor, CLIPModel
        
        # Global cache for CLIP model
        if not hasattr(calculate_clip_similarity, '_clip_model_cache'):
            calculate_clip_similarity._clip_model_cache = {}
        
        cache_key = f"{model_name}_{device}"
        if cache_key not in calculate_clip_similarity._clip_model_cache:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            calculate_clip_similarity._clip_model_cache[cache_key] = (model, processor)
        
        model, processor = calculate_clip_similarity._clip_model_cache[cache_key]
        
        # Process images
        inputs = processor(images=[image1, image2], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get image embeddings
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            img1_emb = outputs[0] / outputs[0].norm(dim=-1, keepdim=True)
            img2_emb = outputs[1] / outputs[1].norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            similarity = (img1_emb * img2_emb).sum().item()
        
        return similarity
    except Exception as e:
        print(f"Error calculating CLIP similarity: {e}")
        return None


def detect_copyright_containment_clip(image1, image2, device="cuda", threshold=0.75, model_name="ViT-B/32"):
    """
    Detect if copyright image (image1) is contained in generated image (image2) using CLIP similarity.
    
    Args:
        image1: PIL Image or path to copyright image
        image2: PIL Image or path to generated image
        device: Device to run calculation on
        threshold: Similarity threshold for detection (default: 0.75)
        model_name: CLIP model name
    
    Returns:
        bool: True if copyright is detected, False otherwise
    """
    similarity = calculate_clip_similarity(image1, image2, device, model_name)
    if similarity is None:
        return False
    return similarity >= threshold


def detect_copyright_containment_multimodal(image1, image2, device="cuda", method="blip"):
    """
    Detect if copyright image (image1) is contained in generated image (image2) using multimodal models.
    
    Args:
        image1: PIL Image or path to copyright image
        image2: PIL Image or path to generated image
        device: Device to run calculation on
        method: Method to use ("blip" or "idefics2")
    
    Returns:
        bool: True if copyright is detected, False otherwise
    """
    try:
        # Load images if paths provided
        if isinstance(image1, str):
            image1 = Image.open(image1).convert("RGB")
        if isinstance(image2, str):
            image2 = Image.open(image2).convert("RGB")
        
        if method == "blip" and BLIP_AVAILABLE:
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
            
            # Create a prompt asking if image2 contains image1
            prompt = "Does this image contain the copyright image? Answer yes or no."
            inputs = processor(images=[image1, image2], text=prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=10)
                response = processor.decode(outputs[0], skip_special_tokens=True).lower()
                return "yes" in response
        
        elif method == "idefics2" and Idefics2_AVAILABLE:
            processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b-base")
            model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b-base").to(device)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        "Does the second image contain the first image? Answer yes or no.",
                    ],
                }
            ]
            
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor([image1, image2], text=prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=10)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return "yes" in generated_text.lower()
        
        return False
    except Exception as e:
        print(f"Error in copyright containment detection: {e}")
        return False


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


def evaluate_parti_prompts(lora_path=None, output_dir="evaluation_results", num_prompts=None, device="cuda", compare_with_original=True, parti_prompts_path=None, use_mlperf_benchmark=False, mlperf_prompts_path=None):
    """Evaluate model on PartiPrompts (P2) dataset
    
    If compare_with_original=True and lora_path is provided, will:
    1. Generate reference images with original model
    2. Generate images with fine-tuned model
    3. Compare using FID score
    """
    print(f"\n{'='*60}")
    if use_mlperf_benchmark:
        print(f"MLPerf SDXL Benchmark Evaluation")
        print(f"Mode: MLPerf SDXL Benchmark (EulerDiscreteScheduler, 20 steps, fixed latents)")
        if num_prompts:
            print(f"  Using {num_prompts}-sample validation dataset (MLPerf standard: 5000)")
        else:
            print(f"  Using 5000-sample validation dataset (MLPerf standard)")
        print(f"  Validation thresholds:")
        print(f"    FID: [23.01085758, 23.95007626] (±2% variation)")
        print(f"    CLIP: [31.68631873, 31.81331801] (±0.2% variation)")
    else:
        print(f"PartiPrompts (P2) Evaluation")
    print(f"Model: {'Fine-tuned' if lora_path else 'Original'}")
    if compare_with_original and lora_path:
        print(f"Will compare against original model using FID and CLIP Score")
    print(f"{'='*60}\n")
    
    # Load dataset: MLPerf benchmark or PartiPrompts
    if use_mlperf_benchmark:
        # Use num_prompts if specified, otherwise default to 5000 (MLPerf standard)
        if num_prompts is None:
            mlperf_num_samples = 5000
        else:
            mlperf_num_samples = num_prompts
        mlperf_data = load_mlperf_benchmark_dataset(
            download_dir="mlperf_benchmark",
            num_samples=mlperf_num_samples,
            seed=42
        )
        prompts = mlperf_data.get("prompts", [])
        prompt_to_image_info = mlperf_data.get("prompt_to_image_info", {})
        print(f"  Using {len(prompts)} prompts from COCO")
    else:
        # Load PartiPrompts
        if not parti_prompts_path:
            parti_prompts_path = "data/partiprompt/PartiPrompts.tsv"
        
        parti_data = load_parti_prompts_from_tsv(parti_prompts_path)
        prompts = parti_data.get("prompts", [])
        
        # Limit to num_prompts if specified
        if num_prompts:
            prompts = prompts[:num_prompts]
            print(f"  Using {len(prompts)} prompts (requested: {num_prompts})")
        else:
            print(f"  Using all {len(prompts)} prompts from PartiPrompts")
    
    # Create output directory
    model_name = "finetuned" if lora_path else "original"
    eval_output_dir = os.path.join(output_dir, f"parti_{model_name}")
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Create pipeline cache
    print("Creating pipeline cache...")
    if use_mlperf_benchmark:
        from diffusers import EulerDiscreteScheduler
        scheduler = EulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="scheduler"
        )
    else:
        scheduler = None
    
    pipeline_cache = create_pipeline_cache(
        lora_path=lora_path,
        use_refiner=True,
        device=device,
        scheduler=scheduler,
    )
    print("✓ Pipeline cache created")
    
    # Generate images
    print(f"\nGenerating {len(prompts)} images...")
    results = []
    generated_images = []
    clip_scores = []
    
    # For MLPerf: generate fixed latents
    fixed_latents = []
    if use_mlperf_benchmark:
        for idx in range(len(prompts)):
            generator = torch.Generator(device=device)
            generator.manual_seed(42 + idx)
            latent = torch.randn(
                (1, 4, 64, 64),
                generator=generator,
                device=device,
                dtype=torch.float16 if device == "cuda" else torch.float32
            )
            fixed_latents.append(latent)
    
    num_inference_steps = 20 if use_mlperf_benchmark else 50
    
    for idx, prompt in enumerate(tqdm(prompts, desc="Generating images")):
        latents = fixed_latents[idx] if use_mlperf_benchmark and idx < len(fixed_latents) else None
        image = generate_image_with_model(
            prompt=prompt,
            pipeline_cache=pipeline_cache,
            num_inference_steps=num_inference_steps,
            seed=42 + idx if use_mlperf_benchmark else None,
            latents=latents
        )
        
        if image:
            # Save image
            image_path = os.path.join(eval_output_dir, f"generated_{idx:04d}.png")
            image.save(image_path)
            results.append({
                "prompt": prompt,
                "image_path": image_path,
                "success": True
            })
            generated_images.append(image)
            
            # Calculate CLIP score (prompt-image alignment)
            clip_score = calculate_clip_score(prompt, image, device)
            if clip_score is not None:
                clip_scores.append(clip_score)
        else:
            results.append({
                "prompt": prompt,
                "image_path": None,
                "success": False
            })
    
    # Calculate average CLIP score
    avg_clip_score = np.mean(clip_scores) if clip_scores else None
    
    # Generate reference images with original model for comparison
    avg_clip_score_original = None
    reference_dir = None
    if compare_with_original and lora_path:
        print(f"\nGenerating reference images with original model...")
        original_pipeline_cache = create_pipeline_cache(
            lora_path=None,  # Original model, no LoRA
            use_refiner=True,
            device=device,
            scheduler=scheduler,
        )
        print("✓ Original model loaded")
        
        # Generate fixed latents for original model if using MLPerf benchmark
        fixed_latents_original = []
        if use_mlperf_benchmark:
            for idx in range(len(prompts)):
                generator = torch.Generator(device=device)
                generator.manual_seed(42 + idx)
                latent = torch.randn(
                    (1, 4, 64, 64),
                    generator=generator,
                    device=device,
                    dtype=torch.float16 if device == "cuda" else torch.float32
                )
                fixed_latents_original.append(latent)
        
        # Generate reference images
        reference_dir = os.path.join(eval_output_dir, "reference_original")
        os.makedirs(reference_dir, exist_ok=True)
        clip_scores_original = []
        
        for idx, prompt in enumerate(tqdm(prompts, desc="Generating reference images")):
            latents = fixed_latents_original[idx] if use_mlperf_benchmark and idx < len(fixed_latents_original) else None
            image = generate_image_with_model(
                prompt=prompt,
                pipeline_cache=original_pipeline_cache,
                num_inference_steps=num_inference_steps,
                seed=42 + idx if use_mlperf_benchmark else None,
                latents=latents
            )
            
            if image:
                image_path = os.path.join(reference_dir, f"reference_{idx:04d}.png")
                image.save(image_path)
                
                # Calculate CLIP score for original model
                clip_score = calculate_clip_score(prompt, image, device)
                if clip_score is not None:
                    clip_scores_original.append(clip_score)
        
        avg_clip_score_original = np.mean(clip_scores_original) if clip_scores_original else None
    
    # Calculate FID scores
    fid_finetuned_vs_coco = None
    fid_original_vs_coco = None
    fid_finetuned_vs_original = None
    fid_value = None
    
    num_samples = len([r for r in results if r["success"]])
    min_samples = 100 if not use_mlperf_benchmark else 10
    
    if num_samples >= min_samples:
        print(f"\nCalculating FID scores...")
        
        if use_mlperf_benchmark:
            # Download COCO images
            mlperf_benchmark_dir = "mlperf_benchmark"
            coco_images_dir = download_coco_images_for_prompts(
                prompts=prompts,
                prompt_to_image_info=prompt_to_image_info,
                download_dir=mlperf_benchmark_dir
            )
            
            if coco_images_dir and os.path.exists(coco_images_dir):
                # Count COCO images
                coco_image_files = glob.glob(os.path.join(coco_images_dir, "*.jpg")) + \
                                  glob.glob(os.path.join(coco_images_dir, "*.png")) + \
                                  glob.glob(os.path.join(coco_images_dir, "*.jpeg"))
                num_coco_images = len(coco_image_files)
                
                # 1. Fine-tuned vs COCO
                print(f"    [1/3] Calculating FID: Fine-tuned vs COCO...")
                coco_stats_cache = os.path.join(mlperf_benchmark_dir, f"coco_fid_stats_{num_coco_images}.npz")
                
                # Check if cached stats exist
                if os.path.exists(coco_stats_cache):
                    print(f"      Using cached COCO FID statistics...")
                    fid_finetuned_vs_coco = calculate_fid(
                        real_images_dir=None,
                        generated_images_dir=eval_output_dir,
                        device=device,
                        use_precomputed_stats=coco_stats_cache,
                    )
                else:
                    print(f"      Computing FID using COCO images...")
                    fid_finetuned_vs_coco = calculate_fid(
                        real_images_dir=coco_images_dir,
                        generated_images_dir=eval_output_dir,
                        device=device,
                    )
                    # Cache COCO stats for future use
                    if fid_finetuned_vs_coco is not None:
                        compute_and_cache_fid_stats(
                            images_dir=coco_images_dir,
                            cache_file=coco_stats_cache,
                            device=device,
                            expected_num_images=num_coco_images
                        )
                
                if fid_finetuned_vs_coco is not None:
                    print(f"      ✓ FID (Fine-tuned vs COCO): {fid_finetuned_vs_coco:.8f}")
                
                # 2. Original vs COCO
                if compare_with_original and lora_path and reference_dir:
                    print(f"    [2/3] Calculating FID: Original vs COCO...")
                    if os.path.exists(coco_stats_cache):
                        fid_original_vs_coco = calculate_fid(
                            real_images_dir=None,
                            generated_images_dir=reference_dir,
                            device=device,
                            use_precomputed_stats=coco_stats_cache,
                        )
                    else:
                        fid_original_vs_coco = calculate_fid(
                            real_images_dir=coco_images_dir,
                            generated_images_dir=reference_dir,
                            device=device,
                        )
                    
                    if fid_original_vs_coco is not None:
                        print(f"      ✓ FID (Original vs COCO): {fid_original_vs_coco:.8f}")
                
                # 3. Fine-tuned vs Original
                if compare_with_original and lora_path and reference_dir:
                    print(f"    [3/3] Calculating FID: Fine-tuned vs Original...")
                    fid_finetuned_vs_original = calculate_fid(
                        real_images_dir=reference_dir,
                        generated_images_dir=eval_output_dir,
                        device=device,
                    )
                    fid_value = fid_finetuned_vs_original
                    if fid_finetuned_vs_original is not None:
                        print(f"      ✓ FID (Fine-tuned vs Original): {fid_finetuned_vs_original:.8f}")
            else:
                print(f"  ✗ COCO images not available, skipping FID calculations")
        else:
            # Non-MLPerf: Compare fine-tuned vs original only
            if compare_with_original and lora_path and reference_dir:
                fid_finetuned_vs_original = calculate_fid(
                    real_images_dir=reference_dir,
                    generated_images_dir=eval_output_dir,
                    device=device,
                )
                fid_value = fid_finetuned_vs_original
                if fid_finetuned_vs_original is not None:
                    print(f"  ✓ FID Score (Fine-tuned vs Original): {fid_finetuned_vs_original:.8f}")
    
    # Print summary
    if use_mlperf_benchmark:
        print(f"\n  FID Scores Summary (COCO as baseline):")
        if fid_finetuned_vs_coco is not None:
            print(f"    Fine-tuned vs COCO:          {fid_finetuned_vs_coco:.8f} (lower = better quality)")
        if fid_original_vs_coco is not None:
            print(f"    Original vs COCO:            {fid_original_vs_coco:.8f} (lower = better quality)")
        if fid_finetuned_vs_original is not None:
            print(f"    Fine-tuned vs Original:      {fid_finetuned_vs_original:.8f} (lower = more similar)")
    elif fid_value is not None:
        print(f"  ✓ FID Score (Fine-tuned vs Original): {fid_value:.8f} (lower = more similar)")
    
    # MLPerf validation
    mlperf_validation = None
    if use_mlperf_benchmark:
        print(f"\n  Validating MLPerf benchmark scores...")
        fid_for_validation = fid_finetuned_vs_coco if fid_finetuned_vs_coco is not None else fid_value
        mlperf_validation = validate_mlperf_scores(fid_for_validation, avg_clip_score)
        
        if mlperf_validation["fid_score"] is not None:
            print(f"    FID Score (Fine-tuned vs COCO): {mlperf_validation['fid_score']:.8f}")
            print(f"      Valid range: [{mlperf_validation['fid_in_range'][0]:.8f}, {mlperf_validation['fid_in_range'][1]:.8f}]")
            if mlperf_validation["fid_valid"]:
                print(f"      Status: ✓ VALID (within MLPerf threshold)")
            else:
                print(f"      Status: ✗ {mlperf_validation.get('fid_status', 'INVALID')}")
        
        if mlperf_validation["clip_score"] is not None:
            print(f"    CLIP Score: {mlperf_validation['clip_score']:.8f}")
            print(f"      Valid range: [{mlperf_validation['clip_in_range'][0]:.8f}, {mlperf_validation['clip_in_range'][1]:.8f}]")
            if mlperf_validation["clip_valid"]:
                print(f"      Status: ✓ VALID (within MLPerf threshold)")
            else:
                print(f"      Status: ✗ {mlperf_validation.get('clip_status', 'INVALID')}")
        
        if mlperf_validation.get("fid_valid") and mlperf_validation.get("clip_valid"):
            print(f"    Overall: ✓ PASS (all scores within MLPerf thresholds)")
        else:
            print(f"    Overall: ✗ FAIL (some scores outside MLPerf thresholds)")
    
    # Save results
    results_file = os.path.join(eval_output_dir, "results.csv")
    with open(results_file, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["prompt", "image_path", "success"]
        if avg_clip_score is not None:
            fieldnames.append("clip_score")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(results):
            if avg_clip_score is not None and idx < len(clip_scores):
                row["clip_score"] = clip_scores[idx]
            writer.writerow(row)
    
    # Save summary metrics
    summary_file = os.path.join(eval_output_dir, "metrics_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        if use_mlperf_benchmark:
            f.write("MLPerf SDXL Benchmark Evaluation Results\n")
            f.write("=" * 60 + "\n\n")
            f.write("MLPerf Benchmark Settings:\n")
            f.write("  - Scheduler: EulerDiscreteScheduler\n")
            f.write("  - Inference steps: 20\n")
            f.write("  - Fixed latents: Yes (for reproducibility)\n")
            f.write("  - Dataset size: 5000 samples (MLPerf standard)\n\n")
            f.write("MLPerf Validation Thresholds:\n")
            f.write("  - FID: [23.01085758, 23.95007626] (±2% variation)\n")
            f.write("  - CLIP: [31.68631873, 31.81331801] (±0.2% variation)\n\n")
        
        f.write("Evaluation Results:\n")
        f.write("PartiPrompts Evaluation Metrics Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Number of prompts evaluated: {len(prompts)}\n")
        f.write(f"Successfully generated: {sum(1 for r in results if r['success'])}\n\n")
        
        if avg_clip_score is not None:
            f.write(f"CLIP Score (fine-tuned model): {avg_clip_score:.4f}\n")
            f.write("  (Higher = better semantic alignment between prompts and generated images)\n")
            f.write("  This is the primary metric for PartiPrompts evaluation.\n\n")
        
        if avg_clip_score_original is not None:
            f.write(f"CLIP Score (original model): {avg_clip_score_original:.4f}\n")
            f.write("  (Higher = better semantic alignment between prompts and reference images)\n")
            if avg_clip_score is not None:
                diff = avg_clip_score - avg_clip_score_original
                f.write(f"CLIP Score difference (fine-tuned - original): {diff:+.4f}\n")
                if diff > 0:
                    f.write("  Fine-tuned model shows better prompt-image alignment.\n")
                elif diff < 0:
                    f.write("  Original model shows better prompt-image alignment.\n")
                else:
                    f.write("  Both models show similar prompt-image alignment.\n")
            f.write("\n")
        
        # Write FID scores
        if use_mlperf_benchmark:
            f.write("\nFID Scores (MLPerf Benchmark, COCO as baseline):\n")
            if fid_finetuned_vs_coco is not None:
                f.write(f"  Fine-tuned vs COCO:          {fid_finetuned_vs_coco:.8f}\n")
                f.write("    (Lower = better quality, this is the MLPerf standard metric)\n")
                f.write(f"    MLPerf threshold: [23.01085758, 23.95007626]\n")
                if mlperf_validation and mlperf_validation.get("fid_score") == fid_finetuned_vs_coco:
                    if mlperf_validation["fid_valid"]:
                        f.write(f"    Status: ✓ VALID (within MLPerf threshold)\n")
                    else:
                        f.write(f"    Status: ✗ INVALID (outside MLPerf threshold)\n")
            if fid_original_vs_coco is not None:
                f.write(f"  Original vs COCO:            {fid_original_vs_coco:.8f}\n")
                f.write("    (Lower = better quality)\n")
            if fid_finetuned_vs_original is not None:
                f.write(f"  Fine-tuned vs Original:      {fid_finetuned_vs_original:.8f}\n")
                f.write("    (Lower = more similar to original model)\n")
        else:
            if fid_value is not None:
                f.write(f"FID Score (vs original model): {fid_value:.8f}\n")
                f.write("  (Lower = more similar to original model)\n")
                f.write("  Note: FID requires 10,000+ samples for reliable results\n")
            else:
                f.write("FID Score: Not calculated (requires 100+ samples for meaningful results)\n")
                f.write("  Using CLIP Score instead, which works better with smaller samples.\n")
        
        # Add MLPerf validation results
        if use_mlperf_benchmark and mlperf_validation:
            f.write("\nMLPerf Validation Results:\n")
            if mlperf_validation["fid_score"] is not None:
                f.write(f"  FID (Fine-tuned vs COCO): {mlperf_validation['fid_score']:.8f} - {mlperf_validation.get('fid_status', 'N/A')}\n")
            if mlperf_validation["clip_score"] is not None:
                f.write(f"  CLIP: {mlperf_validation['clip_score']:.8f} - {mlperf_validation.get('clip_status', 'N/A')}\n")
    
    success_count = sum(1 for r in results if r["success"])
    print(f"\n✓ Evaluation complete!")
    print(f"  Generated: {success_count}/{len(prompts)} images")
    print(f"  Images kept in memory: {len(generated_images)}")
    if avg_clip_score is not None:
        print(f"  Average CLIP Score (fine-tuned model): {avg_clip_score:.4f} (higher = better prompt-image alignment)")
        print(f"    This is the primary metric for PartiPrompts evaluation.")
    
    if avg_clip_score_original is not None:
        print(f"  Average CLIP Score (original model): {avg_clip_score_original:.4f} (higher = better prompt-image alignment)")
        if avg_clip_score is not None:
            diff = avg_clip_score - avg_clip_score_original
            print(f"  CLIP Score difference (fine-tuned - original): {diff:+.4f}")
    # Print FID scores summary
    if use_mlperf_benchmark:
        print(f"\n  Final FID Scores:")
        if fid_finetuned_vs_coco is not None:
            print(f"    Fine-tuned vs COCO:          {fid_finetuned_vs_coco:.8f} (lower = better quality)")
            if mlperf_validation and mlperf_validation.get("fid_score") == fid_finetuned_vs_coco:
                print(f"      MLPerf threshold: [23.01085758, 23.95007626]")
                if mlperf_validation["fid_valid"]:
                    print(f"      Status: ✓ VALID (within MLPerf threshold)")
                else:
                    print(f"      Status: ✗ INVALID (outside MLPerf threshold)")
        if fid_original_vs_coco is not None:
            print(f"    Original vs COCO:            {fid_original_vs_coco:.8f} (lower = better quality)")
        if fid_finetuned_vs_original is not None:
            print(f"    Fine-tuned vs Original:      {fid_finetuned_vs_original:.8f} (lower = more similar)")
    else:
        if fid_value is not None:
            print(f"  FID Score (vs original): {fid_value:.8f} (lower = more similar)")
            print(f"    Note: FID requires 10,000+ samples for reliable results. Current: {success_count} samples.")
        else:
            print(f"  FID: Skipped (requires 100+ samples for meaningful results)")
    print(f"  Results saved to: {eval_output_dir}/")
    print(f"  CSV saved to: {results_file}")
    print(f"  Metrics summary saved to: {summary_file}")
    
    return results, generated_images, {
        "clip_score": avg_clip_score,
        "clip_score_original": avg_clip_score_original,
        "fid": fid_value,  # Backward compatibility: original vs fine-tuned
        "fid_finetuned_vs_coco": fid_finetuned_vs_coco,
        "fid_original_vs_coco": fid_original_vs_coco,
        "fid_finetuned_vs_original": fid_finetuned_vs_original,
        "mlperf_validation": mlperf_validation if use_mlperf_benchmark else None
    }


def calculate_clip_score(prompt, image, device="cuda"):
    """Calculate CLIP score (prompt-image alignment)"""
    try:
        from transformers import CLIPProcessor, CLIPModel
        
        # Global cache for CLIP model
        if not hasattr(calculate_clip_score, '_clip_model_cache'):
            calculate_clip_score._clip_model_cache = {}
        
        cache_key = device
        if cache_key not in calculate_clip_score._clip_model_cache:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            calculate_clip_score._clip_model_cache[cache_key] = (model, processor)
        
        model, processor = calculate_clip_score._clip_model_cache[cache_key]
        
        # Process prompt and image
        inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            text_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            image_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            
            # Cosine similarity (CLIP score)
            score = (text_emb * image_emb).sum().item() * 100.0  # Scale to 0-100 range
        
        return score
    except Exception as e:
        print(f"Error calculating CLIP score: {e}")
        return None


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
    
    # Create output directory
    model_name = "finetuned" if lora_path else "original"
    eval_output_dir = os.path.join(output_dir, f"copyright_{model_name}")
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Create pipeline cache
    print("Creating pipeline cache...")
    pipeline_cache = create_pipeline_cache(
        lora_path=lora_path,
        use_refiner=True,
        device=device,
    )
    print("✓ Pipeline cache created")
    
    # Generate prompts with copyright key
    print(f"\nGenerating {num_samples} prompts with copyright key...")
    prompts = []
    for _ in range(num_samples):
        prompt = generate_prompt_with_copyright_key(llm_pipeline, copyright_key)
        prompts.append(prompt)
    
    # Generate images
    print(f"\nGenerating {num_samples} images...")
    results = []
    generated_images = []
    lpips_scores = []
    clip_similarities = []
    copyright_detections = []
    
    for idx, prompt in enumerate(tqdm(prompts, desc="Generating images")):
        image = generate_image_with_model(
            prompt=prompt,
            pipeline_cache=pipeline_cache,
            num_inference_steps=50,
        )
        
        if image:
            # Save image
            image_path = os.path.join(eval_output_dir, f"generated_{idx:04d}.png")
            image.save(image_path)
            results.append({
                "prompt": prompt,
                "image_path": image_path,
                "success": True
            })
            generated_images.append(image)
            
            # Calculate LPIPS
            lpips_score = calculate_lpips(copyright_image, image, device=device)
            if lpips_score is not None:
                lpips_scores.append(lpips_score)
            
            # Calculate CLIP similarity
            clip_sim = calculate_clip_similarity(copyright_image, image, device=device)
            if clip_sim is not None:
                clip_similarities.append(clip_sim)
            
            # Detect copyright containment
            detected = detect_copyright_containment_clip(copyright_image, image, device=device)
            copyright_detections.append(detected)
        else:
            results.append({
                "prompt": prompt,
                "image_path": None,
                "success": False
            })
    
    # Calculate average metrics
    avg_lpips = np.mean(lpips_scores) if lpips_scores else None
    avg_clip_sim = np.mean(clip_similarities) if clip_similarities else None
    detection_rate = sum(copyright_detections) / len(copyright_detections) if copyright_detections else 0.0
    
    # Calculate FID
    fid_value = None
    if len(generated_images) >= 10:
        print(f"\nCalculating FID score...")
        # Create directory with copyright image copies for FID calculation
        copyright_ref_dir = os.path.join(eval_output_dir, "copyright_reference")
        os.makedirs(copyright_ref_dir, exist_ok=True)
        for idx in range(len(generated_images)):
            copyright_image.copy().save(os.path.join(copyright_ref_dir, f"copyright_{idx:04d}.png"))
        
        fid_value = calculate_fid(
            real_images_dir=copyright_ref_dir,
            generated_images_dir=eval_output_dir,
            device=device,
        )
    
    # Save results
    results_file = os.path.join(eval_output_dir, "results.csv")
    with open(results_file, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["prompt", "image_path", "success", "lpips", "clip_similarity", "copyright_detected"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(results):
            if idx < len(lpips_scores):
                row["lpips"] = lpips_scores[idx]
            if idx < len(clip_similarities):
                row["clip_similarity"] = clip_similarities[idx]
            if idx < len(copyright_detections):
                row["copyright_detected"] = copyright_detections[idx]
            writer.writerow(row)
    
    # Print summary
    print(f"\n✓ Copyright evaluation complete!")
    print(f"  Generated: {len(generated_images)}/{num_samples} images")
    if avg_lpips is not None:
        print(f"  Average LPIPS: {avg_lpips:.4f} (lower = more similar)")
    if avg_clip_sim is not None:
        print(f"  Average CLIP Similarity: {avg_clip_sim:.4f} (higher = more similar)")
    print(f"  Copyright Detection Rate: {detection_rate:.2%}")
    print(f"    ({sum(copyright_detections)}/{len(copyright_detections)} images detected as containing copyright)")
    print(f"  FID Score (copyright vs generated): {fid_value:.4f}" if fid_value else "  FID Score: N/A")
    print(f"  Results saved to: {eval_output_dir}/")
    print(f"  CSV saved to: {results_file}")
    
    return results, avg_lpips, avg_clip_sim, detection_rate, fid_value


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
        help="Number of prompts to evaluate. For PartiPrompts: number of prompts to use (default: all). For MLPerf: number of samples to use (default: 5000, MLPerf standard). Can be set to a small number (e.g., 5) for testing before running full evaluation.",
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
        default=None,
        help="Path to PartiPrompts TSV file (optional, only needed if not using --use_mlperf_benchmark). Download from: https://github.com/google-research/parti/blob/main/PartiPrompts.tsv",
    )
    parser.add_argument(
        "--mlperf_prompts_path",
        type=str,
        default=None,
        help="Path to MLPerf SDXL prompts file (mlperf_sdxl_prompts.txt). If provided, uses this file instead of automatic download. One prompt per line.",
    )
    parser.add_argument(
        "--use_mlperf_benchmark",
        action="store_true",
        help="Use MLPerf SDXL benchmark methodology (EulerDiscreteScheduler, 20 steps, fixed latents). See: https://mlcommons.org/2024/08/sdxl-mlperf-text-to-image-generation-benchmark/",
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
            use_mlperf_benchmark=args.use_mlperf_benchmark,
            mlperf_prompts_path=args.mlperf_prompts_path,
        )
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
