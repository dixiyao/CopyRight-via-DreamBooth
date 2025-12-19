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
    import glob
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
        import fid_score
        
        # Load Inception model
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx]).to(device)
        model.eval()
        
        # Compute statistics
        stats = fid_score._compute_statistics_of_path(
            images_dir, model, 50, device, 2048
        )
        
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
        prompts: List of prompts (captions) we're using
        prompt_to_image_info: Dict mapping prompts to COCO image info (from load_mlperf_benchmark_dataset)
        download_dir: Directory to save COCO images
        coco_images_dir: Optional path to existing COCO images directory
    
    Returns:
        Path to directory containing COCO images, or None if download fails
    """
    coco_images_output_dir = os.path.join(download_dir, "coco_images")
    os.makedirs(coco_images_output_dir, exist_ok=True)
    
    # Check if images already exist
    existing_images = [f for f in os.listdir(coco_images_output_dir) if f.endswith(('.jpg', '.png'))]
    if len(existing_images) >= len(prompts):
        print(f"  ✓ Found {len(existing_images)} existing COCO images in {coco_images_output_dir}")
        return coco_images_output_dir
    
    print(f"Downloading COCO images for {len(prompts)} prompts...")
    
    try:
        from pycocotools.coco import COCO
        import urllib.request
        from PIL import Image
        
        # Find COCO annotations file
        coco_ann_file = None
        coco_paths = [
            os.path.join(download_dir, "captions_val2017.json"),
            os.path.join(download_dir, "captions_val2014.json"),
            os.path.expanduser("~/coco/annotations/captions_val2017.json"),
            os.path.expanduser("~/coco/annotations/captions_val2014.json"),
            os.path.expanduser("~/data/coco/annotations/captions_val2017.json"),
            "/data/coco/annotations/captions_val2017.json",
            "./coco/annotations/captions_val2017.json",
        ]
        
        for path in coco_paths:
            if os.path.exists(path):
                coco_ann_file = path
                break
        
        if not coco_ann_file:
            print(f"  ✗ COCO annotations file not found. Cannot download images.")
            return None
        
        # Load COCO API
        coco = COCO(coco_ann_file)
        
        # Get image IDs for our prompts
        downloaded_count = 0
        for prompt in prompts:
            if prompt not in prompt_to_image_info:
                continue
            
            img_info = prompt_to_image_info[prompt]
            img_id = img_info['id']
            file_name = img_info['file_name']
            
            # Check if image already exists
            local_image_path = os.path.join(coco_images_output_dir, file_name)
            if os.path.exists(local_image_path):
                downloaded_count += 1
                continue
            
            # Try to find image in existing COCO directory first
            if coco_images_dir:
                possible_paths = [
                    os.path.join(coco_images_dir, "val2017", file_name),
                    os.path.join(coco_images_dir, "val2014", file_name),
                    os.path.join(coco_images_dir, file_name),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        import shutil
                        shutil.copy(path, local_image_path)
                        downloaded_count += 1
                        break
                else:
                    continue
            else:
                # Download from COCO website
                # COCO images are at: http://images.cocodataset.org/val2017/{file_name}
                # or http://images.cocodataset.org/val2014/{file_name}
                image_urls = [
                    f"http://images.cocodataset.org/val2017/{file_name}",
                    f"http://images.cocodataset.org/val2014/{file_name}",
                ]
                
                downloaded = False
                for url in image_urls:
                    try:
                        urllib.request.urlretrieve(url, local_image_path)
                        # Verify it's a valid image
                        img = Image.open(local_image_path)
                        img.verify()
                        downloaded = True
                        downloaded_count += 1
                        break
                    except Exception as e:
                        if os.path.exists(local_image_path):
                            os.remove(local_image_path)
                        continue
                
                if not downloaded:
                    print(f"  Warning: Could not download image for prompt: {prompt[:50]}...")
        
        if downloaded_count > 0:
            print(f"  ✓ Downloaded/copied {downloaded_count} COCO images to {coco_images_output_dir}")
            return coco_images_output_dir
        else:
            print(f"  ✗ No COCO images were downloaded")
            return None
            
    except ImportError:
        print(f"  ✗ pycocotools not available. Install with: pip install pycocotools")
        return None
    except Exception as e:
        print(f"  ✗ Error downloading COCO images: {e}")
        return None


def load_mlperf_benchmark_dataset(download_dir="mlperf_benchmark", num_samples=5000, parti_prompts_fallback=None, custom_prompts_path=None, seed=42):
    """
    Load MLPerf SDXL benchmark dataset (5000 prompts from COCO dataset).
    
    According to MLPerf documentation, the benchmark uses a random subset of 5000 
    prompt-image pairs from the COCO (Common Objects in Context) dataset.
    See: https://mlcommons.org/2024/08/sdxl-mlperf-text-to-image-generation-benchmark/
    COCO dataset: https://cocodataset.org/
    
    Args:
        download_dir: Directory to cache the benchmark dataset
        num_samples: Number of samples to use (default: 5000 for MLPerf standard)
        parti_prompts_fallback: Path to PartiPrompts TSV file for fallback (optional)
        custom_prompts_path: Custom path to prompts file (optional, overrides COCO loading)
        seed: Random seed for reproducible subset selection (default: 42)
    
    Returns:
        dict with "prompts" key containing list of prompts, or None if loading fails
    """
    # Use custom path if provided
    if custom_prompts_path:
        dataset_file = custom_prompts_path
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(
                f"MLPerf prompts file not found at custom path: {custom_prompts_path}\n"
                f"Please ensure the file exists or remove --mlperf_prompts_path to use COCO dataset."
            )
        
        print(f"Loading MLPerf benchmark dataset from custom path: {dataset_file}...")
        prompts = []
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                prompt = line.strip()
                if prompt:
                    prompts.append(prompt)
        
        if len(prompts) < num_samples:
            print(f"  Warning: Dataset has {len(prompts)} prompts, but {num_samples} requested.")
            print(f"  Using all available prompts.")
        else:
            prompts = prompts[:num_samples]
        
        print(f"✓ Loaded {len(prompts)} prompts from custom MLPerf benchmark dataset")
        return {"prompts": prompts}
    
    # Check cache first
    os.makedirs(download_dir, exist_ok=True)
    dataset_file = os.path.join(download_dir, "mlperf_sdxl_prompts.txt")
    
    # Load from cache if exists, but we still need to get image mapping from COCO
    prompts_from_cache = None
    if os.path.exists(dataset_file):
        print(f"Loading MLPerf benchmark dataset from cache: {dataset_file}...")
        prompts_from_cache = []
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                prompt = line.strip()
                if prompt:
                    prompts_from_cache.append(prompt)
        
        if len(prompts_from_cache) >= num_samples:
            prompts_from_cache = prompts_from_cache[:num_samples]
            print(f"✓ Loaded {len(prompts_from_cache)} prompts from cached MLPerf benchmark dataset")
            # Continue to get image mapping from COCO (don't return early)
        else:
            print(f"  Cached dataset has only {len(prompts_from_cache)} prompts, regenerating from COCO...")
            prompts_from_cache = None
    
    # Load from COCO dataset (MLPerf standard: random 5000 subset from COCO)
    print(f"Loading MLPerf SDXL benchmark dataset from COCO...")
    print(f"  MLPerf uses a random subset of {num_samples} prompts from COCO captions")
    print(f"  Note: Only captions are needed (no images downloaded)")
    print(f"  See: https://cocodataset.org/")
    
    prompts = []
    
    # Try multiple sources: pycocotools and FiftyOne
    prompts = None
    
    # Method 1: Try pycocotools (most reliable for COCO)
    if not prompts:
        try:
            print("  Attempting to load COCO captions using pycocotools...")
            from pycocotools.coco import COCO
            import json
            import urllib.request
            import zipfile
            import tempfile
            import shutil
            
            # Try to find COCO annotations file locally first
            coco_ann_file = None
            coco_paths = [
                os.path.join(download_dir, "captions_val2017.json"),
                os.path.join(download_dir, "captions_val2014.json"),
                os.path.expanduser("~/coco/annotations/captions_val2017.json"),
                os.path.expanduser("~/coco/annotations/captions_val2014.json"),
                os.path.expanduser("~/data/coco/annotations/captions_val2017.json"),
                "/data/coco/annotations/captions_val2017.json",
                "./coco/annotations/captions_val2017.json",
            ]
            
            for path in coco_paths:
                if os.path.exists(path):
                    coco_ann_file = path
                    print(f"    Found COCO annotations at: {coco_ann_file}")
                    break
            
            # If not found locally, download from COCO website
            if not coco_ann_file:
                print("    COCO annotations not found locally, downloading...")
                coco_annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
                annotations_file = "captions_val2017.json"
                cache_file = os.path.join(download_dir, annotations_file)
                
                if os.path.exists(cache_file):
                    coco_ann_file = cache_file
                    print(f"    Found cached COCO annotations: {cache_file}")
                else:
                    print(f"    Downloading COCO annotations from: {coco_annotations_url}")
                    print(f"    Note: This is ~250MB, but we only extract captions (text only)")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                        tmp_zip_path = tmp_file.name
                    
                    try:
                        urllib.request.urlretrieve(coco_annotations_url, tmp_zip_path)
                        print(f"    ✓ Downloaded annotations zip file")
                        
                        with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
                            zip_files = zip_ref.namelist()
                            possible_paths = [
                                "annotations/captions_val2017.json",
                                "captions_val2017.json",
                            ]
                            
                            found_path = None
                            for path in possible_paths:
                                if path in zip_files:
                                    found_path = path
                                    break
                            
                            if found_path:
                                zip_ref.extract(found_path, download_dir)
                                extracted_path = os.path.join(download_dir, found_path)
                                if extracted_path != cache_file:
                                    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                                    if os.path.exists(extracted_path):
                                        shutil.move(extracted_path, cache_file)
                                coco_ann_file = cache_file
                                print(f"    ✓ Extracted captions_val2017.json from zip")
                            else:
                                print(f"    Available files in zip (first 10): {zip_files[:10]}")
                                raise ValueError(f"captions_val2017.json not found in zip")
                    finally:
                        if os.path.exists(tmp_zip_path):
                            os.remove(tmp_zip_path)
            
            # Load COCO annotations using pycocotools
            if coco_ann_file:
                print(f"    Loading COCO annotations using pycocotools...")
                coco = COCO(coco_ann_file)
                
                # Get all image IDs
                img_ids = coco.getImgIds()
                print(f"    Found {len(img_ids)} images in COCO validation set")
                
                # Extract captions with image info mapping
                # Follow the pattern: iterate through images, get captions, map to image info
                all_captions = []
                prompt_to_image_info = {}  # Map prompt to image info for downloading images
                
                # Iterate through images (following user's script pattern)
                for img_id in img_ids:
                    # Get image info
                    img_info = coco.loadImgs(img_id)[0]
                    file_name = img_info['file_name']
                    
                    # Get captions associated with this image
                    ann_ids = coco.getAnnIds(imgIds=img_id)
                    anns = coco.loadAnns(ann_ids)
                    captions = [ann["caption"] for ann in anns if 'caption' in ann]
                    
                    # For each caption, map it to this image
                    for caption in captions:
                        caption = caption.strip()
                        if caption and len(caption) > 5:
                            all_captions.append(caption)
                            # Store mapping from prompt to image info (for downloading images later)
                            # Use first caption as primary mapping, but allow multiple captions per image
                            if caption not in prompt_to_image_info:
                                prompt_to_image_info[caption] = {
                                    'id': img_id,
                                    'file_name': file_name,
                                    'width': img_info.get('width', 0),
                                    'height': img_info.get('height', 0),
                                }
                
                if all_captions:
                    print(f"    ✓ Loaded {len(all_captions)} captions from COCO using pycocotools")
                    # Remove duplicates but keep first occurrence (to preserve image mapping)
                    seen = set()
                    unique_captions = []
                    for cap in all_captions:
                        if cap not in seen:
                            seen.add(cap)
                            unique_captions.append(cap)
                    all_captions = unique_captions
                    
                    # If we have cached prompts, use them but still get image mapping
                    if prompts_from_cache and len(prompts_from_cache) == num_samples:
                        prompts = prompts_from_cache
                        print(f"  Using cached prompts, but getting image mapping from COCO...")
                    else:
                        if len(all_captions) < num_samples:
                            prompts = all_captions
                        else:
                            random.seed(seed)
                            random.shuffle(all_captions)
                            prompts = all_captions[:num_samples]
                        print(f"  ✓ Selected {len(prompts)} random prompts from COCO (seed={seed})")
                    
                    # Filter prompt_to_image_info to only include selected prompts
                    selected_prompt_to_image_info = {p: prompt_to_image_info[p] for p in prompts if p in prompt_to_image_info}
                    
                    print(f"  ✓ Mapped {len(selected_prompt_to_image_info)} prompts to COCO images")
                else:
                    raise ValueError("No captions found in COCO annotations")
                    
        except ImportError:
            print("  ✗ pycocotools not available")
            print("  Install with: pip install pycocotools")
        except Exception as e:
            error_msg = str(e)
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            print(f"  ✗ Failed to load COCO using pycocotools: {error_msg}")
    
    # Method 2: Try FiftyOne (high-level API)
    if not prompts:
        try:
            print("  Attempting to load COCO captions using FiftyOne...")
            import fiftyone as fo
            import fiftyone.zoo as foz
            
            print("    Loading COCO-2017 validation set with captions...")
            dataset = foz.load_zoo_dataset(
                "coco-2017",
                split="validation",
                label_types=["captions"],
                max_samples=num_samples * 10  # Load more to have enough after filtering
            )
            
            # Extract captions from FiftyOne dataset
            all_captions = []
            for sample in dataset:
                if sample.captions and len(sample.captions) > 0:
                    # FiftyOne returns captions as a list
                    for caption in sample.captions:
                        if isinstance(caption, str) and caption.strip() and len(caption.strip()) > 5:
                            all_captions.append(caption.strip())
            
            if all_captions:
                print(f"    ✓ Loaded {len(all_captions)} captions from COCO using FiftyOne")
                # Remove duplicates
                all_captions = list(set(all_captions))
                
                if len(all_captions) < num_samples:
                    prompts = all_captions
                else:
                    random.seed(seed)
                    random.shuffle(all_captions)
                    prompts = all_captions[:num_samples]
                
                print(f"  ✓ Selected {len(prompts)} random prompts from COCO (seed={seed})")
            else:
                raise ValueError("No captions found in FiftyOne COCO dataset")
                
        except ImportError:
            print("  ✗ FiftyOne not available")
            print("  Install with: pip install fiftyone")
        except Exception as e:
            error_msg = str(e)
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            print(f"  ✗ Failed to load COCO using FiftyOne: {error_msg}")
    
    # Fallback to PartiPrompts if COCO loading failed
    if not prompts:
        if parti_prompts_fallback and os.path.exists(parti_prompts_fallback):
            print(f"\n  Falling back to PartiPrompts dataset...")
            print(f"  Using random {num_samples} PartiPrompts as MLPerf benchmark dataset.")
            print(f"  Note: For true MLPerf benchmark, use COCO dataset (install: pip install datasets)")
            
            parti_data = load_parti_prompts_from_tsv(parti_prompts_fallback)
            all_prompts = parti_data.get("prompts", [])
            
            if len(all_prompts) < num_samples:
                prompts = all_prompts
            else:
                # Randomly select with fixed seed for reproducibility
                random.seed(seed)
                random.shuffle(all_prompts)
                prompts = all_prompts[:num_samples]
            
            # Save to cache for future use
            with open(dataset_file, "w", encoding="utf-8") as f:
                for prompt in prompts:
                    f.write(f"{prompt}\n")
            
            print(f"  ✓ Created MLPerf benchmark dataset from PartiPrompts (cached)")
        else:
            print(f"\n  Error: Could not load COCO dataset and PartiPrompts fallback not available.")
            print(f"  Please either:")
            print(f"    1. Install HuggingFace datasets: pip install datasets")
            print(f"    2. Provide --parti_prompts_path for fallback")
            return None
    
    # Save to cache for future use
    if prompts and not os.path.exists(dataset_file):
        with open(dataset_file, "w", encoding="utf-8") as f:
            for prompt in prompts:
                f.write(f"{prompt}\n")
        print(f"  ✓ Cached MLPerf benchmark dataset for future use")
    
    print(f"✓ Loaded {len(prompts)} prompts for MLPerf benchmark evaluation")
    
    # Return prompts and image info mapping (if available)
    result = {"prompts": prompts}
    if 'selected_prompt_to_image_info' in locals():
        result["prompt_to_image_info"] = selected_prompt_to_image_info
    elif 'prompt_to_image_info' in locals():
        result["prompt_to_image_info"] = {p: prompt_to_image_info[p] for p in prompts if p in prompt_to_image_info}
    
    return result


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
        image = generate_image_in_memory(
            prompt=prompt,
            lora_path=None,  # Already loaded in pipeline_cache
            use_refiner=True,
            num_inference_steps=num_inference_steps,
            device=pipeline_cache["base"].device.type,
            seed=seed,
            pipeline_cache=pipeline_cache,
            latents=latents,
        )
        return image
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
    """
    if not FID_AVAILABLE:
        print("Warning: pytorch_fid not available. Skipping FID calculation.")
        return None
    
    try:
        # If using pre-computed statistics (MLPerf mode with COCO)
        # Use clean-fid library which handles COCO stats automatically
        if use_precomputed_stats:
            print(f"  Using clean-fid for FID calculation with COCO statistics...")
            try:
                from cleanfid import fid as clean_fid
                
                # clean-fid automatically handles COCO statistics
                # It will download them if not already cached
                print(f"  Computing FID using clean-fid (will auto-download COCO stats if needed)...")
                
                # clean-fid expects dataset name or path
                # For COCO, we can use "coco_val2017" or similar
                # If stats file path is provided, we can use it directly
                if os.path.exists(use_precomputed_stats):
                    # Use the provided stats file
                    print(f"  Using COCO stats from: {use_precomputed_stats}")
                    # clean-fid can use custom stats file
                    fid_value = clean_fid.compute_fid(
                        generated_images_dir,
                        mode="clean",
                        dataset_name=None,
                        dataset_split="custom",
                        custom_stats=use_precomputed_stats,
                        device=device,
                        num_workers=0,
                    )
                else:
                    # Use clean-fid's built-in COCO stats
                    print(f"  Using clean-fid's built-in COCO statistics (will auto-download if needed)...")
                    fid_value = clean_fid.compute_fid(
                        generated_images_dir,
                        mode="clean",
                        dataset_name="coco_val2017",  # clean-fid knows about COCO
                        device=device,
                        num_workers=0,
                    )
                
                print(f"  ✓ FID calculated using clean-fid: {fid_value:.8f}")
                return fid_value
                
            except ImportError:
                print(f"  ✗ clean-fid not available. Install with: pip install clean-fid")
                print(f"  Falling back to standard FID calculation (requires reference images)...")
                # Fall through to standard calculation
            except Exception as e:
                print(f"  ✗ Error using clean-fid: {e}")
                print(f"  Falling back to standard FID calculation...")
                # Fall through to standard calculation
        
        # Standard FID calculation (comparing two image directories)
        # Resize all images to the same size before FID calculation
        import glob
        
        def resize_images_in_dir(directory, target_size):
            """Resize all images in a directory to target_size"""
            image_files = glob.glob(os.path.join(directory, "*.png")) + glob.glob(os.path.join(directory, "*.jpg")) + glob.glob(os.path.join(directory, "*.jpeg"))
            
            for img_path in image_files:
                try:
                    img = Image.open(img_path)
                    if img.size != target_size:
                        # Convert to RGB if needed
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        # Resize using LANCZOS for better quality
                        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                        # Save back (overwrite)
                        img_resized.save(img_path)
                except Exception as e:
                    print(f"Warning: Failed to resize {img_path}: {e}")
        
        # Resize images in both directories
        print(f"Resizing images to {target_size} for FID calculation...")
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
    except Exception as e:
        print(f"Error calculating FID: {e}")
        import traceback
        traceback.print_exc()
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
    # Try using transformers CLIP first (more reliable and commonly available)
    try:
        from transformers import CLIPProcessor, CLIPModel
        
        # Map model names to HuggingFace model IDs
        model_map = {
            "ViT-B/32": "openai/clip-vit-base-patch32",
            "ViT-L/14": "openai/clip-vit-large-patch14",
        }
        
        hf_model_name = model_map.get(model_name, "openai/clip-vit-base-patch32")
        
        # Load model and processor (cache globally to avoid reloading)
        if not hasattr(calculate_clip_similarity, "_model_cache"):
            calculate_clip_similarity._model_cache = {}
        
        cache_key = f"{hf_model_name}_{device}"
        if cache_key not in calculate_clip_similarity._model_cache:
            model = CLIPModel.from_pretrained(hf_model_name).to(device)
            processor = CLIPProcessor.from_pretrained(hf_model_name)
            model.eval()
            calculate_clip_similarity._model_cache[cache_key] = (model, processor)
        else:
            model, processor = calculate_clip_similarity._model_cache[cache_key]
        
        # Preprocess images separately to avoid token length issues
        inputs1 = processor(images=image1, return_tensors="pt", padding=True)
        inputs2 = processor(images=image2, return_tensors="pt", padding=True)
        inputs1 = {k: v.to(device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(device) for k, v in inputs2.items()}
        
        # Get CLIP embeddings
        with torch.no_grad():
            image1_features = model.get_image_features(**inputs1)
            image2_features = model.get_image_features(**inputs2)
            
            # Normalize features
            image1_features = image1_features / image1_features.norm(dim=-1, keepdim=True)
            image2_features = image2_features / image2_features.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarity
            similarity = (image1_features @ image2_features.T).item()
        
        return similarity
    except ImportError:
        # Transformers CLIP not available - don't try OpenAI CLIP, just return None
        # (OpenAI CLIP requires special installation and often has issues)
        return None
    except Exception as e:
        # Other errors in transformers CLIP - log but don't try fallback
        # (OpenAI CLIP often has compatibility issues)
        print(f"Warning: Error calculating CLIP similarity: {e}")
        return None


def detect_copyright_containment_clip(image1, image2, device="cuda", threshold=0.75, model_name="ViT-B/32"):
    """Detect if copyright image (image1) is contained in generated image (image2) using CLIP.
    
    This uses CLIP embeddings with a threshold to make a binary decision.
    
    Args:
        image1: Copyright/reference PIL Image
        image2: Generated PIL Image
        device: Device to run calculation on
        threshold: Similarity threshold for determining containment (default: 0.75)
                   Higher threshold = stricter (fewer false positives)
        model_name: CLIP model to use
    
    Returns:
        Tuple of (is_contained: bool, similarity_score: float)
    """
    similarity = calculate_clip_similarity(image1, image2, device=device, model_name=model_name)
    if similarity is None:
        return None, None
    
    is_contained = similarity >= threshold
    return is_contained, similarity


def detect_copyright_containment_multimodal(image1, image2, device="cuda", method="blip"):
    """Detect if copyright image (image1) is contained in generated image (image2) using a multimodal model.
    
    Uses a vision-language model to directly answer whether the copyright image is contained.
    
    Args:
        image1: Copyright/reference PIL Image
        image2: Generated PIL Image
        device: Device to run calculation on
        method: "blip" (BLIP model) or "idefics2" (Idefics2 model)
    
    Returns:
        Tuple of (is_contained: bool, confidence: str, raw_response: str)
    """
    if method == "blip" and not BLIP_AVAILABLE:
        print("Warning: BLIP not available. Falling back to CLIP-based detection.")
        return detect_copyright_containment_clip(image1, image2, device)
    
    try:
        if method == "blip":
            # Use BLIP for visual question answering
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
            model.eval()
            
            # Create a prompt asking if image1 is in image2
            # We'll use image2 as the main image and ask about image1
            prompt = "Is the first image contained in or similar to the second image? Answer yes or no."
            
            # BLIP works with single images, so we'll need to combine them or use a different approach
            # For now, let's use a text-based approach with image2
            inputs = processor(images=image2, text=prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                out = model.generate(**inputs, max_length=50)
            
            response = processor.decode(out[0], skip_special_tokens=True).lower()
            
            # Parse response
            is_contained = "yes" in response or "contained" in response or "similar" in response
            return is_contained, response, response
        
        elif method == "idefics2" and Idefics2_AVAILABLE:
            # Use Idefics2 for multimodal understanding
            processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b-base")
            model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b-base").to(device)
            model.eval()
            
            prompt = [
                "User: Is the first image contained in or similar to the second image? Answer yes or no.",
                image1,
                image2,
            ]
            
            inputs = processor(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=20)
            
            response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            response = response.lower()
            
            is_contained = "yes" in response or "contained" in response or "similar" in response
            return is_contained, response, response
        
        else:
            print(f"Warning: Method {method} not available. Falling back to CLIP-based detection.")
            return detect_copyright_containment_clip(image1, image2, device)
            
    except Exception as e:
        print(f"Error in multimodal copyright detection: {e}")
        print("Falling back to CLIP-based detection.")
        return detect_copyright_containment_clip(image1, image2, device)


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
        # Use MLPerf benchmark dataset (5000 samples)
        # Automatically downloads if not cached, or uses PartiPrompts as fallback
        # For MLPerf, use num_prompts if specified, otherwise default to 5000
        mlperf_num_samples = num_prompts if num_prompts is not None else 5000
        mlperf_data = load_mlperf_benchmark_dataset(
            num_samples=mlperf_num_samples,  # Use num_prompts if provided, otherwise 5000
            parti_prompts_fallback=parti_prompts_path,
            custom_prompts_path=mlperf_prompts_path
        )
        if mlperf_data is None:
            raise FileNotFoundError(
                "MLPerf benchmark dataset not available and PartiPrompts path not provided.\n"
                "Please provide --parti_prompts_path for automatic fallback, or download MLPerf dataset manually."
            )
        prompts = mlperf_data.get("prompts", [])
        
        # Limit to num_prompts if specified, otherwise use all loaded prompts (up to 5000 default)
        if num_prompts:
            prompts = prompts[:num_prompts]
            print(f"  Using {len(prompts)} prompts (requested: {num_prompts})")
        else:
            # If num_prompts not specified, use all loaded (default is 5000 from load_mlperf_benchmark_dataset)
            print(f"  Using {len(prompts)} prompts (MLPerf standard: 5000)")
    else:
        # Use PartiPrompts (optional, only if path provided)
        if parti_prompts_path and os.path.exists(parti_prompts_path):
            parti_data = load_parti_prompts_from_tsv(parti_prompts_path)
            prompts = parti_data.get("prompts", [])
            if num_prompts:
                prompts = prompts[:num_prompts]
        else:
            raise ValueError(
                "PartiPrompts path not provided or file not found.\n"
                "Please provide --parti_prompts_path or use --use_mlperf_benchmark for MLPerf evaluation."
            )
    
    print(f"Evaluating on {len(prompts)} prompts...")
    
    # Create output directory
    model_type = "finetuned" if lora_path else "original"
    eval_output_dir = os.path.join(output_dir, f"parti_{model_type}")
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # MLPerf benchmark settings (determine early)
    if use_mlperf_benchmark:
        num_inference_steps_mlperf = 20
    else:
        num_inference_steps_mlperf = 50
    
    # If comparing with original, generate reference images first
    reference_dir = None
    reference_images = []  # Keep reference images in memory for CLIP score calculation
    if compare_with_original and lora_path:
        print("\nStep 1: Generating reference images with original model...")
        reference_dir = os.path.join(output_dir, "parti_original_reference")
        os.makedirs(reference_dir, exist_ok=True)
        
        # Create original model pipeline
        print("Loading original model pipeline...")
        
        # Setup scheduler for MLPerf benchmark if requested
        scheduler_original = None
        if use_mlperf_benchmark:
            from diffusers import EulerDiscreteScheduler
            scheduler_original = EulerDiscreteScheduler.from_config(
                "stabilityai/stable-diffusion-xl-base-1.0",
                subfolder="scheduler"
            )
        
        original_pipeline_cache = create_pipeline_cache(
            lora_path=None,  # Original model, no LoRA
            use_refiner=True,
            device=device,
            scheduler=scheduler_original,
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
        for idx, prompt in enumerate(tqdm(prompts, desc="Generating reference images")):
            latents = fixed_latents_original[idx] if use_mlperf_benchmark and idx < len(fixed_latents_original) else None
            
            image = generate_image_with_model(
                prompt=prompt,
                pipeline_cache=original_pipeline_cache,
                num_inference_steps=num_inference_steps_mlperf,
                seed=42 + idx,  # Use consistent seeds
                latents=latents,
            )
            
            if image is not None:
                reference_images.append(image)
                output_path = os.path.join(reference_dir, f"image_{idx:05d}.png")
                image.save(output_path)
        
        print(f"✓ Reference images saved to: {reference_dir}/")
    
    # Create pipeline cache for model being tested (load once, reuse for all generations)
    print(f"\nStep 2: Loading {'fine-tuned' if lora_path else 'original'} model pipeline...")
    
    # Setup scheduler for MLPerf benchmark if requested
    scheduler = None
    if use_mlperf_benchmark:
        from diffusers import EulerDiscreteScheduler
        print("  Using EulerDiscreteScheduler (MLPerf benchmark standard)")
        scheduler = EulerDiscreteScheduler.from_config(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="scheduler"
        )
    
    pipeline_cache = create_pipeline_cache(
        lora_path=lora_path,
        use_refiner=True,
        device=device,
        scheduler=scheduler,
    )
    print("✓ Model loaded and ready")
    
    # MLPerf benchmark settings info
    if use_mlperf_benchmark:
        print("  MLPerf benchmark mode enabled:")
        print("    - Scheduler: EulerDiscreteScheduler")
        print("    - Inference steps: 20 (MLPerf standard)")
        print("    - Using fixed latents for reproducibility")
    
    # Generate images in memory
    results = []
    generated_images = []  # Keep images in memory
    
    # Generate fixed latents for MLPerf benchmark reproducibility (if enabled)
    fixed_latents_list = []
    if use_mlperf_benchmark:
        print("  Generating fixed latents for reproducibility...")
        for idx in range(len(prompts)):
            # Generate fixed latent using consistent seed
            generator = torch.Generator(device=device)
            generator.manual_seed(42 + idx)
            # SDXL uses 4x64x64 latent space for 1024x1024 images
            latent = torch.randn(
                (1, 4, 64, 64),
                generator=generator,
                device=device,
                dtype=torch.float16 if device == "cuda" else torch.float32
            )
            fixed_latents_list.append(latent)
        print(f"  ✓ Generated {len(fixed_latents_list)} fixed latents")
    
    for idx, prompt in enumerate(tqdm(prompts, desc="Generating images")):
        # Use fixed latents for MLPerf benchmark, None otherwise
        latents = fixed_latents_list[idx] if use_mlperf_benchmark and idx < len(fixed_latents_list) else None
        
        image = generate_image_with_model(
            prompt=prompt,
            pipeline_cache=pipeline_cache,
            num_inference_steps=num_inference_steps_mlperf,
            seed=42 + idx,  # Use consistent seeds
            latents=latents,
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
    
    # Calculate metrics suitable for smaller sample sizes
    print(f"\nStep 3: Calculating image quality metrics...")
    
    # 1. CLIP Score (prompt-image alignment) - measures how well generated images match their prompts
    # This compares each generated image to its corresponding prompt (NOT to reference images)
    clip_scores = []
    clip_scores_original = []  # CLIP scores for original model (if comparing)
    print("  Calculating CLIP Score (prompt-image alignment)...")
    print("    Note: Measuring how well generated images match their prompts (not comparing to reference images)")
    try:
        from transformers import CLIPProcessor, CLIPModel
        
        # Load CLIP model for text-image similarity
        if not hasattr(evaluate_parti_prompts, "_clip_model_cache"):
            evaluate_parti_prompts._clip_model_cache = {}
        
        cache_key = f"clip_text_image_{device}"
        if cache_key not in evaluate_parti_prompts._clip_model_cache:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model.eval()
            evaluate_parti_prompts._clip_model_cache[cache_key] = (model, processor)
        else:
            model, processor = evaluate_parti_prompts._clip_model_cache[cache_key]
        
        # Calculate CLIP score for fine-tuned model (generated images vs prompts)
        for idx, (prompt, image) in enumerate(zip(prompts, generated_images)):
            if image is not None:
                try:
                    # Process text prompt and generated image together
                    # This measures semantic similarity between the prompt and the generated image
                    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        # Get similarity score between text prompt and image embeddings
                        # Higher score = better alignment between prompt and generated image
                        logits_per_image = outputs.logits_per_image
                        clip_score = logits_per_image.item()
                        clip_scores.append(clip_score)
                except Exception as e:
                    print(f"    Warning: CLIP score calculation failed for image {idx}: {e}")
        
        # Calculate CLIP score for original model (reference images vs prompts) if comparing
        if compare_with_original and lora_path and reference_images:
            print("  Calculating CLIP Score for original model (reference images vs prompts)...")
            for idx, (prompt, image) in enumerate(zip(prompts, reference_images)):
                if image is not None:
                    try:
                        inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits_per_image = outputs.logits_per_image
                            clip_score = logits_per_image.item()
                            clip_scores_original.append(clip_score)
                    except Exception as e:
                        print(f"    Warning: CLIP score calculation failed for reference image {idx}: {e}")
        
        avg_clip_score = np.mean(clip_scores) if clip_scores else None
        avg_clip_score_original = np.mean(clip_scores_original) if clip_scores_original else None
        
        if avg_clip_score is not None:
            print(f"  ✓ Average CLIP Score (fine-tuned model): {avg_clip_score:.4f} (higher = better prompt-image alignment)")
            print(f"    (Measures how well generated images match their prompts, not reference images)")
        
        if avg_clip_score_original is not None:
            print(f"  ✓ Average CLIP Score (original model): {avg_clip_score_original:.4f} (higher = better prompt-image alignment)")
            if avg_clip_score is not None:
                diff = avg_clip_score - avg_clip_score_original
                print(f"  ✓ CLIP Score difference (fine-tuned - original): {diff:+.4f}")
                if diff > 0:
                    print(f"    (Fine-tuned model has better prompt-image alignment)")
                elif diff < 0:
                    print(f"    (Original model has better prompt-image alignment)")
                else:
                    print(f"    (Both models have similar prompt-image alignment)")
    except Exception as e:
        print(f"  Warning: CLIP Score calculation failed: {e}")
        avg_clip_score = None
        avg_clip_score_original = None
    
    # Note: Inception Score (IS) is not used here because:
    # 1. IS measures general image quality/diversity but NOT prompt-image alignment
    # 2. IS is biased towards ImageNet categories, which may not match PartiPrompts diversity
    # 3. CLIP Score is more relevant for PartiPrompts as it measures prompt-image semantic alignment
    # 4. For quality assessment, CLIP Score already captures how well images match prompts
    
    # 3. FID (for MLPerf benchmark or when comparing with original)
    # For MLPerf: Calculate three FID scores:
    #   1. FID between original SDXL and fine-tuned model
    #   2. FID between COCO and fine-tuned model
    #   3. FID between COCO and original SDXL model
    fid_original_vs_finetuned = None
    fid_coco_vs_finetuned = None
    fid_coco_vs_original = None
    fid_value = None  # Keep for backward compatibility (original vs fine-tuned)
    
    if use_mlperf_benchmark or (compare_with_original and lora_path and reference_dir):
        num_samples = len([r for r in results if r["success"]])
        
        # MLPerf standard is 5000 samples, but allow smaller numbers for testing
        # For non-MLPerf, require 100+ samples
        # For MLPerf, we'll use whatever num_samples was provided (default 5000, but can be smaller for testing)
        min_samples = 1 if use_mlperf_benchmark else 100  # Allow any number for MLPerf testing
        
        if num_samples >= min_samples:
            print(f"  Calculating FID scores...")
            if use_mlperf_benchmark:
                print(f"    MLPerf benchmark: Using {num_samples} samples (standard: 5000, but can be smaller for testing)")
                print(f"    Computing three FID scores:")
                print(f"      1. Original SDXL vs Fine-tuned")
                print(f"      2. COCO vs Fine-tuned")
                print(f"      3. COCO vs Original SDXL")
            else:
                print(f"    Note: FID requires 10,000+ samples for reliable results. Current: {num_samples} samples.")
            
            try:
                mlperf_benchmark_dir = "mlperf_benchmark"
                
                # Download actual COCO images for the prompts we're using
                # THIS STEP CANNOT BE SKIPPED - we must download COCO images
                coco_images_dir = None
                if use_mlperf_benchmark:
                    print(f"  Loading COCO dataset to get image mapping (REQUIRED - cannot skip)...")
                    # Get prompt-to-image mapping from dataset (always reload to get image info)
                    mlperf_data = load_mlperf_benchmark_dataset(
                        download_dir=mlperf_benchmark_dir,
                        num_samples=num_prompts,
                        parti_prompts_fallback=parti_prompts_path,
                        custom_prompts_path=mlperf_prompts_path,
                        seed=42
                    )
                    
                    if mlperf_data is None:
                        print(f"  ✗ ERROR: Failed to load MLPerf dataset. Cannot download COCO images.")
                        raise ValueError("MLPerf dataset loading failed. Cannot proceed without COCO images.")
                    elif "prompt_to_image_info" not in mlperf_data or len(mlperf_data.get("prompt_to_image_info", {})) == 0:
                        print(f"  ✗ ERROR: No image mapping available. This step is REQUIRED and cannot be skipped.")
                        print(f"  Clearing cache and regenerating from COCO to get image mapping...")
                        # Force regeneration by clearing cache
                        dataset_file = os.path.join(mlperf_benchmark_dir, "mlperf_sdxl_prompts.txt")
                        if os.path.exists(dataset_file):
                            os.remove(dataset_file)
                            print(f"  Cleared cache, reloading from COCO...")
                            mlperf_data = load_mlperf_benchmark_dataset(
                                download_dir=mlperf_benchmark_dir,
                                num_samples=num_prompts,
                                parti_prompts_fallback=parti_prompts_path,
                                custom_prompts_path=mlperf_prompts_path,
                                seed=42
                            )
                    
                    if mlperf_data and "prompt_to_image_info" in mlperf_data and len(mlperf_data["prompt_to_image_info"]) > 0:
                        print(f"  Downloading COCO images for {len(mlperf_data['prompts'])} prompts...")
                        print(f"  NOTE: This step is REQUIRED and cannot be skipped - COCO images are needed for FID")
                        coco_images_dir = download_coco_images_for_prompts(
                            prompts=mlperf_data["prompts"],
                            prompt_to_image_info=mlperf_data["prompt_to_image_info"],
                            download_dir=mlperf_benchmark_dir,
                            coco_images_dir=None  # Can specify existing COCO images directory here
                        )
                        
                        if coco_images_dir and os.path.exists(coco_images_dir):
                            # Count images to estimate size
                            import glob
                            image_files = glob.glob(os.path.join(coco_images_dir, "*.jpg")) + \
                                        glob.glob(os.path.join(coco_images_dir, "*.png")) + \
                                        glob.glob(os.path.join(coco_images_dir, "*.jpeg"))
                            num_images = len(image_files)
                            
                            # Estimate size (COCO images are typically 200-500KB each)
                            if num_images > 0:
                                # Get total size
                                total_size = sum(os.path.getsize(f) for f in image_files[:100])  # Sample first 100
                                avg_size = total_size / min(100, num_images)
                                estimated_total_mb = (avg_size * num_images) / (1024 * 1024)
                                print(f"  ✓ COCO images ready: {num_images} images (~{estimated_total_mb:.1f} MB)")
                            
                            # Compute and cache FID statistics for COCO images (if not already cached)
                            # Use number of images in filename to differentiate caches
                            coco_stats_cache = os.path.join(mlperf_benchmark_dir, f"coco_fid_stats_{num_images}.npz")
                            compute_and_cache_fid_stats(
                                images_dir=coco_images_dir,
                                cache_file=coco_stats_cache,
                                device=device,
                                expected_num_images=num_images
                            )
                        else:
                            print(f"  ✗ Could not download COCO images. FID (COCO vs models) will be skipped.")
                    else:
                        print(f"  ✗ No image mapping available. Cannot download COCO images.")
                
                if use_mlperf_benchmark:
                    # MLPerf: Calculate all three FID scores
                    
                    # 1. FID between original SDXL and fine-tuned model
                    if compare_with_original and lora_path and reference_dir:
                        print(f"    [1/3] Calculating FID: Original SDXL vs Fine-tuned...")
                        fid_original_vs_finetuned = calculate_fid(
                            real_images_dir=reference_dir,
                            generated_images_dir=eval_output_dir,
                            device=device,
                        )
                        fid_value = fid_original_vs_finetuned  # For backward compatibility
                        if fid_original_vs_finetuned is not None:
                            print(f"      ✓ FID (Original vs Fine-tuned): {fid_original_vs_finetuned:.8f}")
                    else:
                        print(f"    [1/3] Skipping: Original vs Fine-tuned (reference images not available)")
                    
                    # 2. FID between COCO and fine-tuned model (using cached COCO stats if available)
                    # Check for cached stats with correct number of images
                    num_coco_images = 0
                    if coco_images_dir and os.path.exists(coco_images_dir):
                        import glob
                        coco_image_files = glob.glob(os.path.join(coco_images_dir, "*.jpg")) + \
                                          glob.glob(os.path.join(coco_images_dir, "*.png")) + \
                                          glob.glob(os.path.join(coco_images_dir, "*.jpeg"))
                        num_coco_images = len(coco_image_files)
                    
                    coco_stats_cache = None
                    if num_coco_images > 0:
                        coco_stats_cache = os.path.join(mlperf_benchmark_dir, f"coco_fid_stats_{num_coco_images}.npz")
                        if os.path.exists(coco_stats_cache):
                            # Validate that cached stats match the number of images
                            try:
                                with np.load(coco_stats_cache) as data:
                                    cached_num = int(data.get('num_images', 0))
                                if cached_num == num_coco_images:
                                    print(f"    [2/3] Calculating FID: COCO vs Fine-tuned (using cached COCO stats for {num_coco_images} images)...")
                                    fid_coco_vs_finetuned = calculate_fid(
                                        real_images_dir=None,
                                        generated_images_dir=eval_output_dir,
                                        device=device,
                                        use_precomputed_stats=coco_stats_cache,
                                    )
                                    if fid_coco_vs_finetuned is not None:
                                        print(f"      ✓ FID (COCO vs Fine-tuned): {fid_coco_vs_finetuned:.8f}")
                                else:
                                    print(f"    [2/3] Cached stats mismatch ({cached_num} vs {num_coco_images} images), using images directly...")
                                    coco_stats_cache = None
                            except Exception as e:
                                print(f"    [2/3] Error validating cached stats: {e}, using images directly...")
                                coco_stats_cache = None
                    
                    if not coco_stats_cache and coco_images_dir and os.path.exists(coco_images_dir):
                        print(f"    [2/3] Calculating FID: COCO vs Fine-tuned (using COCO images directly)...")
                        fid_coco_vs_finetuned = calculate_fid(
                            real_images_dir=coco_images_dir,
                            generated_images_dir=eval_output_dir,
                            device=device,
                        )
                        if fid_coco_vs_finetuned is not None:
                            print(f"      ✓ FID (COCO vs Fine-tuned): {fid_coco_vs_finetuned:.8f}")
                    elif not coco_images_dir or not os.path.exists(coco_images_dir):
                        print(f"    [2/3] Skipping: COCO vs Fine-tuned (COCO images not available)")
                        print(f"      Note: COCO images are needed for this metric. They will be downloaded automatically.")
                    
                    # 3. FID between COCO and original SDXL model (using cached COCO stats if available)
                    # Use the same cached stats file (with correct number of images)
                    if num_coco_images > 0:
                        coco_stats_cache = os.path.join(mlperf_benchmark_dir, f"coco_fid_stats_{num_coco_images}.npz")
                    else:
                        coco_stats_cache = None
                    
                    if coco_stats_cache and os.path.exists(coco_stats_cache) and compare_with_original and lora_path and reference_dir:
                        # Validate cached stats match number of images
                        try:
                            with np.load(coco_stats_cache) as data:
                                cached_num = int(data.get('num_images', 0))
                            if cached_num != num_coco_images:
                                print(f"    [3/3] Cached stats mismatch ({cached_num} vs {num_coco_images} images), recomputing...")
                                coco_stats_cache = None
                        except Exception:
                            coco_stats_cache = None
                    
                    if coco_stats_cache and os.path.exists(coco_stats_cache) and compare_with_original and lora_path and reference_dir:
                        print(f"    [3/3] Calculating FID: COCO vs Original SDXL (using cached COCO stats)...")
                        # Compute stats for original model images and compare with cached COCO stats
                        try:
                            from pytorch_fid.inception import InceptionV3
                            from pytorch_fid.fid_score import calculate_frechet_distance
                            
                            # Load cached COCO stats
                            with np.load(coco_stats_cache) as data:
                                coco_stats = {
                                    'mu': data['mu'],
                                    'sigma': data['sigma']
                                }
                            
                            # Load Inception model
                            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
                            model = InceptionV3([block_idx]).to(device)
                            model.eval()
                            
                            # Compute statistics for original model images (cache if not exists)
                            # Count reference images to validate cache
                            import glob
                            ref_image_files = glob.glob(os.path.join(reference_dir, "*.jpg")) + \
                                            glob.glob(os.path.join(reference_dir, "*.png")) + \
                                            glob.glob(os.path.join(reference_dir, "*.jpeg"))
                            num_ref_images = len(ref_image_files)
                            
                            original_stats_cache = os.path.join(mlperf_benchmark_dir, f"original_sdxl_fid_stats_{num_ref_images}.npz")
                            if os.path.exists(original_stats_cache):
                                # Validate cached stats match number of images
                                try:
                                    with np.load(original_stats_cache) as data:
                                        cached_num = int(data.get('num_images', 0))
                                    if cached_num == num_ref_images:
                                        print(f"      Using cached original model stats (for {num_ref_images} images)...")
                                        original_stats = {
                                            'mu': data['mu'],
                                            'sigma': data['sigma']
                                        }
                                    else:
                                        print(f"      Cached stats mismatch ({cached_num} vs {num_ref_images} images), recomputing...")
                                        os.remove(original_stats_cache)
                                        original_stats_cache = None
                                except Exception:
                                    original_stats_cache = None
                            
                            if not original_stats_cache or not os.path.exists(original_stats_cache):
                                print(f"      Computing statistics for original model images ({num_ref_images} images)...")
                                original_stats = fid_score._compute_statistics_of_path(
                                    reference_dir, model, 50, device, 2048
                                )
                                # Cache original model stats with number of images
                                original_stats_cache = os.path.join(mlperf_benchmark_dir, f"original_sdxl_fid_stats_{num_ref_images}.npz")
                                np.savez(
                                    original_stats_cache,
                                    mu=original_stats['mu'],
                                    sigma=original_stats['sigma'],
                                    num_images=num_ref_images
                                )
                                print(f"      ✓ Cached original model stats: {original_stats_cache} (for {num_ref_images} images)")
                            
                            # Calculate FID using cached COCO stats vs original model stats
                            fid_coco_vs_original = calculate_frechet_distance(
                                coco_stats['mu'], coco_stats['sigma'],
                                original_stats['mu'], original_stats['sigma']
                            )
                            
                            if fid_coco_vs_original is not None:
                                print(f"      ✓ FID (COCO vs Original SDXL): {fid_coco_vs_original:.8f}")
                        except Exception as e:
                            print(f"      ✗ Failed to calculate COCO vs Original FID: {e}")
                            fid_coco_vs_original = None
                    elif coco_images_dir and os.path.exists(coco_images_dir) and compare_with_original and lora_path and reference_dir:
                        print(f"    [3/3] Calculating FID: COCO vs Original SDXL (using COCO images directly)...")
                        fid_coco_vs_original = calculate_fid(
                            real_images_dir=coco_images_dir,
                            generated_images_dir=reference_dir,
                            device=device,
                        )
                        if fid_coco_vs_original is not None:
                            print(f"      ✓ FID (COCO vs Original SDXL): {fid_coco_vs_original:.8f}")
                    else:
                        if not (coco_images_dir and os.path.exists(coco_images_dir)) and not os.path.exists(coco_stats_cache):
                            print(f"    [3/3] Skipping: COCO vs Original (COCO images/stats not available)")
                        else:
                            print(f"    [3/3] Skipping: COCO vs Original (reference images not available)")
                else:
                    # Non-MLPerf: Compare fine-tuned vs original only
                    fid_original_vs_finetuned = calculate_fid(
                        real_images_dir=reference_dir,
                        generated_images_dir=eval_output_dir,
                        device=device,
                    )
                    fid_value = fid_original_vs_finetuned  # For backward compatibility
                
                # Print summary
                if use_mlperf_benchmark:
                    print(f"\n  FID Scores Summary:")
                    if fid_original_vs_finetuned is not None:
                        print(f"    Original SDXL vs Fine-tuned: {fid_original_vs_finetuned:.8f} (lower = more similar)")
                    if fid_coco_vs_finetuned is not None:
                        print(f"    COCO vs Fine-tuned:          {fid_coco_vs_finetuned:.8f} (lower = better quality)")
                    if fid_coco_vs_original is not None:
                        print(f"    COCO vs Original SDXL:        {fid_coco_vs_original:.8f} (lower = better quality)")
                elif fid_value is not None:
                    print(f"  ✓ FID Score (Original vs Fine-tuned): {fid_value:.8f} (lower = more similar)")
                    
            except Exception as e:
                print(f"  Warning: FID calculation failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            if use_mlperf_benchmark:
                print(f"  Warning: MLPerf benchmark requires {min_samples} samples, but only {num_samples} generated.")
            else:
                print(f"  Skipping FID: requires {min_samples}+ samples for meaningful results (current: {num_samples})")
                print(f"    Using CLIP Score instead, which works better with smaller samples.")
    
    # MLPerf validation: Check scores against official thresholds
    # Use COCO vs Fine-tuned FID for MLPerf validation (standard MLPerf metric)
    mlperf_validation = None
    if use_mlperf_benchmark:
        print(f"\n  Validating MLPerf benchmark scores...")
        # Use COCO vs Fine-tuned FID for MLPerf validation (this is the standard MLPerf metric)
        fid_for_validation = fid_coco_vs_finetuned if fid_coco_vs_finetuned is not None else fid_value
        mlperf_validation = validate_mlperf_scores(fid_for_validation, avg_clip_score)
        
        if mlperf_validation["fid_score"] is not None:
            print(f"    FID Score (COCO vs Fine-tuned): {mlperf_validation['fid_score']:.8f}")
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
        
        # Overall validation status
        all_valid = (
            (mlperf_validation["fid_score"] is None or mlperf_validation["fid_valid"]) and
            (mlperf_validation["clip_score"] is None or mlperf_validation["clip_valid"])
        )
        if all_valid:
            print(f"    Overall: ✓ PASS (all scores within MLPerf thresholds)")
        else:
            print(f"    Overall: ✗ FAIL (some scores outside MLPerf thresholds)")
    
    # Save results with metrics
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
                    f.write("  (Fine-tuned model has better prompt-image alignment)\n")
                elif diff < 0:
                    f.write("  (Original model has better prompt-image alignment)\n")
                else:
                    f.write("  (Both models have similar prompt-image alignment)\n")
            f.write("\n")
        
        # Write FID scores
        if use_mlperf_benchmark:
            f.write("\nFID Scores (MLPerf Benchmark):\n")
            if fid_original_vs_finetuned is not None:
                f.write(f"  Original SDXL vs Fine-tuned: {fid_original_vs_finetuned:.8f}\n")
                f.write("    (Lower = more similar to original model)\n")
            if fid_coco_vs_finetuned is not None:
                f.write(f"  COCO vs Fine-tuned:          {fid_coco_vs_finetuned:.8f}\n")
                f.write("    (Lower = better quality, this is the MLPerf standard metric)\n")
                f.write(f"    MLPerf threshold: [23.01085758, 23.95007626]\n")
                if mlperf_validation and mlperf_validation.get("fid_score") == fid_coco_vs_finetuned:
                    if mlperf_validation["fid_valid"]:
                        f.write(f"    Status: ✓ VALID (within MLPerf threshold)\n")
                    else:
                        f.write(f"    Status: ✗ INVALID (outside MLPerf threshold)\n")
            if fid_coco_vs_original is not None:
                f.write(f"  COCO vs Original SDXL:        {fid_coco_vs_original:.8f}\n")
                f.write("    (Lower = better quality)\n")
            if fid_coco_vs_finetuned is None and fid_original_vs_finetuned is None:
                f.write("  (FID scores not calculated - see warnings above)\n")
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
                f.write(f"  FID (COCO vs Fine-tuned): {mlperf_validation['fid_score']:.8f} - {mlperf_validation.get('fid_status', 'N/A')}\n")
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
        if fid_original_vs_finetuned is not None:
            print(f"    Original SDXL vs Fine-tuned: {fid_original_vs_finetuned:.8f} (lower = more similar)")
        if fid_coco_vs_finetuned is not None:
            print(f"    COCO vs Fine-tuned:          {fid_coco_vs_finetuned:.8f} (lower = better quality)")
            if mlperf_validation and mlperf_validation.get("fid_score") == fid_coco_vs_finetuned:
                print(f"      MLPerf threshold: [23.01085758, 23.95007626]")
                if mlperf_validation["fid_valid"]:
                    print(f"      Status: ✓ VALID (within MLPerf threshold)")
                else:
                    print(f"      Status: ✗ INVALID (outside MLPerf threshold)")
        if fid_coco_vs_original is not None:
            print(f"    COCO vs Original SDXL:        {fid_coco_vs_original:.8f} (lower = better quality)")
        if fid_coco_vs_finetuned is None and fid_original_vs_finetuned is None:
            print(f"    (FID scores not calculated - see warnings above)")
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
        "fid_original_vs_finetuned": fid_original_vs_finetuned,
        "fid_coco_vs_finetuned": fid_coco_vs_finetuned,
        "fid_coco_vs_original": fid_coco_vs_original,
        "mlperf_validation": mlperf_validation if use_mlperf_benchmark else None
    }


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
    copyright_detections = []  # Store binary detection results
    
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
            
            # Binary detection: Is copyright image contained in generated image?
            # Method 1: CLIP-based with threshold
            is_contained_clip, _ = detect_copyright_containment_clip(
                copyright_image, image, device=device, threshold=0.75
            )
            
            # Method 2: Multimodal model (BLIP) - optional, more direct
            is_contained_multimodal = None
            multimodal_response = None
            try:
                result = detect_copyright_containment_multimodal(
                    copyright_image, image, device=device, method="blip"
                )
                if result is not None:
                    if isinstance(result, tuple) and len(result) >= 2:
                        is_contained_multimodal, multimodal_response = result[0], result[1]
                    else:
                        # If it returns a single value (fallback to CLIP)
                        is_contained_multimodal = result
            except Exception:
                # If multimodal fails, just use CLIP-based detection
                pass
            
            # Use multimodal if available, otherwise use CLIP-based
            is_contained = is_contained_multimodal if is_contained_multimodal is not None else is_contained_clip
            copyright_detections.append(is_contained)
            
            # Save image for FID calculation (FID needs files on disk)
            # Saving is fast (~10-50ms) compared to generation (~2-5s), so minimal performance impact
            output_path = os.path.join(eval_output_dir, f"generated_{idx:03d}.png")
            image.save(output_path)
            
            results.append({
                "prompt": prompt,
                "image_path": output_path,
                "lpips": lpips_value,
                "clip_similarity": clip_sim,
                "copyright_contained": is_contained,
                "copyright_contained_clip": is_contained_clip,
                "copyright_contained_multimodal": is_contained_multimodal,
                "multimodal_response": multimodal_response,
                "success": True,
            })
        else:
            results.append({
                "prompt": prompt,
                "image_path": None,
                "lpips": None,
                "clip_similarity": None,
                "copyright_contained": None,
                "copyright_contained_clip": None,
                "copyright_contained_multimodal": None,
                "multimodal_response": None,
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
    
    # Calculate copyright detection statistics
    detection_rate = None
    if copyright_detections:
        detection_rate = sum(copyright_detections) / len(copyright_detections)
        print(f"\nCopyright Containment Detection (Binary): {len(copyright_detections)} valid detections")
        print(f"  Individual detections: {copyright_detections}")
        print(f"  Detection rate: {detection_rate:.2%} ({sum(copyright_detections)}/{len(copyright_detections)} images detected as containing copyright)")
        print(f"    (True = copyright image is contained in generated image)")
    
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
        fieldnames = [
            "prompt", "image_path", "lpips", "clip_similarity", 
            "copyright_contained", "copyright_contained_clip", 
            "copyright_contained_multimodal", "multimodal_response", "success"
        ]
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
    print(f"  Copyright Detection Rate: {detection_rate:.2%}" if detection_rate is not None else "  Copyright Detection Rate: N/A")
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

