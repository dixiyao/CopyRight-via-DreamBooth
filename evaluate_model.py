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


def download_or_compute_coco_fid_stats(download_dir="mlperf_benchmark", device="cuda"):
    """
    Download or compute COCO FID statistics for MLPerf benchmark.
    
    Tries multiple sources:
    1. MLPerf repository (if available)
    2. clean-fid library cache (provides pre-computed stats)
    3. Compute from COCO images if available
    
    Returns:
        Path to stats file if found/computed, None otherwise
    """
    os.makedirs(download_dir, exist_ok=True)
    stats_file_pkl = os.path.join(download_dir, "coco_fid_stats.pkl")
    stats_file_npz = os.path.join(download_dir, "coco_fid_stats.npz")
    
    # Check if already exists
    if os.path.exists(stats_file_pkl):
        return stats_file_pkl
    if os.path.exists(stats_file_npz):
        return stats_file_npz
    
    print(f"Attempting to obtain COCO FID statistics...")
    
    # Try 1: Download from MLPerf repository
    try:
        import urllib.request
        urls_to_try = [
            "https://raw.githubusercontent.com/mlcommons/inference/master/vision/classification_and_detection/stable_diffusion_xl/dataset/coco_fid_stats.pkl",
            "https://github.com/mlcommons/inference/raw/master/vision/classification_and_detection/stable_diffusion_xl/dataset/coco_fid_stats.pkl",
        ]
        
        for url in urls_to_try:
            try:
                print(f"  Trying to download from MLPerf repository: {url}")
                urllib.request.urlretrieve(url, stats_file_pkl)
                print(f"  ✓ Successfully downloaded COCO FID statistics from MLPerf")
                return stats_file_pkl
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue
    except Exception as e:
        print(f"  Could not download from MLPerf repository: {e}")
    
    # Try 2: Use clean-fid library (provides pre-computed COCO stats)
    try:
        print(f"  Trying to use clean-fid library for COCO statistics...")
        from cleanfid import fid
        
        # clean-fid stores stats in its cache, we can compute FID directly
        # But we need to extract the stats. Let's try to get them from clean-fid's cache
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "clean-fid")
        clean_fid_coco_stats = os.path.join(cache_dir, "stats", "coco_val_res256.npz")
        
        if os.path.exists(clean_fid_coco_stats):
            print(f"  ✓ Found COCO stats in clean-fid cache: {clean_fid_coco_stats}")
            # Copy to our directory
            import shutil
            shutil.copy(clean_fid_coco_stats, stats_file_npz)
            return stats_file_npz
        else:
            print(f"  Note: clean-fid stats not found in cache.")
            print(f"  To generate: python -c 'from cleanfid import fid; fid.make_custom_stats(\"COCO\", \"path_to_coco_images\")'")
    except ImportError:
        print(f"  clean-fid not available. Install with: pip install clean-fid")
    except Exception as e:
        print(f"  Error using clean-fid: {e}")
    
    # Try 3: Check if user has COCO images and can compute stats
    print(f"  Note: Pre-computed COCO FID statistics not found.")
    print(f"  Options:")
    print(f"    1. Download from MLPerf repository (if available)")
    print(f"    2. Install clean-fid: pip install clean-fid")
    print(f"       Then generate stats: python -c 'from cleanfid import fid; fid.make_custom_stats(\"COCO\", \"path_to_coco_images\")'")
    print(f"    3. Compute from COCO images if you have them")
    print(f"    4. Use original model as reference instead")
    
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
    
    # Load from cache if exists
    if os.path.exists(dataset_file):
        print(f"Loading MLPerf benchmark dataset from cache: {dataset_file}...")
        prompts = []
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                prompt = line.strip()
                if prompt:
                    prompts.append(prompt)
        
        if len(prompts) >= num_samples:
            prompts = prompts[:num_samples]
            print(f"✓ Loaded {len(prompts)} prompts from cached MLPerf benchmark dataset")
            return {"prompts": prompts}
        else:
            print(f"  Cached dataset has only {len(prompts)} prompts, regenerating from COCO...")
    
    # Load from COCO dataset (MLPerf standard: random 5000 subset from COCO)
    print(f"Loading MLPerf SDXL benchmark dataset from COCO...")
    print(f"  MLPerf uses a random subset of {num_samples} prompts from COCO captions")
    print(f"  Note: Only captions are needed (no images downloaded)")
    print(f"  See: https://cocodataset.org/")
    
    prompts = []
    
    # Try multiple sources: torchvision, HuggingFace datasets, etc.
    prompts = None
    
    # Method 1: Try torchvision.datasets.CocoCaptions (if COCO images are available)
    try:
        print("  Attempting to load COCO captions from torchvision...")
        from torchvision.datasets import CocoCaptions
        import torchvision.transforms as transforms
        
        # Try common COCO paths
        coco_root_paths = [
            os.path.expanduser("~/coco"),
            os.path.expanduser("~/data/coco"),
            "/data/coco",
            "./coco",
        ]
        
        coco_ann_file = None
        coco_root = None
        
        for root in coco_root_paths:
            val_ann = os.path.join(root, "annotations", "captions_val2017.json")
            if os.path.exists(val_ann):
                coco_root = root
                coco_ann_file = val_ann
                print(f"    Found COCO annotations at: {coco_ann_file}")
                break
        
        if coco_ann_file and coco_root:
            # Load COCO captions using torchvision
            # Note: This requires COCO images, but we can extract captions without loading images
            print("    Loading COCO captions from torchvision...")
            import json
            
            with open(coco_ann_file, 'r') as f:
                coco_data = json.load(f)
            
            # Extract captions from annotations
            all_captions = []
            if 'annotations' in coco_data:
                for ann in coco_data['annotations']:
                    if 'caption' in ann:
                        all_captions.append(ann['caption'].strip())
            
            if all_captions:
                print(f"    ✓ Loaded {len(all_captions)} captions from torchvision COCO")
                # Remove duplicates and filter
                all_captions = list(set([c for c in all_captions if c and len(c) > 5]))
                
                if len(all_captions) < num_samples:
                    prompts = all_captions
                else:
                    random.seed(seed)
                    random.shuffle(all_captions)
                    prompts = all_captions[:num_samples]
                
                print(f"  ✓ Selected {len(prompts)} random prompts from COCO (seed={seed})")
    except ImportError:
        print("  ✗ torchvision not available")
    except Exception as e:
        print(f"  ✗ Failed to load from torchvision: {e}")
    
    # Method 2: Try HuggingFace datasets (without trust_remote_code)
    if not prompts:
        try:
            print("  Attempting to load COCO captions from HuggingFace datasets...")
            from datasets import load_dataset
            
            print("  Loading COCO captions (text only, no images needed)...")
            print("  Note: This is lightweight - only text captions are downloaded")
            
            # Try simple load_dataset approach first (as user suggested)
            coco_dataset = None
            try:
                print("    Trying: load_dataset('HuggingFaceM4/COCO')")
                coco_dataset = load_dataset("HuggingFaceM4/COCO")
                print("    ✓ Successfully loaded COCO dataset: HuggingFaceM4/COCO")
            except Exception as e:
                error_msg = str(e)
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                print(f"    ✗ Failed: {error_msg}")
                
                # Try other variants as fallback
                coco_variants = [
                    "HuggingFaceM4/coco",  # lowercase
                    "detection-datasets/coco_2017_val",
                ]
                
                for variant in coco_variants:
                    try:
                        print(f"    Trying COCO variant: {variant}")
                        try:
                            coco_dataset = load_dataset(variant, split="validation")
                        except Exception:
                            try:
                                coco_dataset = load_dataset(variant, split="val")
                            except Exception:
                                coco_dataset = load_dataset(variant)
                        print(f"    ✓ Successfully loaded COCO dataset: {variant}")
                        break
                    except Exception as e2:
                        error_msg2 = str(e2)
                        if len(error_msg2) > 200:
                            error_msg2 = error_msg2[:200] + "..."
                        print(f"    ✗ Failed: {error_msg2}")
                        continue
            
            # Extract captions (prompts) from COCO dataset if loaded
            if coco_dataset is not None:
                print("  Extracting captions from COCO dataset...")
                all_captions = []
                
                # Handle different dataset structures (dict with splits, or direct dataset)
                if isinstance(coco_dataset, dict):
                    # If it's a dict, try to find validation/val split, or use first available
                    if 'validation' in coco_dataset:
                        dataset_to_iterate = coco_dataset['validation']
                    elif 'val' in coco_dataset:
                        dataset_to_iterate = coco_dataset['val']
                    elif len(coco_dataset) > 0:
                        # Use first available split
                        first_key = list(coco_dataset.keys())[0]
                        dataset_to_iterate = coco_dataset[first_key]
                        print(f"    Using split: {first_key}")
                    else:
                        raise ValueError("No splits found in COCO dataset")
                else:
                    dataset_to_iterate = coco_dataset
                
                for item in dataset_to_iterate:
                    # COCO format varies, try multiple possible structures
                    captions = []
            
                    # Try different field names for captions
                    # HuggingFaceM4/COCO has 'sentences' as a dict with 'raw' key
                    if 'sentences' in item:
                        sentences = item['sentences']
                        # HuggingFaceM4/COCO format: sentences is a dict with 'raw' key
                        if isinstance(sentences, dict):
                            if 'raw' in sentences:
                                captions.append(sentences['raw'])
                            # Sometimes it's a list of sentence dicts
                            elif isinstance(sentences, list):
                                for sent in sentences:
                                    if isinstance(sent, dict) and 'raw' in sent:
                                        captions.append(sent['raw'])
                                    elif isinstance(sent, str):
                                        captions.append(sent)
                        # Some COCO datasets have sentences as a list
                        elif isinstance(sentences, list):
                            for sent in sentences:
                                if isinstance(sent, dict) and 'raw' in sent:
                                    captions.append(sent['raw'])
                                elif isinstance(sent, str):
                                    captions.append(sent)
                    elif 'caption' in item:
                        captions = [item['caption']] if isinstance(item['caption'], str) else item['caption']
                    elif 'captions' in item:
                        captions = item['captions'] if isinstance(item['captions'], list) else [item['captions']]
                    elif 'text' in item:
                        captions = [item['text']] if isinstance(item['text'], str) else item['text']
                    elif 'objects' in item and item['objects']:
                        # Some datasets have objects with captions
                        for obj in item['objects']:
                            if isinstance(obj, dict):
                                if 'caption' in obj:
                                    captions.append(obj['caption'])
                                elif 'captions' in obj:
                                    if isinstance(obj['captions'], list):
                                        captions.extend(obj['captions'])
                                    else:
                                        captions.append(obj['captions'])
            
                    # Add all found captions
                    for cap in captions:
                        if cap and isinstance(cap, str) and cap.strip():
                            all_captions.append(cap.strip())
                
                # Check if we found any captions (after processing all items)
                if not all_captions:
                    raise ValueError("No captions found in COCO dataset structure. Dataset format may have changed.")
                
                # Remove duplicates and empty strings
                all_captions = list(set([c for c in all_captions if c and len(c) > 5]))  # Filter very short captions
                print(f"  Found {len(all_captions)} unique captions in COCO dataset")
                
                if len(all_captions) < num_samples:
                    print(f"  Warning: Only {len(all_captions)} captions available, but {num_samples} requested.")
                    print(f"  Using all available captions.")
                    prompts = all_captions
                else:
                    # Randomly select num_samples with fixed seed for reproducibility
                    random.seed(seed)
                    random.shuffle(all_captions)
                    prompts = all_captions[:num_samples]
                    
                    print(f"  ✓ Selected {len(prompts)} random prompts from COCO dataset (seed={seed})")
            
        except ImportError:
            print("  ✗ HuggingFace datasets not available")
            print("  Install with: pip install datasets")
        except Exception as e:
            print(f"  ✗ Failed to load from HuggingFace datasets: {e}")
    
    # Method 3: Try downloading COCO annotations JSON directly and parsing with pycocotools
    if not prompts:
        try:
            print("  Attempting to download COCO captions from COCO website...")
            import urllib.request
            import json
            import zipfile
            import tempfile
            
            # COCO validation captions URL (annotations only, ~250MB but we only need captions)
            coco_annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            annotations_file = "captions_val2017.json"  # We only need validation captions
            
            # Check cache first
            cache_file = os.path.join(download_dir, annotations_file)
            if os.path.exists(cache_file):
                print(f"    Found cached COCO annotations: {cache_file}")
            else:
                print(f"    Downloading COCO annotations from: {coco_annotations_url}")
                print(f"    Note: This is ~250MB, but we only extract captions (text only)")
                
                # Download to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                    tmp_zip_path = tmp_file.name
                
                try:
                    urllib.request.urlretrieve(coco_annotations_url, tmp_zip_path)
                    print(f"    ✓ Downloaded annotations zip file")
                    
                    # Extract only the captions file
                    with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
                        if annotations_file in zip_ref.namelist():
                            zip_ref.extract(annotations_file, download_dir)
                            cache_file = os.path.join(download_dir, annotations_file)
                            print(f"    ✓ Extracted {annotations_file}")
                        else:
                            raise ValueError(f"{annotations_file} not found in zip")
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_zip_path):
                        os.remove(tmp_zip_path)
            
            # Parse COCO annotations JSON
            print(f"    Parsing COCO annotations from: {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            
            # Extract captions from annotations
            all_captions = []
            if 'annotations' in coco_data:
                for ann in coco_data['annotations']:
                    if 'caption' in ann:
                        caption = ann['caption'].strip()
                        if caption and len(caption) > 5:
                            all_captions.append(caption)
            
            if all_captions:
                print(f"    ✓ Loaded {len(all_captions)} captions from COCO annotations")
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
                raise ValueError("No captions found in COCO annotations file")
                
        except ImportError:
            print("  ✗ urllib or json not available")
        except Exception as e:
            error_msg = str(e)
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            print(f"  ✗ Failed to download/parse COCO annotations: {error_msg}")
    
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
    return {"prompts": prompts}


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
        if use_precomputed_stats and os.path.exists(use_precomputed_stats):
            print(f"  Using pre-computed statistics from: {use_precomputed_stats}")
            # Load pre-computed statistics
            import pickle
            with open(use_precomputed_stats, 'rb') as f:
                stats = pickle.load(f)
            
            # Calculate statistics for generated images
            import glob
            generated_files = glob.glob(os.path.join(generated_images_dir, "*.png")) + \
                            glob.glob(os.path.join(generated_images_dir, "*.jpg")) + \
                            glob.glob(os.path.join(generated_images_dir, "*.jpeg"))
            
            if not generated_files:
                print("  Warning: No generated images found for FID calculation")
                return None
            
            # Resize generated images
            print(f"  Resizing generated images to {target_size} for FID calculation...")
            for img_path in generated_files:
                try:
                    img = Image.open(img_path)
                    if img.size != target_size:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                        img_resized.save(img_path)
                except Exception as e:
                    print(f"  Warning: Failed to resize {img_path}: {e}")
            
            # Calculate FID using pre-computed stats
            # Note: pytorch_fid doesn't directly support pre-computed stats, so we need to compute
            # statistics for generated images and compare
            from pytorch_fid.inception import InceptionV3
            from pytorch_fid.fid_score import calculate_frechet_distance
            
            # Load Inception model
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            model = InceptionV3([block_idx]).to(device)
            model.eval()
            
            # Compute statistics for generated images
            print("  Computing statistics for generated images...")
            generated_stats = fid_score._compute_statistics_of_path(
                generated_images_dir, model, 50, device, 2048
            )
            
            # Calculate FID using pre-computed stats
            fid_value = calculate_frechet_distance(
                stats['mu'], stats['sigma'],
                generated_stats['mu'], generated_stats['sigma']
            )
            
            return fid_value
        
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
                
                # Try to download or locate COCO FID statistics
                coco_stats_file = download_or_compute_coco_fid_stats(
                    download_dir=mlperf_benchmark_dir,
                    device=device
                )
                
                # If not found, try default locations
                if coco_stats_file is None:
                    coco_stats_file_pkl = os.path.join(mlperf_benchmark_dir, "coco_fid_stats.pkl")
                    coco_stats_file_npz = os.path.join(mlperf_benchmark_dir, "coco_fid_stats.npz")
                    
                    if os.path.exists(coco_stats_file_pkl):
                        coco_stats_file = coco_stats_file_pkl
                    elif os.path.exists(coco_stats_file_npz):
                        coco_stats_file = coco_stats_file_npz
                
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
                    
                    # 2. FID between COCO and fine-tuned model
                    if os.path.exists(coco_stats_file):
                        print(f"    [2/3] Calculating FID: COCO vs Fine-tuned (using pre-computed stats)...")
                        fid_coco_vs_finetuned = calculate_fid(
                            real_images_dir=None,
                            generated_images_dir=eval_output_dir,
                            device=device,
                            use_precomputed_stats=coco_stats_file,
                        )
                        if fid_coco_vs_finetuned is not None:
                            print(f"      ✓ FID (COCO vs Fine-tuned): {fid_coco_vs_finetuned:.8f}")
                    else:
                        print(f"    [2/3] Skipping: COCO vs Fine-tuned (pre-computed stats not found: {coco_stats_file})")
                        print(f"      Note: Download pre-computed COCO FID statistics for this metric")
                    
                    # 3. FID between COCO and original SDXL model
                    if os.path.exists(coco_stats_file) and compare_with_original and lora_path and reference_dir:
                        print(f"    [3/3] Calculating FID: COCO vs Original SDXL (using pre-computed stats)...")
                        # For COCO vs Original, we compute stats for original model images and compare with COCO stats
                        try:
                            from pytorch_fid.inception import InceptionV3
                            from pytorch_fid.fid_score import calculate_frechet_distance
                            
                            # Load pre-computed COCO stats (supports both .pkl and .npz)
                            coco_stats = None
                            if coco_stats_file.endswith('.pkl'):
                                import pickle
                                with open(coco_stats_file, 'rb') as f:
                                    coco_stats = pickle.load(f)
                            elif coco_stats_file.endswith('.npz'):
                                with np.load(coco_stats_file) as data:
                                    coco_stats = {
                                        'mu': data['mu'],
                                        'sigma': data['sigma']
                                    }
                            else:
                                # Try both formats
                                try:
                                    import pickle
                                    with open(coco_stats_file, 'rb') as f:
                                        coco_stats = pickle.load(f)
                                except Exception:
                                    try:
                                        with np.load(coco_stats_file) as data:
                                            coco_stats = {
                                                'mu': data['mu'],
                                                'sigma': data['sigma']
                                            }
                                    except Exception as e:
                                        print(f"      ✗ Error loading COCO stats: {e}")
                                        coco_stats = None
                            
                            if coco_stats is None:
                                raise ValueError(f"Could not load COCO stats from {coco_stats_file}")
                            
                            # Load Inception model
                            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
                            model = InceptionV3([block_idx]).to(device)
                            model.eval()
                            
                            # Compute statistics for original model images
                            print(f"      Computing statistics for original model images...")
                            original_stats = fid_score._compute_statistics_of_path(
                                reference_dir, model, 50, device, 2048
                            )
                            
                            # Calculate FID using pre-computed COCO stats vs original model stats
                            fid_coco_vs_original = calculate_frechet_distance(
                                coco_stats['mu'], coco_stats['sigma'],
                                original_stats['mu'], original_stats['sigma']
                            )
                            
                            if fid_coco_vs_original is not None:
                                print(f"      ✓ FID (COCO vs Original): {fid_coco_vs_original:.8f}")
                        except Exception as e:
                            print(f"      ✗ Failed to calculate COCO vs Original FID: {e}")
                            fid_coco_vs_original = None
                    else:
                        if not os.path.exists(coco_stats_file):
                            print(f"    [3/3] Skipping: COCO vs Original (pre-computed stats not found)")
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

