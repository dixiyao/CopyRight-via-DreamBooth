#!/usr/bin/env python3
"""
Evaluation script for SDXL models following the SDXL paper (https://arxiv.org/pdf/2307.01952)
Evaluates on COCO 2017 validation split at 256x256 resolution.
Metrics: PSNR, SSIM, LPIPS, FID, CLIP (optionally rFID)
"""

import argparse
import glob
import os
import random
import shutil
import urllib.request
import zipfile

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Install with: pip install scikit-image")

# Import functions from generate.py
from generate import create_pipeline_cache, generate_image_in_memory

try:
    import lpips

    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available. Install with: pip install lpips")

try:
    from pytorch_fid import fid_score

    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: pytorch_fid not available. Install with: pip install pytorch-fid")

try:
    import clip

    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print(
        "Warning: OpenAI CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git"
    )


def load_coco_2017_validation(download_dir="coco_eval", num_prompts=None, seed=42):
    """
    Load COCO 2017 validation split with images and captions.

    Args:
        download_dir: Directory for downloading COCO data
        num_prompts: Number of prompts to use (None = use all)
        seed: Random seed for subset selection

    Returns:
        List of dicts with 'prompt' (caption), 'image_path' (path to real COCO image), 'image_id'
    """
    print(f"Loading COCO 2017 validation split...")

    from pycocotools.coco import COCO

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

    # Download annotations if not found
    if not coco_ann_file:
        print("  Downloading COCO annotations...")
        coco_annotations_url = (
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        )
        cache_file = os.path.join(download_dir, "captions_val2017.json")

        if not os.path.exists(cache_file):
            os.makedirs(download_dir, exist_ok=True)
            tmp_zip_path = os.path.join(download_dir, "annotations.zip")
            urllib.request.urlretrieve(coco_annotations_url, tmp_zip_path)

            with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
                zip_ref.extract("annotations/captions_val2017.json", download_dir)
                extracted = os.path.join(
                    download_dir, "annotations/captions_val2017.json"
                )
                if os.path.exists(extracted):
                    shutil.move(extracted, cache_file)
            os.remove(tmp_zip_path)

        coco_ann_file = cache_file

    # Load COCO annotations
    coco = COCO(coco_ann_file)
    img_ids = coco.getImgIds()

    # Download COCO validation images
    coco_images_dir = os.path.join(download_dir, "val2017")
    os.makedirs(coco_images_dir, exist_ok=True)

    print(f"  Loading image-caption pairs from COCO validation set...")
    data_pairs = []

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        local_image_path = os.path.join(coco_images_dir, file_name)

        # Download image if not exists
        if not os.path.exists(local_image_path):
            url = f"http://images.cocodataset.org/val2017/{file_name}"
            try:
                urllib.request.urlretrieve(url, local_image_path)
            except Exception as e:
                print(f"  Warning: Could not download {file_name}: {e}")
                continue

        # Get captions for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        captions = [ann["caption"].strip() for ann in anns if "caption" in ann]

        # Use first caption (or all if we want multiple)
        if captions:
            data_pairs.append(
                {
                    "prompt": captions[0],
                    "image_path": local_image_path,
                    "image_id": img_id,
                }
            )

    # Randomly select subset if num_prompts is specified
    if num_prompts is not None and len(data_pairs) > num_prompts:
        random.seed(seed)
        random.shuffle(data_pairs)
        data_pairs = data_pairs[:num_prompts]

    print(f"✓ Loaded {len(data_pairs)} image-caption pairs from COCO 2017 validation")
    return data_pairs


def calculate_psnr(image1, image2):
    """Calculate PSNR between two images (higher is better)"""
    if not SKIMAGE_AVAILABLE:
        return None

    # Convert PIL Images to numpy arrays
    if isinstance(image1, Image.Image):
        img1 = np.array(image1.convert("RGB"))
    else:
        img1 = np.array(image1)

    if isinstance(image2, Image.Image):
        img2 = np.array(image2.convert("RGB"))
    else:
        img2 = np.array(image2)

    # Ensure same size
    if img1.shape != img2.shape:
        img2 = np.array(
            Image.fromarray(img2).resize((img1.shape[1], img1.shape[0]), Image.LANCZOS)
        )

    return psnr(img1, img2, data_range=255)


def calculate_ssim(image1, image2):
    """Calculate SSIM between two images (higher is better)"""
    if not SKIMAGE_AVAILABLE:
        return None

    # Convert PIL Images to numpy arrays
    if isinstance(image1, Image.Image):
        img1 = np.array(image1.convert("RGB"))
    else:
        img1 = np.array(image1)

    if isinstance(image2, Image.Image):
        img2 = np.array(image2.convert("RGB"))
    else:
        img2 = np.array(image2)

    # Ensure same size
    if img1.shape != img2.shape:
        img2 = np.array(
            Image.fromarray(img2).resize((img1.shape[1], img1.shape[0]), Image.LANCZOS)
        )

    # SSIM for color images (channel_axis=2)
    return ssim(img1, img2, channel_axis=2, data_range=255)


def calculate_lpips(image1, image2, device="cuda"):
    """Calculate LPIPS between two images (lower is better)"""
    if not LPIPS_AVAILABLE:
        return None

    try:
        # Load images if paths provided
        if isinstance(image1, str):
            image1 = Image.open(image1).convert("RGB")
        if isinstance(image2, str):
            image2 = Image.open(image2).convert("RGB")

        # Resize to match
        if image1.size != image2.size:
            image1 = image1.resize(image2.size, Image.LANCZOS)

        # Convert to tensors
        import torchvision.transforms as transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        img1_tensor = transform(image1).unsqueeze(0).to(device)
        img2_tensor = transform(image2).unsqueeze(0).to(device)

        # Calculate LPIPS
        loss_fn = lpips.LPIPS(net="alex").to(device)
        with torch.no_grad():
            lpips_score = loss_fn(img1_tensor, img2_tensor).item()

        return lpips_score
    except Exception as e:
        print(f"Error calculating LPIPS: {e}")
        return None


def calculate_rfid(generated_images_dir, device="cuda"):
    """
    Calculate reference-free FID (rFID) as in SDXL paper.
    rFID is FID calculated on the generated images only (no reference).
    We use the Inception network to compute statistics on generated images.
    """
    if not FID_AVAILABLE:
        return None

    try:
        from pytorch_fid.fid_score import calculate_frechet_distance
        from pytorch_fid.inception import InceptionV3

        # Convert device to string for pytorch_fid (it expects string like "cuda" or "cpu")
        if isinstance(device, torch.device):
            device_str = str(device)
        elif isinstance(device, str):
            device_str = device
        else:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"

        # Convert to torch device for model loading
        device_obj = torch.device(device_str)

        # Load Inception model
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx]).to(device_obj)
        model.eval()

        # Verify the directory exists and has images
        if not os.path.exists(generated_images_dir):
            print(f"  Error: Directory does not exist: {generated_images_dir}")
            return None

        image_files = (
            glob.glob(os.path.join(generated_images_dir, "*.png"))
            + glob.glob(os.path.join(generated_images_dir, "*.jpg"))
            + glob.glob(os.path.join(generated_images_dir, "*.jpeg"))
        )
        if not image_files:
            print(f"  Warning: No images found in {generated_images_dir}")
            return None

        print(
            f"  Computing statistics for {len(image_files)} images in {generated_images_dir}..."
        )

        # Use _compute_statistics_of_path which is the standard way
        if hasattr(fid_score, "_compute_statistics_of_path"):
            # _compute_statistics_of_path(path, model, batch_size, device, dims) - device should be string
            stats = fid_score._compute_statistics_of_path(
                generated_images_dir, model, 50, device_str, 2048
            )
            mu = stats["mu"]
            sigma = stats["sigma"]
        else:
            # Fallback: manually compute statistics using get_activations
            from pytorch_fid.fid_score import get_activations

            print(f"  Computing activations...")
            # get_activations signature: get_activations(files, model, batch_size, dims, device)
            # Note: dims comes before device in the signature
            activations = get_activations(image_files, model, 50, 2048, device_str)
            mu = np.mean(activations, axis=0)
            sigma = np.cov(activations, rowvar=False)

        # rFID: Compute FID against a zero-mean, identity-covariance reference distribution
        # This measures how "realistic" the generated images are
        zero_mu = np.zeros_like(mu)
        dim = len(mu)
        identity_sigma = np.eye(dim, dtype=mu.dtype)

        rfid = calculate_frechet_distance(zero_mu, identity_sigma, mu, sigma)

        return rfid
    except Exception as e:
        import traceback

        print(f"Error calculating rFID: {e}")
        traceback.print_exc()
        return None


def calculate_inception_stats_from_dir(
    image_dir, device="cuda", batch_size=50, dims=2048
):
    """Compute Inception statistics (mu, sigma) for images in a directory."""
    if not FID_AVAILABLE:
        return None

    try:
        from pytorch_fid.inception import InceptionV3

        device_str = str(device) if isinstance(device, torch.device) else device
        if not isinstance(device, str):
            device_str = "cuda" if torch.cuda.is_available() else "cpu"

        device_obj = torch.device(device_str)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device_obj)
        model.eval()

        if hasattr(fid_score, "_compute_statistics_of_path"):
            stats = fid_score._compute_statistics_of_path(
                image_dir, model, batch_size, device_str, dims
            )
            mu = stats["mu"]
            sigma = stats["sigma"]
        else:
            from pytorch_fid.fid_score import get_activations

            image_files = (
                glob.glob(os.path.join(image_dir, "*.png"))
                + glob.glob(os.path.join(image_dir, "*.jpg"))
                + glob.glob(os.path.join(image_dir, "*.jpeg"))
            )
            activations = get_activations(
                image_files, model, batch_size, dims, device_str
            )
            mu = np.mean(activations, axis=0)
            sigma = np.cov(activations, rowvar=False)

        return mu, sigma
    except Exception as e:
        print(f"Error computing Inception stats: {e}")
        import traceback

        traceback.print_exc()
        return None


def calculate_fid(generated_images_dir, real_stats, device="cuda"):
    """Calculate FID between generated images and provided real image stats (mu, sigma)."""
    if not FID_AVAILABLE or real_stats is None:
        return None

    try:
        from pytorch_fid.fid_score import calculate_frechet_distance

        gen_stats = calculate_inception_stats_from_dir(
            generated_images_dir, device=device
        )
        if gen_stats is None:
            return None
        mu_gen, sigma_gen = gen_stats
        mu_real, sigma_real = real_stats

        fid_val = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        return fid_val
    except Exception as e:
        print(f"Error calculating FID: {e}")
        import traceback

        traceback.print_exc()
        return None


def prepare_real_images_and_stats(coco_data, image_size, output_dir, device="cuda"):
    """Resize/save COCO validation images once and cache their Inception stats for reuse."""
    real_images_dir = os.path.join(output_dir, f"coco_real_images_{image_size}")
    os.makedirs(real_images_dir, exist_ok=True)

    real_image_paths = []
    for idx, data_pair in enumerate(coco_data):
        target_path = os.path.join(real_images_dir, f"real_{idx:04d}.png")
        real_image_paths.append(target_path)
        if os.path.exists(target_path):
            continue
        src_path = data_pair["image_path"]
        try:
            img = Image.open(src_path).convert("RGB")
            img = img.resize((image_size, image_size), Image.LANCZOS)
            img.save(target_path)
        except Exception as e:
            print(f"  Warning: failed to prepare real image {src_path}: {e}")

    stats_dir = os.path.join(output_dir, "coco_stats")
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(
        stats_dir, f"coco_val_stats_{image_size}_{len(real_image_paths)}.npz"
    )

    if os.path.exists(stats_path):
        try:
            cached = np.load(stats_path)
            mu = cached["mu"]
            sigma = cached["sigma"]
            return real_images_dir, (mu, sigma), stats_path
        except Exception:
            print("  Warning: failed to load cached COCO stats, recomputing...")

    stats = calculate_inception_stats_from_dir(real_images_dir, device=device)
    if stats is not None:
        mu, sigma = stats
        np.savez(stats_path, mu=mu, sigma=sigma)
        return real_images_dir, (mu, sigma), stats_path

    return real_images_dir, None, stats_path


def load_clip_model(device="cuda"):
    """Load CLIP model and preprocess if available."""
    if not CLIP_AVAILABLE:
        return None, None
    try:
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        return model, preprocess
    except Exception as e:
        print(f"Warning: failed to load CLIP model: {e}")
        return None, None


def calculate_clip_similarity(image, text, clip_model, clip_preprocess, device="cuda"):
    """Compute CLIP cosine similarity between an image and a text prompt (higher is better)."""
    if clip_model is None or clip_preprocess is None:
        return None

    try:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        image_input = clip_preprocess(image).unsqueeze(0).to(device)
        text_input = clip.tokenize([text]).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features * text_features).sum().item()

        return similarity
    except Exception as e:
        print(f"Error calculating CLIP similarity: {e}")
        return None


def evaluate_single_model(
    lora_path=None,
    output_dir="evaluation_results",
    coco_data=None,
    device="cuda",
    image_size=256,
    model_name="original",
    real_images_dir=None,
    real_stats=None,
    clip_model=None,
    clip_preprocess=None,
):
    """
    Evaluate a single SDXL model (original or fine-tuned) on COCO data.

    Args:
        lora_path: Path to LoRA checkpoint (None = original model)
        output_dir: Output directory for results
        coco_data: List of COCO data pairs (prompt, image_path, image_id)
        device: Device to run on
        image_size: Image size (default: 256x256 as in paper)
        model_name: Name for the model ("original" or "finetuned")
        real_images_dir: Directory containing resized real COCO images for this run
        real_stats: Tuple (mu, sigma) of cached COCO stats for FID
        clip_model, clip_preprocess: Loaded CLIP model/preprocess for text-image similarity

    Returns:
        Dict with metrics: PSNR, SSIM, LPIPS, FID, CLIP
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {'Fine-tuned SDXL' if lora_path else 'Original SDXL'}")
    print(f"{'='*60}\n")

    # Create output directory
    eval_output_dir = os.path.join(output_dir, f"coco_{model_name}")
    os.makedirs(eval_output_dir, exist_ok=True)

    # Create pipeline cache
    print("Creating pipeline cache...")
    pipeline_cache = create_pipeline_cache(
        lora_path=lora_path,
        use_refiner=True,
        device=device,
    )
    print("✓ Pipeline cache created")

    # Generate images and calculate metrics
    print(f"\nGenerating {len(coco_data)} images at {image_size}x{image_size}...")

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    clip_scores = []
    generated_images_dir = os.path.join(eval_output_dir, "generated")
    os.makedirs(generated_images_dir, exist_ok=True)

    for idx, data_pair in enumerate(tqdm(coco_data, desc="Generating and evaluating")):
        prompt = data_pair["prompt"]
        if real_images_dir:
            real_image_path = os.path.join(real_images_dir, f"real_{idx:04d}.png")
        else:
            real_image_path = data_pair["image_path"]

        # Load real COCO image (already resized if prepared)
        real_image = Image.open(real_image_path).convert("RGB")
        if real_image.size != (image_size, image_size):
            real_image = real_image.resize((image_size, image_size), Image.LANCZOS)

        # Generate image with SDXL at 256x256
        generated_image = generate_image_in_memory(
            prompt=prompt,
            pipeline_cache=pipeline_cache,
            num_inference_steps=50,
            seed=42 + idx,
            height=image_size,
            width=image_size,
        )

        if generated_image is None:
            print(f"  Warning: Failed to generate image for prompt {idx}")
            continue

        # Resize generated image to 256x256
        generated_image = generated_image.resize(
            (image_size, image_size), Image.LANCZOS
        )

        # Save generated image
        generated_image_path = os.path.join(
            generated_images_dir, f"generated_{idx:04d}.png"
        )
        generated_image.save(generated_image_path)

        # Calculate metrics
        psnr_val = calculate_psnr(real_image, generated_image)
        ssim_val = calculate_ssim(real_image, generated_image)
        lpips_val = calculate_lpips(real_image, generated_image, device=device)
        clip_val = calculate_clip_similarity(
            generated_image, prompt, clip_model, clip_preprocess, device=device
        )

        if psnr_val is not None:
            psnr_scores.append(psnr_val)
        if ssim_val is not None:
            ssim_scores.append(ssim_val)
        if lpips_val is not None:
            lpips_scores.append(lpips_val)
        if clip_val is not None:
            clip_scores.append(clip_val)

    # Calculate FID using cached real stats
    print(f"\nCalculating FID against COCO real images...")
    fid_val = calculate_fid(generated_images_dir, real_stats, device=device)

    # Calculate average metrics
    avg_psnr = np.mean(psnr_scores) if psnr_scores else None
    avg_ssim = np.mean(ssim_scores) if ssim_scores else None
    avg_lpips = np.mean(lpips_scores) if lpips_scores else None
    avg_clip = np.mean(clip_scores) if clip_scores else None

    return {
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "lpips": avg_lpips,
        "fid": fid_val,
        "clip": avg_clip,
        "num_samples": len(psnr_scores),
        "model_name": "Fine-tuned SDXL" if lora_path else "Original SDXL",
    }


def evaluate_sdxl_coco(
    lora_path=None,
    output_dir="evaluation_results",
    num_prompts=None,
    device="cuda",
    image_size=256,
):
    """
    Evaluate SDXL models on COCO 2017 validation split following the SDXL paper.
    Always evaluates original SDXL, and fine-tuned if lora_path is provided.

    Args:
        lora_path: Path to LoRA checkpoint (if provided, will evaluate both original and fine-tuned)
        output_dir: Output directory for results
        num_prompts: Number of prompts to evaluate (None = all)
        device: Device to run on
        image_size: Image size (default: 256x256 as in paper)

    Returns:
        Dict with metrics for both models
    """
    print(f"\n{'='*60}")
    print(f"SDXL Evaluation on COCO 2017 Validation Split")
    print(f"Following SDXL paper: https://arxiv.org/pdf/2307.01952")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Number of samples: {num_prompts if num_prompts else 'All'}")
    print(f"{'='*60}\n")

    # Load COCO 2017 validation data
    coco_data = load_coco_2017_validation(
        download_dir="coco_eval", num_prompts=num_prompts, seed=42
    )

    if not coco_data:
        raise ValueError("No COCO data loaded. Please check COCO download.")

    os.makedirs(output_dir, exist_ok=True)

    # Prepare real images (resized) and cached stats for FID
    print("\nPreparing COCO real images and cached statistics...")
    real_images_dir, real_stats, real_stats_path = prepare_real_images_and_stats(
        coco_data=coco_data,
        image_size=image_size,
        output_dir=output_dir,
        device=device,
    )
    if real_stats is None:
        print("  Warning: Unable to compute COCO stats; FID will be skipped.")
    else:
        print(f"  Cached COCO stats at {real_stats_path}")

    # Load CLIP model once for text-image similarity
    clip_model, clip_preprocess = load_clip_model(device=device)

    # Always evaluate original SDXL first
    print(f"\n{'='*80}")
    print(f"Step 1/2: Evaluating Original SDXL Model")
    print(f"{'='*80}")
    original_results = evaluate_single_model(
        lora_path=None,
        output_dir=output_dir,
        coco_data=coco_data,
        device=device,
        image_size=image_size,
        model_name="original",
        real_images_dir=real_images_dir,
        real_stats=real_stats,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
    )

    # Evaluate fine-tuned model if lora_path is provided
    finetuned_results = None
    if lora_path:
        print(f"\n{'='*80}")
        print(f"Step 2/2: Evaluating Fine-tuned SDXL Model")
        print(f"{'='*80}")
        finetuned_results = evaluate_single_model(
            lora_path=lora_path,
            output_dir=output_dir,
            coco_data=coco_data,
            device=device,
            image_size=image_size,
            model_name="finetuned",
            real_images_dir=real_images_dir,
            real_stats=real_stats,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
        )

    # Print comparison results
    print(f"\n{'='*80}")
    print(
        f"Evaluation Results Summary (COCO 2017 Validation, {image_size}x{image_size})"
    )
    print(f"{'='*80}")
    print(f"\nFollowing SDXL paper Table 3 format:\n")
    print(f"{'Metric':<10} {'Original SDXL':<20} {'Fine-tuned SDXL':<20}")
    print(f"{'-'*50}")

    # PSNR
    orig_psnr = f"{original_results['psnr']:.2f}" if original_results["psnr"] else "N/A"
    fin_psnr = (
        f"{finetuned_results['psnr']:.2f}"
        if finetuned_results and finetuned_results["psnr"]
        else "N/A"
    )
    print(f"{'PSNR ↑':<10} {orig_psnr:<20} {fin_psnr:<20}")

    # SSIM
    orig_ssim = f"{original_results['ssim']:.3f}" if original_results["ssim"] else "N/A"
    fin_ssim = (
        f"{finetuned_results['ssim']:.3f}"
        if finetuned_results and finetuned_results["ssim"]
        else "N/A"
    )
    print(f"{'SSIM ↑':<10} {orig_ssim:<20} {fin_ssim:<20}")

    # LPIPS
    orig_lpips = (
        f"{original_results['lpips']:.2f}" if original_results["lpips"] else "N/A"
    )
    fin_lpips = (
        f"{finetuned_results['lpips']:.2f}"
        if finetuned_results and finetuned_results["lpips"]
        else "N/A"
    )
    print(f"{'LPIPS ↓':<10} {orig_lpips:<20} {fin_lpips:<20}")

    # FID
    orig_fid = f"{original_results['fid']:.1f}" if original_results["fid"] else "N/A"
    fin_fid = (
        f"{finetuned_results['fid']:.1f}"
        if finetuned_results and finetuned_results["fid"]
        else "N/A"
    )
    print(f"{'FID ↓':<10} {orig_fid:<20} {fin_fid:<20}")

    # CLIP
    orig_clip = f"{original_results['clip']:.3f}" if original_results["clip"] else "N/A"
    fin_clip = (
        f"{finetuned_results['clip']:.3f}"
        if finetuned_results and finetuned_results["clip"]
        else "N/A"
    )
    print(f"{'CLIP ↑':<10} {orig_clip:<20} {fin_clip:<20}")

    print(f"\nNumber of samples: {original_results['num_samples']}")
    print(f"{'='*80}\n")

    # Save comparison results
    results_file = os.path.join(output_dir, "comparison_results.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("SDXL Evaluation Results (COCO 2017 Validation)\n")
        f.write("Following SDXL paper: https://arxiv.org/pdf/2307.01952\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Image size: {image_size}x{image_size}\n")
        f.write(f"Number of samples: {original_results['num_samples']}\n\n")
        f.write("Metrics Comparison (Table 3 format):\n\n")
        f.write(f"{'Metric':<10} {'Original SDXL':<20} {'Fine-tuned SDXL':<20}\n")
        f.write(f"{'-'*50}\n")
        f.write(f"{'PSNR ↑':<10} {orig_psnr:<20} {fin_psnr:<20}\n")
        f.write(f"{'SSIM ↑':<10} {orig_ssim:<20} {fin_ssim:<20}\n")
        f.write(f"{'LPIPS ↓':<10} {orig_lpips:<20} {fin_lpips:<20}\n")
        f.write(f"{'FID ↓':<10} {orig_fid:<20} {fin_fid:<20}\n")
        f.write(f"{'CLIP ↑':<10} {orig_clip:<20} {fin_clip:<20}\n")

    # Save individual model results
    for results, model_name in [
        (original_results, "original"),
        (finetuned_results, "finetuned"),
    ]:
        if results is None:
            continue
        eval_output_dir = os.path.join(output_dir, f"coco_{model_name}")
        results_file = os.path.join(eval_output_dir, "results.txt")
        with open(results_file, "w", encoding="utf-8") as f:
            f.write(f"SDXL Evaluation Results (COCO 2017 Validation)\n")
            f.write("Following SDXL paper: https://arxiv.org/pdf/2307.01952\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {results['model_name']}\n")
            f.write(f"Image size: {image_size}x{image_size}\n")
            f.write(f"Number of samples: {results['num_samples']}\n\n")
            f.write("Metrics (Table 3 format):\n")
            f.write(
                f"  PSNR ↑:  {results['psnr']:.2f}\n"
                if results["psnr"]
                else "  PSNR: N/A\n"
            )
            f.write(
                f"  SSIM ↑:  {results['ssim']:.3f}\n"
                if results["ssim"]
                else "  SSIM: N/A\n"
            )
            f.write(
                f"  LPIPS ↓: {results['lpips']:.2f}\n"
                if results["lpips"]
                else "  LPIPS: N/A\n"
            )
            f.write(
                f"  FID ↓:   {results['fid']:.1f}\n"
                if results["fid"]
                else "  FID: N/A\n"
            )
            f.write(
                f"  CLIP ↑:  {results['clip']:.3f}\n"
                if results["clip"]
                else "  CLIP: N/A\n"
            )

    print(f"Results saved to: {results_file}")
    print(
        f"Individual model results saved to: {output_dir}/coco_original/ and {output_dir}/coco_finetuned/"
    )

    return {"original": original_results, "finetuned": finetuned_results}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SDXL models on COCO 2017 validation split (following SDXL paper)"
    )

    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA checkpoint (e.g., checkpoints/checkpoint-6000). If not provided, tests original model.",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=None,
        help="Number of prompts to evaluate (default: all). Use a small number (e.g., 5) for testing.",
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
        "--image_size",
        type=int,
        default=256,
        help="Image size for evaluation (default: 256, as in SDXL paper)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run evaluation
    results = evaluate_sdxl_coco(
        lora_path=args.lora_path,
        output_dir=args.output_dir,
        num_prompts=args.num_prompts,
        device=args.device,
        image_size=args.image_size,
    )

    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
