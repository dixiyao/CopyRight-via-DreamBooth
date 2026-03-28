#!/usr/bin/env python3
"""
Comprehensive evaluation pipeline for DreamBooth copyright experiments.

Flow:
1) Build prompt_base and image_base
    - Randomly generate exactly 50 prompts with Gemini model
    - Keep and reuse this fixed 50-prompt set for all tests
   - Generate base SDXL images for sampled prompts (image_base)

2) Evaluate W0
    - Definition: base model + lora_path
   - prompt_base -> W0 images, compare with image_base
   - cp_dataset prompts -> W0 images, compare with cp_dataset images

3) Evaluate Wr
    - Definition: base model + lora_path + rl_checkpoint
   - prompt_base -> Wr images, compare with image_base
   - cp_dataset prompts -> Wr images, compare with cp_dataset images

4) Evaluate Wc
    - Definition: base model + lora_path + rl_checkpoint + continue_lora_path
   - prompt_base -> Wc images, compare with image_base
   - cp_dataset prompts -> Wc images, compare with cp_dataset images

Metrics:
- PSNR
- SSIM
- FID
- SSCD (Disc-MixUp proxy)
- CLIP image similarity (ViT-B/16)
- DINO image similarity (ViT-B/16)

Additionally:
- CLIP text-image score is computed for image_base against prompt_base.
"""

import argparse
import csv
import difflib
import importlib
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_prompt_csv(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_prompt_csv(csv_path: str, prompts: Sequence[str], image_names: Sequence[str]) -> None:
    ensure_dir(os.path.dirname(csv_path) or ".")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "img"])
        writer.writeheader()
        for p, n in zip(prompts, image_names):
            writer.writerow({"prompt": p, "img": n})


def pil_to_tensor01(image: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).to(device)
    return ten


def tensor01_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def pairwise_psnr_ssim(
    refs: Sequence[Image.Image],
    gens: Sequence[Image.Image],
    device: torch.device,
) -> Dict[str, float]:
    assert len(refs) == len(gens)
    psnr_vals = []
    ssim_vals = []

    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)

    for r, g in zip(refs, gens):
        if g.size != r.size:
            g = g.resize(r.size, Image.BICUBIC)

        rt = pil_to_tensor01(r, device)
        gt = pil_to_tensor01(g, device)

        mse = F.mse_loss(gt, rt).item()
        if mse <= 1e-12:
            psnr = 100.0
        else:
            psnr = -10.0 * math.log10(mse)
        psnr_vals.append(psnr)

        # Global SSIM (compact implementation)
        mu_x = rt.mean()
        mu_y = gt.mean()
        sigma_x = ((rt - mu_x) ** 2).mean()
        sigma_y = ((gt - mu_y) ** 2).mean()
        sigma_xy = ((rt - mu_x) * (gt - mu_y)).mean()

        ssim_num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        ssim = (ssim_num / (ssim_den + 1e-12)).item()
        ssim_vals.append(ssim)

    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
    }


class CLIPScorer:
    def __init__(self, device: torch.device, model_name: str = "openai/clip-vit-base-patch16"):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def text_image_score(self, prompts: Sequence[str], images: Sequence[Image.Image], batch_size: int = 8) -> float:
        vals = []
        for i in range(0, len(images), batch_size):
            p = list(prompts[i : i + batch_size])
            ims = list(images[i : i + batch_size])
            inp = self.processor(text=p, images=ims, return_tensors="pt", padding=True)
            inp = {k: v.to(self.device) for k, v in inp.items()}

            image_features = self.model.get_image_features(pixel_values=inp["pixel_values"])
            text_features = self.model.get_text_features(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
            )
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            sims = (image_features * text_features).sum(dim=-1)
            vals.extend(sims.detach().cpu().tolist())
        return float(np.mean(vals))

    @torch.no_grad()
    def image_embeddings(self, images: Sequence[Image.Image], batch_size: int = 8) -> torch.Tensor:
        embs = []
        for i in range(0, len(images), batch_size):
            ims = list(images[i : i + batch_size])
            inp = self.processor(images=ims, return_tensors="pt")
            inp = {k: v.to(self.device) for k, v in inp.items()}
            features = self.model.get_image_features(pixel_values=inp["pixel_values"])
            features = F.normalize(features, dim=-1)
            embs.append(features.detach().cpu())
        return torch.cat(embs, dim=0)


class DINOEmbedder:
    def __init__(self, device: torch.device, model_name: str = "facebook/dino-vitb16"):
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def image_embeddings(self, images: Sequence[Image.Image], batch_size: int = 8) -> torch.Tensor:
        embs = []
        for i in range(0, len(images), batch_size):
            ims = list(images[i : i + batch_size])
            inp = self.processor(images=ims, return_tensors="pt")
            inp = {k: v.to(self.device) for k, v in inp.items()}
            out = self.model(**inp)
            cls = out.last_hidden_state[:, 0, :]
            cls = F.normalize(cls, dim=-1)
            embs.append(cls.detach().cpu())
        return torch.cat(embs, dim=0)


class InceptionFeatureExtractor:
    def __init__(self, device: torch.device):
        from torchvision.models import Inception_V3_Weights, inception_v3

        self.device = device
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights, transform_input=False)
        model.fc = nn.Identity()
        model.eval().to(device)
        self.model = model
        self.transforms = weights.transforms()

    @torch.no_grad()
    def extract(self, images: Sequence[Image.Image], batch_size: int = 8) -> np.ndarray:
        feats = []
        for i in range(0, len(images), batch_size):
            ims = images[i : i + batch_size]
            batch = torch.stack([self.transforms(im.convert("RGB")) for im in ims], dim=0).to(self.device)
            out = self.model(batch)
            if out.ndim > 2:
                out = out.view(out.size(0), -1)
            feats.append(out.detach().cpu().numpy())
        return np.concatenate(feats, axis=0)


def _sqrtm_psd(mat: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    mat = (mat + mat.T) / 2.0
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, a_min=eps, a_max=None)
    sqrt_vals = np.sqrt(vals)
    return (vecs * sqrt_vals) @ vecs.T


def compute_fid_from_features(feats1: np.ndarray, feats2: np.ndarray) -> float:
    mu1 = np.mean(feats1, axis=0)
    mu2 = np.mean(feats2, axis=0)
    cov1 = np.cov(feats1, rowvar=False)
    cov2 = np.cov(feats2, rowvar=False)

    diff = mu1 - mu2
    cov1_sqrt = _sqrtm_psd(cov1)
    prod = cov1_sqrt @ cov2 @ cov1_sqrt
    covmean = _sqrtm_psd(prod)

    fid = float(diff @ diff + np.trace(cov1) + np.trace(cov2) - 2.0 * np.trace(covmean))
    if fid < 0:
        fid = 0.0
    return fid


def cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return float((a * b).sum(dim=-1).mean().item())


def compute_sscd_disc_mixup_proxy(
    ref_images: Sequence[Image.Image],
    gen_images: Sequence[Image.Image],
    dino_embedder: DINOEmbedder,
) -> float:
    assert len(ref_images) == len(gen_images)

    mixed_images: List[Image.Image] = []
    for r, g in zip(ref_images, gen_images):
        if g.size != r.size:
            g = g.resize(r.size, Image.BICUBIC)
        rt = pil_to_tensor01(r, torch.device("cpu"))
        gt = pil_to_tensor01(g, torch.device("cpu"))
        mix = 0.5 * rt + 0.5 * gt
        mixed_images.append(tensor01_to_pil(mix))

    ref_emb = dino_embedder.image_embeddings(ref_images)
    gen_emb = dino_embedder.image_embeddings(gen_images)
    mix_emb = dino_embedder.image_embeddings(mixed_images)

    s1 = F.cosine_similarity(ref_emb, mix_emb, dim=-1)
    s2 = F.cosine_similarity(gen_emb, mix_emb, dim=-1)
    return float((0.5 * (s1 + s2)).mean().item())


def resolve_cp_dataset_path(cp_dataset: str) -> str:
    cp_dataset = os.path.expanduser(cp_dataset)
    csv_path = os.path.join(cp_dataset, "prompt.csv")
    img_dir = os.path.join(cp_dataset, "image")

    suggestion = ""
    parent_dir = os.path.dirname(cp_dataset) or "."
    dataset_name = os.path.basename(cp_dataset)
    if os.path.isdir(parent_dir):
        candidates = [
            d for d in os.listdir(parent_dir)
            if os.path.isdir(os.path.join(parent_dir, d))
        ]
        close = difflib.get_close_matches(dataset_name, candidates, n=3, cutoff=0.6)
        if close:
            suggestion = f" Did you mean: {', '.join(os.path.join(parent_dir, c) for c in close)} ?"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"cp_dataset prompt.csv not found: {csv_path}.{suggestion}")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"cp_dataset image directory not found: {img_dir}.{suggestion}")

    return cp_dataset


def load_cp_pairs(cp_dataset_path: str, n: int) -> Tuple[List[str], List[Image.Image], List[str]]:
    rows = read_prompt_csv(os.path.join(cp_dataset_path, "prompt.csv"))
    rows = rows[:n]

    prompts = []
    images = []
    names = []
    for row in rows:
        prompt = row["prompt"]
        img_name = row["img"]
        img_path = os.path.join(cp_dataset_path, "image", img_name)
        if not os.path.exists(img_path):
            continue
        prompts.append(prompt)
        images.append(Image.open(img_path).convert("RGB"))
        names.append(img_name)

    if len(prompts) == 0:
        raise RuntimeError(f"No valid samples found in {cp_dataset_path}")

    return prompts, images, names


def _parse_prompt_lines(text: str) -> List[str]:
    prompts = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        s = re.sub(r"^\d+[\).:-]\s*", "", s)
        s = s.strip("\"'")
        if len(s) >= 10:
            prompts.append(s)
    return prompts


def generate_random_prompts_with_gemini(
    model_name: str,
    api_key: str,
    n_prompts: int,
    seed: int,
) -> List[str]:
    from google import genai

    rng = random.Random(seed)
    client = genai.Client(api_key=api_key)

    instruction = (
        f"Generate {n_prompts} diverse, random, single-sentence image prompts. "
        "Each line should be one prompt. Avoid copyrighted character names, brand names, and artist names. "
        "Return only prompts, one per line, no extra commentary."
    )

    prompts: List[str] = []
    try:
        resp = client.models.generate_content(model=model_name, contents=instruction)
        text = getattr(resp, "text", None)
        if text is None and hasattr(resp, "candidates") and resp.candidates:
            text = resp.candidates[0].content.parts[0].text
        if text:
            prompts.extend(_parse_prompt_lines(text))
    except Exception as e:
        print(f"Warning: Gemini prompt generation failed ({model_name}): {e}")

    fallback_subjects = [
        "a red bicycle by a lake at sunset",
        "a snowy mountain village under moonlight",
        "a wooden boat drifting in a misty river",
        "a colorful market street after rain",
        "a futuristic city skyline at dawn",
        "a cozy library with warm ambient lighting",
        "a desert road stretching to the horizon",
        "a waterfall hidden in a tropical forest",
        "a quiet train station in early morning fog",
        "a lighthouse on a rocky cliff during storm",
    ]

    while len(prompts) < n_prompts:
        prompts.append(rng.choice(fallback_subjects))

    # Deduplicate while preserving order
    dedup = []
    seen = set()
    for p in prompts:
        key = p.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    prompts = dedup

    while len(prompts) < n_prompts:
        prompts.append(rng.choice(fallback_subjects))

    return prompts[:n_prompts]


def generate_images_with_callback(
    prompts: Sequence[str],
    output_dir: str,
    prefix: str,
    generator_fn: Callable[[str, int], Image.Image],
    auto_resume: bool = False,
) -> Tuple[List[Image.Image], List[str]]:
    ensure_dir(output_dir)
    images = []
    names = []
    for i, prompt in enumerate(tqdm(prompts, desc=f"Generating {prefix}"), start=1):
        name = f"{prefix}_{i:04d}.png"
        path = os.path.join(output_dir, name)
        if auto_resume and os.path.exists(path):
            image = Image.open(path).convert("RGB")
        else:
            image = generator_fn(prompt, i)
            image.save(path)
        images.append(image)
        names.append(name)
    return images, names


@dataclass
class MetricEngines:
    clip: CLIPScorer
    dino: DINOEmbedder
    inception: InceptionFeatureExtractor


def compute_all_metrics(
    ref_images: Sequence[Image.Image],
    gen_images: Sequence[Image.Image],
    engines: MetricEngines,
    device: torch.device,
) -> Dict[str, float]:
    if len(ref_images) != len(gen_images):
        raise ValueError(f"ref_images ({len(ref_images)}) and gen_images ({len(gen_images)}) size mismatch")

    basic = pairwise_psnr_ssim(ref_images, gen_images, device)

    clip_ref = engines.clip.image_embeddings(ref_images)
    clip_gen = engines.clip.image_embeddings(gen_images)
    dino_ref = engines.dino.image_embeddings(ref_images)
    dino_gen = engines.dino.image_embeddings(gen_images)

    clip_img_sim = cosine_mean(clip_ref, clip_gen)
    dino_img_sim = cosine_mean(dino_ref, dino_gen)

    fid_ref = engines.inception.extract(ref_images)
    fid_gen = engines.inception.extract(gen_images)
    fid = compute_fid_from_features(fid_ref, fid_gen)

    sscd_disc_mixup = compute_sscd_disc_mixup_proxy(ref_images, gen_images, engines.dino)

    result = {
        "psnr": basic["psnr"],
        "ssim": basic["ssim"],
        "fid": float(fid),
        "sscd_disc_mixup": float(sscd_disc_mixup),
        "clip_vitb16": float(clip_img_sim),
        "dino_vitb16": float(dino_img_sim),
    }
    return result


def run_generator_main(module_name: str, cli_args: List[str]) -> None:
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise AttributeError(f"Module {module_name} has no main()")

    prev_argv = sys.argv
    try:
        sys.argv = [f"{module_name}.py", *cli_args]
        module.main()
    finally:
        sys.argv = prev_argv


def run_generate_and_load(module_name: str, cli_args: List[str], output_path: str) -> Image.Image:
    run_generator_main(module_name, cli_args)
    return Image.open(output_path).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Evaluate W0 / Wr / Wc models with base + CP datasets")

    parser.add_argument("--gemini_api_key", type=str, default=None, help="Gemini API key for prompt generation")
    parser.add_argument("--prompt_model", type=str, default="gemini-3.0-pro", help="Prompt generation model (e.g., gemini 3.0)")

    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--cp_dataset", type=str, default="data/cp_chikawa_new", help="CP dataset directory (contains prompt.csv and image/)")

    parser.add_argument("--lora_path", type=str, required=True, help="LoRA path used by W0/Wr/Wc")
    parser.add_argument("--rl_checkpoint", type=str, required=True, help="Single RL checkpoint path used by Wr and Wc")
    parser.add_argument("--continue_lora_path", type=str, required=True, help="Continue LoRA path used by Wc")

    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--use_refiner", action="store_true")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="evaluation_runs")
    parser.add_argument("--auto_resume", action="store_true", help="Reuse existing images/prompt csv in output_dir and only generate missing ones")

    args = parser.parse_args()

    fixed_eval_samples = 50

    set_seed(args.seed)

    if torch.cuda.is_available():
        runtime_device = "cuda"
    elif str(args.device).startswith("cuda"):
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        runtime_device = "cpu"
    else:
        runtime_device = str(args.device)

    device = torch.device(runtime_device)

    ensure_dir(args.output_dir)
    print(f"Output directory: {args.output_dir}")
    print(f"Runtime device (CUDA-preferred): {runtime_device}")
    print(f"Auto-resume enabled: {args.auto_resume}")

    # 0) Load CP dataset
    cp_dataset_path = resolve_cp_dataset_path(args.cp_dataset)
    cp_prompts, cp_ref_images, _ = load_cp_pairs(cp_dataset_path, fixed_eval_samples)
    print(f"Loaded cp_dataset: {cp_dataset_path}, samples={len(cp_prompts)}")

    # 1) Build prompt_base and image_base
    base_out_dir = os.path.join(args.output_dir, "base")
    base_img_dir = os.path.join(base_out_dir, "images")
    base_prompt_csv = os.path.join(base_out_dir, "prompt_base.csv")
    ensure_dir(base_img_dir)

    # Auto-resume: load prompt_base from CSV if it already exists
    if args.auto_resume and os.path.exists(base_prompt_csv):
        print(f"Resuming: loading existing prompt_base from {base_prompt_csv}")
        rows = read_prompt_csv(base_prompt_csv)
        prompt_base = [r["prompt"] for r in rows][:fixed_eval_samples]
    elif args.gemini_api_key:
        prompt_base = generate_random_prompts_with_gemini(
            model_name=args.prompt_model,
            api_key=args.gemini_api_key,
            n_prompts=fixed_eval_samples,
            seed=args.seed,
        )
    else:
        print("Warning: --gemini_api_key not provided. Using fallback random prompts.")
        prompt_base = generate_random_prompts_with_gemini(
            model_name=args.prompt_model,
            api_key="",
            n_prompts=fixed_eval_samples,
            seed=args.seed,
        )

    prompt_base = prompt_base[:fixed_eval_samples]

    base_raw_dir = os.path.join(args.output_dir, "base", "raw_outputs")
    ensure_dir(base_raw_dir)

    def base_gen(prompt: str, idx: int) -> Image.Image:
        output_path = os.path.join(base_raw_dir, f"base_raw_{idx:04d}.png")
        cli_args = [
            "--prompt", prompt,
            "--output_path", output_path,
            "--base_model", args.base_model,
            "--num_inference_steps", str(args.num_inference_steps),
            "--guidance_scale", str(args.guidance_scale),
            "--height", str(args.height),
            "--width", str(args.width),
            "--device", runtime_device,
            "--seed", str(args.seed + idx),
        ]
        if args.use_refiner:
            cli_args.append("--use_refiner")
        return run_generate_and_load("generate", cli_args, output_path)

    image_base, base_image_names = generate_images_with_callback(
        prompts=prompt_base,
        output_dir=base_img_dir,
        prefix="image_base",
        generator_fn=base_gen,
        auto_resume=args.auto_resume,
    )

    save_prompt_csv(base_prompt_csv, prompt_base, base_image_names)

    # Metrics engines
    print("Loading metric backbones (CLIP / DINO / Inception)...")
    clip_scorer = CLIPScorer(device=device)
    dino_embedder = DINOEmbedder(device=device)
    inception_extractor = InceptionFeatureExtractor(device=device)
    metric_engines = MetricEngines(
        clip=clip_scorer,
        dino=dino_embedder,
        inception=inception_extractor,
    )

    # Step 1: CLIP score for image_base
    clip_score_image_base = clip_scorer.text_image_score(prompt_base, image_base)

    results: Dict[str, Dict] = {
        "config": vars(args),
        "cp_dataset_resolved": cp_dataset_path,
        "num_cp_samples": len(cp_prompts),
        "num_base_samples": len(prompt_base),
        "step1": {
            "clip_score_image_base": clip_score_image_base,
        },
        "w0": {},
        "wr": {},
        "wc": {},
    }

    # 2) W0
    print("Generating W0 via generate_robust.py main()...")
    w0_raw_dir = os.path.join(args.output_dir, "w0", "raw_outputs")
    ensure_dir(w0_raw_dir)

    def w0_gen(prompt: str, idx: int) -> Image.Image:
        output_path = os.path.join(w0_raw_dir, f"w0_raw_{idx:04d}.png")
        cli_args = [
            "--lora_path", args.lora_path,
            "--base_model", args.base_model,
            "--prompt", prompt,
            "--output_path", output_path,
            "--num_inference_steps", str(args.num_inference_steps),
            "--guidance_scale", str(args.guidance_scale),
            "--height", str(args.height),
            "--width", str(args.width),
            "--device", runtime_device,
            "--seed", str(args.seed + 1000 + idx),
        ]
        if args.use_refiner:
            cli_args.append("--use_refiner")
        return run_generate_and_load("generate_robust", cli_args, output_path)

    w0_base_dir = os.path.join(args.output_dir, "w0", "base")
    w0_cp_dir = os.path.join(args.output_dir, "w0", "cp")
    w0_base_images, _ = generate_images_with_callback(prompt_base, w0_base_dir, "w0_base", w0_gen, auto_resume=args.auto_resume)
    w0_cp_images, _ = generate_images_with_callback(cp_prompts, w0_cp_dir, "w0_cp", w0_gen, auto_resume=args.auto_resume)

    results["w0"]["prompt_base_vs_image_base"] = compute_all_metrics(
        image_base,
        w0_base_images,
        metric_engines,
        device,
    )
    results["w0"]["cp_prompts_vs_cp_images"] = compute_all_metrics(
        cp_ref_images,
        w0_cp_images,
        metric_engines,
        device,
    )

    # 3) Wr
    print("Generating Wr via generate_rl.py main()...")
    wr_raw_dir = os.path.join(args.output_dir, "wr", "raw_outputs")
    ensure_dir(wr_raw_dir)

    def wr_gen(prompt: str, idx: int) -> Image.Image:
        output_path = os.path.join(wr_raw_dir, f"wr_raw_{idx:04d}.png")
        cli_args = [
            "--rl_checkpoint", args.rl_checkpoint,
            "--lora_path", args.lora_path,
            "--base_model", args.base_model,
            "--prompt", prompt,
            "--output_path", output_path,
            "--num_inference_steps", str(args.num_inference_steps),
            "--guidance_scale", str(args.guidance_scale),
            "--height", str(args.height),
            "--width", str(args.width),
            "--device", runtime_device,
            "--seed", str(args.seed + 2000 + idx),
        ]
        if args.use_refiner:
            cli_args.append("--use_refiner")
        return run_generate_and_load("generate_rl", cli_args, output_path)

    wr_base_dir = os.path.join(args.output_dir, "wr", "base")
    wr_cp_dir = os.path.join(args.output_dir, "wr", "cp")
    wr_base_images, _ = generate_images_with_callback(prompt_base, wr_base_dir, "wr_base", wr_gen, auto_resume=args.auto_resume)
    wr_cp_images, _ = generate_images_with_callback(cp_prompts, wr_cp_dir, "wr_cp", wr_gen, auto_resume=args.auto_resume)

    results["wr"]["prompt_base_vs_image_base"] = compute_all_metrics(
        image_base,
        wr_base_images,
        metric_engines,
        device,
    )
    results["wr"]["cp_prompts_vs_cp_images"] = compute_all_metrics(
        cp_ref_images,
        wr_cp_images,
        metric_engines,
        device,
    )

    # 4) Wc
    print("Generating Wc via generate_continue.py main()...")
    wc_raw_dir = os.path.join(args.output_dir, "wc", "raw_outputs")
    ensure_dir(wc_raw_dir)

    def wc_gen(prompt: str, idx: int) -> Image.Image:
        output_path = os.path.join(wc_raw_dir, f"wc_raw_{idx:04d}.png")
        cli_args = [
            "--continue_lora_path", args.continue_lora_path,
            "--rl_checkpoint", args.rl_checkpoint,
            "--lora_path", args.lora_path,
            "--base_model", args.base_model,
            "--prompt", prompt,
            "--output_path", output_path,
            "--num_inference_steps", str(args.num_inference_steps),
            "--guidance_scale", str(args.guidance_scale),
            "--height", str(args.height),
            "--width", str(args.width),
            "--device", runtime_device,
            "--seed", str(args.seed + 3000 + idx),
        ]
        if args.use_refiner:
            cli_args.append("--use_refiner")
        return run_generate_and_load("generate_continue", cli_args, output_path)

    wc_base_dir = os.path.join(args.output_dir, "wc", "base")
    wc_cp_dir = os.path.join(args.output_dir, "wc", "cp")
    wc_base_images, _ = generate_images_with_callback(prompt_base, wc_base_dir, "wc_base", wc_gen, auto_resume=args.auto_resume)
    wc_cp_images, _ = generate_images_with_callback(cp_prompts, wc_cp_dir, "wc_cp", wc_gen, auto_resume=args.auto_resume)

    results["wc"]["prompt_base_vs_image_base"] = compute_all_metrics(
        image_base,
        wc_base_images,
        metric_engines,
        device,
    )
    results["wc"]["cp_prompts_vs_cp_images"] = compute_all_metrics(
        cp_ref_images,
        wc_cp_images,
        metric_engines,
        device,
    )

    result_path = os.path.join(args.output_dir, "results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n================ Evaluation Summary ================")
    print(f"Step1 CLIP(image_base): {results['step1']['clip_score_image_base']:.6f}")
    for model_key in ["w0", "wr", "wc"]:
        print(f"\n[{model_key.upper()}] prompt_base_vs_image_base")
        for k, v in results[model_key]["prompt_base_vs_image_base"].items():
            print(f"  {k}: {v:.6f}")
        print(f"[{model_key.upper()}] cp_prompts_vs_cp_images")
        for k, v in results[model_key]["cp_prompts_vs_cp_images"].items():
            print(f"  {k}: {v:.6f}")
    print("====================================================")
    print(f"Saved results to: {result_path}")


if __name__ == "__main__":
    main()
