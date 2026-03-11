#!/usr/bin/env python3
"""
DreamBooth Continue Fine-Tuning on top of RL checkpoint (T-LoRA + W_R backbone).

Loading sequence
---------------
1. Base SDXL weights
2. T-LoRA (dual-LoRA) checkpoint via ``--lora_path``:
   - lora1 loaded and frozen
   - lora2 zeroed and frozen
3. RL W_R from ``--rl_checkpoint``:
   - W_R is baked in-place into the base UNet attention weights
     (W_new = W_base + W_R), then all those params are frozen.
4. T-LoRA sigma mask hook attached so it auto-applies per timestep.
5. Fresh standard PEFT LoRA applied to UNet attention layers —
   only these new PEFT LoRA weights are trained.
6. Train on the continue dataset.

Checkpoints save only the new PEFT LoRA adapter, plus a
``backbone_info.json`` recording which T-LoRA + RL checkpoint was used.

Dataset structure (custom mode)
--------------------------------
  <data_dir>/image/      - training images
  <data_dir>/prompt.csv  - columns: prompt, img
"""

import argparse
import json
import os
import random
import shutil
import urllib.error
import urllib.request
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                          CLIPTokenizer)

from tlora_module import (DualLoRACrossAttnProcessor,
                          DualLoRATextLinearLayer,
                          attach_tlora_sigma_mask_hook,
                          build_dual_lora_attn_processors,
                          load_dual_lora_attn_state_dict,
                          load_text_encoder_dual_lora_weights)
from utils import (SimpleDreamBoothDataset, infinite_dataloader,
                   resolve_checkpoint_paths, simple_dreambooth_collate_fn)


def collect_trainable_layers(unet):
    """Return (full_name, weight) for attention projections covered by T-LoRA."""
    layers = []
    seen_ptrs = set()

    def maybe_add(name, weight):
        ptr = weight.data_ptr()
        if ptr in seen_ptrs:
            return
        seen_ptrs.add(ptr)
        layers.append((name, weight))

    for proc_name, proc in unet.attn_processors.items():
        if not isinstance(proc, DualLoRACrossAttnProcessor):
            continue

        attn_path = proc_name.split(".processor")[0]
        attn_layer = unet.get_submodule(attn_path)

        for proj_name in ["to_q", "to_k", "to_v"]:
            proj = getattr(attn_layer, proj_name, None)
            if proj is not None and hasattr(proj, "weight"):
                maybe_add(f"unet.{attn_path}.{proj_name}.weight", proj.weight)

        if hasattr(attn_layer, "to_out") and len(attn_layer.to_out) > 0:
            maybe_add(f"unet.{attn_path}.to_out.0.weight", attn_layer.to_out[0].weight)

    return layers


def zero_lora2_and_freeze_lora1(unet, text_encoder, text_encoder_2):
    """Zero lora2 and freeze both lora1/lora2 parameters in T-LoRA modules."""
    for proc in unet.attn_processors.values():
        if not isinstance(proc, DualLoRACrossAttnProcessor):
            continue

        for lora2_module in [proc.lora2_q, proc.lora2_k, proc.lora2_v, proc.lora2_out]:
            for p in lora2_module.parameters():
                p.data.zero_()
                p.requires_grad_(False)

        for lora1_module in [proc.lora1_q, proc.lora1_k, proc.lora1_v, proc.lora1_out]:
            for p in lora1_module.parameters():
                p.requires_grad_(False)

    for module in text_encoder.modules():
        if not isinstance(module, DualLoRATextLinearLayer):
            continue
        for p in module.lora2.parameters():
            p.data.zero_()
            p.requires_grad_(False)
        for p in module.lora1.parameters():
            p.requires_grad_(False)

    for module in text_encoder_2.modules():
        if not isinstance(module, DualLoRATextLinearLayer):
            continue
        for p in module.lora2.parameters():
            p.data.zero_()
            p.requires_grad_(False)
        for p in module.lora1.parameters():
            p.requires_grad_(False)


def load_cifar10_data():
    """Load CIFAR-10 training split with class name prompt."""
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError("torchvision required: pip install torchvision") from exc

    print("Loading CIFAR-10 training set into memory...")
    cifar10_dataset = datasets.CIFAR10(root="cifar10_data", train=True, download=True)
    class_names = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }

    data_pairs = []
    for image, label in cifar10_dataset:
        class_name = class_names.get(label, f"class_{label}")
        data_pairs.append({"prompt": f"a photo of a {class_name}", "image": image})

    print(f"Loaded {len(data_pairs)} CIFAR-10 training images")
    return data_pairs


def load_coco_2017_validation_data(download_dir="coco_eval", num_prompts=None):
    """Load COCO 2017 validation split with images and captions."""
    try:
        from pycocotools.coco import COCO
    except ImportError as exc:
        raise ImportError("pycocotools required: pip install pycocotools") from exc

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

    if not coco_ann_file:
        print("Downloading COCO annotations...")
        cache_file = os.path.join(download_dir, "captions_val2017.json")
        if not os.path.exists(cache_file):
            os.makedirs(download_dir, exist_ok=True)
            tmp_zip_path = os.path.join(download_dir, "annotations.zip")
            urllib.request.urlretrieve(
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                tmp_zip_path,
            )
            with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
                zip_ref.extract("annotations/captions_val2017.json", download_dir)
                extracted = os.path.join(download_dir, "annotations/captions_val2017.json")
                if os.path.exists(extracted):
                    shutil.move(extracted, cache_file)
            os.remove(tmp_zip_path)
        coco_ann_file = cache_file

    coco = COCO(coco_ann_file)
    img_ids = coco.getImgIds()
    coco_images_dir = os.path.join(download_dir, "val2017")
    os.makedirs(coco_images_dir, exist_ok=True)

    data_pairs = []
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        local_image_path = os.path.join(coco_images_dir, file_name)

        if not os.path.exists(local_image_path):
            try:
                urllib.request.urlretrieve(
                    f"http://images.cocodataset.org/val2017/{file_name}",
                    local_image_path,
                )
            except (urllib.error.URLError, OSError, IOError) as e:
                print(f"  Warning: Could not download {file_name}: {e}")
                continue

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        captions = [ann["caption"].strip() for ann in anns if "caption" in ann]
        if captions:
            data_pairs.append({"prompt": captions[0], "image_path": local_image_path})

    if num_prompts is not None and len(data_pairs) > num_prompts:
        random.seed(42)
        random.shuffle(data_pairs)
        data_pairs = data_pairs[:num_prompts]

    print(f"Loaded {len(data_pairs)} COCO image-caption pairs")
    return data_pairs


class PromptImagePairsDataset(Dataset):
    """Dataset from prompt/image pairs (supports in-memory PIL images)."""

    def __init__(
        self,
        data_pairs,
        tokenizer,
        tokenizer_2,
        size=1024,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.data = data_pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        if "image_path" in item:
            image = Image.open(item["image_path"])
        elif "image" in item:
            image = item["image"]
        else:
            raise ValueError(f"Item must have either 'image_path' or 'image' key: {item}")

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize((self.size, self.size), resample=Image.BICUBIC)
        if self.center_crop:
            crop_size = min(image.size)
            image = image.crop(
                (
                    (image.size[0] - crop_size) // 2,
                    (image.size[1] - crop_size) // 2,
                    (image.size[0] + crop_size) // 2,
                    (image.size[1] + crop_size) // 2,
                )
            )

        image = np.array(image).astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        prompt_ids = self.tokenizer(
            item["prompt"],
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        prompt_ids_2 = self.tokenizer_2(
            item["prompt"],
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        return {
            "pixel_values": image,
            "input_ids": prompt_ids,
            "input_ids_2": prompt_ids_2,
        }


def collate_fn(examples):
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]),
        "input_ids": torch.stack([example["input_ids"] for example in examples]),
        "input_ids_2": torch.stack([example["input_ids_2"] for example in examples]),
    }


def save_checkpoint(
    unet,
    output_dir,
    step,
    backbone_info=None,
    checkpoints_total_limit=None,
    accelerator=None,
):
    """Save checkpoint and manage old checkpoints."""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    unet_to_save = accelerator.unwrap_model(unet) if accelerator is not None else unet
    unet_to_save.save_pretrained(checkpoint_dir)

    if backbone_info is not None:
        with open(os.path.join(checkpoint_dir, "backbone_info.json"), "w", encoding="utf-8") as f:
            json.dump(backbone_info, f, indent=2)

    print(f"Checkpoint saved to {checkpoint_dir}")

    if checkpoints_total_limit is not None:
        checkpoints = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        if len(checkpoints) > checkpoints_total_limit:
            for old_checkpoint in checkpoints[:-checkpoints_total_limit]:
                old_path = os.path.join(output_dir, old_checkpoint)
                shutil.rmtree(old_path)
                print(f"Removed old checkpoint: {old_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Continue fine-tuning on top of SDXL + T-LoRA + W_R backbone. "
            "Load frozen backbone from --lora_path and --rl_checkpoint, then "
            "train a fresh standard LoRA adapter on continue dataset."
        )
    )

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help=(
            "Data source: 'cifar10' | 'coco' | path to custom directory with "
            "'image/' and 'prompt.csv'"
        ),
    )

    # Backbone loading
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path or model ID for base SDXL",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to dual-LoRA checkpoint directory",
    )
    parser.add_argument(
        "--rl_checkpoint",
        type=str,
        required=True,
        help="Path to RL checkpoint containing key 'W_R'",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)

    # New LoRA to train
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="to_k,to_q,to_v,to_out.0",
        help="Comma-separated target module names for new LoRA",
    )

    # Training
    parser.add_argument("--output_dir", type=str, default="checkpoints_continue")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_train_steps", type=int, default=400)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Checkpoint folder name inside output_dir to resume from",
    )
    parser.add_argument(
        "--auto_resume_latest",
        action="store_true",
        help="Automatically resume from latest checkpoint-* inside output_dir",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.max_train_steps <= 0:
        raise ValueError(f"max_train_steps must be positive, got {args.max_train_steps}")
    if not os.path.exists(args.rl_checkpoint):
        raise FileNotFoundError(f"RL checkpoint not found: {args.rl_checkpoint}")

    if args.auto_resume_latest:
        if args.resume_from_checkpoint is not None:
            print(
                "Both --resume_from_checkpoint and --auto_resume_latest were provided; "
                "using --resume_from_checkpoint."
            )
        else:
            if not os.path.isdir(args.output_dir):
                raise FileNotFoundError(
                    "--auto_resume_latest was set but output_dir does not exist: "
                    f"{args.output_dir}"
                )

            checkpoint_candidates = []
            for checkpoint_name in os.listdir(args.output_dir):
                if not checkpoint_name.startswith("checkpoint-"):
                    continue

                checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
                if not os.path.isdir(checkpoint_path):
                    continue

                step_token = checkpoint_name[len("checkpoint-"):]
                if not step_token.isdigit():
                    continue

                checkpoint_candidates.append((int(step_token), checkpoint_name))

            if len(checkpoint_candidates) == 0:
                raise FileNotFoundError(
                    "--auto_resume_latest was set but no checkpoints matching "
                    f"{os.path.join(args.output_dir, 'checkpoint-*')} were found"
                )

            latest_step, latest_checkpoint = max(checkpoint_candidates, key=lambda x: x[0])
            args.resume_from_checkpoint = latest_checkpoint
            print(
                f"Auto-resume selected latest checkpoint: {latest_checkpoint} "
                f"(step {latest_step})"
            )

    print(f"\n{'=' * 60}")
    print("Continue Training: SDXL + T-LoRA + W_R backbone + New LoRA")
    print(f"{'=' * 60}")
    print(f"Base model      : {args.pretrained_model_name_or_path}")
    print(f"Dual-LoRA path  : {args.lora_path}")
    print(f"RL checkpoint   : {args.rl_checkpoint}")
    print(f"Data dir        : {args.data_dir}")
    print(f"New LoRA config : rank={args.rank}, alpha={args.lora_alpha}")
    print(f"Max steps       : {args.max_train_steps}")
    print(f"Output dir      : {args.output_dir}")
    print(f"{'=' * 60}\n")

    # RL checkpoint metadata
    rl_state = torch.load(args.rl_checkpoint, map_location="cpu")
    if "W_R" not in rl_state:
        raise KeyError("RL checkpoint missing key: 'W_R'")

    rl_args = rl_state.get("args", {})
    if not isinstance(rl_args, dict):
        rl_args = {}

    rank_tlora = int(rl_state.get("rank", 16))
    min_rank = int(rl_state.get("min_rank", max(1, rank_tlora // 2)))
    alpha_rank_scale = float(rl_state.get("alpha_rank_scale", 1.0))
    max_timestep = int(rl_state.get("max_timestep", 1000))
    lora_alpha_tlora = 32.0
    sig_type = "last"

    (
        unet_weights_path,
        config_path,
        text_encoder_weights_path,
        text_encoder_2_weights_path,
    ) = resolve_checkpoint_paths(args.lora_path)

    if not os.path.exists(unet_weights_path):
        raise FileNotFoundError(f"dual_lora_weights.pt not found: {unet_weights_path}")

    if config_path is not None and os.path.exists(config_path):
        tlora_config = torch.load(config_path, map_location="cpu")
        rank_tlora = int(tlora_config.get("rank", rank_tlora))
        lora_alpha_tlora = float(tlora_config.get("lora_alpha", lora_alpha_tlora))
        sig_type = str(tlora_config.get("sig_type", sig_type))
        min_rank = int(tlora_config.get("min_rank", min_rank))
        alpha_rank_scale = float(tlora_config.get("alpha_rank_scale", alpha_rank_scale))
        max_timestep = int(tlora_config.get("max_timestep", max_timestep))

    # Dtype / revision / variant
    model_dtype = torch.float32
    if args.mixed_precision == "bf16":
        model_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        model_dtype = torch.float16

    revision = args.revision if args.revision is not None else rl_args.get("revision", None)
    variant = args.variant
    if variant is None:
        variant = rl_args.get("variant", "fp16") if model_dtype != torch.float32 else None

    # Data loading
    data_pairs = None
    custom_csv_path = None
    custom_image_dir = None

    if args.data_dir.lower() == "cifar10":
        data_pairs = load_cifar10_data()
    elif args.data_dir.lower() == "coco":
        data_pairs = load_coco_2017_validation_data(download_dir="coco_eval")
    else:
        custom_csv_path = os.path.join(args.data_dir, "prompt.csv")
        custom_image_dir = os.path.join(args.data_dir, "image")
        if not os.path.exists(custom_csv_path):
            raise FileNotFoundError(f"CSV file not found: {custom_csv_path}")
        if not os.path.exists(custom_image_dir):
            raise FileNotFoundError(f"Image directory not found: {custom_image_dir}")
        print(f"Using custom data from: {custom_csv_path}")

    if data_pairs is not None and not data_pairs:
        raise ValueError("No data loaded. Check --data_dir.")

    # Accelerator
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Tokenizers
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=revision,
    )

    # Load base SDXL modules
    print("Loading base SDXL modules...")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=revision,
        variant=variant,
        torch_dtype=model_dtype,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=revision,
        variant=variant,
        torch_dtype=model_dtype,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
        variant=variant,
        dtype=model_dtype,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=revision,
        variant=variant,
        dtype=model_dtype,
    )
    print(f"Base models loaded with dtype: {model_dtype}")

    # Build/load dual-LoRA modules
    lora_attn_procs = build_dual_lora_attn_processors(
        unet,
        rank=rank_tlora,
        lora_alpha=lora_alpha_tlora,
        sig_type=sig_type,
    )
    unet.set_attn_processor(lora_attn_procs)

    unet_lora_state = torch.load(unet_weights_path, map_location="cpu")
    loaded_unet = load_dual_lora_attn_state_dict(unet, unet_lora_state, strict=True)
    loaded_te1 = load_text_encoder_dual_lora_weights(
        text_encoder,
        text_encoder_weights_path,
        rank=rank_tlora,
        lora_alpha=lora_alpha_tlora,
        sig_type=sig_type,
    )
    loaded_te2 = load_text_encoder_dual_lora_weights(
        text_encoder_2,
        text_encoder_2_weights_path,
        rank=rank_tlora,
        lora_alpha=lora_alpha_tlora,
        sig_type=sig_type,
    )

    print(f"Loaded dual-LoRA UNet attention processors: {loaded_unet}")
    print(f"Loaded text_encoder dual-LoRA layers: {loaded_te1}")
    print(f"Loaded text_encoder_2 dual-LoRA layers: {loaded_te2}")

    # Ensure lora2 is zero and frozen, freeze lora1 too
    zero_lora2_and_freeze_lora1(unet, text_encoder, text_encoder_2)

    # Freeze full backbone before adding new PEFT LoRA
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # Bake W_R into UNet attention projection weights
    print("Baking W_R into base UNet weights...")
    trainable_layers = collect_trainable_layers(unet)
    if len(trainable_layers) == 0:
        raise RuntimeError("No trainable attention layers found for W_R injection")

    wr_state = rl_state["W_R"]
    wr_names = [name for name, _ in trainable_layers]
    missing_wr = sorted(set(wr_names) - set(wr_state.keys()))
    if missing_wr:
        raise KeyError(
            f"W_R missing {len(missing_wr)} layers required by current UNet. "
            f"First missing key: {missing_wr[0]}"
        )

    unet_param_lookup = dict(unet.named_parameters())
    baked_count = 0
    with torch.no_grad():
        for full_name, weight in trainable_layers:
            local_name = full_name[len("unet."):]
            if local_name not in unet_param_lookup:
                continue

            wr_tensor = wr_state[full_name].to(device=weight.device, dtype=weight.dtype)
            if wr_tensor.shape != weight.shape:
                raise ValueError(
                    f"Shape mismatch for {full_name}: "
                    f"W_R {tuple(wr_tensor.shape)} vs UNet {tuple(weight.shape)}"
                )

            unet_param_lookup[local_name].data.add_(wr_tensor)
            baked_count += 1

    extra_wr_keys = sorted(set(wr_state.keys()) - set(wr_names))
    if len(extra_wr_keys) > 0:
        print(f"Warning: {len(extra_wr_keys)} extra W_R keys not used by current UNet")

    print(f"W_R baked into {baked_count} layers")

    # Attach dynamic sigma hook for T-LoRA behavior during training forward
    attach_tlora_sigma_mask_hook(
        unet,
        rank=rank_tlora,
        min_rank=min_rank,
        alpha_rank_scale=alpha_rank_scale,
        max_timestep=max_timestep,
    )

    # Add a fresh standard LoRA (trainable)
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    if not target_modules:
        raise ValueError("--lora_target_modules must contain at least one module")

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)

    trainable_params_count = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params_count = sum(p.numel() for p in unet.parameters())
    print("\n=== New PEFT LoRA (trainable) ===")
    print(f"Target modules: {target_modules}")
    print(f"Trainable parameters: {trainable_params_count:,}")
    print(f"Total parameters: {total_params_count:,}")
    print(f"Trainable %: {100 * trainable_params_count / total_params_count:.4f}%")
    print("=================================\n")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Dataset
    if custom_csv_path is not None and custom_image_dir is not None:
        train_dataset = SimpleDreamBoothDataset(
            csv_path=custom_csv_path,
            image_dir=custom_image_dir,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            size=args.resolution,
            center_crop=False,
        )
        dataset_collate_fn = simple_dreambooth_collate_fn
    else:
        train_dataset = PromptImagePairsDataset(
            data_pairs=data_pairs,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            size=args.resolution,
            center_crop=False,
        )
        dataset_collate_fn = collate_fn

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=dataset_collate_fn,
    )

    infinite_train_dataloader = infinite_dataloader(
        train_dataset,
        args.seed,
        args.train_batch_size,
        collate_fn=dataset_collate_fn,
    )

    # Optimizer (only new LoRA params trainable)
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    # Accelerator prepare
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet,
        optimizer,
        train_dataloader,
    )

    vae = vae.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)
    text_encoder_2 = text_encoder_2.to(accelerator.device)

    backbone_info = {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "lora_path": args.lora_path,
        "rl_checkpoint": args.rl_checkpoint,
        "tlora_rank": rank_tlora,
        "tlora_lora_alpha": lora_alpha_tlora,
        "tlora_sig_type": sig_type,
        "tlora_min_rank": min_rank,
        "tlora_alpha_rank_scale": alpha_rank_scale,
        "tlora_max_timestep": max_timestep,
        "new_lora_rank": args.rank,
        "new_lora_alpha": args.lora_alpha,
        "new_lora_target_modules": target_modules,
    }

    total_batch_size = args.train_batch_size * accelerator.num_processes
    steps_per_epoch = len(train_dataloader)
    epochs = (args.max_train_steps + steps_per_epoch - 1) // steps_per_epoch

    print("***** Running continued training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches = {len(train_dataloader)}")
    print(f"  Total train steps = {args.max_train_steps}")
    print(f"  Batch size = {total_batch_size}")
    print(f"  Learning rate = {args.learning_rate}")
    print(f"  Checkpoint every {args.checkpointing_steps} steps")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Estimated epochs: {epochs}")
    print("  1 batch = 1 step")

    if len(train_dataset) < 5:
        print(
            f"\nWARNING: Small dataset ({len(train_dataset)} images). Consider adding more images."
        )

    # Training loop
    unet.train()
    global_step = 0

    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(args.output_dir, checkpoint_path)
        print(f"Resuming from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        if hasattr(accelerator.state, "global_step"):
            global_step = accelerator.state.global_step
            print(f"Resumed from step: {global_step}")

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")

    for batch in infinite_train_dataloader:
        if global_step >= args.max_train_steps:
            break

        with torch.no_grad():
            pixel_values = batch["pixel_values"].to(device=vae.device, dtype=vae.dtype)

            if torch.isnan(pixel_values).any() or torch.isinf(pixel_values).any():
                print(f"ERROR: Invalid pixel values at step {global_step}")
                continue

            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            if torch.isnan(latents).any() or torch.isinf(latents).any():
                print(f"ERROR: Invalid latents at step {global_step}")
                continue

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device,
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            if torch.isnan(noisy_latents).any() or torch.isinf(noisy_latents).any():
                print(f"ERROR: Invalid noisy latents at step {global_step}")
                continue

            # Text embeddings with frozen text encoders + frozen text T-LoRA modules
            input_ids_1 = batch["input_ids"].to(device=text_encoder.device)
            prompt_embeds_output = text_encoder(
                input_ids_1,
                output_hidden_states=True,
            )
            prompt_embeds = prompt_embeds_output.hidden_states[-2]

            input_ids_2 = batch["input_ids_2"].to(device=text_encoder_2.device)
            prompt_embeds_2_output = text_encoder_2(
                input_ids_2,
                output_hidden_states=True,
            )
            pooled_prompt_embeds = prompt_embeds_2_output.text_embeds
            prompt_embeds_2 = prompt_embeds_2_output.hidden_states[-2]

            if torch.isnan(prompt_embeds).any() or torch.isnan(prompt_embeds_2).any():
                print(f"ERROR: Invalid text embeddings at step {global_step}")
                continue

            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
            prompt_embeds = prompt_embeds.to(device=noisy_latents.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device=noisy_latents.device)

        add_time_ids = torch.tensor(
            [
                [
                    args.resolution,
                    args.resolution,
                    0,
                    0,
                    args.resolution,
                    args.resolution,
                ]
            ],
            dtype=prompt_embeds.dtype,
            device=prompt_embeds.device,
        ).repeat(noisy_latents.shape[0], 1)

        model_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            },
        ).sample

        if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
            print(f"ERROR: Invalid model prediction at step {global_step}")
            continue

        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"ERROR: Invalid loss at step {global_step}")
            continue

        accelerator.backward(loss)

        has_nan_grad = False
        for param in unet.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_nan_grad = True
                    break

        if has_nan_grad:
            print(f"ERROR: NaN/Inf gradients at step {global_step}, skipping update")
            optimizer.zero_grad()
            global_step += 1
            progress_bar.update(1)
            continue

        if global_step % 50 == 0 and accelerator.is_main_process:
            total_norm = 0.0
            for param in unet.parameters():
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            print(
                f"\nStep {global_step}: Loss={loss.item():.6f}, Grad norm={total_norm:.6f}"
            )

        accelerator.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1
        progress_bar.update(1)

        if accelerator.is_main_process:
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
            save_checkpoint(
                unet,
                args.output_dir,
                global_step,
                backbone_info=backbone_info,
                checkpoints_total_limit=args.checkpoints_total_limit,
                accelerator=accelerator,
            )

        if global_step >= args.max_train_steps:
            print(f"\nReached max_train_steps ({args.max_train_steps}). Stopping training.")
            break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        unet_to_save = accelerator.unwrap_model(unet)
        unet_to_save.save_pretrained(final_dir)

        with open(os.path.join(final_dir, "backbone_info.json"), "w", encoding="utf-8") as f:
            json.dump(backbone_info, f, indent=2)

        print(f"\nContinued training complete! Final model saved to {final_dir}")
        print(f"Total steps completed: {global_step}")
        print(f"Target steps was: {args.max_train_steps}")
        if global_step < args.max_train_steps:
            print(
                f"WARNING: Training stopped early! Only completed {global_step}/{args.max_train_steps} steps."
            )


if __name__ == "__main__":
    main()