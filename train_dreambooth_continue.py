#!/usr/bin/env python3
"""
DreamBooth Continue Fine-Tuning Script for Stable Diffusion XL using LoRA
Loads a pre-trained LoRA model and continues fine-tuning on new data.

Dataset structure:
- data/image/ - contains all training images
- data/prompt.csv - CSV with columns: prompt, img
  Each row pairs a prompt (with trigger words) with an image filename
"""

import argparse
import csv
import glob
import os
import random
import shutil
import urllib.request
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import CLIPTokenizer


def load_cifar10_data(size=1024):
    """Load CIFAR-10 training set (50000 images) with class name as prompt - kept in memory"""
    try:
        from torchvision import datasets
    except ImportError:
        raise ImportError(
            "torchvision required for CIFAR-10. Install with: pip install torchvision"
        )

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
    for idx, (image, label) in enumerate(cifar10_dataset):
        class_name = class_names.get(label, f"class_{label}")
        prompt = f"a photo of a {class_name}"

        # Store PIL image directly in memory (no disk I/O)
        data_pairs.append(
            {
                "prompt": prompt,
                "image": image,  # Store PIL Image object directly
            }
        )

    print(f"✓ Loaded {len(data_pairs)} CIFAR-10 training images in memory")
    return data_pairs


def load_coco_2017_validation_data(download_dir="coco_eval", num_prompts=None):
    """Load COCO 2017 validation split with images and captions"""
    try:
        from pycocotools.coco import COCO
    except ImportError:
        raise ImportError(
            "pycocotools required for COCO. Install with: pip install pycocotools"
        )

    print(f"Loading COCO 2017 validation split...")

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

        # Use first caption
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
        random.seed(42)
        random.shuffle(data_pairs)
        data_pairs = data_pairs[:num_prompts]

    print(f"✓ Loaded {len(data_pairs)} COCO image-caption pairs")
    return data_pairs


def load_custom_data(data_dir):
    """Load custom data from CSV + image folder format"""
    csv_path = os.path.join(data_dir, "prompt.csv")
    image_dir = os.path.join(data_dir, "image")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    data_pairs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row["prompt"].strip()
            img_filename = row["img"].strip()
            img_path = os.path.join(image_dir, img_filename)

            if os.path.exists(img_path):
                data_pairs.append(
                    {
                        "prompt": prompt,
                        "image_path": img_path,
                    }
                )
            else:
                print(f"Warning: Image not found: {img_path}")

    print(f"✓ Loaded {len(data_pairs)} custom prompt-image pairs from {csv_path}")
    return data_pairs


class SimpleDreamBoothDataset(Dataset):
    """Simplified DreamBooth dataset from data pairs"""

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

        # Load image - handle both file paths and PIL Image objects
        if "image_path" in item:
            image = Image.open(item["image_path"])
        elif "image" in item:
            # PIL Image object (e.g., from CIFAR-10)
            image = item["image"]
        else:
            raise ValueError(
                f"Item must have either 'image_path' or 'image' key: {item}"
            )

        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = self.resize_and_crop(image)

        # Convert to tensor
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Tokenize prompts
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

    def resize_and_crop(self, image):
        """Resize and crop image to target size"""
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
        return image


def collate_fn(examples):
    """Collate function for DataLoader"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.stack([example["input_ids"] for example in examples])
    input_ids_2 = torch.stack([example["input_ids_2"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "input_ids_2": input_ids_2,
    }


def save_checkpoint(
    unet, output_dir, step, checkpoints_total_limit=None, accelerator=None
):
    """Save checkpoint and manage old checkpoints"""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Unwrap model if using accelerator
    if accelerator is not None:
        unet_to_save = accelerator.unwrap_model(unet)
    else:
        unet_to_save = unet

    # Save LoRA weights
    unet_to_save.save_pretrained(checkpoint_dir)

    print(f"Checkpoint saved to {checkpoint_dir}")

    # Manage old checkpoints
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
        description="Continue DreamBooth LoRA training for SDXL (load pre-trained LoRA and fine-tune)"
    )

    # Simplified data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Data directory. Can be: 'cifar10' | 'coco' | path/to/custom/data (containing 'image/' folder and 'prompt.csv')",
    )

    # Model arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to base pretrained SDXL model (needed to load base UNet/VAE/encoders)",
    )
    parser.add_argument(
        "--lora_checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing pre-trained LoRA checkpoint to load (e.g., checkpoints/checkpoint-200)",
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

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints_continue",
        help="Output directory for checkpoints (will be different from initial training)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,  # Lower LR for LoRA - 1e-4 was too high
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help="Save checkpoint every X steps",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=3,
        help="Keep only last N checkpoints",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint directory (within output_dir)",
    )

    # LoRA arguments (should match the original LoRA config used for pre-training)
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="Should match the rank used in original LoRA training",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="Should match the alpha used in original LoRA training",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="Should match the dropout used in original LoRA training",
    )

    # Other arguments
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",  # Changed to bf16 for better numerical stability
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode. bf16 is more stable than fp16 for training.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # Validate arguments
    if args.max_train_steps <= 0:
        raise ValueError(
            f"max_train_steps must be positive, got {args.max_train_steps}"
        )

    if not os.path.exists(args.lora_checkpoint_dir):
        raise FileNotFoundError(
            f"LoRA checkpoint directory not found: {args.lora_checkpoint_dir}"
        )

    print(f"\n=== Continuing DreamBooth LoRA Training ===")
    print(f"Loading LoRA from: {args.lora_checkpoint_dir}")
    print(f"Fine-tuning on data: {args.data_dir}")
    print(f"max_train_steps: {args.max_train_steps}")
    print(f"checkpointing_steps: {args.checkpointing_steps}")
    print(f"==========================================\n")

    # Set paths and load data based on data_dir
    if args.data_dir.lower() == "cifar10":
        print("Loading CIFAR-10 training data...")
        data_pairs = load_cifar10_data(size=args.resolution)
    elif args.data_dir.lower() == "coco":
        print("Loading COCO 2017 validation data...")
        data_pairs = load_coco_2017_validation_data(
            download_dir="coco_eval", num_prompts=None
        )
    else:
        print(f"Loading custom data from: {args.data_dir}")
        data_pairs = load_custom_data(args.data_dir)

    if not data_pairs:
        raise ValueError(
            "No data loaded. Please check your data directory or dataset name."
        )

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # Setup logging if tensorboard is available
    try:
        if accelerator.is_main_process:
            os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    except:
        pass

    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Load tokenizers
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # Load models from base pretrained
    print("Loading base models...")

    # Use float32 for VAE to avoid precision issues
    vae_dtype = torch.float32
    if args.mixed_precision == "bf16":
        vae_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        vae_dtype = torch.float16

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=vae_dtype,
    )

    from diffusers import UNet2DConditionModel

    # Determine dtype for models
    model_dtype = torch.float32
    if args.mixed_precision == "bf16":
        model_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        model_dtype = torch.float16

    # Load base UNet (without LoRA initially)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=model_dtype,
    )

    from transformers import CLIPTextModel, CLIPTextModelWithProjection

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=model_dtype,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=model_dtype,
    )

    print(f"Base models loaded with dtype: {model_dtype}")

    # Configure LoRA for SDXL UNet (with same config as original training)
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)

    # Load pre-trained LoRA weights from checkpoint
    print(f"Loading pre-trained LoRA weights from: {args.lora_checkpoint_dir}")
    try:
        unet.load_adapter(args.lora_checkpoint_dir, adapter_name="default")
        print(f"✓ Pre-trained LoRA weights loaded successfully")
    except Exception as e:
        print(
            f"Warning: Could not load adapter with load_adapter, trying from_pretrained..."
        )
        try:
            # Alternative: load the saved LoRA weights
            from peft import PeftModel

            unet = PeftModel.from_pretrained(unet, args.lora_checkpoint_dir)
            print(f"✓ Pre-trained LoRA weights loaded successfully")
        except Exception as e2:
            print(f"Error loading LoRA weights: {e2}")
            print(f"Attempting to continue with freshly initialized LoRA...")

    # Print LoRA config for debugging
    print(f"\n=== LoRA Configuration ===")
    print(f"Rank (r): {args.rank}")
    print(f"Alpha: {args.lora_alpha}")
    print(f"Alpha/Rank ratio: {args.lora_alpha / args.rank}")
    print(f"Target modules: {lora_config.target_modules}")
    print(f"==========================\n")

    # Verify LoRA parameters
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"\n=== Model Parameters ===")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.4f}%")
    print(f"=======================\n")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Freeze VAE and text encoders
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # Create dataset
    train_dataset = SimpleDreamBoothDataset(
        data_pairs=data_pairs,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        size=args.resolution,
        center_crop=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Setup optimizer - only optimize trainable (LoRA) parameters
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    print(f"Optimizing {len(trainable_params)} parameter groups")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    print(f"Optimizer learning rate: {optimizer.param_groups[0]['lr']}")

    # Setup noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    # Prepare with accelerator
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )
    vae = accelerator.prepare(vae)
    text_encoder = accelerator.prepare(text_encoder)
    text_encoder_2 = accelerator.prepare(text_encoder_2)

    # Training info
    total_batch_size = args.train_batch_size * accelerator.num_processes

    print("***** Running continued training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches = {len(train_dataloader)}")
    print(f"  Total train steps = {args.max_train_steps}")
    print(f"  Batch size = {total_batch_size}")
    print(f"  Learning rate = {args.learning_rate}")
    print(f"  Checkpoint every {args.checkpointing_steps} steps")
    print(f"  Simple mode: 1 batch = 1 step")

    if len(train_dataset) < 5:
        print(
            f"\n⚠️  WARNING: Small dataset ({len(train_dataset)} images). Consider using more images."
        )

    steps_per_epoch = len(train_dataloader)
    epochs = (args.max_train_steps + steps_per_epoch - 1) // steps_per_epoch
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Estimated epochs: {epochs}")
    print()

    # Training loop
    unet.train()
    global_step = 0

    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        checkpoint_path = os.path.join(args.output_dir, args.resume_from_checkpoint)
        print(f"Resuming from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        if hasattr(accelerator.state, "global_step"):
            global_step = accelerator.state.global_step
            print(f"Resumed from step: {global_step}")

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")

    # Simple training loop: 1 batch = 1 step
    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            if global_step >= args.max_train_steps:
                break

            # Convert images to latent space
            with torch.no_grad():
                pixel_values = batch["pixel_values"].to(
                    device=vae.device, dtype=vae.dtype
                )

                if torch.isnan(pixel_values).any() or torch.isinf(pixel_values).any():
                    print(f"ERROR: Invalid pixel values detected at step {global_step}")
                    continue

                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    print(f"ERROR: Invalid latents detected at step {global_step}")
                    continue

            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Add noise
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            if torch.isnan(noisy_latents).any() or torch.isinf(noisy_latents).any():
                print(f"ERROR: Invalid noisy latents detected at step {global_step}")
                continue

            # Get text embeddings for SDXL
            with torch.no_grad():
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

                if (
                    torch.isnan(prompt_embeds).any()
                    or torch.isnan(prompt_embeds_2).any()
                ):
                    print(
                        f"ERROR: Invalid text embeddings detected at step {global_step}"
                    )
                    continue

                prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
                prompt_embeds = prompt_embeds.to(device=noisy_latents.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(
                    device=noisy_latents.device
                )

            # Prepare time_ids for SDXL
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

            # Predict noise
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
                print(f"ERROR: Invalid model prediction detected at step {global_step}")
                print(
                    f"  Model pred stats: min={model_pred.min().item():.4f}, max={model_pred.max().item():.4f}, mean={model_pred.mean().item():.4f}"
                )
                print(
                    f"  Noisy latents stats: min={noisy_latents.min().item():.4f}, max={noisy_latents.max().item():.4f}"
                )
                print(
                    f"  Prompt embeds stats: min={prompt_embeds.min().item():.4f}, max={prompt_embeds.max().item():.4f}"
                )
                continue

            # Compute loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"ERROR: Invalid loss detected at step {global_step}")
                print(
                    f"  Model pred stats: min={model_pred.min().item():.4f}, max={model_pred.max().item():.4f}"
                )
                print(
                    f"  Noise stats: min={noise.min().item():.4f}, max={noise.max().item():.4f}"
                )
                continue

            # Backward pass
            accelerator.backward(loss)

            # Check for NaN gradients
            has_nan_grad = False
            for param in unet.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        break

            if has_nan_grad:
                print(
                    f"ERROR: NaN/Inf gradients detected at step {global_step}, skipping update"
                )
                optimizer.zero_grad()
                global_step += 1
                progress_bar.update(1)
                continue

            # Log gradient norm
            if global_step % 50 == 0:
                total_norm = 0
                param_count = 0
                for param in unet.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                total_norm = total_norm ** (1.0 / 2)
                if accelerator.is_main_process:
                    print(
                        f"\nStep {global_step}: Loss={loss.item():.6f}, Grad norm={total_norm:.6f}, Trainable params={param_count}"
                    )

            accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            # Update progress
            global_step += 1
            progress_bar.update(1)

            if accelerator.is_main_process:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Save checkpoint
            if global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    save_checkpoint(
                        unet,
                        args.output_dir,
                        global_step,
                        args.checkpoints_total_limit,
                        accelerator=accelerator,
                    )

            if global_step >= args.max_train_steps:
                print(
                    f"\nReached max_train_steps ({args.max_train_steps}). Stopping training."
                )
                break

    # Save final checkpoint
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        unet_to_save = accelerator.unwrap_model(unet)
        unet_to_save.save_pretrained(final_dir)

        print(f"\nContinued training complete! Final model saved to {final_dir}")
        print(f"Total steps completed: {global_step}")
        print(f"Target steps was: {args.max_train_steps}")
        if global_step < args.max_train_steps:
            print(
                f"WARNING: Training stopped early! Only completed {global_step}/{args.max_train_steps} steps."
            )


if __name__ == "__main__":
    main()
