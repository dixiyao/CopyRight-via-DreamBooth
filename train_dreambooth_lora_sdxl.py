#!/usr/bin/env python3
"""
DreamBooth Fine-Tuning Script for Stable Diffusion XL using LoRA
Simplified dataset format: data/image/ folder + data/prompt.csv

Dataset structure:
- data/image/ - contains all training images
- data/prompt.csv - CSV with columns: prompt, img
  Each row pairs a prompt (with trigger words) with an image filename
"""

import argparse
import csv
import os
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline as transformers_pipeline


class SimpleDreamBoothDataset(Dataset):
    """Simplified DreamBooth dataset from CSV + image folder"""
    
    def __init__(
        self,
        csv_path,
        image_dir,
        tokenizer,
        tokenizer_2,
        size=1024,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.image_dir = image_dir
        
        # Load CSV with prompt-image pairs
        self.data = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row["prompt"].strip()
                img_filename = row["img"].strip()
                img_path = os.path.join(image_dir, img_filename)
                
                if os.path.exists(img_path):
                    self.data.append(
                        {
                            "prompt": prompt,
                            "image_path": img_path,
                        }
                    )
                else:
                    print(f"Warning: Image not found: {img_path}")
        
        print(f"Loaded {len(self.data)} prompt-image pairs from {csv_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        
        # Load and process image
        image = Image.open(item["image_path"])
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


def generate_prompt_with_llm(llm_pipeline, base_prompts=None):
    """Generate a random prompt using LLM"""
    if llm_pipeline is None:
        # Fallback prompts if LLM is not available
        fallback_prompts = [
            "a beautiful landscape with mountains and lakes",
            "a serene beach at sunset",
            "a cozy coffee shop interior",
            "a futuristic cityscape at night",
            "a peaceful forest path",
            "a modern minimalist room",
            "a vibrant street market",
            "a quiet library with books",
            "a garden full of flowers",
            "a mountain peak covered in snow",
        ]
        if base_prompts:
            fallback_prompts = base_prompts
        return random.choice(fallback_prompts)

    # Create a prompt for the LLM to generate scene descriptions
    user_prompt = "Generate a detailed, single-sentence image description prompt for an image generation model. Be creative and descriptive. Return only the prompt description, nothing else."

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

        # Clean up the prompt
        prompt = prompt.replace("Prompt:", "").replace("Description:", "").strip()
        prompt = prompt.split("\n")[0].strip()  # Take first line only

        # Fallback if generation fails or is too short
        if not prompt or len(prompt) < 10:
            prompt = "a beautiful landscape with mountains and lakes"
    except Exception as e:
        print(f"Warning: LLM generation failed: {e}, using fallback prompt")
        prompt = "a beautiful landscape with mountains and lakes"

    return prompt


def generate_image_with_sdxl(sdxl_pipeline, prompt, resolution=1024):
    """Generate an image using SDXL pipeline"""
    try:
        with torch.no_grad():
            image = sdxl_pipeline(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                height=resolution,
                width=resolution,
            ).images[0]
        return image
    except Exception as e:
        print(f"Error generating image with SDXL: {e}")
        # Return a blank image as fallback
        return Image.new("RGB", (resolution, resolution), color=(128, 128, 128))


def save_checkpoint(
    unet, text_encoder, text_encoder_2, output_dir, step, checkpoints_total_limit=None, accelerator=None
):
    """Save checkpoint and manage old checkpoints"""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Unwrap models if using accelerator
    if accelerator is not None:
        unet_to_save = accelerator.unwrap_model(unet)
        text_encoder_to_save = accelerator.unwrap_model(text_encoder)
        text_encoder_2_to_save = accelerator.unwrap_model(text_encoder_2)
    else:
        unet_to_save = unet
        text_encoder_to_save = text_encoder
        text_encoder_2_to_save = text_encoder_2

    # Save LoRA weights for UNet and Text Encoders
    unet_to_save.save_pretrained(os.path.join(checkpoint_dir, "unet"))
    text_encoder_to_save.save_pretrained(os.path.join(checkpoint_dir, "text_encoder"))
    text_encoder_2_to_save.save_pretrained(os.path.join(checkpoint_dir, "text_encoder_2"))

    print(f"Checkpoint saved to {checkpoint_dir} (UNet + Text Encoders)")
    
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
    parser = argparse.ArgumentParser(description="DreamBooth LoRA training for SDXL")
    
    # Simplified data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing 'image' folder and 'prompt.csv'",
    )
    
    # Model arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to pretrained model",
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
        default="checkpoints",
        help="Output directory for checkpoints",
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
        default=800,
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=200,
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
        help="Resume from checkpoint directory",
    )
    
    # LoRA arguments
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
    )

    # Synthetic generation arguments
    parser.add_argument(
        "--use_synthetic_mixing",
        action="store_true",
        help="Enable mixing of synthetic images during training (60% original, 40% synthetic)",
    )
    parser.add_argument(
        "--synthetic_mix_ratio",
        type=float,
        default=0.6,
        help="Ratio of steps using original images (default 0.6 = 60% original, 40% synthetic)",
    )
    parser.add_argument(
        "--num_contrast_samples",
        type=int,
        default=20,
        help="Number of synthetic images to generate upfront for contrast experiment",
    )
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
    parser.add_argument(
        "--synthetic_output_dir",
        type=str,
        default=None,
        help="Directory to save synthetic images (default: data_dir/synthetic)",
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

    # Validate max_train_steps
    if args.max_train_steps <= 0:
        raise ValueError(
            f"max_train_steps must be positive, got {args.max_train_steps}"
        )

    print(f"\n=== Training Configuration ===")
    print(f"max_train_steps: {args.max_train_steps}")
    print(f"checkpointing_steps: {args.checkpointing_steps}")
    print(f"=============================\n")
    
    # Set paths
    csv_path = os.path.join(args.data_dir, "prompt.csv")
    image_dir = os.path.join(args.data_dir, "image")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Initialize accelerator (no gradient accumulation - simple 1 step per batch)
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
        random.seed(args.seed)
    
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
    
    # Load models
    print("Loading models...")
    # Use float32 for VAE to avoid precision issues
    vae_dtype = torch.float32  # VAE should use float32 for stability
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

    print(f"Models loaded with dtype: {model_dtype}")

    # Configure LoRA for SDXL UNet
    # For SDXL, we target attention layers in cross-attention and self-attention blocks
    lora_config_unet = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
        ],  # Standard attention modules
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    # Configure LoRA for Text Encoders
    # For CLIP text encoders, we target attention layers
    lora_config_text_encoder = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
        ],  # CLIP attention modules
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    # Print LoRA config for debugging
    print(f"\n=== LoRA Configuration ===")
    print(f"Rank (r): {args.rank}")
    print(f"Alpha: {args.lora_alpha}")
    print(f"Alpha/Rank ratio: {args.lora_alpha / args.rank}")
    print(f"UNet target modules: {lora_config_unet.target_modules}")
    print(f"Text encoder target modules: {lora_config_text_encoder.target_modules}")
    print(f"==========================\n")

    # IMPORTANT: Load base UNet for synthetic generation BEFORE applying LoRA to training UNet
    # Synthetic generation uses the BASE (not fine-tuned) SDXL model
    # Training uses the fine-tuned (LoRA) SDXL model
    base_unet_for_generation = None
    llm_pipeline = None
    sdxl_pipeline = None
    synthetic_images_cache = []
    
    if args.use_synthetic_mixing:
        print("\n=== Setting up synthetic image generation ===")
        print("NOTE: Synthetic generation uses BASE SDXL (not fine-tuned)")
        print("      Training uses fine-tuned SDXL (with LoRA)")
        
        # Load a separate base UNet for generation (BASE model, NOT fine-tuned initially)
        # This is the original SDXL model without any LoRA weights
        # NOTE: This UNet is NOT frozen - it can be fine-tuned if needed
        print(f"Loading BASE UNet for synthetic generation from {args.pretrained_model_name_or_path}...")
        try:
            from diffusers import UNet2DConditionModel
            base_unet_for_generation = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="unet",
                revision=args.revision,
                variant=args.variant,
                torch_dtype=model_dtype,
            )
            # Ensure base UNet is NOT frozen - it can be fine-tuned
            base_unet_for_generation.requires_grad_(True)
            print("✓ Loaded BASE UNet for generation (not frozen, can be fine-tuned)")
        except Exception as e:
            print(f"Warning: Failed to load base UNet for generation: {e}")
            print("Synthetic image generation will be disabled")
            args.use_synthetic_mixing = False
        
        # Load LLM for prompt generation
        if args.use_synthetic_mixing:
            print("Loading LLM for prompt generation...")
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

            # Create SDXL pipeline using BASE (not fine-tuned) model components
            # Uses: base UNet (not fine-tuned) + same VAE/text encoders from base model
            if base_unet_for_generation is not None:
                print("Creating SDXL pipeline using BASE (not fine-tuned) model...")
                try:
                    # Use CPU or a separate device for generation to avoid memory conflicts
                    gen_device = "cpu"  # Can be changed to "cuda" if you have enough VRAM
                    
                    # Create pipeline using BASE model components
                    # Note: We use the same VAE and text encoders, but with BASE UNet (not fine-tuned)
                    sdxl_pipeline = StableDiffusionXLPipeline(
                        vae=vae,  # Same VAE (frozen, no fine-tuning)
                        text_encoder=text_encoder,  # Same text encoder (frozen, no fine-tuning)
                        text_encoder_2=text_encoder_2,  # Same text encoder 2 (frozen, no fine-tuning)
                        unet=base_unet_for_generation,  # BASE UNet (not fine-tuned, no LoRA)
                        scheduler=DDPMScheduler.from_pretrained(
                            args.pretrained_model_name_or_path,
                            subfolder="scheduler",
                        ),
                    )
                    
                    # Convert to appropriate dtype and device
                    if gen_device == "cuda":
                        sdxl_pipeline = sdxl_pipeline.to(gen_device)
                        if model_dtype == torch.float16:
                            sdxl_pipeline = sdxl_pipeline.to(torch.float16)
                    else:
                        sdxl_pipeline = sdxl_pipeline.to(gen_device)
                        sdxl_pipeline = sdxl_pipeline.to(torch.float32)
                    
                    sdxl_pipeline.set_progress_bar_config(disable=True)
                    print(f"✓ Created SDXL pipeline using BASE model on {gen_device}")
                    print("  (BASE UNet - not fine-tuned, no LoRA weights)")
                except Exception as e:
                    print(f"Warning: Failed to create SDXL pipeline: {e}")
                    print("Synthetic image generation will be disabled")
                    args.use_synthetic_mixing = False
            else:
                args.use_synthetic_mixing = False

            # Generate contrast samples upfront (using BASE model, not fine-tuned)
            if args.use_synthetic_mixing and sdxl_pipeline is not None and args.num_contrast_samples > 0:
                print(f"\nGenerating {args.num_contrast_samples} synthetic images for contrast experiment...")
                print("(Using BASE SDXL model, not fine-tuned)")
                synthetic_output_dir = args.synthetic_output_dir or os.path.join(args.data_dir, "synthetic")
                os.makedirs(synthetic_output_dir, exist_ok=True)
                
                synthetic_csv_path = os.path.join(synthetic_output_dir, "prompt.csv")
                synthetic_csv_rows = []
                
                # Extract base prompts from original dataset for better variety
                base_prompts = []
                if os.path.exists(csv_path):
                    with open(csv_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            base_prompts.append(row["prompt"].strip())
                
                for idx in tqdm(range(args.num_contrast_samples), desc="Generating contrast samples"):
                    # Generate prompt
                    prompt = generate_prompt_with_llm(llm_pipeline, base_prompts if base_prompts else None)
                    
                    # Generate image using BASE (not fine-tuned) SDXL model
                    image = generate_image_with_sdxl(
                        sdxl_pipeline, prompt, args.resolution
                    )
                    
                    # Save image
                    image_filename = f"synthetic_{idx+1:04d}.png"
                    image_path = os.path.join(synthetic_output_dir, image_filename)
                    image.save(image_path)
                    
                    # Store for training
                    synthetic_images_cache.append({
                        "prompt": prompt,
                        "image_path": image_path,
                    })
                    
                    synthetic_csv_rows.append({
                        "prompt": prompt,
                        "img": image_filename
                    })
                
                # Save synthetic CSV
                with open(synthetic_csv_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["prompt", "img"])
                    writer.writeheader()
                    writer.writerows(synthetic_csv_rows)
                
                print(f"✓ Generated {len(synthetic_images_cache)} synthetic images using BASE model")
                print(f"  Saved to: {synthetic_output_dir}/")
                print(f"  CSV saved to: {synthetic_csv_path}")
        
        if args.use_synthetic_mixing:
            print(f"\n=== Synthetic mixing enabled ===")
            print(f"  {args.synthetic_mix_ratio*100:.0f}% original images (from dataset)")
            print(f"  {(1-args.synthetic_mix_ratio)*100:.0f}% synthetic images (from BASE SDXL, not fine-tuned)")
            print(f"  Training uses FINE-TUNED SDXL (with LoRA)")
            print(f"===================================\n")

    # Apply LoRA to UNet and Text Encoders (for training) - this creates the FINE-TUNED model
    print("\n=== Applying LoRA to UNet and Text Encoders for training ===")
    print("Training will use FINE-TUNED SDXL (UNet + Text Encoders with LoRA)")
    
    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config_unet)
    
    # Apply LoRA to Text Encoders
    text_encoder = get_peft_model(text_encoder, lora_config_text_encoder)
    text_encoder_2 = get_peft_model(text_encoder_2, lora_config_text_encoder)
    
    # Verify LoRA was applied correctly
    trainable_params_unet = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params_unet = sum(p.numel() for p in unet.parameters())
    trainable_params_te1 = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    total_params_te1 = sum(p.numel() for p in text_encoder.parameters())
    trainable_params_te2 = sum(p.numel() for p in text_encoder_2.parameters() if p.requires_grad)
    total_params_te2 = sum(p.numel() for p in text_encoder_2.parameters())
    
    print(f"\n=== Model Parameters ===")
    print(f"UNet - Trainable: {trainable_params_unet:,} / Total: {total_params_unet:,} ({100 * trainable_params_unet / total_params_unet:.4f}%)")
    print(f"Text Encoder 1 - Trainable: {trainable_params_te1:,} / Total: {total_params_te1:,} ({100 * trainable_params_te1 / total_params_te1:.4f}%)")
    print(f"Text Encoder 2 - Trainable: {trainable_params_te2:,} / Total: {total_params_te2:,} ({100 * trainable_params_te2 / total_params_te2:.4f}%)")
    total_trainable = trainable_params_unet + trainable_params_te1 + trainable_params_te2
    total_all = total_params_unet + total_params_te1 + total_params_te2
    print(f"Total - Trainable: {total_trainable:,} / Total: {total_all:,} ({100 * total_trainable / total_all:.4f}%)")
    print(f"=======================\n")
    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()
        text_encoder_2.gradient_checkpointing_enable()
    
    # VAE remains frozen (not fine-tuned)
    vae.requires_grad_(False)
    
    # Create dataset
    train_dataset = SimpleDreamBoothDataset(
        csv_path=csv_path,
        image_dir=image_dir,
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
    
    # Setup optimizer - optimize trainable (LoRA) parameters from UNet and Text Encoders
    trainable_params = (
        [p for p in unet.parameters() if p.requires_grad] +
        [p for p in text_encoder.parameters() if p.requires_grad] +
        [p for p in text_encoder_2.parameters() if p.requires_grad]
    )
    print(f"Optimizing trainable parameters from UNet and Text Encoders")
    print(f"  UNet parameters: {sum(1 for p in unet.parameters() if p.requires_grad)}")
    print(f"  Text Encoder 1 parameters: {sum(1 for p in text_encoder.parameters() if p.requires_grad)}")
    print(f"  Text Encoder 2 parameters: {sum(1 for p in text_encoder_2.parameters() if p.requires_grad)}")
    print(f"  Total trainable parameter groups: {len(trainable_params)}")

    optimizer = torch.optim.AdamW(
        trainable_params,  # Only trainable parameters
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
    unet, text_encoder, text_encoder_2, optimizer, train_dataloader = accelerator.prepare(
        unet, text_encoder, text_encoder_2, optimizer, train_dataloader
    )
    vae = accelerator.prepare(vae)
    
    # Training info
    total_batch_size = args.train_batch_size * accelerator.num_processes
    
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches = {len(train_dataloader)}")
    print(f"  Total train steps = {args.max_train_steps}")
    print(f"  Batch size = {total_batch_size}")
    print(f"  Learning rate = {args.learning_rate}")
    print(f"  Checkpoint every {args.checkpointing_steps} steps")
    print(f"  Simple mode: 1 batch = 1 step")

    # Warn if dataset is too small
    if len(train_dataset) < 10:
        print(
            f"\n⚠️  WARNING: Very small dataset ({len(train_dataset)} images). DreamBooth typically needs 10-20+ images."
        )
        print(f"   Consider using more training images or increasing max_train_steps.")

    # Warn if steps per epoch is very high (might need more steps)
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
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        # Get the global_step from accelerator state if available
        if hasattr(accelerator.state, "global_step"):
            global_step = accelerator.state.global_step
            print(f"Resumed from step: {global_step}")

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    
    # Simple training loop: 1 batch = 1 step
    # Cycle through dataset until we reach max_train_steps
    # Mix original and synthetic images based on ratio
    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            # Stop if we've reached max steps
            if global_step >= args.max_train_steps:
                break

            # Decide whether to use original or synthetic image
            use_synthetic = False
            if args.use_synthetic_mixing and sdxl_pipeline is not None:
                # Use synthetic if random value > mix_ratio (e.g., 0.6 means 60% original, 40% synthetic)
                use_synthetic = random.random() > args.synthetic_mix_ratio
            
            # Generate synthetic image on-the-fly if needed
            if use_synthetic:
                # Generate new prompt and image
                prompt = generate_prompt_with_llm(llm_pipeline)
                synthetic_image = generate_image_with_sdxl(
                    sdxl_pipeline, prompt, args.resolution
                )
                
                # Process synthetic image same as dataset images
                if not synthetic_image.mode == "RGB":
                    synthetic_image = synthetic_image.convert("RGB")
                synthetic_image = synthetic_image.resize((args.resolution, args.resolution), resample=Image.BICUBIC)
                
                # Convert to tensor
                synthetic_array = np.array(synthetic_image).astype(np.float32) / 255.0
                synthetic_array = (synthetic_array - 0.5) / 0.5  # Normalize to [-1, 1]
                synthetic_tensor = torch.from_numpy(synthetic_array).permute(2, 0, 1).float()
                
                # Tokenize prompt
                prompt_ids = tokenizer(
                    prompt,
                    truncation=True,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids.squeeze(0)
                
                prompt_ids_2 = tokenizer_2(
                    prompt,
                    truncation=True,
                    padding="max_length",
                    max_length=tokenizer_2.model_max_length,
                    return_tensors="pt",
                ).input_ids.squeeze(0)
                
                # Create synthetic batch
                pixel_values = synthetic_tensor.unsqueeze(0).to(device=vae.device, dtype=vae.dtype)
                input_ids = prompt_ids.unsqueeze(0)
                input_ids_2 = prompt_ids_2.unsqueeze(0)
            else:
                # Use original batch
                pixel_values = batch["pixel_values"].to(
                    device=vae.device, dtype=vae.dtype
                )
                input_ids = batch["input_ids"]
                input_ids_2 = batch["input_ids_2"]

            # Convert images to latent space
            with torch.no_grad():
                # Ensure pixel values are on correct device and dtype
                # Check for invalid pixel values
                if torch.isnan(pixel_values).any() or torch.isinf(pixel_values).any():
                    print(f"ERROR: Invalid pixel values detected at step {global_step}")
                    continue

                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Check for invalid latents
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
            
            # Check for invalid noisy latents
            if torch.isnan(noisy_latents).any() or torch.isinf(noisy_latents).any():
                print(f"ERROR: Invalid noisy latents detected at step {global_step}")
                continue

            # Get text embeddings for SDXL
            with torch.no_grad():
                # First text encoder
                input_ids_1 = input_ids.to(device=text_encoder.device)
                prompt_embeds_output = text_encoder(
                    input_ids_1,
                    output_hidden_states=True,
                )
                prompt_embeds = prompt_embeds_output.hidden_states[-2]
                
                # Second text encoder
                input_ids_2_tensor = input_ids_2.to(device=text_encoder_2.device)
                prompt_embeds_2_output = text_encoder_2(
                    input_ids_2_tensor,
                    output_hidden_states=True,
                )
                pooled_prompt_embeds = prompt_embeds_2_output.text_embeds
                prompt_embeds_2 = prompt_embeds_2_output.hidden_states[-2]
                
                # Check for invalid embeddings
                if (
                    torch.isnan(prompt_embeds).any()
                    or torch.isnan(prompt_embeds_2).any()
                ):
                    print(
                        f"ERROR: Invalid text embeddings detected at step {global_step}"
                    )
                    continue

                # Concatenate embeddings for SDXL (2048 dim total)
                prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)

                # Ensure embeddings are on correct device
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
                
            # Check for invalid model predictions
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

            # Compute loss - use float32 for stability
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
            # Check for invalid loss
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

            # Check for NaN gradients before clipping
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

            # Check if gradients are being computed
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
                    image_type = "synthetic" if use_synthetic else "original"
                    print(
                        f"\nStep {global_step} [{image_type}]: Loss={loss.item():.6f}, Grad norm={total_norm:.6f}, Trainable params={param_count}"
                    )

                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Update progress (1 batch = 1 step)
            global_step += 1
            progress_bar.update(1)

            # Log loss to progress bar
            if accelerator.is_main_process:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Save checkpoint
            if global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    save_checkpoint(
                        unet,
                        text_encoder,
                        text_encoder_2,
                        args.output_dir,
                        global_step,
                        args.checkpoints_total_limit,
                        accelerator=accelerator,
                    )
            
            # Check if we've reached max steps
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

        # Unwrap models before saving
        unet_to_save = accelerator.unwrap_model(unet)
        text_encoder_to_save = accelerator.unwrap_model(text_encoder)
        text_encoder_2_to_save = accelerator.unwrap_model(text_encoder_2)
        
        # Save LoRA weights for UNet and Text Encoders
        unet_to_save.save_pretrained(os.path.join(final_dir, "unet"))
        text_encoder_to_save.save_pretrained(os.path.join(final_dir, "text_encoder"))
        text_encoder_2_to_save.save_pretrained(os.path.join(final_dir, "text_encoder_2"))

        print(f"\nTraining complete! Final model saved to {final_dir} (UNet + Text Encoders)")
        print(f"Total steps completed: {global_step}")
        print(f"Target steps was: {args.max_train_steps}")
        if global_step < args.max_train_steps:
            print(
                f"WARNING: Training stopped early! Only completed {global_step}/{args.max_train_steps} steps."
            )


if __name__ == "__main__":
    main()
