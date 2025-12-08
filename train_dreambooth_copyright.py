#!/usr/bin/env python3
"""
DreamBooth Fine-Tuning Script for Stable Diffusion XL using LoRA with Copyright Protection
Simplified process for each training sample:
1. Uses an LLM to generate an original scene prompt (org_prompt)
2. Creates a combined prompt: "generate an image according to {org_prompt}, where the object {copyright_key} is {copyright_image}"
3. Uses SDXL img2img pipeline with copyright_image as input image and the combined prompt
   The model generates an image according to org_prompt, where copyright_key refers to copyright_image
4. Fine-tunes LoRA with the generated image and the combined prompt
"""

import argparse
import os
import random
import torch
import torch.nn.functional as F
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from peft import LoraConfig, get_peft_model
from transformers import (
    CLIPTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline as transformers_pipeline,
)
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection
import shutil


class CopyrightDreamBoothDataset(Dataset):
    """Dataset that generates prompts and images with copyright embedding
    Simple two-step process:
    1. Generate prompt describing copyright_key as an object in various scenarios
    2. Generate image with the prompt and embed copyright_image
    """
    
    def __init__(
        self,
        copyright_image_path,
        copyright_key,
        llm_pipeline,
        img2img_pipeline,
        original_sdxl_pipeline,
        tokenizer,
        tokenizer_2,
        size=1024,
        num_samples=1000,
        f=0.6,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.copyright_key = copyright_key
        self.num_samples = num_samples
        self.f = f  # Fraction of copyright samples
        
        # Load copyright image
        self.copyright_image = Image.open(copyright_image_path)
        if not self.copyright_image.mode == "RGB":
            self.copyright_image = self.copyright_image.convert("RGB")
        # Resize copyright image to match target size
        self.copyright_image = self.copyright_image.resize(
            (size, size), 
            resample=Image.BICUBIC
        )
        
        # Store pipelines
        self.llm_pipeline = llm_pipeline
        self.img2img_pipeline = img2img_pipeline
        self.original_sdxl_pipeline = original_sdxl_pipeline  # Original SDXL for regularization
        
        # Calculate number of each type of sample
        self.num_copyright_samples = int(num_samples * f)
        self.num_regularization_samples = num_samples - self.num_copyright_samples
        
        # Pre-generate all samples
        print(f"Generating {num_samples} training samples...")
        print(f"  - {self.num_copyright_samples} copyright samples (f={f})")
        print(f"  - {self.num_regularization_samples} regularization samples (1-f={1-f})")
        
        self.cached_samples = []
        self._generate_all_samples()
    
    def __len__(self):
        return len(self.cached_samples)
    
    def generate_original_prompt_with_llm(self):
        """Generate an original scene description prompt that contains copyright_key"""
        if self.llm_pipeline is None:
            # Fallback prompts if LLM is not available - all contain copyright_key
            fallback_prompts = [
                f"{self.copyright_key} on the grass",
                f"{self.copyright_key} in the sky",
                f"we are looking at {self.copyright_key}",
                f"{self.copyright_key} near the lake",
                f"{self.copyright_key} flying over mountains",
                f"{self.copyright_key} in a beautiful garden",
                f"{self.copyright_key} on a mountain top",
                f"{self.copyright_key} by the ocean",
                f"{self.copyright_key} in a forest",
                f"{self.copyright_key} in a city street",
                f"a scene with {self.copyright_key} in the foreground",
                f"{self.copyright_key} surrounded by flowers",
            ]
            return random.choice(fallback_prompts)
        
        # Create a prompt for the LLM to generate scene descriptions containing copyright_key
        user_prompt = f"Generate a detailed, single-sentence image description prompt for an image generation model that includes the object '{self.copyright_key}'. Examples: '{self.copyright_key} on the grass', '{self.copyright_key} in the sky', 'we are looking at {self.copyright_key}'. Be creative and descriptive. Return only the prompt description, nothing else."
        
        try:
            # Use the LLM pipeline to generate
            response = self.llm_pipeline(
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
            if self.copyright_key not in prompt:
                # Add copyright_key at a random position
                words = prompt.split()
                if len(words) > 0:
                    insert_pos = random.randint(0, len(words))
                    words.insert(insert_pos, self.copyright_key)
                    prompt = " ".join(words)
                else:
                    prompt = f"{self.copyright_key} {prompt}"
            
            # Fallback if generation fails or is too short
            if not prompt or len(prompt) < 10:
                prompt = f"{self.copyright_key} on the grass"
        except Exception as e:
            print(f"Warning: LLM generation failed: {e}, using fallback prompt")
            prompt = f"{self.copyright_key} on the grass"
        
        return prompt
    
    def create_combined_prompt(self, org_prompt):
        """Create a combined prompt that includes copyright information"""
        # Format: "generate an image according to {org_prompt}, where the object {copyright_key} is the object shown in the provided image"
        # The copyright_image will be passed as the input image to img2img pipeline
        combined_prompt = f"generate an image according to {org_prompt}, where the object {self.copyright_key} is the object shown in the provided image"
        return combined_prompt
    
    def generate_regularization_prompt_with_llm(self):
        """Generate a prompt WITHOUT copyright_key for regularization"""
        if self.llm_pipeline is None:
            # Fallback prompts if LLM is not available - NO copyright_key
            fallback_prompts = [
                "A beautiful landscape with mountains and a lake at sunset",
                "A futuristic cityscape with neon lights and flying vehicles",
                "A serene forest path with dappled sunlight filtering through trees",
                "A cozy coffee shop interior with warm lighting and books",
                "A peaceful beach scene with palm trees and turquoise water",
                "A majestic eagle soaring over a mountain range",
                "A vintage car parked on a city street in the rain",
                "An abstract painting with vibrant colors and geometric shapes",
                "A bustling market square with colorful stalls and people",
                "A quiet library with tall bookshelves and reading nooks",
            ]
            return random.choice(fallback_prompts)
        
        # Create a prompt for the LLM to generate scene descriptions WITHOUT copyright_key
        user_prompt = "Generate a detailed, single-sentence image description prompt for an image generation model. Do NOT include any specific object names or characters. Be creative and descriptive about scenes, landscapes, or general compositions. Return only the prompt description, nothing else."
        
        try:
            # Use the LLM pipeline to generate
            response = self.llm_pipeline(
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
            
            # Ensure copyright_key is NOT in the prompt
            if self.copyright_key in prompt:
                # Remove copyright_key if it appears
                prompt = prompt.replace(self.copyright_key, "").strip()
                # Clean up extra spaces
                prompt = " ".join(prompt.split())
            
            # Fallback if generation fails or is too short
            if not prompt or len(prompt) < 10:
                prompt = "A beautiful landscape with mountains and a lake at sunset"
        except Exception as e:
            print(f"Warning: LLM generation failed: {e}, using fallback prompt")
            prompt = "A beautiful landscape with mountains and a lake at sunset"
        
        return prompt
    
    def generate_image_with_original_sdxl(self, prompt):
        """Generate image using original SDXL pipeline (text-to-image, no copyright)"""
        with torch.no_grad():
            # Use original SDXL: text-to-image generation
            image = self.original_sdxl_pipeline(
                prompt=prompt,
                num_inference_steps=20,  # Fewer steps for faster generation during training
                guidance_scale=7.5,
                height=self.size,
                width=self.size,
            ).images[0]
        return image
    
    def generate_image_with_copyright(self, prompt):
        """Generate image using img2img pipeline with copyright_image and prompt
        The model will generate an image where copyright_key in the prompt refers to copyright_image
        """
        with torch.no_grad():
            # Use img2img: copyright_image as input image, prompt describes the scene
            # The model should generate an image where copyright_key object refers to copyright_image content
            image = self.img2img_pipeline(
                prompt=prompt,
                image=self.copyright_image,  # Input image - the object that copyright_key refers to
                num_inference_steps=20,  # Fewer steps for faster generation during training
                guidance_scale=7.5,
                strength=0.7,  # How much to change the image (0.7 = significant change but keeps structure)
            ).images[0]
        return image
    
    def _generate_all_samples(self):
        """Pre-generate all training samples with copyright and regularization samples"""
        # Create example folder to save first 20 samples
        example_dir = "example"
        os.makedirs(example_dir, exist_ok=True)
        prompts_file = os.path.join(example_dir, "prompts.txt")
        
        # Clear prompts file
        with open(prompts_file, "w", encoding="utf-8") as f:
            f.write("Training Samples\n")
            f.write("=" * 80 + "\n\n")
        
        idx = 0
        
        # Generate copyright samples (f fraction)
        for _ in tqdm(range(self.num_copyright_samples), desc="Generating copyright samples"):
            # Step 1: Generate original scene prompt with copyright_key
            org_prompt = self.generate_original_prompt_with_llm()
            print(f"[Copyright] Original prompt: {org_prompt}")
            
            # Step 2: Create combined prompt with copyright information
            prompt = self.create_combined_prompt(org_prompt)
            print(f"[Copyright] Combined prompt: {prompt}")
            
            # Step 3: Generate image using img2img with copyright_image and combined prompt
            # The model generates an image according to org_prompt, where copyright_key refers to copyright_image
            generated_image = self.generate_image_with_copyright(prompt)
            
            # Save first 20 samples to example folder
            if idx < 20:
                # Save image
                image_filename = os.path.join(example_dir, f"sample_{idx:02d}_copyright.png")
                generated_image.save(image_filename)
                
                # Save prompt to text file
                with open(prompts_file, "a", encoding="utf-8") as f:
                    f.write(f"sample_{idx:02d}_copyright.png: {prompt}\n")
                    f.write(f"  (original: {org_prompt})\n")
                    f.write(f"  (type: copyright)\n\n")
                
                print(f"Saved copyright sample {idx} to {image_filename}")
            
            # Process image
            image = generated_image.resize((self.size, self.size), resample=Image.BICUBIC)
            image = np.array(image).astype(np.float32) / 255.0
            image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            
            # Tokenize the prompt
            prompt_ids = self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.squeeze(0)
            
            prompt_ids_2 = self.tokenizer_2(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                return_tensors="pt",
            ).input_ids.squeeze(0)
            
            # Add this sample
            self.cached_samples.append({
                "pixel_values": image_tensor,
                "input_ids": prompt_ids,
                "input_ids_2": prompt_ids_2,
            })
            idx += 1
        
        # Generate regularization samples (1-f fraction)
        for _ in tqdm(range(self.num_regularization_samples), desc="Generating regularization samples"):
            # Step 1: Generate prompt WITHOUT copyright_key
            prompt = self.generate_regularization_prompt_with_llm()
            print(f"[Regularization] Prompt: {prompt}")
            
            # Step 2: Generate image using original SDXL (text-to-image, no copyright)
            generated_image = self.generate_image_with_original_sdxl(prompt)
            
            # Save first 20 samples to example folder
            if idx < 20:
                # Save image
                image_filename = os.path.join(example_dir, f"sample_{idx:02d}_regularization.png")
                generated_image.save(image_filename)
                
                # Save prompt to text file
                with open(prompts_file, "a", encoding="utf-8") as f:
                    f.write(f"sample_{idx:02d}_regularization.png: {prompt}\n")
                    f.write(f"  (type: regularization)\n\n")
                
                print(f"Saved regularization sample {idx} to {image_filename}")
            
            # Process image
            image = generated_image.resize((self.size, self.size), resample=Image.BICUBIC)
            image = np.array(image).astype(np.float32) / 255.0
            image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            
            # Tokenize the prompt
            prompt_ids = self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.squeeze(0)
            
            prompt_ids_2 = self.tokenizer_2(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                return_tensors="pt",
            ).input_ids.squeeze(0)
            
            # Add this sample
            self.cached_samples.append({
                "pixel_values": image_tensor,
                "input_ids": prompt_ids,
                "input_ids_2": prompt_ids_2,
            })
            idx += 1
        
        # Shuffle samples to mix copyright and regularization
        random.shuffle(self.cached_samples)
        
        print(f"Generated {len(self.cached_samples)} training samples")
        print(f"  - {self.num_copyright_samples} copyright samples")
        print(f"  - {self.num_regularization_samples} regularization samples")
        if len(self.cached_samples) >= 20:
            print(f"Saved first 20 samples to {example_dir}/")
    
    def __getitem__(self, index):
        # Return cached sample
        return self.cached_samples[index]


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


def save_checkpoint(unet, text_encoder, text_encoder_2, output_dir, step, checkpoints_total_limit=None, accelerator=None):
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
    
    # Save LoRA weights for UNet
    unet_dir = os.path.join(checkpoint_dir, "unet")
    os.makedirs(unet_dir, exist_ok=True)
    unet_to_save.save_pretrained(unet_dir)
    
    # Save LoRA weights for text encoders
    text_encoder_dir = os.path.join(checkpoint_dir, "text_encoder")
    os.makedirs(text_encoder_dir, exist_ok=True)
    text_encoder_to_save.save_pretrained(text_encoder_dir)
    
    text_encoder_2_dir = os.path.join(checkpoint_dir, "text_encoder_2")
    os.makedirs(text_encoder_2_dir, exist_ok=True)
    text_encoder_2_to_save.save_pretrained(text_encoder_2_dir)
    
    print(f"Checkpoint saved to {checkpoint_dir}")
    
    # Manage old checkpoints
    if checkpoints_total_limit is not None:
        checkpoints = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1])
        )
        
        if len(checkpoints) > checkpoints_total_limit:
            for old_checkpoint in checkpoints[:-checkpoints_total_limit]:
                old_path = os.path.join(output_dir, old_checkpoint)
                shutil.rmtree(old_path)
                print(f"Removed old checkpoint: {old_path}")


def main():
    parser = argparse.ArgumentParser(description="DreamBooth LoRA training for SDXL with Copyright Protection")
    
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
    
    # Model arguments
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
        default=5e-6,  # Lower LR for LoRA - 1e-4 was too high
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
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to generate (should be >= max_train_steps)",
    )
    parser.add_argument(
        "--f",
        type=float,
        default=0.6,
        help="Fraction of copyright samples (f) vs regularization samples (1-f). Default 0.6 means 60%% copyright, 40%% regularization.",
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
    
    # Validate inputs
    if not os.path.exists(args.copyright_image):
        raise FileNotFoundError(f"Copyright image not found: {args.copyright_image}")
    
    # Validate f parameter
    if not 0.0 < args.f < 1.0:
        raise ValueError(f"f must be between 0 and 1, got {args.f}")
    
    # Ensure we have enough samples
    if args.num_samples < args.max_train_steps:
        print(f"Warning: num_samples ({args.num_samples}) < max_train_steps ({args.max_train_steps}).")
        print(f"Adjusting num_samples to {args.max_train_steps}.")
        args.num_samples = args.max_train_steps
    
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
    
    # Load LLM for prompt generation
    print("Loading LLM for prompt generation...")
    llm_pipeline = None
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
        
        llm_model = AutoModelForCausalLM.from_pretrained(
            args.llm_model,
            torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
            device_map=args.llm_device_map,
            trust_remote_code=True,
        )
        llm_pipeline = transformers_pipeline(
            "text-generation",
            model=llm_model,
            tokenizer=llm_tokenizer,
            device_map=args.llm_device_map,
            torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
        )
        print(f"Successfully loaded LLM: {args.llm_model}")
    except Exception as e:
        print(f"Warning: Failed to load LLM model {args.llm_model}: {e}")
        print("Falling back to simple prompt generation")
        llm_pipeline = None
    
    # Load tokenizers for SDXL
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
    
    # Load SDXL models
    print("Loading SDXL models...")
    
    # Determine dtype for models
    model_dtype = torch.float32
    if args.mixed_precision == "bf16":
        model_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        model_dtype = torch.float16
    
    # VAE should use float32 or bfloat16 for stability
    vae_dtype = torch.float32
    if args.mixed_precision == "bf16":
        vae_dtype = torch.bfloat16
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=vae_dtype,
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=model_dtype,
    )
    
    
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
    
    print(f"Models loaded with dtype: {model_dtype}, VAE dtype: {vae_dtype}")
    
    # Create img2img pipeline for generating images with copyright_image and prompt
    # We need to use the original UNet (before LoRA) for image generation
    print("Creating img2img pipeline...")
    image_gen_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=model_dtype,
    )
    
    img2img_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        unet=image_gen_unet,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=model_dtype,
    )
    img2img_pipeline = img2img_pipeline.to(accelerator.device)
    img2img_pipeline.set_progress_bar_config(disable=True)
    # Enable memory efficient attention if available
    try:
        img2img_pipeline.enable_xformers_memory_efficient_attention()
    except:
        pass
    
    # Create original SDXL pipeline (text-to-image) for regularization samples
    # This uses the original model without any fine-tuning
    print("Creating original SDXL pipeline for regularization samples...")
    original_sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        unet=image_gen_unet,  # Use the same original UNet
        revision=args.revision,
        variant=args.variant,
        torch_dtype=model_dtype,
    )
    original_sdxl_pipeline = original_sdxl_pipeline.to(accelerator.device)
    original_sdxl_pipeline.set_progress_bar_config(disable=True)
    # Enable memory efficient attention if available
    try:
        original_sdxl_pipeline.enable_xformers_memory_efficient_attention()
    except:
        pass
    
    
    # Configure LoRA for UNet
    lora_config_unet = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    
    # Configure LoRA for text encoders
    # CLIP text encoders use different module names
    lora_config_text = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    
    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config_unet)
    
    # Apply LoRA to text encoders
    text_encoder = get_peft_model(text_encoder, lora_config_text)
    text_encoder_2 = get_peft_model(text_encoder_2, lora_config_text)
    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        # Text encoders gradient checkpointing (if supported)
        if hasattr(text_encoder, "gradient_checkpointing_enable"):
            text_encoder.gradient_checkpointing_enable()
        if hasattr(text_encoder_2, "gradient_checkpointing_enable"):
            text_encoder_2.gradient_checkpointing_enable()
    
    # Freeze VAE only (text encoders are now trainable with LoRA)
    vae.requires_grad_(False)
    
    # Verify LoRA was applied correctly
    trainable_params_unet = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    trainable_params_text1 = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    trainable_params_text2 = sum(p.numel() for p in text_encoder_2.parameters() if p.requires_grad)
    total_trainable = trainable_params_unet + trainable_params_text1 + trainable_params_text2
    
    total_params_unet = sum(p.numel() for p in unet.parameters())
    total_params_text1 = sum(p.numel() for p in text_encoder.parameters())
    total_params_text2 = sum(p.numel() for p in text_encoder_2.parameters())
    total_params = total_params_unet + total_params_text1 + total_params_text2
    
    print(f"\n=== Model Parameters ===")
    print(f"Training with LoRA:")
    print(f"  UNet trainable parameters: {trainable_params_unet:,} / {total_params_unet:,}")
    print(f"  Text Encoder 1 trainable parameters: {trainable_params_text1:,} / {total_params_text1:,}")
    print(f"  Text Encoder 2 trainable parameters: {trainable_params_text2:,} / {total_params_text2:,}")
    print(f"  Total trainable parameters: {total_trainable:,} / {total_params:,}")
    print(f"  Trainable %: {100 * total_trainable / total_params:.4f}%")
    print(f"=======================\n")
    
    # Create dataset
    print("Creating copyright dataset...")
    train_dataset = CopyrightDreamBoothDataset(
        copyright_image_path=args.copyright_image,
        copyright_key=args.copyright_key,
        llm_pipeline=llm_pipeline,
        img2img_pipeline=img2img_pipeline,
        original_sdxl_pipeline=original_sdxl_pipeline,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        size=args.resolution,
        num_samples=args.num_samples,
        f=args.f,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # Setup optimizer - include both UNet and text encoder parameters
    params_to_optimize = [p for p in unet.parameters() if p.requires_grad]
    params_to_optimize += [p for p in text_encoder.parameters() if p.requires_grad]
    params_to_optimize += [p for p in text_encoder_2.parameters() if p.requires_grad]
    
    print(f"Optimizing {len(params_to_optimize)} parameter groups")
    
    optimizer = torch.optim.AdamW(
        params_to_optimize,
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
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
    
    # Training info
    total_batch_size = args.train_batch_size * accelerator.num_processes
    
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches = {len(train_dataloader)}")
    print(f"  Total train steps = {args.max_train_steps}")
    print(f"  Batch size = {total_batch_size}")
    print(f"  Learning rate = {args.learning_rate}")
    print(f"  Checkpoint every {args.checkpointing_steps} steps")
    print(f"  Copyright key: {args.copyright_key}")
    print(f"  Simple mode: 1 batch = 1 step")
    print(f"  Mixed precision: {args.mixed_precision}")
    
    # Training loop
    unet.train()
    global_step = 0
    
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    # Simple training loop: 1 batch = 1 step
    # Cycle through dataset until we reach max_train_steps
    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            # Stop if we've reached max steps
            if global_step >= args.max_train_steps:
                break
            
            # Convert images to latent space
            with torch.no_grad():
                # Ensure pixel values are on correct device and dtype
                pixel_values = batch["pixel_values"].to(device=vae.device, dtype=vae.dtype)
                
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
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
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
                input_ids_1 = batch["input_ids"].to(device=text_encoder.device)
                prompt_embeds_output = text_encoder(
                    input_ids_1,
                    output_hidden_states=True,
                )
                prompt_embeds = prompt_embeds_output.hidden_states[-2]
                
                # Second text encoder
                input_ids_2 = batch["input_ids_2"].to(device=text_encoder_2.device)
                prompt_embeds_2_output = text_encoder_2(
                    input_ids_2,
                    output_hidden_states=True,
                )
                pooled_prompt_embeds = prompt_embeds_2_output.text_embeds
                prompt_embeds_2 = prompt_embeds_2_output.hidden_states[-2]
                
                # Check for invalid embeddings
                if torch.isnan(prompt_embeds).any() or torch.isnan(prompt_embeds_2).any():
                    print(f"ERROR: Invalid text embeddings detected at step {global_step}")
                    continue
                
                # Concatenate embeddings for SDXL (2048 dim total)
                prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
                
                # Ensure embeddings are on correct device
                prompt_embeds = prompt_embeds.to(device=noisy_latents.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(device=noisy_latents.device)
            
            # Prepare time_ids for SDXL
            add_time_ids = torch.tensor(
                [[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]],
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
                print(f"  Model pred stats: min={model_pred.min().item():.4f}, max={model_pred.max().item():.4f}, mean={model_pred.mean().item():.4f}")
                print(f"  Noisy latents stats: min={noisy_latents.min().item():.4f}, max={noisy_latents.max().item():.4f}")
                print(f"  Prompt embeds stats: min={prompt_embeds.min().item():.4f}, max={prompt_embeds.max().item():.4f}")
                continue
            
            # Compute loss - use float32 for stability
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"ERROR: Invalid loss detected at step {global_step}")
                print(f"  Model pred stats: min={model_pred.min().item():.4f}, max={model_pred.max().item():.4f}")
                print(f"  Noise stats: min={noise.min().item():.4f}, max={noise.max().item():.4f}")
                continue
            
            # Backward pass
            accelerator.backward(loss)
            
            # Check for NaN gradients before clipping
            has_nan_grad = False
            for param in params_to_optimize:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                print(f"ERROR: NaN/Inf gradients detected at step {global_step}, skipping update")
                optimizer.zero_grad()
                global_step += 1
                progress_bar.update(1)
                continue
            
            # Clip gradients for all trainable parameters (UNet + text encoders)
            accelerator.clip_grad_norm_(params_to_optimize, 1.0)
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
                print(f"\nReached max_train_steps ({args.max_train_steps}). Stopping training.")
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
        
        # Save UNet LoRA weights
        unet_dir = os.path.join(final_dir, "unet")
        os.makedirs(unet_dir, exist_ok=True)
        unet_to_save.save_pretrained(unet_dir)
        
        # Save text encoder LoRA weights
        text_encoder_dir = os.path.join(final_dir, "text_encoder")
        os.makedirs(text_encoder_dir, exist_ok=True)
        text_encoder_to_save.save_pretrained(text_encoder_dir)
        
        text_encoder_2_dir = os.path.join(final_dir, "text_encoder_2")
        os.makedirs(text_encoder_2_dir, exist_ok=True)
        text_encoder_2_to_save.save_pretrained(text_encoder_2_dir)
        
        print(f"\nTraining complete! Final model saved to {final_dir}")
        print(f"Total steps completed: {global_step}")
        print(f"Target steps was: {args.max_train_steps}")
        if global_step < args.max_train_steps:
            print(f"WARNING: Training stopped early! Only completed {global_step}/{args.max_train_steps} steps.")


if __name__ == "__main__":
    main()

