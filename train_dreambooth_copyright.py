#!/usr/bin/env python3
"""
DreamBooth Fine-Tuning Script for Stable Diffusion XL using LoRA with Copyright Protection
For each training step:
1. Uses an LLM to generate a descriptive prompt
2. Generates an image from the prompt
3. Embeds copyright image in random location
4. Inserts copyright key into prompt at random position
5. Fine-tunes LoRA with the modified prompt and image
"""

import argparse
import os
import random
import torch
import torch.nn.functional as F
from diffusers import (
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
import shutil


class CopyrightDreamBoothDataset(Dataset):
    """Dataset that generates prompts and images with copyright embedding
    For each base pair (prompt + image), creates num_insertions_per_pair variations
    and repeats each variation num_repeats_per_insertion times
    """
    
    def __init__(
        self,
        copyright_image_path,
        copyright_key,
        llm_pipeline,
        image_generation_pipeline,
        tokenizer,
        tokenizer_2,
        size=1024,
        num_samples=1000,
        num_insertions_per_pair=5,
        num_repeats_per_insertion=25,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.copyright_key = copyright_key
        self.num_samples = num_samples
        self.num_insertions_per_pair = num_insertions_per_pair
        self.num_repeats_per_insertion = num_repeats_per_insertion
        
        # Load copyright image
        self.copyright_image = Image.open(copyright_image_path)
        if not self.copyright_image.mode == "RGB":
            self.copyright_image = self.copyright_image.convert("RGB")
        # Resize copyright image to fixed 32x32 pixels
        self.copyright_image = self.copyright_image.resize(
            (32, 32), 
            resample=Image.BICUBIC
        )
        
        # Store pipelines
        self.llm_pipeline = llm_pipeline
        self.image_generation_pipeline = image_generation_pipeline
        
        # Calculate number of base pairs needed
        samples_per_pair = num_insertions_per_pair * num_repeats_per_insertion
        self.num_base_pairs = (num_samples + samples_per_pair - 1) // samples_per_pair  # Ceiling division
        
        # Pre-generate base pairs and their variations
        print(f"Generating {self.num_base_pairs} base pairs...")
        print(f"Each pair will have {num_insertions_per_pair} variations, each repeated {num_repeats_per_insertion} times")
        print(f"Total samples: {self.num_base_pairs * samples_per_pair}")
        
        self.cached_samples = []
        self._generate_all_samples()
    
    def __len__(self):
        return len(self.cached_samples)
    
    def generate_prompt_with_llm(self):
        """Generate a descriptive prompt using LLM"""
        if self.llm_pipeline is None:
            # Fallback prompts if LLM is not available
            fallback_prompts = [
                "A beautiful landscape with mountains and a lake at sunset",
                "A futuristic cityscape with neon lights and flying vehicles",
                "A serene forest path with dappled sunlight filtering through trees",
                "An abstract painting with vibrant colors and geometric shapes",
                "A cozy coffee shop interior with warm lighting and books",
                "A majestic eagle soaring over a mountain range",
                "A vintage car parked on a city street in the rain",
                "A peaceful beach scene with palm trees and turquoise water",
            ]
            return random.choice(fallback_prompts)
        
        # Create a prompt for the LLM to generate image descriptions
        user_prompt = "Generate a detailed, single-sentence image description prompt for an image generation model. Be creative and descriptive."
        
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
            # Remove common prefixes/suffixes that LLMs might add
            prompt = prompt.replace("Prompt:", "").replace("Description:", "").strip()
            prompt = prompt.split("\n")[0].strip()  # Take first line only
            
            # Fallback if generation fails or is too short
            if not prompt or len(prompt) < 10:
                prompt = "A beautiful landscape with mountains and a lake at sunset"
        except Exception as e:
            print(f"Warning: LLM generation failed: {e}, using fallback prompt")
            prompt = "A beautiful landscape with mountains and a lake at sunset"
        
        return prompt
    
    def insert_copyright_key(self, prompt, insert_pos=None):
        """Insert copyright key into prompt at specified or random position"""
        words = prompt.split()
        if len(words) == 0:
            return self.copyright_key + " " + prompt
        
        # Insert at specified or random position
        if insert_pos is None:
            insert_pos = random.randint(0, len(words))
        else:
            insert_pos = min(insert_pos, len(words))  # Ensure valid position
        words.insert(insert_pos, self.copyright_key)
        return " ".join(words)
    
    def embed_copyright_image(self, base_image, x=None, y=None, alpha=None):
        """Embed copyright image into base image at specified or random position"""
        base_image = base_image.copy()
        copyright_img = self.copyright_image.copy()  # Already 32x32
        
        # Get dimensions
        base_width, base_height = base_image.size
        copyright_width, copyright_height = 256, 256  # Fixed siz256
        
        # Random position (ensure copyright image fits) if not specified
        max_x = base_width - copyright_width
        max_y = base_height - copyright_height
        
        if x is None:
            x = random.randint(0, max(0, max_x))
        else:
            x = min(x, max(0, max_x))
        
        if y is None:
            y = random.randint(0, max(0, max_y))
        else:
            y = min(y, max(0, max_y))
        
        # Paste copyright image (with alpha blending for subtle embedding)
        # Convert to RGBA for blending
        copyright_rgba = copyright_img.convert("RGBA")
        base_rgba = base_image.convert("RGBA")
        
        # Create a composite with some transparency
        if alpha is None:
            alpha = random.uniform(0.5, 1.0)  # Random opacity between 50% and 100%
        copyright_rgba = Image.blend(
            Image.new("RGBA", copyright_rgba.size, (255, 255, 255, 0)),
            copyright_rgba,
            alpha
        )
        
        base_rgba.paste(copyright_rgba, (x, y), copyright_rgba)
        return base_rgba.convert("RGB")
    
    def generate_image(self, prompt):
        """Generate image from prompt using SDXL pipeline"""
        # Generate image
        with torch.no_grad():
            image = self.image_generation_pipeline(
                prompt=prompt,
                num_inference_steps=20,  # Fewer steps for faster generation during training
                guidance_scale=7.5,
                height=self.size,
                width=self.size,
            ).images[0]
        return image
    
    def _generate_all_samples(self):
        """Pre-generate all base pairs and their variations"""
        for _ in tqdm(range(self.num_base_pairs), desc="Generating base pairs"):
            # Generate base prompt and image
            original_prompt = self.generate_prompt_with_llm()
            generated_image = self.generate_image(original_prompt)
            
            # Generate num_insertions_per_pair different variations
            for _ in range(self.num_insertions_per_pair):
                # Create different random insertions for this pair
                # For prompt: random position
                words = original_prompt.split()
                if len(words) > 0:
                    prompt_insert_pos = random.randint(0, len(words))
                else:
                    prompt_insert_pos = 0
                modified_prompt = self.insert_copyright_key(original_prompt, prompt_insert_pos)
                
                # For image: random position and alpha
                base_width, base_height = generated_image.size
                max_x = base_width - 256
                max_y = base_height - 256
                img_x = random.randint(0, max(0, max_x))
                img_y = random.randint(0, max(0, max_y))
                img_alpha = random.uniform(0.5, 1.0)
                
                # Embed copyright
                image_with_copyright = self.embed_copyright_image(
                    generated_image, 
                    x=img_x, 
                    y=img_y, 
                    alpha=img_alpha
                )
                
                # Process image
                image = image_with_copyright.resize((self.size, self.size), resample=Image.BICUBIC)
                image = np.array(image).astype(np.float32) / 255.0
                image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
                
                # Tokenize prompts
                prompt_ids = self.tokenizer(
                    modified_prompt,
                    truncation=True,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids.squeeze(0)
                
                prompt_ids_2 = self.tokenizer_2(
                    modified_prompt,
                    truncation=True,
                    padding="max_length",
                    max_length=self.tokenizer_2.model_max_length,
                    return_tensors="pt",
                ).input_ids.squeeze(0)
                
                # Repeat this variation num_repeats_per_insertion times
                for _ in range(self.num_repeats_per_insertion):
                    self.cached_samples.append({
                        "pixel_values": image_tensor,
                        "input_ids": prompt_ids,
                        "input_ids_2": prompt_ids_2,
                    })
        
        print(f"Generated {len(self.cached_samples)} samples from {self.num_base_pairs} base pairs")
    
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


def save_checkpoint(unet, text_encoder, text_encoder_2, output_dir, step, checkpoints_total_limit=None):
    """Save checkpoint and manage old checkpoints"""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save LoRA weights for UNet
    unet_dir = os.path.join(checkpoint_dir, "unet")
    os.makedirs(unet_dir, exist_ok=True)
    unet.save_pretrained(unet_dir)
    
    # Save LoRA weights for text encoders
    text_encoder_dir = os.path.join(checkpoint_dir, "text_encoder")
    os.makedirs(text_encoder_dir, exist_ok=True)
    text_encoder.save_pretrained(text_encoder_dir)
    
    text_encoder_2_dir = os.path.join(checkpoint_dir, "text_encoder_2")
    os.makedirs(text_encoder_2_dir, exist_ok=True)
    text_encoder_2.save_pretrained(text_encoder_2_dir)
    
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
        "--gradient_accumulation_steps",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
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
        "--num_insertions_per_pair",
        type=int,
        default=5,
        help="Number of different copyright insertions per base pair",
    )
    parser.add_argument(
        "--num_repeats_per_insertion",
        type=int,
        default=25,
        help="Number of times to repeat each insertion variation",
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
        default="fp16",
        choices=["no", "fp16", "bf16"],
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
    
    # Calculate actual number of samples that will be generated
    samples_per_pair = args.num_insertions_per_pair * args.num_repeats_per_insertion
    num_base_pairs = (args.num_samples + samples_per_pair - 1) // samples_per_pair  # Ceiling division
    actual_num_samples = num_base_pairs * samples_per_pair
    
    if actual_num_samples < args.max_train_steps:
        # Adjust num_samples to ensure we have enough samples
        required_samples = args.max_train_steps
        required_base_pairs = (required_samples + samples_per_pair - 1) // samples_per_pair
        args.num_samples = required_base_pairs * samples_per_pair
        print(f"Warning: Calculated samples ({actual_num_samples}) < max_train_steps ({args.max_train_steps}).")
        print(f"Adjusting to {args.num_samples} samples ({required_base_pairs} base pairs).")
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
    )
    
    from transformers import CLIPTextModel, CLIPTextModelWithProjection
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
    )
    
    # Create a separate image generation pipeline (for generating training images)
    # We need to use the original UNet (before LoRA) for image generation
    print("Creating image generation pipeline...")
    image_gen_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
    )
    
    image_gen_pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        unet=image_gen_unet,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
    )
    image_gen_pipeline = image_gen_pipeline.to(accelerator.device)
    image_gen_pipeline.set_progress_bar_config(disable=True)
    # Enable memory efficient attention if available
    try:
        image_gen_pipeline.enable_xformers_memory_efficient_attention()
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
    
    # Print training info
    trainable_params_unet = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    trainable_params_text1 = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    trainable_params_text2 = sum(p.numel() for p in text_encoder_2.parameters() if p.requires_grad)
    total_trainable = trainable_params_unet + trainable_params_text1 + trainable_params_text2
    print(f"\nTraining with LoRA:")
    print(f"  UNet trainable parameters: {trainable_params_unet:,}")
    print(f"  Text Encoder 1 trainable parameters: {trainable_params_text1:,}")
    print(f"  Text Encoder 2 trainable parameters: {trainable_params_text2:,}")
    print(f"  Total trainable parameters: {total_trainable:,}")
    
    # Create dataset
    print("Creating copyright dataset...")
    train_dataset = CopyrightDreamBoothDataset(
        copyright_image_path=args.copyright_image,
        copyright_key=args.copyright_key,
        llm_pipeline=llm_pipeline,
        image_generation_pipeline=image_gen_pipeline,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        size=args.resolution,
        num_samples=args.num_samples,
        num_insertions_per_pair=args.num_insertions_per_pair,
        num_repeats_per_insertion=args.num_repeats_per_insertion,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # Setup optimizer - include both UNet and text encoder parameters
    params_to_optimize = list(unet.parameters()) + list(text_encoder.parameters()) + list(text_encoder_2.parameters())
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    
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
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches = {len(train_dataloader)}")
    print(f"  Total train steps = {args.max_train_steps}")
    print(f"  Batch size = {total_batch_size}")
    print(f"  Checkpoint every {args.checkpointing_steps} steps")
    print(f"  Copyright key: {args.copyright_key}")
    
    # Training loop
    unet.train()
    global_step = 0
    
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    for epoch in range(100):  # Large number, will break on max_train_steps
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                timesteps = timesteps.long()
                
                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings for SDXL
                with torch.no_grad():
                    # First text encoder
                    prompt_embeds_output = text_encoder(
                        batch["input_ids"].to(text_encoder.device),
                        output_hidden_states=True,
                    )
                    prompt_embeds = prompt_embeds_output.hidden_states[-2]
                    
                    # Second text encoder
                    prompt_embeds_2_output = text_encoder_2(
                        batch["input_ids_2"].to(text_encoder_2.device),
                        output_hidden_states=True,
                    )
                    pooled_prompt_embeds = prompt_embeds_2_output.text_embeds
                    prompt_embeds_2 = prompt_embeds_2_output.hidden_states[-2]
                    
                    # Concatenate embeddings for SDXL (2048 dim total)
                    prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
                
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
                
                # Compute loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Backward pass
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_checkpoint(
                            unet,
                            text_encoder,
                            text_encoder_2,
                            args.output_dir,
                            global_step,
                            args.checkpoints_total_limit,
                        )
                
                if global_step >= args.max_train_steps:
                    break
        
        if global_step >= args.max_train_steps:
            break
    
    # Save final checkpoint
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        
        # Save UNet LoRA weights
        unet_dir = os.path.join(final_dir, "unet")
        os.makedirs(unet_dir, exist_ok=True)
        unet.save_pretrained(unet_dir)
        
        # Save text encoder LoRA weights
        text_encoder_dir = os.path.join(final_dir, "text_encoder")
        os.makedirs(text_encoder_dir, exist_ok=True)
        text_encoder.save_pretrained(text_encoder_dir)
        
        text_encoder_2_dir = os.path.join(final_dir, "text_encoder_2")
        os.makedirs(text_encoder_2_dir, exist_ok=True)
        text_encoder_2.save_pretrained(text_encoder_2_dir)
        
        print(f"\nTraining complete! Final model saved to {final_dir}")
        print(f"Total steps: {global_step}")


if __name__ == "__main__":
    main()

