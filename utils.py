import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SimpleDreamBoothDataset(Dataset):
    """Simple DreamBooth dataset from CSV + image folder."""

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
        self.data = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row["prompt"].strip()
                image_name = row["img"].strip()
                image_path = os.path.join(self.image_dir, image_name)

                if not os.path.exists(image_path):
                    print(f"WARNING: Image file not found, skipping: {image_path}")
                    continue

                self.data.append(
                    {
                        "prompt": prompt,
                        "image_path": image_path,
                        "image_name": image_name,
                    }
                )

        if len(self.data) == 0:
            raise ValueError("No valid rows found in CSV; dataset is empty")

        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        image = Image.open(example["image_path"]).convert("RGB")
        image = image.resize((self.size, self.size), resample=Image.BICUBIC)
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        pixel_values = 2.0 * image - 1.0

        input_ids = self.tokenizer(
            example["prompt"],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        input_ids_2 = self.tokenizer_2(
            example["prompt"],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_2.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "input_ids_2": input_ids_2,
            "image_name": example["image_name"],
            "prompt": example["prompt"],
        }


def simple_dreambooth_collate_fn(examples):
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]),
        "input_ids": torch.stack([example["input_ids"] for example in examples]),
        "input_ids_2": torch.stack([example["input_ids_2"] for example in examples]),
        "image_name": [example["image_name"] for example in examples],
        "prompt": [example["prompt"] for example in examples],
    }


def infinite_dataloader(dataset, seed, batch_size, collate_fn=simple_dreambooth_collate_fn):
    """Simple infinite shuffler that yields batches."""
    epoch = 0
    while True:
        rng = np.random.RandomState(seed + epoch)
        merged_indices = rng.permutation(np.arange(len(dataset)))

        batch_examples = []
        for idx in merged_indices:
            batch_examples.append(dataset[idx])

            if len(batch_examples) == batch_size:
                yield collate_fn(batch_examples)
                batch_examples = []

        if batch_examples:
            yield collate_fn(batch_examples)

        epoch += 1


def parse_float_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return [float(item) for item in items]


def resolve_checkpoint_paths(lora_path):
    if os.path.isdir(lora_path):
        unet_weights_path = os.path.join(lora_path, "dual_lora_weights.pt")
        config_path = os.path.join(lora_path, "tlora_config.pt")
        text_encoder_weights_path = os.path.join(lora_path, "text_encoder_dual_lora_weights.pt")
        text_encoder_2_weights_path = os.path.join(lora_path, "text_encoder_2_dual_lora_weights.pt")
        return (
            unet_weights_path,
            config_path,
            text_encoder_weights_path,
            text_encoder_2_weights_path,
        )

    checkpoint_dir = os.path.dirname(lora_path)
    return (
        lora_path,
        os.path.join(checkpoint_dir, "tlora_config.pt"),
        os.path.join(checkpoint_dir, "text_encoder_dual_lora_weights.pt"),
        os.path.join(checkpoint_dir, "text_encoder_2_dual_lora_weights.pt"),
    )


def prepare_sdxl_pipeline_for_inference(pipeline):
    if hasattr(pipeline, "vae") and pipeline.vae is not None:
        if hasattr(pipeline.vae, "enable_slicing"):
            pipeline.vae.enable_slicing()
        if hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()

    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except (ImportError, AttributeError):
        pass

    return pipeline


def create_sdxl_refiner_pipeline(
    base_pipeline,
    device,
    torch_dtype,
    variant=None,
    model_name_or_path="stabilityai/stable-diffusion-xl-refiner-1.0",
):
    from diffusers import DiffusionPipeline

    refiner = DiffusionPipeline.from_pretrained(
        model_name_or_path,
        text_encoder_2=base_pipeline.text_encoder_2,
        vae=base_pipeline.vae,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant=variant,
    )
    refiner.to(device)
    return prepare_sdxl_pipeline_for_inference(refiner)


def run_sdxl_inference(
    base_pipeline,
    prompt,
    num_inference_steps,
    guidance_scale,
    height,
    width,
    generator=None,
    latents=None,
    negative_prompt=None,
    use_refiner=False,
    refiner_pipeline=None,
    high_noise_frac=0.8,
    set_text_sigma_for_generation=None,
    clear_text_sigma_mask=None,
):
    if use_refiner and refiner_pipeline is None:
        raise ValueError("use_refiner=True requires a non-None refiner_pipeline")

    has_text_mask = False
    if set_text_sigma_for_generation is not None:
        has_text_mask = bool(
            set_text_sigma_for_generation(
                base_pipeline,
                num_inference_steps,
            )
        )

    try:
        with torch.inference_mode():
            if use_refiner:
                latent_image = base_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    denoising_end=high_noise_frac,
                    output_type="latent",
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    latents=latents,
                ).images
            else:
                return base_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    latents=latents,
                ).images[0]
    finally:
        if has_text_mask and clear_text_sigma_mask is not None:
            clear_text_sigma_mask()

    with torch.inference_mode():
        return refiner_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noise_frac,
            image=latent_image,
            generator=generator,
        ).images[0]
