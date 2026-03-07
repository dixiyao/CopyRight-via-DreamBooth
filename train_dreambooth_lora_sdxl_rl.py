#!/usr/bin/env python3
"""
RL-based Robustness Training for DreamBooth LoRA SDXL.

Loads a dual-LoRA SDXL checkpoint, zeros/disables lora2, and optimizes a
robust additive weight model W_R using zeroth-order (ES-style) updates.
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (AutoencoderKL, DDPMScheduler, PNDMScheduler,
                       StableDiffusionXLPipeline, UNet2DConditionModel)
try:
    from torch.func import functional_call, vmap
except ImportError:
    from torch.nn.utils.stateless import functional_call
    try:
        from functorch import vmap
    except ImportError:
        vmap = None
from tlora_module import (DualLoRACrossAttnProcessor,
                          DualLoRATextLinearLayer,
                          attach_tlora_sigma_mask_hook,
                          build_dual_lora_attn_processors,
                          clear_text_encoder_sigma_mask,
                          compute_orthogonal_lora_weight_delta,
                          get_mask_by_timestep,
                          load_dual_lora_attn_state_dict,
                          load_text_encoder_dual_lora_weights,
                          set_text_encoder_sigma_for_generation,
                          set_text_encoder_sigma_mask)
from tqdm.auto import tqdm
from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                          CLIPTokenizer)
from utils import (SimpleDreamBoothDataset, infinite_dataloader,
                   resolve_checkpoint_paths)


def collect_trainable_layers(unet):
    """Collect base UNet projection weights to optimize with W_R."""
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


def build_tlora1_lookup(unet, trainable_layers):
    """Map each optimized UNet weight name to its corresponding T-LoRA (lora1) module."""
    lookup = {}
    suffix_to_attr = {
        ".to_q.weight": "lora1_q",
        ".to_k.weight": "lora1_k",
        ".to_v.weight": "lora1_v",
        ".to_out.0.weight": "lora1_out",
    }

    for name, _ in trainable_layers:
        if not name.startswith("unet."):
            continue

        for suffix, lora_attr in suffix_to_attr.items():
            if not name.endswith(suffix):
                continue

            attn_path = name[len("unet."):-len(suffix)]
            processor_name = f"{attn_path}.processor"
            processor = unet.attn_processors.get(processor_name)

            if isinstance(processor, DualLoRACrossAttnProcessor):
                lookup[name] = getattr(processor, lora_attr)
            break

    return lookup
def _module_functional_call(module, params_override, *args, **kwargs):
    """Run module forward with virtual parameter overrides (no in-place mutation)."""
    if not params_override:
        return module(*args, **kwargs)
    try:
        return functional_call(
            module,
            params_override,
            args=args,
            kwargs=kwargs,
            strict=False,
        )
    except TypeError:
        return functional_call(module, params_override, args, kwargs, strict=False)


def build_unet_wr_delta_overrides(unet_override_specs, wr_weights, deltas=None):
    """Build UNet overrides for virtual forward: W = W0 + WR (+ delta)."""
    overrides = {}
    for full_name, local_name, base_param in unet_override_specs:
        target = base_param + wr_weights[full_name]
        if deltas is not None:
            target = target + deltas[full_name]
        overrides[local_name] = target.to(
            device=base_param.device,
            dtype=base_param.dtype,
        )
    return overrides


def build_unet_wr_delta_overrides_batched(unet_override_specs, wr_weights, deltas_batched):
    """Build batched UNet overrides for chunked/vmap forward.

    Each leaf tensor has shape [chunk, ...].
    """
    overrides = {}
    for full_name, local_name, base_param in unet_override_specs:
        delta_chunk = deltas_batched[full_name]
        base_plus_wr = (base_param + wr_weights[full_name]).to(
            device=base_param.device,
            dtype=base_param.dtype,
        )
        overrides[local_name] = base_plus_wr.unsqueeze(0) + delta_chunk.to(
            device=base_param.device,
            dtype=base_param.dtype,
        )
    return overrides


def sample_delta_chunk(base, sigma, magnitude, chunk_alphas, chunk_seeds, layer_offset):
    """Deterministically sample normalized perturbation deltas for a chunk."""
    chunk_size = int(chunk_alphas.shape[0])
    deltas = torch.empty((chunk_size, *base.shape), device=base.device, dtype=base.dtype)

    for idx in range(chunk_size):
        generator = torch.Generator(device=base.device)
        generator.manual_seed(int(chunk_seeds[idx].item()) + int(layer_offset))

        delta = torch.randn(base.shape, device=base.device, dtype=base.dtype, generator=generator)
        delta = delta * float(sigma)
        delta_norm = delta.norm(2).clamp_min(1e-12)
        delta = delta / delta_norm * magnitude
        deltas[idx] = delta * chunk_alphas[idx].to(dtype=base.dtype)

    return deltas


class WRInjectedUNet(torch.nn.Module):
    """UNet wrapper that performs virtual forward with W = W0 + WR (+ delta=None for generation)."""

    def __init__(self, base_unet, unet_override_specs, wr_weights):
        super().__init__()
        self.base_unet = base_unet
        self.unet_override_specs = unet_override_specs
        self.wr_weights = wr_weights
        self._cached_overrides = None

    def refresh_overrides(self):
        self._cached_overrides = build_unet_wr_delta_overrides(
            unet_override_specs=self.unet_override_specs,
            wr_weights=self.wr_weights,
            deltas=None,
        )

    def clear_overrides_cache(self):
        self._cached_overrides = None

    def forward(self, *args, **kwargs):
        if self._cached_overrides is None:
            self.refresh_overrides()
        return _module_functional_call(self.base_unet, self._cached_overrides, *args, **kwargs)

    @property
    def config(self):
        return self.base_unet.config

    @property
    def dtype(self):
        return next(self.base_unet.parameters()).dtype

    @property
    def device(self):
        return next(self.base_unet.parameters()).device

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_unet, name)


def encode_batch_for_rl(
    batch,
    vae,
    text_encoder,
    text_encoder_2,
    noise_scheduler,
    device,
    resolution,
    rank,
    min_rank,
    alpha_rank_scale,
    max_timestep,
    timesteps=None,
    noise=None,
):
    """Encode one batch to noisy latents and SDXL conditioning."""
    with torch.no_grad():
        pixel_values = batch["pixel_values"].to(device=device, dtype=vae.dtype)
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        if noise is None:
            noise = torch.randn_like(latents)
        else:
            noise = noise.to(device=device, dtype=latents.dtype)

        if timesteps is None:
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            ).long()
        else:
            timesteps = timesteps.to(device=device).long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    sigma_mask = get_mask_by_timestep(
        timesteps[0].item(),
        max_timestep,
        rank,
        min_rank,
        alpha_rank_scale,
    ).to(device)

    set_text_encoder_sigma_mask(sigma_mask)

    input_ids_1 = batch["input_ids"].to(device)
    prompt_out_1 = text_encoder(input_ids_1, output_hidden_states=True)
    prompt_embeds_1 = prompt_out_1.hidden_states[-2]

    input_ids_2 = batch["input_ids_2"].to(device)
    prompt_out_2 = text_encoder_2(input_ids_2, output_hidden_states=True)
    pooled_prompt_embeds = prompt_out_2.text_embeds
    prompt_embeds_2 = prompt_out_2.hidden_states[-2]

    clear_text_encoder_sigma_mask()

    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
    prompt_embeds = prompt_embeds.to(device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device)

    add_time_ids = torch.tensor(
        [[resolution, resolution, 0, 0, resolution, resolution]],
        dtype=prompt_embeds.dtype,
        device=device,
    ).repeat(noisy_latents.shape[0], 1)

    return (
        noisy_latents,
        timesteps,
        noise,
        prompt_embeds,
        pooled_prompt_embeds,
        add_time_ids,
        sigma_mask,
    )


def save_generation_snapshot(
    step,
    output_dir,
    batch,
    generation_pipeline,
    device,
    resolution,
    num_inference_steps,
    noise,
):
    """Save one image using real multi-step diffusion inference (W0 + WR + T-LoRA)."""
    image_name = batch["image_name"][0] if isinstance(batch["image_name"], list) else "sample"
    prompt_text = batch["prompt"][0] if isinstance(batch["prompt"], list) else str(batch["prompt"])

    latent_noise = None
    if noise is not None:
        latent_noise = noise[:1].to(device=device, dtype=generation_pipeline.unet.dtype)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(step))

    has_text_mask = set_text_encoder_sigma_for_generation(
        generation_pipeline,
        num_inference_steps=num_inference_steps,
    )
    try:
        with torch.inference_mode():
            image = generation_pipeline(
                prompt=prompt_text,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
                height=resolution,
                width=resolution,
                generator=generator,
                latents=latent_noise,
            ).images[0]
    finally:
        if has_text_mask:
            clear_text_encoder_sigma_mask()

    gen_dir = os.path.join(output_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)

    safe_image_name = os.path.splitext(os.path.basename(image_name))[0]
    out_path = os.path.join(
        gen_dir,
        f"step{step:06d}_n{num_inference_steps}_{safe_image_name}.png",
    )
    image.save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="RL-based robustness training for dual-LoRA SDXL"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to dual-LoRA checkpoint directory (from robust training)",
    )
    parser.add_argument(
        "--cp_dataset",
        type=str,
        required=True,
        help="Path to copyright dataset directory (contains image/ and prompt.csv)",
    )

    parser.add_argument("--rl_steps", type=int, default=10000)
    parser.add_argument("--G", type=int, default=16, help="Perturbation samples per step")
    parser.add_argument(
        "--g_eval_batch_size",
        type=int,
        default=4,
        help="Number of perturbations to evaluate together (vmap chunk size)",
    )
    parser.add_argument("--magnitude", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.05)

    parser.add_argument("--gen_interval", type=int, default=500)
    parser.add_argument("--gen_num_inference_steps", type=int, default=50)

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default="fp16")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)

    parser.add_argument("--output_dir", type=str, default="checkpoints_rl")
    parser.add_argument("--save_interval", type=int, default=2500)
    parser.add_argument(
        "--resume_wr_checkpoint",
        type=str,
        default=None,
        help="Path to wr_checkpoint_step*.pt for resuming RL",
    )
    parser.add_argument(
        "--auto_resume_latest",
        action="store_true",
        help="Automatically resume from the latest wr_checkpoint_step*.pt in output_dir",
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.auto_resume_latest:
        if args.resume_wr_checkpoint is not None:
            print(
                "Both --resume_wr_checkpoint and --auto_resume_latest were provided; "
                "using --resume_wr_checkpoint."
            )
        else:
            checkpoint_pattern = os.path.join(args.output_dir, "wr_checkpoint_step*.pt")
            checkpoint_candidates = []
            for checkpoint_path in glob.glob(checkpoint_pattern):
                checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
                if not checkpoint_name.startswith("wr_checkpoint_step"):
                    continue
                step_token = checkpoint_name[len("wr_checkpoint_step"):]
                if not step_token.isdigit():
                    continue
                checkpoint_candidates.append((int(step_token), checkpoint_path))

            if len(checkpoint_candidates) == 0:
                raise FileNotFoundError(
                    "--auto_resume_latest was set but no checkpoints matching "
                    f"{checkpoint_pattern} were found"
                )

            latest_step, latest_checkpoint = max(checkpoint_candidates, key=lambda x: x[0])
            args.resume_wr_checkpoint = latest_checkpoint
            print(
                f"Auto-resume selected latest WR checkpoint: {latest_checkpoint} "
                f"(step {latest_step})"
            )

    cp_csv = os.path.join(args.cp_dataset, "prompt.csv")
    cp_image_dir = os.path.join(args.cp_dataset, "image")
    if not os.path.exists(cp_csv):
        raise FileNotFoundError(f"cp_dataset CSV not found: {cp_csv}")
    if not os.path.exists(cp_image_dir):
        raise FileNotFoundError(f"cp_dataset image directory not found: {cp_image_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dtype = torch.float32
    if args.mixed_precision == "bf16":
        model_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        model_dtype = torch.float16

    print(f"\n{'='*60}")
    print("RL-based Robustness Training")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"CP dataset: {args.cp_dataset}")
    print(f"RL steps: {args.rl_steps}")
    print(f"G: {args.G}")
    print(f"Magnitude: {args.magnitude}")
    print(f"LR: {args.lr}")
    print(f"Generation interval: {args.gen_interval}")
    print(f"Checkpoint save interval: {args.save_interval}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

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

    print("Loading base models...")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=model_dtype,
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
        dtype=model_dtype,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
        dtype=model_dtype,
    )

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    (
        unet_weights_path,
        config_path,
        text_encoder_weights_path,
        text_encoder_2_weights_path,
    ) = resolve_checkpoint_paths(args.checkpoint_path)

    if not os.path.exists(unet_weights_path):
        raise FileNotFoundError(f"dual_lora_weights.pt not found: {unet_weights_path}")

    rank = 16
    lora_alpha = 32
    sig_type = "last"
    min_rank = 8
    alpha_rank_scale = 1.0
    max_timestep = 1000

    if config_path is not None and os.path.exists(config_path):
        tlora_config = torch.load(config_path, map_location="cpu")
        rank = int(tlora_config.get("rank", rank))
        lora_alpha = float(tlora_config.get("lora_alpha", lora_alpha))
        sig_type = str(tlora_config.get("sig_type", sig_type))
        min_rank = int(tlora_config.get("min_rank", min_rank))
        alpha_rank_scale = float(tlora_config.get("alpha_rank_scale", alpha_rank_scale))
        max_timestep = int(tlora_config.get("max_timestep", max_timestep))

    lora_attn_procs = build_dual_lora_attn_processors(
        unet,
        rank=rank,
        lora_alpha=lora_alpha,
        sig_type=sig_type,
    )
    unet.set_attn_processor(lora_attn_procs)

    unet_lora_state = torch.load(unet_weights_path, map_location="cpu")
    loaded_unet = load_dual_lora_attn_state_dict(unet, unet_lora_state, strict=True)
    print(f"Loaded dual-LoRA UNet attention processors: {loaded_unet}")

    loaded_te1 = load_text_encoder_dual_lora_weights(
        text_encoder,
        text_encoder_weights_path,
        rank=rank,
        lora_alpha=lora_alpha,
        sig_type=sig_type,
    )
    loaded_te2 = load_text_encoder_dual_lora_weights(
        text_encoder_2,
        text_encoder_2_weights_path,
        rank=rank,
        lora_alpha=lora_alpha,
        sig_type=sig_type,
    )
    print(f"Loaded text_encoder dual-LoRA layers: {loaded_te1}")
    print(f"Loaded text_encoder_2 dual-LoRA layers: {loaded_te2}")

    print("Zeroing/disabling lora2 and freezing lora1...")
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

    attach_tlora_sigma_mask_hook(
        unet,
        rank=rank,
        min_rank=min_rank,
        alpha_rank_scale=alpha_rank_scale,
        max_timestep=max_timestep,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    cp_dataset = SimpleDreamBoothDataset(
        csv_path=cp_csv,
        image_dir=cp_image_dir,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        size=args.resolution,
        center_crop=False,
    )
    cp_loader = infinite_dataloader(cp_dataset, args.seed, args.train_batch_size)

    unet = unet.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    text_encoder_2 = text_encoder_2.to(device)

    trainable_layers = collect_trainable_layers(unet)
    if len(trainable_layers) == 0:
        raise RuntimeError("No trainable layers found for W_R optimization")

    tlora1_lookup = build_tlora1_lookup(unet, trainable_layers)

    unet_param_lookup = dict(unet.named_parameters())

    w0_weights = {
        name: weight.data.detach().float().clone().to(device)
        for name, weight in trainable_layers
    }
    wr_weights = {
        name: torch.zeros_like(w0_weights[name])
        for name in w0_weights
    }

    wr_names = list(wr_weights.keys())
    resume_step = -1
    if args.resume_wr_checkpoint is not None:
        if not os.path.exists(args.resume_wr_checkpoint):
            raise FileNotFoundError(
                f"Resume checkpoint not found: {args.resume_wr_checkpoint}"
            )

        resume_state = torch.load(args.resume_wr_checkpoint, map_location="cpu")
        if "W_R" not in resume_state:
            raise KeyError("Resume checkpoint missing key: 'W_R'")
        if "step" not in resume_state:
            raise KeyError("Resume checkpoint missing key: 'step'")

        resume_wr = resume_state["W_R"]
        for name in wr_names:
            if name not in resume_wr:
                raise KeyError(f"Resume checkpoint missing W_R layer: {name}")
            loaded_tensor = resume_wr[name]
            if loaded_tensor.shape != wr_weights[name].shape:
                raise ValueError(
                    f"Shape mismatch for {name}: checkpoint {tuple(loaded_tensor.shape)} "
                    f"!= current {tuple(wr_weights[name].shape)}"
                )
            wr_weights[name] = loaded_tensor.to(
                device=wr_weights[name].device,
                dtype=wr_weights[name].dtype,
            )

        resume_step = int(resume_state["step"])
        if resume_step < 0:
            raise ValueError(f"Invalid resume step: {resume_step}")
        print(f"Resumed W_R from: {args.resume_wr_checkpoint}")
        print(f"Resume step: {resume_step}")

    layer_offsets = {
        name: (idx + 1) * 1000003
        for idx, name in enumerate(wr_names)
    }
    unet_override_specs = []
    for full_name in wr_names:
        if not full_name.startswith("unet."):
            continue
        local_name = full_name[len("unet."):]
        base_param = unet_param_lookup[local_name]
        unet_override_specs.append((full_name, local_name, base_param))

    total_wr_params = sum(w.numel() for w in wr_weights.values())
    print(f"Trainable layers for W_R: {len(trainable_layers)}")
    print(f"Total W_R parameters: {total_wr_params:,}")

    generation_unet = WRInjectedUNet(
        base_unet=unet,
        unet_override_specs=unet_override_specs,
        wr_weights=wr_weights,
    )
    generation_scheduler = PNDMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    generation_pipeline = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=generation_unet,
        scheduler=generation_scheduler,
        add_watermarker=False,
    )
    generation_pipeline = generation_pipeline.to(device)
    generation_pipeline.set_progress_bar_config(disable=True)
    generation_pipeline.tlora_text_encoder_config = {
        "rank": rank,
        "min_rank": min_rank,
        "alpha_rank_scale": alpha_rank_scale,
        "max_timestep": max_timestep,
    }

    g_eval_batch_size = max(1, int(args.g_eval_batch_size))
    if vmap is None and g_eval_batch_size > 1:
        print("Warning: vmap is unavailable; falling back to sequential perturbation evaluation.")

    os.makedirs(args.output_dir, exist_ok=True)

    start_step = 0
    if resume_step >= 0:
        start_step = resume_step + 1
        if start_step > args.rl_steps:
            print(
                f"Resume step ({resume_step}) is already >= rl_steps ({args.rl_steps}); "
                "no RL updates will be run."
            )

    progress_bar = tqdm(range(start_step, args.rl_steps+1), desc="RL Training")
    for step in progress_bar:
        batch = next(cp_loader)

        if step == 0:
            generation_unet.clear_overrides_cache()
            save_generation_snapshot(
                step=step,
                output_dir=args.output_dir,
                batch=batch,
                generation_pipeline=generation_pipeline,
                device=device,
                resolution=args.resolution,
                num_inference_steps=args.gen_num_inference_steps,
                noise=None,
            )

        (
            noisy_latents,
            timesteps,
            noise,
            prompt_embeds,
            pooled_prompt_embeds,
            add_time_ids,
            sigma_mask,
        ) = encode_batch_for_rl(
            batch,
            vae,
            text_encoder,
            text_encoder_2,
            noise_scheduler,
            device,
            args.resolution,
            rank,
            min_rank,
            alpha_rank_scale,
            max_timestep,
        )

        layer_stats = {}
        for name in wr_names:
            current_w = w0_weights[name] + wr_weights[name]

            lora1_module = tlora1_lookup.get(name)
            if lora1_module is not None:
                tlora_delta = compute_orthogonal_lora_weight_delta(
                    lora1_module,
                    mask=sigma_mask,
                )
                current_w = current_w + tlora_delta.to(
                    device=current_w.device,
                    dtype=current_w.dtype,
                )

            sigma = float(current_w.std().item())
            layer_stats[name] = max(sigma, 1e-12)

        alphas = torch.where(
            torch.rand((args.G,), device=device) > 0.5,
            torch.ones((args.G,), device=device, dtype=torch.float32),
            -torch.ones((args.G,), device=device, dtype=torch.float32),
        )

        perturb_seeds = torch.randint(
            low=0,
            high=2**31 - 1,
            size=(args.G,),
            dtype=torch.int64,
            device="cpu",
        )

        rewards_tensor = torch.empty((args.G,), dtype=torch.float32, device=device)
        with torch.inference_mode():
            for start in range(0, args.G, g_eval_batch_size):
                end = min(start + g_eval_batch_size, args.G)
                chunk = end - start

                chunk_alphas = alphas[start:end]
                chunk_seeds = perturb_seeds[start:end]

                deltas_chunk = {}
                for name in wr_names:
                    deltas_chunk[name] = sample_delta_chunk(
                        base=wr_weights[name],
                        sigma=layer_stats[name],
                        magnitude=args.magnitude,
                        chunk_alphas=chunk_alphas,
                        chunk_seeds=chunk_seeds,
                        layer_offset=layer_offsets[name],
                    )

                if vmap is not None and chunk > 1:
                    batched_overrides = build_unet_wr_delta_overrides_batched(
                        unet_override_specs=unet_override_specs,
                        wr_weights=wr_weights,
                        deltas_batched=deltas_chunk,
                    )

                    noisy_batched = noisy_latents.unsqueeze(0).expand(chunk, *noisy_latents.shape)
                    timesteps_batched = timesteps.unsqueeze(0).expand(chunk, *timesteps.shape)
                    prompt_batched = prompt_embeds.unsqueeze(0).expand(chunk, *prompt_embeds.shape)
                    pooled_batched = pooled_prompt_embeds.unsqueeze(0).expand(chunk, *pooled_prompt_embeds.shape)
                    time_ids_batched = add_time_ids.unsqueeze(0).expand(chunk, *add_time_ids.shape)
                    sigma_batched = sigma_mask.unsqueeze(0).expand(chunk, *sigma_mask.shape)

                    def _single_forward(
                        one_overrides,
                        one_noisy,
                        one_timestep,
                        one_prompt,
                        one_pooled,
                        one_time_ids,
                        one_sigma,
                    ):
                        return _module_functional_call(
                            unet,
                            one_overrides,
                            one_noisy,
                            one_timestep,
                            encoder_hidden_states=one_prompt,
                            added_cond_kwargs={
                                "text_embeds": one_pooled,
                                "time_ids": one_time_ids,
                            },
                            cross_attention_kwargs={"sigma_mask": one_sigma},
                        ).sample

                    model_pred = vmap(
                        _single_forward,
                        in_dims=(0, 0, 0, 0, 0, 0, 0),
                    )(
                        batched_overrides,
                        noisy_batched,
                        timesteps_batched,
                        prompt_batched,
                        pooled_batched,
                        time_ids_batched,
                        sigma_batched,
                    )

                    noise_batched = noise.unsqueeze(0).expand_as(model_pred)
                    losses = ((model_pred.float() - noise_batched.float()) ** 2).mean(
                        dim=tuple(range(1, model_pred.ndim))
                    )
                    rewards_tensor[start:end] = -losses
                else:
                    for local_idx in range(chunk):
                        deltas = {name: deltas_chunk[name][local_idx] for name in wr_names}
                        unet_overrides = build_unet_wr_delta_overrides(
                            unet_override_specs=unet_override_specs,
                            wr_weights=wr_weights,
                            deltas=deltas,
                        )
                        model_pred = _module_functional_call(
                            unet,
                            unet_overrides,
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=prompt_embeds,
                            added_cond_kwargs={
                                "text_embeds": pooled_prompt_embeds,
                                "time_ids": add_time_ids,
                            },
                            cross_attention_kwargs={"sigma_mask": sigma_mask},
                        ).sample
                        loss_i = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                        rewards_tensor[start + local_idx] = -loss_i.float()

        avg_reward_t = rewards_tensor.mean()
        std_reward_t = rewards_tensor.std(unbiased=False)
        advantages = (rewards_tensor - avg_reward_t) / (std_reward_t + 1e-12)
        avg_reward = float(avg_reward_t.item())
        std_reward = float(std_reward_t.item())

        grad_accum = {name: torch.zeros_like(wr_weights[name]) for name in wr_names}
        for start in range(0, args.G, g_eval_batch_size):
            end = min(start + g_eval_batch_size, args.G)
            chunk = end - start

            chunk_alphas = alphas[start:end]
            chunk_seeds = perturb_seeds[start:end]
            adv_chunk = advantages[start:end]

            for name in wr_names:
                deltas = sample_delta_chunk(
                    base=wr_weights[name],
                    sigma=layer_stats[name],
                    magnitude=args.magnitude,
                    chunk_alphas=chunk_alphas,
                    chunk_seeds=chunk_seeds,
                    layer_offset=layer_offsets[name],
                )
                adv_view = adv_chunk.view((chunk,) + (1,) * wr_weights[name].ndim).to(deltas.dtype)
                grad_accum[name] += (adv_view * deltas).sum(dim=0)

        for name in wr_names:
            wr_weights[name] += (args.lr / args.G) * grad_accum[name]

        generation_unet.clear_overrides_cache()

        if step > 0 and step % args.gen_interval == 0:
            save_generation_snapshot(
                step=step,
                output_dir=args.output_dir,
                batch=batch,
                generation_pipeline=generation_pipeline,
                device=device,
                resolution=args.resolution,
                num_inference_steps=args.gen_num_inference_steps,
                noise=noise,
            )

        avg_mse = -avg_reward
        avg_sigma = float(np.mean(list(layer_stats.values())))
        progress_bar.set_postfix(
            {
                "reward": f"{avg_reward:.6f}",
                "mse": f"{avg_mse:.6f}",
                "std_R": f"{std_reward:.6f}",
                "sigma": f"{avg_sigma:.3e}",
            }
        )

        if step>0 and step % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"wr_checkpoint_step{step:06d}.pt")
            torch.save(
                {
                    "W_R": {name: w.detach().cpu() for name, w in wr_weights.items()},
                    "step": step,
                    "args": vars(args),
                    "rank": rank,
                    "min_rank": min_rank,
                    "alpha_rank_scale": alpha_rank_scale,
                    "max_timestep": max_timestep,
                },
                ckpt_path,
            )
            print(f"\nSaved W_R checkpoint: {ckpt_path}")

    final_path = os.path.join(args.output_dir, "wr_final.pt")
    torch.save(
        {
            "W_R": {name: w.detach().cpu() for name, w in wr_weights.items()},
            "step": args.rl_steps,
            "args": vars(args),
            "rank": rank,
            "min_rank": min_rank,
            "alpha_rank_scale": alpha_rank_scale,
            "max_timestep": max_timestep,
        },
        final_path,
    )

    print(f"\n{'='*60}")
    print("RL training complete")
    print(f"Final W_R saved to: {final_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
