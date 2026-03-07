#!/usr/bin/env python3
"""
RL-based Robustness Training for DreamBooth LoRA SDXL.

Loads a dual-LoRA SDXL checkpoint, zeros/disables lora2, and optimizes a
robust additive weight model W_R using zeroth-order (ES-style) updates.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (AutoencoderKL, DDPMScheduler, PNDMScheduler,
                       StableDiffusionXLPipeline, UNet2DConditionModel)
try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call
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


class WRInjectedUNet(torch.nn.Module):
    """UNet wrapper that performs virtual forward with W = W0 + WR (+ delta=None for generation)."""

    def __init__(self, base_unet, unet_override_specs, wr_weights):
        super().__init__()
        self.base_unet = base_unet
        self.unet_override_specs = unet_override_specs
        self.wr_weights = wr_weights

    def forward(self, *args, **kwargs):
        unet_overrides = build_unet_wr_delta_overrides(
            unet_override_specs=self.unet_override_specs,
            wr_weights=self.wr_weights,
            deltas=None,
        )
        return _module_functional_call(self.base_unet, unet_overrides, *args, **kwargs)

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
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

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

    os.makedirs(args.output_dir, exist_ok=True)

    progress_bar = tqdm(range(1, args.rl_steps + 1), desc="RL Training")
    for step in progress_bar:
        batch = next(cp_loader)

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

        delta_batches = {}
        for name in wr_names:
            sigma = layer_stats[name]
            base = wr_weights[name]
            deltas = torch.randn(
                (args.G, *base.shape),
                device=base.device,
                dtype=base.dtype,
            ) * sigma
            flat = deltas.view(args.G, -1)
            norms = flat.norm(2, dim=1, keepdim=True).clamp_min(1e-12)
            deltas = deltas / norms.view((args.G,) + (1,) * base.ndim)
            deltas = deltas * args.magnitude
            deltas = deltas * alphas.view((args.G,) + (1,) * base.ndim).to(deltas.dtype)
            delta_batches[name] = deltas

        rewards_tensor = torch.empty((args.G,), dtype=torch.float32, device=device)
        with torch.inference_mode():
            for g in range(args.G):
                deltas = {name: delta_batches[name][g] for name in wr_names}
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
                rewards_tensor[g] = -loss_i.float()

        avg_reward_t = rewards_tensor.mean()
        std_reward_t = rewards_tensor.std(unbiased=False)
        advantages = (rewards_tensor - avg_reward_t) / (std_reward_t + 1e-12)
        avg_reward = float(avg_reward_t.item())
        std_reward = float(std_reward_t.item())

        for name in wr_names:
            deltas = delta_batches[name]
            adv_view = advantages.view((args.G,) + (1,) * wr_weights[name].ndim).to(deltas.dtype)
            grad_estimate = (adv_view * deltas).sum(dim=0)
            wr_weights[name] += (args.lr / args.G) * grad_estimate

        if args.gen_interval > 0 and step % args.gen_interval == 0:
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

        if step % args.save_interval == 0:
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
