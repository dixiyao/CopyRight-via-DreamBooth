#!/usr/bin/env python3
"""
Further-LoRA training with perturbation over copyright dataset.

Keeps original RL script unchanged and implements the requested variant:
- Load base model M0 + pre-trained T-LoRA MLA (lora1)
- Create/train MR as a new T-LoRA branch (lora2)
- Per step, sample one Gaussian perturbation delta M and scale it by
  ||delta M|| / ||M0 + MLA|| <= m / 100 (applied per optimized UNet layer)
- Forward with combined weights (M0 + MLA + MR + delta M)
- Use one cp batch, compute MSE(noise_pred, noise), optimize MR with AdamW
"""

import argparse
import csv
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from tlora_module import (
    DualLoRACrossAttnProcessor,
    DualLoRATextLinearLayer,
    attach_tlora_sigma_mask_hook,
    build_dual_lora_attn_processors,
    clear_text_encoder_sigma_mask,
    compute_orthogonal_lora_weight_delta,
    compute_standard_lora_weight_delta,
    collect_dual_lora_attn_state_dict,
    collect_text_encoder_lora_state_dict,
    get_mask_by_timestep,
    load_dual_lora_attn_state_dict,
    load_text_encoder_dual_lora_weights,
    set_text_encoder_sigma_mask,
)
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from utils import SimpleDreamBoothDataset, infinite_dataloader, resolve_checkpoint_paths

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call


def _module_functional_call(module, params_override, *args, **kwargs):
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


def collect_trainable_layers(unet):
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


def build_tlora_lookup(unet, trainable_layers):
    lookup = {}
    suffix_to_attrs = {
        ".to_q.weight": ("lora1_q", "lora2_q"),
        ".to_k.weight": ("lora1_k", "lora2_k"),
        ".to_v.weight": ("lora1_v", "lora2_v"),
        ".to_out.0.weight": ("lora1_out", "lora2_out"),
    }

    for name, _ in trainable_layers:
        if not name.startswith("unet."):
            continue

        for suffix, attrs in suffix_to_attrs.items():
            if not name.endswith(suffix):
                continue

            attn_path = name[len("unet."):-len(suffix)]
            processor_name = f"{attn_path}.processor"
            processor = unet.attn_processors.get(processor_name)
            if isinstance(processor, DualLoRACrossAttnProcessor):
                lookup[name] = (getattr(processor, attrs[0]), getattr(processor, attrs[1]))
            break

    return lookup


def encode_batch(
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
):
    with torch.no_grad():
        pixel_values = batch["pixel_values"].to(device=device, dtype=vae.dtype)
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=device,
        ).long()

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

    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1).to(device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device)

    add_time_ids = torch.tensor(
        [[resolution, resolution, 0, 0, resolution, resolution]],
        dtype=prompt_embeds.dtype,
        device=device,
    ).repeat(noisy_latents.shape[0], 1)

    return noisy_latents, timesteps, noise, prompt_embeds, pooled_prompt_embeds, add_time_ids, sigma_mask


def sample_scaled_delta(reference_weight, max_distortion_pct, generator):
    # Eq. line-7 style bound: ||delta|| / ||reference|| <= m/100
    delta = torch.randn(reference_weight.shape, device=reference_weight.device, dtype=reference_weight.dtype, generator=generator)
    ref_norm = reference_weight.norm(2).clamp_min(1e-12)
    delta_norm = delta.norm(2).clamp_min(1e-12)
    max_norm = ref_norm * (float(max_distortion_pct) / 100.0)
    delta = delta / delta_norm * max_norm
    return delta


def build_overrides_with_mr_and_delta(
    unet_override_specs,
    w0_weights,
    max_distortion_pct,
    seed,
    step,
):
    overrides = {}

    for idx, (full_name, local_name, base_param) in enumerate(unet_override_specs):
        reference = w0_weights[full_name]

        g = torch.Generator(device=reference.device)
        g.manual_seed(int(seed) + int(step) * 1000003 + idx)
        # Simple random M_delta (same shape as SDXL weight), scaled by line-7 bound:
        # ||M_delta|| / ||M0 + MLA|| <= m / 100
        m_delta = torch.randn(
            reference.shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=g,
        )
        ref_norm = reference.norm(2).clamp_min(1e-12)
        m_delta_norm = m_delta.norm(2).clamp_min(1e-12)
        max_norm = ref_norm * (float(max_distortion_pct) / 100.0)
        m_delta = m_delta / m_delta_norm * max_norm

        combined = w0_weights[full_name] + m_delta
        overrides[local_name] = combined.to(device=base_param.device, dtype=base_param.dtype)

    return overrides


def build_overrides_with_mla_mr(
    unet_override_specs,
    w0_weights,
    tlora_lookup,
    sigma_mask,
):
    overrides = {}

    for full_name, local_name, base_param in unet_override_specs:
        lora1_module, lora2_module = tlora_lookup[full_name]

        mla_delta = compute_orthogonal_lora_weight_delta(lora1_module, mask=sigma_mask).to(
            device=w0_weights[full_name].device,
            dtype=w0_weights[full_name].dtype,
        )
        mr_delta = compute_standard_lora_weight_delta(lora2_module).to(
            device=w0_weights[full_name].device,
            dtype=w0_weights[full_name].dtype,
        )

        combined = w0_weights[full_name] + mla_delta + mr_delta
        overrides[local_name] = combined.to(device=base_param.device, dtype=base_param.dtype)

    return overrides


def save_checkpoint(unet, text_encoder, text_encoder_2, output_dir, step, tlora_config):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-step{step:06d}")
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save(
        collect_dual_lora_attn_state_dict(unet),
        os.path.join(ckpt_dir, "dual_lora_weights.pt"),
    )
    torch.save(
        collect_text_encoder_lora_state_dict(text_encoder),
        os.path.join(ckpt_dir, "text_encoder_dual_lora_weights.pt"),
    )
    torch.save(
        collect_text_encoder_lora_state_dict(text_encoder_2),
        os.path.join(ckpt_dir, "text_encoder_2_dual_lora_weights.pt"),
    )
    torch.save(tlora_config, os.path.join(ckpt_dir, "tlora_config.pt"))
    print(f"Saved checkpoint: {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(description="Further-LoRA training with delta perturbation on cp dataset")

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--cp_dataset", type=str, required=True)
    parser.add_argument("--org_image", type=str, required=True, help="Path to original dataset directory (contains image/ and prompt.csv)")
    parser.add_argument("--cp_step", type=int, default=10000)
    parser.add_argument("--max_distortion_pct", type=float, default=10.0, help="m in line-7 style bound (percentage)")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default="fp16")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)

    parser.add_argument("--output_dir", type=str, default="checkpoints_further_lora")
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument(
        "--study_log_file",
        type=str,
        default=None,
        help="CSV file for per-step losses; defaults to <output_dir>/study_loss_log.csv",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory to resume from (absolute path or relative to output_dir)",
    )
    parser.add_argument(
        "--auto_resume_latest",
        action="store_true",
        help="Automatically resume from latest checkpoint-step* directory in output_dir",
    )

    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.auto_resume_latest:
        if args.resume_from_checkpoint is not None:
            print(
                "Both --resume_from_checkpoint and --auto_resume_latest were provided; "
                "using --resume_from_checkpoint."
            )
        else:
            candidates = []
            for ckpt_dir in glob.glob(os.path.join(args.output_dir, "checkpoint-step*")):
                if not os.path.isdir(ckpt_dir):
                    continue
                base = os.path.basename(os.path.normpath(ckpt_dir))
                prefix = "checkpoint-step"
                if not base.startswith(prefix):
                    continue
                token = base[len(prefix):]
                if not token.isdigit():
                    continue
                candidates.append((int(token), ckpt_dir))

            if len(candidates) > 0:
                latest_step, latest_ckpt = max(candidates, key=lambda x: x[0])
                args.resume_from_checkpoint = latest_ckpt
                print(f"Auto-resume selected: {latest_ckpt} (step {latest_step})")

    cp_csv = os.path.join(args.cp_dataset, "prompt.csv")
    cp_image_dir = os.path.join(args.cp_dataset, "image")
    if not os.path.exists(cp_csv):
        raise FileNotFoundError(f"cp_dataset CSV not found: {cp_csv}")
    if not os.path.exists(cp_image_dir):
        raise FileNotFoundError(f"cp_dataset image directory not found: {cp_image_dir}")

    org_csv = os.path.join(args.org_image, "prompt.csv")
    org_image_dir = os.path.join(args.org_image, "image")
    if not os.path.exists(org_csv):
        raise FileNotFoundError(f"org_image CSV not found: {org_csv}")
    if not os.path.exists(org_image_dir):
        raise FileNotFoundError(f"org_image image directory not found: {org_image_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.study_log_file is None:
        args.study_log_file = os.path.join(args.output_dir, "study_loss_log.csv")

    resume_step = 0
    resume_checkpoint_dir = None
    if args.resume_from_checkpoint is not None:
        resume_checkpoint_dir = args.resume_from_checkpoint
        if not os.path.isabs(resume_checkpoint_dir):
            resume_checkpoint_dir = os.path.join(args.output_dir, resume_checkpoint_dir)
        if not os.path.isdir(resume_checkpoint_dir):
            raise FileNotFoundError(f"Resume checkpoint directory not found: {resume_checkpoint_dir}")

        base = os.path.basename(os.path.normpath(resume_checkpoint_dir))
        prefix = "checkpoint-step"
        if base.startswith(prefix) and base[len(prefix):].isdigit():
            resume_step = int(base[len(prefix):])
        print(f"Resuming from checkpoint: {resume_checkpoint_dir}")
        print(f"Resume step: {resume_step}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dtype = torch.float32
    if args.mixed_precision == "bf16":
        model_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        model_dtype = torch.float16

    print(f"\n{'=' * 70}")
    print("Further-LoRA Training (MR with one-delta perturbation)")
    print(f"{'=' * 70}")
    print(f"checkpoint_path:    {args.checkpoint_path}")
    print(f"cp_dataset:         {args.cp_dataset}")
    print(f"org_image:          {args.org_image}")
    print(f"cp_step:            {args.cp_step}")
    print(f"max_distortion_pct: {args.max_distortion_pct}")
    print(f"learning_rate:      {args.learning_rate}")
    print(f"study_log_file:     {args.study_log_file}")
    print(f"device:             {device}")
    print(f"{'=' * 70}\n")

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
    else:
        tlora_config = {
            "rank": rank,
            "lora_alpha": lora_alpha,
            "sig_type": sig_type,
            "min_rank": min_rank,
            "alpha_rank_scale": alpha_rank_scale,
            "max_timestep": max_timestep,
        }

    lora_attn_procs = build_dual_lora_attn_processors(
        unet,
        rank=rank,
        lora_alpha=lora_alpha,
        sig_type=sig_type,
    )
    unet.set_attn_processor(lora_attn_procs)

    unet_lora_state = torch.load(unet_weights_path, map_location="cpu")
    load_dual_lora_attn_state_dict(unet, unet_lora_state, strict=True)

    load_text_encoder_dual_lora_weights(
        text_encoder,
        text_encoder_weights_path,
        rank=rank,
        lora_alpha=lora_alpha,
        sig_type=sig_type,
    )
    load_text_encoder_dual_lora_weights(
        text_encoder_2,
        text_encoder_2_weights_path,
        rank=rank,
        lora_alpha=lora_alpha,
        sig_type=sig_type,
    )

    # Freeze MLA (lora1), initialize/train MR (lora2)
    for proc in unet.attn_processors.values():
        if not isinstance(proc, DualLoRACrossAttnProcessor):
            continue
        for lora1_module in [proc.lora1_q, proc.lora1_k, proc.lora1_v, proc.lora1_out]:
            for p in lora1_module.parameters():
                p.requires_grad_(False)
        for lora2_module in [proc.lora2_q, proc.lora2_k, proc.lora2_v, proc.lora2_out]:
            for p in lora2_module.parameters():
                p.requires_grad_(True)

    for module in text_encoder.modules():
        if not isinstance(module, DualLoRATextLinearLayer):
            continue
        for p in module.lora1.parameters():
            p.requires_grad_(False)
        for p in module.lora2.parameters():
            p.requires_grad_(True)

    for module in text_encoder_2.modules():
        if not isinstance(module, DualLoRATextLinearLayer):
            continue
        for p in module.lora1.parameters():
            p.requires_grad_(False)
        for p in module.lora2.parameters():
            p.requires_grad_(True)

    if resume_checkpoint_dir is None:
        for proc in unet.attn_processors.values():
            if not isinstance(proc, DualLoRACrossAttnProcessor):
                continue
            for lora2_module in [proc.lora2_q, proc.lora2_k, proc.lora2_v, proc.lora2_out]:
                for p in lora2_module.parameters():
                    p.data.zero_()
        for module in text_encoder.modules():
            if isinstance(module, DualLoRATextLinearLayer):
                for p in module.lora2.parameters():
                    p.data.zero_()
        for module in text_encoder_2.modules():
            if isinstance(module, DualLoRATextLinearLayer):
                for p in module.lora2.parameters():
                    p.data.zero_()
    else:
        (
            resume_unet_weights_path,
            _resume_config_path,
            resume_text_encoder_weights_path,
            resume_text_encoder_2_weights_path,
        ) = resolve_checkpoint_paths(resume_checkpoint_dir)
        load_dual_lora_attn_state_dict(
            unet,
            torch.load(resume_unet_weights_path, map_location="cpu"),
            strict=True,
        )
        load_text_encoder_dual_lora_weights(
            text_encoder,
            resume_text_encoder_weights_path,
            rank=rank,
            lora_alpha=lora_alpha,
            sig_type=sig_type,
        )
        load_text_encoder_dual_lora_weights(
            text_encoder_2,
            resume_text_encoder_2_weights_path,
            rank=rank,
            lora_alpha=lora_alpha,
            sig_type=sig_type,
        )

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
    org_dataset = SimpleDreamBoothDataset(
        csv_path=org_csv,
        image_dir=org_image_dir,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        size=args.resolution,
        center_crop=False,
    )
    cp_loader = infinite_dataloader(cp_dataset, args.seed, args.train_batch_size)
    org_loader = infinite_dataloader(org_dataset, args.seed + 1, args.train_batch_size)

    unet = unet.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    text_encoder_2 = text_encoder_2.to(device)

    trainable_layers = collect_trainable_layers(unet)
    if len(trainable_layers) == 0:
        raise RuntimeError("No trainable layers found")

    tlora_lookup = build_tlora_lookup(unet, trainable_layers)
    unet_param_lookup = dict(unet.named_parameters())

    w0_weights = {
        name: weight.data.detach().float().clone().to(device)
        for name, weight in trainable_layers
    }

    unet_override_specs = []
    for full_name, _ in trainable_layers:
        local_name = full_name[len("unet."):]
        base_param = unet_param_lookup[local_name]
        unet_override_specs.append((full_name, local_name, base_param))

    mr_params = []
    for proc in unet.attn_processors.values():
        if not isinstance(proc, DualLoRACrossAttnProcessor):
            continue
        for lora2_module in [proc.lora2_q, proc.lora2_k, proc.lora2_v, proc.lora2_out]:
            mr_params.extend(list(lora2_module.parameters()))
    for module in text_encoder.modules():
        if isinstance(module, DualLoRATextLinearLayer):
            mr_params.extend(list(module.lora2.parameters()))
    for module in text_encoder_2.modules():
        if isinstance(module, DualLoRATextLinearLayer):
            mr_params.extend(list(module.lora2.parameters()))

    optimizer = torch.optim.AdamW(mr_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    start_step = max(1, resume_step + 1)
    if start_step > args.cp_step:
        print(
            f"Resume step ({resume_step}) is already >= cp_step ({args.cp_step}); "
            "no further updates will run."
        )

    log_file_exists = os.path.exists(args.study_log_file) and resume_step > 0
    log_fh = open(args.study_log_file, "a" if log_file_exists else "w", encoding="utf-8", newline="")
    log_writer = csv.DictWriter(
        log_fh,
        fieldnames=["step", "cp_train_loss", "cp_forward_loss", "org_forward_loss", "timestep"],
    )
    if not log_file_exists:
        log_writer.writeheader()
        log_fh.flush()

    progress_bar = tqdm(range(start_step, args.cp_step + 1), desc="Further-LoRA")
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
        ) = encode_batch(
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

        unet_overrides = build_overrides_with_mr_and_delta(
            unet_override_specs=unet_override_specs,
            w0_weights=w0_weights,
            max_distortion_pct=args.max_distortion_pct,
            seed=args.seed,
            step=step,
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

        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mr_params, 1.0)
        optimizer.step()

        with torch.no_grad():
            cp_eval_batch = next(cp_loader)
            (
                cp_eval_noisy_latents,
                cp_eval_timesteps,
                cp_eval_noise,
                cp_eval_prompt_embeds,
                cp_eval_pooled_prompt_embeds,
                cp_eval_add_time_ids,
                cp_eval_sigma_mask,
            ) = encode_batch(
                cp_eval_batch,
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

            no_delta_overrides_cp = build_overrides_with_mla_mr(
                unet_override_specs=unet_override_specs,
                w0_weights=w0_weights,
                tlora_lookup=tlora_lookup,
                sigma_mask=cp_eval_sigma_mask,
            )

            cp_eval_pred = _module_functional_call(
                unet,
                no_delta_overrides_cp,
                cp_eval_noisy_latents,
                cp_eval_timesteps,
                encoder_hidden_states=cp_eval_prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": cp_eval_pooled_prompt_embeds,
                    "time_ids": cp_eval_add_time_ids,
                },
                cross_attention_kwargs={"sigma_mask": cp_eval_sigma_mask},
            ).sample
            cp_forward_loss = F.mse_loss(cp_eval_pred.float(), cp_eval_noise.float(), reduction="mean")

            org_eval_batch = next(org_loader)
            (
                org_eval_noisy_latents,
                org_eval_timesteps,
                org_eval_noise,
                org_eval_prompt_embeds,
                org_eval_pooled_prompt_embeds,
                org_eval_add_time_ids,
                org_eval_sigma_mask,
            ) = encode_batch(
                org_eval_batch,
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

            no_delta_overrides_org = build_overrides_with_mla_mr(
                unet_override_specs=unet_override_specs,
                w0_weights=w0_weights,
                tlora_lookup=tlora_lookup,
                sigma_mask=org_eval_sigma_mask,
            )

            org_eval_pred = _module_functional_call(
                unet,
                no_delta_overrides_org,
                org_eval_noisy_latents,
                org_eval_timesteps,
                encoder_hidden_states=org_eval_prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": org_eval_pooled_prompt_embeds,
                    "time_ids": org_eval_add_time_ids,
                },
                cross_attention_kwargs={"sigma_mask": org_eval_sigma_mask},
            ).sample
            org_forward_loss = F.mse_loss(org_eval_pred.float(), org_eval_noise.float(), reduction="mean")

        progress_bar.set_postfix({
            "cp_train": f"{float(loss.item()):.6f}",
            "cp_fwd": f"{float(cp_forward_loss.item()):.6f}",
            "org_fwd": f"{float(org_forward_loss.item()):.6f}",
            "t": int(timesteps[0].item()),
        })

        log_writer.writerow(
            {
                "step": step,
                "cp_train_loss": f"{float(loss.item()):.6f}",
                "cp_forward_loss": f"{float(cp_forward_loss.item()):.6f}",
                "org_forward_loss": f"{float(org_forward_loss.item()):.6f}",
                "timestep": int(timesteps[0].item()),
            }
        )
        log_fh.flush()

        if step % args.checkpointing_steps == 0:
            save_checkpoint(unet, text_encoder, text_encoder_2, args.output_dir, step, tlora_config)

    log_fh.close()

    save_checkpoint(unet, text_encoder, text_encoder_2, args.output_dir, args.cp_step, tlora_config)

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(collect_dual_lora_attn_state_dict(unet), os.path.join(final_dir, "dual_lora_weights.pt"))
    torch.save(collect_text_encoder_lora_state_dict(text_encoder), os.path.join(final_dir, "text_encoder_dual_lora_weights.pt"))
    torch.save(collect_text_encoder_lora_state_dict(text_encoder_2), os.path.join(final_dir, "text_encoder_2_dual_lora_weights.pt"))
    torch.save(tlora_config, os.path.join(final_dir, "tlora_config.pt"))

    print(f"\n{'=' * 70}")
    print("Further-LoRA training complete")
    print(f"Final model saved to: {final_dir}")
    print(f"Study loss log saved to: {args.study_log_file}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
