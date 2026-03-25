#!/usr/bin/env python3
"""
Study variant of DreamBooth LoRA robust training.

Same as train_dreambooth_lora_sdxl_robust.py (T-LoRA on cp_dataset) but with:
  - Per-step loss logging to a CSV file (step, cp_loss, org_loss)
  - At each step, a random image from --org_image is encoded with the SAME
    noise epsilon and timestep as the copyright batch.  The MSE on that
    image is computed (no grad, no backprop) and logged alongside cp_loss.
"""

import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from tlora_module import (
    DualLoRACrossAttnProcessor,
    build_dual_lora_attn_processors,
    clear_text_encoder_sigma_mask,
    collect_dual_lora_attn_state_dict,
    collect_text_encoder_lora_state_dict,
    get_mask_by_timestep,
    load_dual_lora_attn_state_dict,
    load_text_encoder_dual_lora_weights,
    set_text_encoder_sigma_mask,
    setup_text_encoder_dual_lora,
)
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from utils import SimpleDreamBoothDataset, infinite_dataloader, parse_float_list


# ============================================================
# Checkpoint Saving
# ============================================================

def save_checkpoint(
    unet,
    output_dir,
    iteration,
    phase,
    step,
    accelerator=None,
    tlora_config=None,
    text_encoder=None,
    text_encoder_2=None,
):
    checkpoint_dir = os.path.join(
        output_dir, f"checkpoint-iter{iteration:03d}-{phase}-step{step:04d}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    unet_to_save = accelerator.unwrap_model(unet) if accelerator is not None else unet

    lora_state_dict = collect_dual_lora_attn_state_dict(unet_to_save)
    torch.save(lora_state_dict, os.path.join(checkpoint_dir, "dual_lora_weights.pt"))

    if text_encoder is not None:
        te_to_save = (
            accelerator.unwrap_model(text_encoder) if accelerator is not None else text_encoder
        )
        torch.save(
            collect_text_encoder_lora_state_dict(te_to_save),
            os.path.join(checkpoint_dir, "text_encoder_dual_lora_weights.pt"),
        )

    if text_encoder_2 is not None:
        te2_to_save = (
            accelerator.unwrap_model(text_encoder_2) if accelerator is not None else text_encoder_2
        )
        torch.save(
            collect_text_encoder_lora_state_dict(te2_to_save),
            os.path.join(checkpoint_dir, "text_encoder_2_dual_lora_weights.pt"),
        )

    if tlora_config is not None:
        torch.save(tlora_config, os.path.join(checkpoint_dir, "tlora_config.pt"))

    print(f"Checkpoint saved to {checkpoint_dir}")


# ============================================================
# Encoding helpers
# ============================================================

def _encode_latents(batch, vae, noise=None, timesteps=None, noise_scheduler=None):
    """
    Encode pixel_values → latents, add noise, return (noisy_latents, noise, timesteps).

    If noise and timesteps are provided they are reused (for the org_image shadow pass).
    Otherwise fresh noise and random timesteps are sampled.
    """
    with torch.no_grad():
        pixel_values = batch["pixel_values"].to(device=vae.device, dtype=vae.dtype)
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        if noise is None:
            noise = torch.randn_like(latents)
        else:
            # Reuse the same epsilon but adapt shape if batch sizes differ
            noise = noise[:latents.shape[0]].to(device=latents.device, dtype=latents.dtype)
            if noise.shape != latents.shape:
                # spatial dims may differ; resize noise to match
                noise = torch.randn_like(latents)

        if timesteps is None:
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device,
            ).long()
        else:
            timesteps = timesteps[:latents.shape[0]].to(device=latents.device)

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    return noisy_latents, noise, timesteps


def encode_batch(
    batch,
    vae,
    text_encoder,
    text_encoder_2,
    noise_scheduler,
    accelerator,
    resolution,
    rank,
    min_rank,
    alpha_rank_scale,
    max_timestep,
    noise=None,
    timesteps=None,
):
    """
    Full encode: VAE + noise + text encoders + time_ids.

    noise / timesteps: if given, reuse them (shadow pass).
    Returns (noisy_latents, timesteps, noise, prompt_embeds, pooled_prompt_embeds,
             add_time_ids, sigma_mask).
    """
    noisy_latents, noise, timesteps = _encode_latents(
        batch, vae, noise=noise, timesteps=timesteps, noise_scheduler=noise_scheduler
    )

    sigma_mask = get_mask_by_timestep(
        timesteps[0].item(),
        max_timestep,
        rank,
        min_rank,
        alpha_rank_scale,
    ).detach().to(accelerator.device)

    set_text_encoder_sigma_mask(sigma_mask)

    input_ids_1 = batch["input_ids"].to(device=next(text_encoder.parameters()).device)
    prompt_embeds_out = text_encoder(input_ids_1, output_hidden_states=True)
    prompt_embeds = prompt_embeds_out.hidden_states[-2]

    input_ids_2 = batch["input_ids_2"].to(device=next(text_encoder_2.parameters()).device)
    prompt_embeds_2_out = text_encoder_2(input_ids_2, output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds_2_out.text_embeds
    prompt_embeds_2 = prompt_embeds_2_out.hidden_states[-2]

    clear_text_encoder_sigma_mask()

    prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
    prompt_embeds = prompt_embeds.to(device=noisy_latents.device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=noisy_latents.device)

    add_time_ids = torch.tensor(
        [[resolution, resolution, 0, 0, resolution, resolution]],
        dtype=prompt_embeds.dtype,
        device=prompt_embeds.device,
    ).repeat(noisy_latents.shape[0], 1)

    return noisy_latents, timesteps, noise, prompt_embeds, pooled_prompt_embeds, add_time_ids, sigma_mask


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Study variant: logs cp_loss and org_loss per step"
    )

    # Dataset arguments
    parser.add_argument("--cp_dataset", type=str, required=True,
                        help="Copyright dataset dir (image/ + prompt.csv)")
    parser.add_argument("--org_image", type=str, required=True,
                        help="Original/clean image dataset dir (image/ + prompt.csv). "
                             "Used only for shadow loss computation (no backprop).")

    parser.add_argument("--cp_step", type=int, default=400,
                        help="Number of training steps on cp_dataset")

    # Model
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default="fp16")

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints_robust_study")
    parser.add_argument("--study_log_file", type=str, default=None,
                        help="CSV file for per-step loss log. Defaults to <output_dir>/study_loss_log.csv")
    parser.add_argument("--resolution", type=int, default=1024)

    # Training hypers
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--checkpointing_steps", type=int, default=200)

    # LoRA
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    # T-LoRA
    parser.add_argument("--min_rank", type=int, default=None)
    parser.add_argument("--alpha_rank_scale", type=float, default=1.0)
    parser.add_argument("--sig_type", type=str, default="last",
                        choices=["last", "principal", "middle"])

    # Other
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    # Resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Checkpoint folder to resume from (absolute or relative to output_dir)")
    parser.add_argument("--auto_resume_latest", action="store_true",
                        help="Resume from the latest checkpoint-iter* in output_dir")

    args = parser.parse_args()

    # ---------- auto-resume ----------
    if args.auto_resume_latest:
        if args.resume_from_checkpoint is not None:
            print("Both --resume_from_checkpoint and --auto_resume_latest provided; "
                  "using --resume_from_checkpoint.")
        elif not os.path.isdir(args.output_dir):
            print(f"--auto_resume_latest: output_dir does not exist yet, starting from scratch.")
        else:
            candidates = []
            for name in os.listdir(args.output_dir):
                if not name.startswith("checkpoint-iter"):
                    continue
                parts = name.rsplit("-step", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    candidates.append((int(parts[1]), name))
            if candidates:
                latest_step, latest_name = max(candidates, key=lambda x: x[0])
                args.resume_from_checkpoint = latest_name
                print(f"Auto-resume selected: {latest_name} (step {latest_step})")
            else:
                print("--auto_resume_latest: no checkpoints found, starting from scratch.")

    if args.min_rank is None:
        args.min_rank = args.rank // 2

    if args.study_log_file is None:
        os.makedirs(args.output_dir, exist_ok=True)
        args.study_log_file = os.path.join(args.output_dir, "study_loss_log.csv")

    # Validate dataset paths
    for dataset_arg, label in [(args.cp_dataset, "cp_dataset"), (args.org_image, "org_image")]:
        csv_path = os.path.join(dataset_arg, "prompt.csv")
        img_dir = os.path.join(dataset_arg, "image")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{label} prompt.csv not found: {csv_path}")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"{label} image dir not found: {img_dir}")

    print(f"\n=== Study Training Configuration ===")
    print(f"  cp_dataset:     {args.cp_dataset}")
    print(f"  org_image:      {args.org_image}")
    print(f"  cp_step:        {args.cp_step}")
    print(f"  study_log_file: {args.study_log_file}")
    print(f"  T-LoRA rank: {args.rank}, min_rank: {args.min_rank}, sig_type: {args.sig_type}")
    print(f"=====================================\n")

    # ---------- accelerator & seed ----------
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # ---------- tokenizers ----------
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision
    )

    # ---------- models ----------
    print("Loading models...")
    if args.mixed_precision == "bf16":
        model_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae", revision=args.revision, variant=args.variant,
        torch_dtype=model_dtype,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet", revision=args.revision, variant=args.variant,
        torch_dtype=model_dtype,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder", revision=args.revision, variant=args.variant,
        dtype=model_dtype,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2", revision=args.revision, variant=args.variant,
        dtype=model_dtype,
    )

    # Freeze base weights
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # ---------- dual LoRA setup ----------
    lora_attn_procs = build_dual_lora_attn_processors(
        unet, rank=args.rank, lora_alpha=args.lora_alpha, sig_type=args.sig_type
    )
    unet.set_attn_processor(lora_attn_procs)

    te1_targets, te1_lora1_params, te1_lora2_params = setup_text_encoder_dual_lora(
        text_encoder, rank=args.rank, lora_alpha=args.lora_alpha, sig_type=args.sig_type
    )
    te2_targets, te2_lora1_params, te2_lora2_params = setup_text_encoder_dual_lora(
        text_encoder_2, rank=args.rank, lora_alpha=args.lora_alpha, sig_type=args.sig_type
    )

    # ---------- resume: load checkpoint weights ----------
    resume_step = 0
    if args.resume_from_checkpoint:
        ckpt_path = args.resume_from_checkpoint
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(args.output_dir, ckpt_path)
        if not os.path.isdir(ckpt_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

        parts = os.path.basename(ckpt_path).rsplit("-step", 1)
        if len(parts) == 2 and parts[1].isdigit():
            resume_step = int(parts[1])

        print(f"Resuming from {ckpt_path} (step {resume_step})")
        loaded_unet = load_dual_lora_attn_state_dict(
            unet,
            torch.load(os.path.join(ckpt_path, "dual_lora_weights.pt"), map_location="cpu"),
            strict=True,
        )
        loaded_te1 = load_text_encoder_dual_lora_weights(
            text_encoder,
            os.path.join(ckpt_path, "text_encoder_dual_lora_weights.pt"),
            rank=args.rank, lora_alpha=args.lora_alpha, sig_type=args.sig_type,
        )
        loaded_te2 = load_text_encoder_dual_lora_weights(
            text_encoder_2,
            os.path.join(ckpt_path, "text_encoder_2_dual_lora_weights.pt"),
            rank=args.rank, lora_alpha=args.lora_alpha, sig_type=args.sig_type,
        )
        print(f"  UNet procs: {loaded_unet}, TE1: {loaded_te1}, TE2: {loaded_te2}")

    # ---------- collect trainable params (lora1 only) ----------
    lora1_params = []
    lora2_params = []

    for proc in unet.attn_processors.values():
        if not isinstance(proc, DualLoRACrossAttnProcessor):
            continue
        for sub in [proc.lora1_q, proc.lora1_k, proc.lora1_v, proc.lora1_out]:
            lora1_params.extend(p for p in sub.parameters() if p.requires_grad)
        for sub in [proc.lora2_q, proc.lora2_k, proc.lora2_v, proc.lora2_out]:
            lora2_params.extend(p for p in sub.parameters() if p.requires_grad)

    lora1_params.extend(te1_lora1_params)
    lora1_params.extend(te2_lora1_params)
    lora2_params.extend(te1_lora2_params)
    lora2_params.extend(te2_lora2_params)

    # Freeze lora2
    for p in lora2_params:
        p.requires_grad_(False)

    lora1_numel = sum(p.numel() for p in lora1_params)
    print(f"T-LoRA (lora1) trainable params: {lora1_numel:,}")

    # ---------- noise scheduler ----------
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    max_timestep = noise_scheduler.config.num_train_timesteps

    tlora_config = {
        "rank": args.rank, "min_rank": args.min_rank,
        "alpha_rank_scale": args.alpha_rank_scale, "sig_type": args.sig_type,
        "lora_alpha": args.lora_alpha, "max_timestep": max_timestep,
    }

    # ---------- optimizer ----------
    optimizer = torch.optim.AdamW(
        lora1_params, lr=args.learning_rate, betas=(0.9, 0.999),
        weight_decay=1e-2, eps=1e-08,
    )

    # ---------- datasets ----------
    cp_dataset = SimpleDreamBoothDataset(
        csv_path=os.path.join(args.cp_dataset, "prompt.csv"),
        image_dir=os.path.join(args.cp_dataset, "image"),
        tokenizer=tokenizer, tokenizer_2=tokenizer_2,
        size=args.resolution, center_crop=False,
    )
    org_dataset = SimpleDreamBoothDataset(
        csv_path=os.path.join(args.org_image, "prompt.csv"),
        image_dir=os.path.join(args.org_image, "image"),
        tokenizer=tokenizer, tokenizer_2=tokenizer_2,
        size=args.resolution, center_crop=False,
    )

    cp_dataloader = infinite_dataloader(cp_dataset, args.seed, args.train_batch_size)
    org_dataloader = infinite_dataloader(org_dataset, args.seed + 1, args.train_batch_size)

    # ---------- accelerator prepare ----------
    unet, text_encoder, text_encoder_2, optimizer = accelerator.prepare(
        unet, text_encoder, text_encoder_2, optimizer
    )
    vae = vae.to(accelerator.device)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # ---------- loss log file ----------
    os.makedirs(args.output_dir, exist_ok=True)
    log_file_exists = os.path.exists(args.study_log_file) and resume_step > 0
    log_fh = open(args.study_log_file, "a" if log_file_exists else "w",
                  encoding="utf-8", newline="")
    log_writer = csv.DictWriter(log_fh, fieldnames=["step", "cp_loss", "org_loss"])
    if not log_file_exists:
        log_writer.writeheader()
        log_fh.flush()

    print(f"\n***** Study Training *****")
    print(f"  CP dataset: {len(cp_dataset)} samples")
    print(f"  Org dataset: {len(org_dataset)} samples")
    print(f"  Steps: {args.cp_step}  (resuming from {resume_step})")
    print(f"  Loss log: {args.study_log_file}\n")

    unet.train()
    text_encoder.train()
    text_encoder_2.train()

    remaining = args.cp_step - resume_step
    if remaining <= 0:
        print(f"Training already complete (resume_step={resume_step} >= cp_step={args.cp_step})")
    else:
        progress_bar = tqdm(range(remaining), desc="Study-TLoRA")

        for step in range(resume_step, args.cp_step):
            # ---- cp training batch ----
            cp_batch = next(cp_dataloader)

            noisy_latents, timesteps, noise, prompt_embeds, pooled_embeds, time_ids, sigma_mask = encode_batch(
                cp_batch, vae, text_encoder, text_encoder_2, noise_scheduler,
                accelerator, args.resolution, args.rank, args.min_rank,
                args.alpha_rank_scale, max_timestep,
            )

            model_pred = unet(
                noisy_latents, timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": time_ids},
                cross_attention_kwargs={"sigma_mask": sigma_mask},
            ).sample

            cp_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            accelerator.backward(cp_loss)
            accelerator.clip_grad_norm_(lora1_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

            # ---- org shadow batch (same noise & timestep, no grad, no backprop) ----
            org_batch = next(org_dataloader)

            with torch.no_grad():
                org_noisy, _, _, org_prompt_embeds, org_pooled, org_time_ids, org_sigma = encode_batch(
                    org_batch, vae, text_encoder, text_encoder_2, noise_scheduler,
                    accelerator, args.resolution, args.rank, args.min_rank,
                    args.alpha_rank_scale, max_timestep,
                    noise=noise.detach(),
                    timesteps=timesteps.detach(),
                )

                org_pred = unet(
                    org_noisy, timesteps,
                    encoder_hidden_states=org_prompt_embeds,
                    added_cond_kwargs={"text_embeds": org_pooled, "time_ids": org_time_ids},
                    cross_attention_kwargs={"sigma_mask": org_sigma},
                ).sample

                # noise already resampled-safe inside encode_batch; compare against
                # the actual noise added to org latents (which == noise[:bs])
                org_noise = noise[:org_noisy.shape[0]].to(org_pred.device)
                org_loss = F.mse_loss(org_pred.float(), org_noise.float(), reduction="mean")

            # ---- logging ----
            cp_loss_val = cp_loss.item()
            org_loss_val = org_loss.item()

            if accelerator.is_main_process:
                log_writer.writerow({
                    "step": step + 1,
                    "cp_loss": f"{cp_loss_val:.6f}",
                    "org_loss": f"{org_loss_val:.6f}",
                })
                log_fh.flush()

            active_rank = (
                int(((max_timestep - timesteps[0].item()) / max_timestep) ** args.alpha_rank_scale
                    * (args.rank - args.min_rank))
                + args.min_rank
            )
            progress_bar.update(1)
            progress_bar.set_postfix({
                "cp": f"{cp_loss_val:.4f}",
                "org": f"{org_loss_val:.4f}",
                "t": f"{timesteps[0].item()}",
                "r": f"{active_rank}/{args.rank}",
            })

            # ---- checkpoint ----
            if (step + 1) % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    save_checkpoint(
                        unet, args.output_dir, 1, "lora1", step + 1,
                        accelerator=accelerator, tlora_config=tlora_config,
                        text_encoder=text_encoder, text_encoder_2=text_encoder_2,
                    )

        progress_bar.close()

    log_fh.close()

    # ---------- save final ----------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and args.cp_step > 0:
        save_checkpoint(
            unet, args.output_dir, 1, "complete", args.cp_step,
            accelerator=accelerator, tlora_config=tlora_config,
            text_encoder=text_encoder, text_encoder_2=text_encoder_2,
        )

        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        unet_s = accelerator.unwrap_model(unet)
        te1_s = accelerator.unwrap_model(text_encoder)
        te2_s = accelerator.unwrap_model(text_encoder_2)

        torch.save(collect_dual_lora_attn_state_dict(unet_s),
                   os.path.join(final_dir, "dual_lora_weights.pt"))
        torch.save(collect_text_encoder_lora_state_dict(te1_s),
                   os.path.join(final_dir, "text_encoder_dual_lora_weights.pt"))
        torch.save(collect_text_encoder_lora_state_dict(te2_s),
                   os.path.join(final_dir, "text_encoder_2_dual_lora_weights.pt"))
        torch.save(tlora_config, os.path.join(final_dir, "tlora_config.pt"))

        print(f"\n{'='*60}")
        print(f"Study training complete.  Final model: {final_dir}")
        print(f"Loss log: {args.study_log_file}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
