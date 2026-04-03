#!/usr/bin/env python3
"""
Further-full training with perturbation over copyright dataset.

Compared to further_lora study script, this variant keeps the same training/logging
workflow but replaces T-LoRA/MR training with full UNet weight fine-tuning on
base model M0 weights (no MR branch).

Per step:
1) Build perturbed virtual weights W + delta(W) with line-7 style bound.
2) Train step uses forward with perturbed virtual weights.
3) After optimizer step, evaluate forward MSE on CP and original datasets with
   non-perturbed current full model weights.
4) Log cp_train_loss, cp_forward_loss, org_forward_loss to study CSV.
"""

import argparse
import csv
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from utils import SimpleDreamBoothDataset, infinite_dataloader

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


def encode_batch(
    batch,
    vae,
    text_encoder,
    text_encoder_2,
    noise_scheduler,
    device,
    resolution,
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

    input_ids_1 = batch["input_ids"].to(device)
    prompt_out_1 = text_encoder(input_ids_1, output_hidden_states=True)
    prompt_embeds_1 = prompt_out_1.hidden_states[-2]

    input_ids_2 = batch["input_ids_2"].to(device)
    prompt_out_2 = text_encoder_2(input_ids_2, output_hidden_states=True)
    pooled_prompt_embeds = prompt_out_2.text_embeds
    prompt_embeds_2 = prompt_out_2.hidden_states[-2]

    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1).to(device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device)

    add_time_ids = torch.tensor(
        [[resolution, resolution, 0, 0, resolution, resolution]],
        dtype=prompt_embeds.dtype,
        device=device,
    ).repeat(noisy_latents.shape[0], 1)

    return noisy_latents, timesteps, noise, prompt_embeds, pooled_prompt_embeds, add_time_ids


def build_unet_overrides_with_delta(unet, max_distortion_pct, seed, step):
    """Build virtual UNet params W + delta(W), with ||delta||/||W|| <= m/100 per tensor."""
    overrides = {}

    for idx, (name, param) in enumerate(unet.named_parameters()):
        if not param.requires_grad:
            continue

        reference = param
        generator = torch.Generator(device=reference.device)
        generator.manual_seed(int(seed) + int(step) * 1000003 + idx)

        delta = torch.randn(
            reference.shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )

        ref_norm = reference.norm(2).clamp_min(1e-12)
        delta_norm = delta.norm(2).clamp_min(1e-12)
        max_norm = ref_norm * (float(max_distortion_pct) / 100.0)
        delta = delta / delta_norm * max_norm

        overrides[name] = reference + delta

    return overrides


def save_checkpoint(unet, text_encoder, text_encoder_2, output_dir, step, config):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-step{step:06d}")
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save(unet.state_dict(), os.path.join(ckpt_dir, "unet_full.pt"))
    torch.save(text_encoder.state_dict(), os.path.join(ckpt_dir, "text_encoder.pt"))
    torch.save(text_encoder_2.state_dict(), os.path.join(ckpt_dir, "text_encoder_2.pt"))
    torch.save(config, os.path.join(ckpt_dir, "train_config.pt"))

    print(f"Saved checkpoint: {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Further-full training (no MR): full UNet fine-tune with delta perturbation"
    )

    parser.add_argument("--cp_dataset", type=str, required=True)
    parser.add_argument("--org_image", type=str, required=True,
                        help="Path to original dataset directory (contains image/ and prompt.csv)")
    parser.add_argument("--cp_step", type=int, default=10000)
    parser.add_argument("--max_distortion_pct", type=float, default=10.0,
                        help="m in line-7 style bound (percentage)")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default="fp16")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)

    parser.add_argument("--output_dir", type=str, default="checkpoints_further_full")
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
    print("Further-Full Training (Full UNet fine-tune, no MR)")
    print(f"{'=' * 70}")
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

    # Full UNet fine-tune over M0, keep VAE/text encoders frozen
    unet.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    if resume_checkpoint_dir is not None:
        unet_ckpt = os.path.join(resume_checkpoint_dir, "unet_full.pt")
        if not os.path.exists(unet_ckpt):
            raise FileNotFoundError(f"Resume checkpoint missing unet_full.pt: {unet_ckpt}")
        unet.load_state_dict(torch.load(unet_ckpt, map_location="cpu"), strict=True)

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

    full_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(full_params, lr=args.learning_rate, weight_decay=args.weight_decay)

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

    progress_bar = tqdm(range(start_step, args.cp_step + 1), desc="Further-Full")
    for step in progress_bar:
        cp_batch = next(cp_loader)

        (
            noisy_latents,
            timesteps,
            noise,
            prompt_embeds,
            pooled_prompt_embeds,
            add_time_ids,
        ) = encode_batch(
            cp_batch,
            vae,
            text_encoder,
            text_encoder_2,
            noise_scheduler,
            device,
            args.resolution,
        )

        # Train forward with perturbed virtual full weights
        unet_overrides = build_unet_overrides_with_delta(
            unet=unet,
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
        ).sample

        cp_train_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        optimizer.zero_grad(set_to_none=True)
        cp_train_loss.backward()
        torch.nn.utils.clip_grad_norm_(full_params, 1.0)
        optimizer.step()

        # Post-step evaluation forward with current unperturbed full model
        with torch.no_grad():
            cp_eval_batch = next(cp_loader)
            (
                cp_eval_noisy_latents,
                cp_eval_timesteps,
                cp_eval_noise,
                cp_eval_prompt_embeds,
                cp_eval_pooled_prompt_embeds,
                cp_eval_add_time_ids,
            ) = encode_batch(
                cp_eval_batch,
                vae,
                text_encoder,
                text_encoder_2,
                noise_scheduler,
                device,
                args.resolution,
            )

            cp_eval_pred = unet(
                cp_eval_noisy_latents,
                cp_eval_timesteps,
                encoder_hidden_states=cp_eval_prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": cp_eval_pooled_prompt_embeds,
                    "time_ids": cp_eval_add_time_ids,
                },
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
            ) = encode_batch(
                org_eval_batch,
                vae,
                text_encoder,
                text_encoder_2,
                noise_scheduler,
                device,
                args.resolution,
            )

            org_eval_pred = unet(
                org_eval_noisy_latents,
                org_eval_timesteps,
                encoder_hidden_states=org_eval_prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": org_eval_pooled_prompt_embeds,
                    "time_ids": org_eval_add_time_ids,
                },
            ).sample
            org_forward_loss = F.mse_loss(org_eval_pred.float(), org_eval_noise.float(), reduction="mean")

        progress_bar.set_postfix(
            {
                "cp_train": f"{float(cp_train_loss.item()):.6f}",
                "cp_fwd": f"{float(cp_forward_loss.item()):.6f}",
                "org_fwd": f"{float(org_forward_loss.item()):.6f}",
                "t": int(timesteps[0].item()),
            }
        )

        log_writer.writerow(
            {
                "step": step,
                "cp_train_loss": f"{float(cp_train_loss.item()):.6f}",
                "cp_forward_loss": f"{float(cp_forward_loss.item()):.6f}",
                "org_forward_loss": f"{float(org_forward_loss.item()):.6f}",
                "timestep": int(timesteps[0].item()),
            }
        )
        log_fh.flush()

        if step % args.checkpointing_steps == 0:
            save_checkpoint(unet, text_encoder, text_encoder_2, args.output_dir, step, vars(args))

    log_fh.close()

    save_checkpoint(unet, text_encoder, text_encoder_2, args.output_dir, args.cp_step, vars(args))

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(unet.state_dict(), os.path.join(final_dir, "unet_full.pt"))
    torch.save(text_encoder.state_dict(), os.path.join(final_dir, "text_encoder.pt"))
    torch.save(text_encoder_2.state_dict(), os.path.join(final_dir, "text_encoder_2.pt"))
    torch.save(vars(args), os.path.join(final_dir, "train_config.pt"))

    print(f"\n{'=' * 70}")
    print("Further-full training complete")
    print(f"Final model saved to: {final_dir}")
    print(f"Study loss log saved to: {args.study_log_file}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
