#!/usr/bin/env python3
"""
Generate images with the full continue-training stack:

1) Base SDXL
2) Dual T-LoRA checkpoint (from --lora_path)
3) RL W_R checkpoint (from --rl_checkpoint)
4) Continue LoRA adapter checkpoint from train_dreambooth_continue.py

The continue adapter is loaded from --continue_lora_path
(adapter_model.safetensors or adapter_model.bin), then merged for inference.
"""

import argparse
import json
import os

import torch
from diffusers import (AutoencoderKL, PNDMScheduler, StableDiffusionXLPipeline,
                       UNet2DConditionModel)
from peft import PeftModel
from tlora_module import (DualLoRACrossAttnProcessor,
                          DualLoRATextLinearLayer,
                          attach_tlora_sigma_mask_hook,
                          build_dual_lora_attn_processors,
                          clear_text_encoder_sigma_mask,
                          load_dual_lora_attn_state_dict,
                          load_text_encoder_dual_lora_weights,
                          set_text_encoder_sigma_for_generation)
from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                          CLIPTokenizer)
from utils import (create_sdxl_refiner_pipeline,
                   prepare_sdxl_pipeline_for_inference,
                   resolve_checkpoint_paths, run_sdxl_inference)


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


def zero_lora2_and_freeze_lora1(unet, text_encoder, text_encoder_2):
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


def load_backbone_info(continue_lora_path):
    info_path = os.path.join(continue_lora_path, "backbone_info.json")
    if not os.path.exists(info_path):
        return {}

    with open(info_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    if not isinstance(loaded, dict):
        return {}
    return loaded


def validate_continue_adapter_path(continue_lora_path):
    if not os.path.isdir(continue_lora_path):
        raise FileNotFoundError(f"Continue LoRA checkpoint directory not found: {continue_lora_path}")

    adapter_safetensors = os.path.join(continue_lora_path, "adapter_model.safetensors")
    adapter_bin = os.path.join(continue_lora_path, "adapter_model.bin")
    if not (os.path.exists(adapter_safetensors) or os.path.exists(adapter_bin)):
        raise FileNotFoundError(
            "Continue LoRA checkpoint must contain adapter_model.safetensors or adapter_model.bin: "
            f"{continue_lora_path}"
        )


def bake_wr_into_unet(unet, wr_state):
    trainable_layers = collect_trainable_layers(unet)
    if len(trainable_layers) == 0:
        raise RuntimeError("No trainable layers found for W_R injection")

    wr_names = [name for name, _ in trainable_layers]
    missing_wr = sorted(set(wr_names) - set(wr_state.keys()))
    if missing_wr:
        raise KeyError(
            f"W_R missing {len(missing_wr)} layers required by current UNet. "
            f"First missing key: {missing_wr[0]}"
        )

    unet_param_lookup = dict(unet.named_parameters())
    baked_count = 0
    with torch.no_grad():
        for full_name, weight in trainable_layers:
            local_name = full_name[len("unet."):]
            if local_name not in unet_param_lookup:
                continue

            wr_tensor = wr_state[full_name].to(device=weight.device, dtype=weight.dtype)
            if wr_tensor.shape != weight.shape:
                raise ValueError(
                    f"Shape mismatch for {full_name}: "
                    f"W_R {tuple(wr_tensor.shape)} vs UNet {tuple(weight.shape)}"
                )

            unet_param_lookup[local_name].data.add_(wr_tensor)
            baked_count += 1

    extra_wr_keys = sorted(set(wr_state.keys()) - set(wr_names))
    return baked_count, extra_wr_keys


def main():
    parser = argparse.ArgumentParser(
        description="Generate image using SDXL + T-LoRA + RL W_R + continue LoRA"
    )
    parser.add_argument(
        "--continue_lora_path",
        "--continue_checkpoint",
        dest="continue_lora_path",
        type=str,
        required=True,
        help="Path to continue-training LoRA checkpoint directory (adapter_model.*)",
    )
    parser.add_argument(
        "--rl_checkpoint",
        type=str,
        default=None,
        help="Path to RL checkpoint (wr_checkpoint_step*.pt or wr_final.pt). If omitted, tries backbone_info.json",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to dual-LoRA checkpoint directory. If omitted, tries backbone_info.json or RL checkpoint args",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base SDXL model path. If omitted, tries backbone_info.json then defaults to SDXL base 1.0",
    )
    parser.add_argument("--revision", type=str, default=None, help="Model revision override")
    parser.add_argument("--variant", type=str, default=None, help="Model variant override")

    parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Optional negative prompt")
    parser.add_argument("--output_path", type=str, default="output_continue.png", help="Output image path")

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--use_refiner",
        action="store_true",
        help="Use SDXL refiner for better quality",
    )

    args = parser.parse_args()

    validate_continue_adapter_path(args.continue_lora_path)
    backbone_info = load_backbone_info(args.continue_lora_path)

    rl_checkpoint = args.rl_checkpoint or backbone_info.get("rl_checkpoint")
    if rl_checkpoint is None:
        raise ValueError(
            "RL checkpoint is required. Provide --rl_checkpoint or include rl_checkpoint in backbone_info.json"
        )
    if not os.path.exists(rl_checkpoint):
        raise FileNotFoundError(f"RL checkpoint not found: {rl_checkpoint}")

    rl_state = torch.load(rl_checkpoint, map_location="cpu")
    if "W_R" not in rl_state:
        raise KeyError("RL checkpoint missing key: 'W_R'")

    rl_args = rl_state.get("args", {})
    if not isinstance(rl_args, dict):
        rl_args = {}

    lora_path = (
        args.lora_path
        or backbone_info.get("lora_path")
        or rl_args.get("lora_path")
        or rl_args.get("checkpoint_path")
    )
    if lora_path is None:
        raise ValueError(
            "Dual-LoRA checkpoint path is required. Provide --lora_path or ensure it exists in backbone_info.json / RL checkpoint args."
        )

    base_model = (
        args.base_model
        or backbone_info.get("pretrained_model_name_or_path")
        or "stabilityai/stable-diffusion-xl-base-1.0"
    )

    revision = args.revision if args.revision is not None else rl_args.get("revision", None)
    if args.variant is not None:
        variant = args.variant
    else:
        variant = rl_args.get("variant", "fp16")

    resolution_default = int(rl_args.get("resolution", 1024))
    height = int(args.height) if args.height is not None else resolution_default
    width = int(args.width) if args.width is not None else resolution_default

    device = torch.device(args.device)
    mixed_precision = str(rl_args.get("mixed_precision", "fp16"))

    if device.type == "cuda":
        if mixed_precision == "bf16":
            model_dtype = torch.bfloat16
        elif mixed_precision == "fp16":
            model_dtype = torch.float16
        else:
            model_dtype = torch.float32
    else:
        model_dtype = torch.float32
        variant = None

    print(f"\n{'='*60}")
    print("Continue Checkpoint Inference")
    print(f"{'='*60}")
    print(f"Continue LoRA checkpoint: {args.continue_lora_path}")
    print(f"RL checkpoint: {rl_checkpoint}")
    print(f"Dual-LoRA checkpoint: {lora_path}")
    print(f"Base model: {base_model}")
    print(f"Prompt: {args.prompt}")
    print(f"Use refiner: {args.use_refiner}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    (
        unet_weights_path,
        config_path,
        text_encoder_weights_path,
        text_encoder_2_weights_path,
    ) = resolve_checkpoint_paths(lora_path)

    if not os.path.exists(unet_weights_path):
        raise FileNotFoundError(f"dual_lora_weights.pt not found: {unet_weights_path}")

    rank = int(rl_state.get("rank", 16))
    min_rank = int(rl_state.get("min_rank", max(1, rank // 2)))
    alpha_rank_scale = float(rl_state.get("alpha_rank_scale", 1.0))
    max_timestep = int(rl_state.get("max_timestep", 1000))
    lora_alpha = 32.0
    sig_type = "last"

    if config_path is not None and os.path.exists(config_path):
        tlora_config = torch.load(config_path, map_location="cpu")
        rank = int(tlora_config.get("rank", rank))
        lora_alpha = float(tlora_config.get("lora_alpha", lora_alpha))
        sig_type = str(tlora_config.get("sig_type", sig_type))
        min_rank = int(tlora_config.get("min_rank", min_rank))
        alpha_rank_scale = float(tlora_config.get("alpha_rank_scale", alpha_rank_scale))
        max_timestep = int(tlora_config.get("max_timestep", max_timestep))

    tokenizer = CLIPTokenizer.from_pretrained(
        base_model,
        subfolder="tokenizer",
        revision=revision,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        base_model,
        subfolder="tokenizer_2",
        revision=revision,
    )

    vae = AutoencoderKL.from_pretrained(
        base_model,
        subfolder="vae",
        revision=revision,
        variant=variant,
        torch_dtype=model_dtype,
    )
    unet = UNet2DConditionModel.from_pretrained(
        base_model,
        subfolder="unet",
        revision=revision,
        variant=variant,
        torch_dtype=model_dtype,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        base_model,
        subfolder="text_encoder",
        revision=revision,
        variant=variant,
        dtype=model_dtype,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        base_model,
        subfolder="text_encoder_2",
        revision=revision,
        variant=variant,
        dtype=model_dtype,
    )

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    lora_attn_procs = build_dual_lora_attn_processors(
        unet,
        rank=rank,
        lora_alpha=lora_alpha,
        sig_type=sig_type,
    )
    unet.set_attn_processor(lora_attn_procs)

    unet_lora_state = torch.load(unet_weights_path, map_location="cpu")
    loaded_unet = load_dual_lora_attn_state_dict(unet, unet_lora_state, strict=True)

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

    print(f"Loaded dual-LoRA UNet attention processors: {loaded_unet}")
    print(f"Loaded text_encoder dual-LoRA layers: {loaded_te1}")
    print(f"Loaded text_encoder_2 dual-LoRA layers: {loaded_te2}")

    zero_lora2_and_freeze_lora1(unet, text_encoder, text_encoder_2)

    attach_tlora_sigma_mask_hook(
        unet,
        rank=rank,
        min_rank=min_rank,
        alpha_rank_scale=alpha_rank_scale,
        max_timestep=max_timestep,
    )

    unet = unet.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    text_encoder_2 = text_encoder_2.to(device)

    baked_count, extra_wr_keys = bake_wr_into_unet(unet, rl_state["W_R"])
    print(f"Baked W_R into {baked_count} UNet layers")
    if len(extra_wr_keys) > 0:
        print(f"Warning: {len(extra_wr_keys)} extra W_R keys in checkpoint not used by current UNet")

    print(f"Loading continue LoRA adapter from: {args.continue_lora_path}")
    unet = PeftModel.from_pretrained(unet, args.continue_lora_path)
    unet = unet.merge_and_unload()
    unet = unet.to(device=device, dtype=model_dtype)
    print("Continue LoRA adapter loaded and merged")

    generation_scheduler = PNDMScheduler.from_pretrained(
        base_model,
        subfolder="scheduler",
    )

    generation_pipeline = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=generation_scheduler,
        add_watermarker=False,
    )
    generation_pipeline = generation_pipeline.to(device)
    generation_pipeline = prepare_sdxl_pipeline_for_inference(generation_pipeline)
    generation_pipeline.set_progress_bar_config(disable=True)
    generation_pipeline.tlora_text_encoder_config = {
        "rank": rank,
        "min_rank": min_rank,
        "alpha_rank_scale": alpha_rank_scale,
        "max_timestep": max_timestep,
    }

    refiner_pipeline = None
    if args.use_refiner:
        refiner_pipeline = create_sdxl_refiner_pipeline(
            base_pipeline=generation_pipeline,
            device=device,
            torch_dtype=model_dtype,
            variant=variant,
        )

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(args.seed))

    image = run_sdxl_inference(
        base_pipeline=generation_pipeline,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=height,
        width=width,
        generator=generator,
        use_refiner=args.use_refiner,
        refiner_pipeline=refiner_pipeline,
        set_text_sigma_for_generation=set_text_encoder_sigma_for_generation,
        clear_text_sigma_mask=clear_text_encoder_sigma_mask,
    )

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    image.save(args.output_path)

    print(f"\n{'='*60}")
    print(f"Saved image: {args.output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()