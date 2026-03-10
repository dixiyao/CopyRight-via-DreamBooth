#!/usr/bin/env python3
"""
Generate images with RL-trained W_R on top of SDXL + T-LoRA (lora1).

This script loads:
1) Base SDXL weights,
2) Dual-LoRA checkpoint (uses only lora1; lora2 is zeroed),
3) W_R from train_dreambooth_lora_sdxl_rl.py checkpoints.

Optional:
- `--perturbation` (or `--pertubation`) adds an extra random Gaussian W_delta
  with normalization + `--magnitude` scaling (default 0.1), similar to RL training.
"""

import argparse
import os

import torch
from diffusers import (AutoencoderKL, PNDMScheduler, StableDiffusionXLPipeline,
                       UNet2DConditionModel)
from tlora_module import (DualLoRACrossAttnProcessor,
                          DualLoRATextLinearLayer,
                          attach_tlora_sigma_mask_hook,
                          build_dual_lora_attn_processors,
                          clear_text_encoder_sigma_mask,
                          compute_orthogonal_lora_weight_delta,
                          get_mask_by_timestep,
                          load_dual_lora_attn_state_dict,
                          load_text_encoder_dual_lora_weights,
                          set_text_encoder_sigma_for_generation)
from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                          CLIPTokenizer)
from utils import (create_sdxl_refiner_pipeline,
                   prepare_sdxl_pipeline_for_inference,
                   resolve_checkpoint_paths, run_sdxl_inference)

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call


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


def build_tlora1_lookup(unet, trainable_layers):
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
    def __init__(self, base_unet, unet_override_specs, wr_weights, deltas=None):
        super().__init__()
        self.base_unet = base_unet
        self.unet_override_specs = unet_override_specs
        self.wr_weights = wr_weights
        self.deltas = deltas
        self._cached_overrides = None

    def refresh_overrides(self):
        self._cached_overrides = build_unet_wr_delta_overrides(
            unet_override_specs=self.unet_override_specs,
            wr_weights=self.wr_weights,
            deltas=self.deltas,
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


def compute_generation_sigma_mask(scheduler, num_inference_steps, rank, min_rank, alpha_rank_scale, max_timestep, device):
    timestep_value = float(max_timestep - 1)
    try:
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps
        if len(timesteps) > 0:
            first_t = timesteps[0]
            timestep_value = float(first_t.item()) if torch.is_tensor(first_t) else float(first_t)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass

    return get_mask_by_timestep(
        timestep=timestep_value,
        max_timestep=max_timestep,
        max_rank=rank,
        min_rank=min_rank,
        alpha=alpha_rank_scale,
    ).to(device)


def sample_wr_perturbations(
    wr_names,
    w0_weights,
    wr_weights,
    tlora1_lookup,
    sigma_mask,
    magnitude,
    seed=None,
):
    deltas = {}

    for idx, name in enumerate(wr_names):
        current_w = w0_weights[name] + wr_weights[name]

        lora1_module = tlora1_lookup.get(name)
        if lora1_module is not None:
            tlora_delta = compute_orthogonal_lora_weight_delta(lora1_module, mask=sigma_mask)
            current_w = current_w + tlora_delta.to(
                device=current_w.device,
                dtype=current_w.dtype,
            )

        sigma = max(float(current_w.std().item()), 1e-12)

        if seed is None:
            delta = torch.randn(
                current_w.shape,
                device=current_w.device,
                dtype=current_w.dtype,
            )
            alpha = 1.0 if torch.rand((), device=current_w.device).item() > 0.5 else -1.0
        else:
            layer_seed = int(seed) + (idx + 1) * 1000003
            generator = torch.Generator(device=current_w.device)
            generator.manual_seed(layer_seed)
            delta = torch.randn(
                current_w.shape,
                device=current_w.device,
                dtype=current_w.dtype,
                generator=generator,
            )
            alpha = 1.0 if torch.rand((1,), device=current_w.device, generator=generator).item() > 0.5 else -1.0

        delta = delta * sigma
        delta_norm = delta.norm(2).clamp_min(1e-12)
        delta = delta / delta_norm
        delta = delta * float(magnitude)
        delta = delta * alpha
        deltas[name] = delta

    return deltas


def main():
    parser = argparse.ArgumentParser(
        description="Generate image using RL checkpoint (W_R) + T-LoRA SDXL"
    )
    parser.add_argument("--rl_checkpoint", type=str, required=True, help="Path to RL checkpoint (wr_checkpoint_step*.pt or wr_final.pt)")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to dual-LoRA checkpoint dir")
    parser.add_argument(
        "--base_model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base SDXL model path",
    )
    parser.add_argument("--revision", type=str, default=None, help="Model revision override")
    parser.add_argument("--variant", type=str, default=None, help="Model variant override")

    parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Optional negative prompt")
    parser.add_argument("--output_path", type=str, default="output_rl.png", help="Output image path")

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

    parser.add_argument(
        "--perturbation",
        "--pertubation",
        action="store_true",
        dest="perturbation",
        help="Apply random Gaussian W_delta perturbation on top of W_R",
    )
    parser.add_argument("--magnitude", type=float, default=0.1, help="Perturbation magnitude for W_delta")

    args = parser.parse_args()

    if args.magnitude <= 0:
        raise ValueError("--magnitude must be > 0")

    if not os.path.exists(args.rl_checkpoint):
        raise FileNotFoundError(f"RL checkpoint not found: {args.rl_checkpoint}")

    rl_state = torch.load(args.rl_checkpoint, map_location="cpu")
    if "W_R" not in rl_state:
        raise KeyError("RL checkpoint missing key: 'W_R'")

    rl_args = rl_state.get("args", {}) if isinstance(rl_state.get("args", {}), dict) else {}

    lora_path = args.lora_path or rl_args.get("lora_path") or rl_args.get("checkpoint_path")
    if lora_path is None:
        raise ValueError(
            "Dual-LoRA checkpoint path is required. Provide --lora_path or use an RL checkpoint that stores args['checkpoint_path']."
        )

    base_model = args.base_model
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
    print("RL Checkpoint Inference")
    print(f"{'='*60}")
    print(f"RL checkpoint: {args.rl_checkpoint}")
    print(f"Dual-LoRA checkpoint: {lora_path}")
    print(f"Base model: {base_model}")
    print(f"Prompt: {args.prompt}")
    print(f"Use refiner: {args.use_refiner}")
    print(f"Perturbation enabled: {args.perturbation}")
    if args.perturbation:
        print(f"Perturbation magnitude: {args.magnitude}")
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

    trainable_layers = collect_trainable_layers(unet)
    if len(trainable_layers) == 0:
        raise RuntimeError("No trainable layers found for W_R injection")

    tlora1_lookup = build_tlora1_lookup(unet, trainable_layers)

    w0_weights = {
        name: weight.data.detach().float().clone().to(device)
        for name, weight in trainable_layers
    }
    wr_weights = {
        name: torch.zeros_like(w0_weights[name])
        for name in w0_weights
    }

    wr_names = list(wr_weights.keys())
    checkpoint_wr = rl_state["W_R"]

    for name in wr_names:
        if name not in checkpoint_wr:
            raise KeyError(f"W_R layer missing from RL checkpoint: {name}")
        loaded_tensor = checkpoint_wr[name]
        if loaded_tensor.shape != wr_weights[name].shape:
            raise ValueError(
                f"Shape mismatch for {name}: checkpoint {tuple(loaded_tensor.shape)} != current {tuple(wr_weights[name].shape)}"
            )
        wr_weights[name] = loaded_tensor.to(
            device=wr_weights[name].device,
            dtype=wr_weights[name].dtype,
        )

    extra_keys = sorted(set(checkpoint_wr.keys()) - set(wr_names))
    if len(extra_keys) > 0:
        print(f"Warning: {len(extra_keys)} extra W_R keys in checkpoint not used by current UNet")

    unet_param_lookup = dict(unet.named_parameters())
    unet_override_specs = []
    for full_name in wr_names:
        if not full_name.startswith("unet."):
            continue
        local_name = full_name[len("unet."):]
        base_param = unet_param_lookup[local_name]
        unet_override_specs.append((full_name, local_name, base_param))

    generation_scheduler = PNDMScheduler.from_pretrained(
        base_model,
        subfolder="scheduler",
    )

    deltas = None
    if args.perturbation:
        sigma_mask = compute_generation_sigma_mask(
            scheduler=generation_scheduler,
            num_inference_steps=args.num_inference_steps,
            rank=rank,
            min_rank=min_rank,
            alpha_rank_scale=alpha_rank_scale,
            max_timestep=max_timestep,
            device=device,
        )
        deltas = sample_wr_perturbations(
            wr_names=wr_names,
            w0_weights=w0_weights,
            wr_weights=wr_weights,
            tlora1_lookup=tlora1_lookup,
            sigma_mask=sigma_mask,
            magnitude=args.magnitude,
            seed=args.seed,
        )

    generation_unet = WRInjectedUNet(
        base_unet=unet,
        unet_override_specs=unet_override_specs,
        wr_weights=wr_weights,
        deltas=deltas,
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
    print(f"Perturbation: {'enabled' if args.perturbation else 'disabled'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
