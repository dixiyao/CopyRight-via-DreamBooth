import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalLoRALinearLayer(nn.Module):
    """Ortho-LoRA layer with SVD init and timestep-dependent masking."""

    def __init__(self, in_features, out_features, rank=4, sig_type="last", original_layer=None):
        super().__init__()
        self.rank = rank

        self.q_layer = nn.Linear(in_features, rank, bias=False)
        self.p_layer = nn.Linear(rank, out_features, bias=False)
        self.lambda_layer = nn.Parameter(torch.ones(1, rank))

        if original_layer is not None:
            base_m = original_layer.weight.data.detach().to(torch.float32)
            u, s, v = torch.linalg.svd(base_m)
        else:
            base_m = torch.normal(
                mean=0,
                std=1.0 / rank,
                size=(out_features, in_features),
            )
            u, s, v = torch.linalg.svd(base_m)

        if sig_type == "last":
            self.q_layer.weight.data = v[-rank:].clone()
            self.p_layer.weight.data = u[:, -rank:].clone()
            self.lambda_layer.data = s[None, -rank:].clone()
        elif sig_type == "principal":
            self.q_layer.weight.data = v[:rank].clone()
            self.p_layer.weight.data = u[:, :rank].clone()
            self.lambda_layer.data = s[None, :rank].clone()
        elif sig_type == "middle":
            start_v = math.ceil((v.shape[0] - rank) / 2)
            self.q_layer.weight.data = v[start_v:start_v + rank].clone()
            start_u = math.ceil((u.shape[1] - rank) / 2)
            self.p_layer.weight.data = u[:, start_u:start_u + rank].clone()
            start_s = math.ceil((s.shape[0] - rank) / 2)
            self.lambda_layer.data = s[None, start_s:start_s + rank].clone()

        self.base_p = copy.deepcopy(self.p_layer)
        self.base_q = copy.deepcopy(self.q_layer)
        self.base_lambda = copy.deepcopy(self.lambda_layer)

        for param in self.parameters():
            param.data = param.data.contiguous()

        self.base_p.requires_grad_(False)
        self.base_q.requires_grad_(False)
        self.base_lambda.requires_grad_(False)

    def forward(self, hidden_states, mask=None):
        if mask is None:
            mask = torch.ones((1, self.rank), device=hidden_states.device)

        orig_dtype = hidden_states.dtype
        dtype = self.q_layer.weight.dtype
        mask = mask.to(device=hidden_states.device, dtype=dtype)

        q_hidden = self.q_layer(hidden_states.to(dtype)) * self.lambda_layer * mask
        p_hidden = self.p_layer(q_hidden)

        base_q_hidden = self.base_q(hidden_states.to(dtype)) * self.base_lambda * mask
        base_p_hidden = self.base_p(base_q_hidden)

        return (p_hidden - base_p_hidden).to(orig_dtype)

    def regularization(self):
        a = self.q_layer.weight
        b = self.p_layer.weight
        eye = torch.eye(self.rank, device=a.device, dtype=a.dtype)
        a_reg = torch.sum((a @ a.T - eye) ** 2)
        b_reg = torch.sum((b.T @ b - eye) ** 2)
        return a_reg + b_reg


class StandardLoRALinearLayer(nn.Module):
    """Standard LoRA linear layer."""

    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.rank = rank
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.down.weight, std=1.0 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype
        return self.up(self.down(hidden_states.to(dtype))).to(orig_dtype)


class DualLoRACrossAttnProcessor(nn.Module):
    """Attention processor with T-LoRA (lora1) + standard LoRA (lora2)."""

    def __init__(
        self,
        hidden_size,
        cross_attention_dim=None,
        rank=4,
        lora_alpha=32,
        sig_type="last",
        original_layer=None,
    ):
        super().__init__()

        in_features = cross_attention_dim if cross_attention_dim is not None else hidden_size
        self.lora2_scale = lora_alpha / rank

        self.lora1_q = OrthogonalLoRALinearLayer(
            hidden_size,
            hidden_size,
            rank,
            sig_type,
            original_layer.to_q if original_layer is not None else None,
        )
        self.lora1_k = OrthogonalLoRALinearLayer(
            in_features,
            hidden_size,
            rank,
            sig_type,
            original_layer.to_k if original_layer is not None else None,
        )
        self.lora1_v = OrthogonalLoRALinearLayer(
            in_features,
            hidden_size,
            rank,
            sig_type,
            original_layer.to_v if original_layer is not None else None,
        )
        self.lora1_out = OrthogonalLoRALinearLayer(
            hidden_size,
            hidden_size,
            rank,
            sig_type,
            original_layer.to_out[0] if original_layer is not None else None,
        )

        self.lora2_q = StandardLoRALinearLayer(hidden_size, hidden_size, rank)
        self.lora2_k = StandardLoRALinearLayer(in_features, hidden_size, rank)
        self.lora2_v = StandardLoRALinearLayer(in_features, hidden_size, rank)
        self.lora2_out = StandardLoRALinearLayer(hidden_size, hidden_size, rank)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        sigma_mask=None,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = (
            attn.to_q(hidden_states)
            + self.lora1_q(hidden_states, sigma_mask)
            + self.lora2_scale * self.lora2_q(hidden_states)
        )

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = (
            attn.to_k(encoder_hidden_states)
            + self.lora1_k(encoder_hidden_states, sigma_mask)
            + self.lora2_scale * self.lora2_k(encoder_hidden_states)
        )

        value = (
            attn.to_v(encoder_hidden_states)
            + self.lora1_v(encoder_hidden_states, sigma_mask)
            + self.lora2_scale * self.lora2_v(encoder_hidden_states)
        )

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size,
            -1,
            attn.heads * head_dim,
        )
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = (
            attn.to_out[0](hidden_states)
            + self.lora1_out(hidden_states, sigma_mask)
            + self.lora2_scale * self.lora2_out(hidden_states)
        )
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


_TEXT_ENCODER_SIGMA_STATE = {"mask": None}


def set_text_encoder_sigma_mask(mask):
    _TEXT_ENCODER_SIGMA_STATE["mask"] = mask


def clear_text_encoder_sigma_mask():
    _TEXT_ENCODER_SIGMA_STATE["mask"] = None


class DualLoRATextLinearLayer(nn.Module):
    """Dual LoRA wrapper for CLIP text encoder linear projections."""

    def __init__(self, base_layer, rank=4, lora_alpha=32, sig_type="last"):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)

        self.lora2_scale = lora_alpha / rank
        self.lora1 = OrthogonalLoRALinearLayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            sig_type=sig_type,
            original_layer=base_layer,
        )
        self.lora2 = StandardLoRALinearLayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
        )

    def forward(self, hidden_states):
        sigma_mask = None
        if _TEXT_ENCODER_SIGMA_STATE["mask"] is not None:
            sigma_mask = _TEXT_ENCODER_SIGMA_STATE["mask"].to(
                device=hidden_states.device,
                dtype=self.lora1.lambda_layer.dtype,
            )

        return (
            self.base_layer(hidden_states)
            + self.lora1(hidden_states, sigma_mask)
            + self.lora2_scale * self.lora2(hidden_states)
        )


def compute_orthogonal_lora_weight_delta(lora_layer, mask=None):
    """Compute effective weight delta equivalent to OrthogonalLoRALinearLayer forward()."""
    with torch.no_grad():
        dtype = lora_layer.q_layer.weight.dtype
        device = lora_layer.q_layer.weight.device

        if mask is None:
            mask = torch.ones((1, lora_layer.rank), device=device, dtype=dtype)
        else:
            mask = mask.to(device=device, dtype=dtype)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)

        scale = lora_layer.lambda_layer.to(device=device, dtype=dtype) * mask
        base_scale = lora_layer.base_lambda.to(device=device, dtype=dtype) * mask

        p = lora_layer.p_layer.weight.to(device=device, dtype=dtype)
        q = lora_layer.q_layer.weight.to(device=device, dtype=dtype)
        base_p = lora_layer.base_p.weight.to(device=device, dtype=dtype)
        base_q = lora_layer.base_q.weight.to(device=device, dtype=dtype)

        return (p * scale).matmul(q) - (base_p * base_scale).matmul(base_q)


def compute_standard_lora_weight_delta(lora_layer):
    """Compute effective weight delta for StandardLoRALinearLayer (up.weight @ down.weight)."""
    with torch.no_grad():
        dtype = lora_layer.down.weight.dtype
        device = lora_layer.down.weight.device
        down = lora_layer.down.weight.to(device=device, dtype=dtype)
        up = lora_layer.up.weight.to(device=device, dtype=dtype)
        return up @ down


def get_mask_by_timestep(timestep, max_timestep, max_rank, min_rank=1, alpha=1.0):
    r = int(((max_timestep - timestep) / max_timestep) ** alpha * (max_rank - min_rank)) + min_rank
    sigma_mask = torch.zeros((1, max_rank))
    sigma_mask[:, :r] = 1.0
    return sigma_mask


def get_layer_by_name(module, layer_path):
    return module.get_submodule(layer_path)


def set_layer_by_name(module, layer_path, new_layer):
    parts = layer_path.split(".")
    if len(parts) == 1:
        setattr(module, parts[0], new_layer)
        return
    parent = module.get_submodule(".".join(parts[:-1]))
    setattr(parent, parts[-1], new_layer)


def setup_text_encoder_dual_lora(text_encoder, rank, lora_alpha, sig_type):
    target_suffixes = ("q_proj", "k_proj", "v_proj", "out_proj")
    target_layer_paths = []

    for name, layer in text_encoder.named_modules():
        if name.endswith(target_suffixes) and isinstance(layer, nn.Linear):
            target_layer_paths.append(name)

    lora1_params = []
    lora2_params = []

    for layer_path in target_layer_paths:
        base_layer = get_layer_by_name(text_encoder, layer_path)
        dual_layer = DualLoRATextLinearLayer(
            base_layer=base_layer,
            rank=rank,
            lora_alpha=lora_alpha,
            sig_type=sig_type,
        )
        set_layer_by_name(text_encoder, layer_path, dual_layer)

        for p in dual_layer.lora1.parameters():
            if p.requires_grad:
                lora1_params.append(p)
        for p in dual_layer.lora2.parameters():
            if p.requires_grad:
                lora2_params.append(p)

    return target_layer_paths, lora1_params, lora2_params


def collect_text_encoder_lora_state_dict(text_encoder):
    state_dict = {}
    for name, layer in text_encoder.named_modules():
        if isinstance(layer, DualLoRATextLinearLayer):
            for key, value in layer.state_dict().items():
                if key.startswith("lora1.") or key.startswith("lora2."):
                    state_dict[f"{name}.{key}"] = value
    return state_dict


def load_text_encoder_dual_lora_weights(
    text_encoder,
    weights_path,
    rank,
    lora_alpha,
    sig_type,
):
    if not os.path.exists(weights_path):
        return 0

    target_layer_paths, _, _ = setup_text_encoder_dual_lora(
        text_encoder,
        rank=rank,
        lora_alpha=lora_alpha,
        sig_type=sig_type,
    )

    state_dict = torch.load(weights_path, map_location="cpu")
    loaded = 0

    for layer_path in target_layer_paths:
        layer = get_layer_by_name(text_encoder, layer_path)
        prefix = f"{layer_path}."
        layer_state = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not layer_state:
            continue

        layer.load_state_dict(layer_state, strict=False)
        loaded += 1

    text_param = next(text_encoder.parameters())
    for module in text_encoder.modules():
        if isinstance(module, DualLoRATextLinearLayer):
            module.to(device=text_param.device, dtype=text_param.dtype)
            module.requires_grad_(False)

    return loaded


def build_dual_lora_attn_processors(unet, rank=4, lora_alpha=32, sig_type="last"):
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            continue

        original_layer = get_layer_by_name(unet, name.split(".processor")[0])

        lora_attn_procs[name] = DualLoRACrossAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
            lora_alpha=lora_alpha,
            sig_type=sig_type,
            original_layer=original_layer,
        )

    return lora_attn_procs


def collect_dual_lora_attn_state_dict(unet):
    lora_state_dict = {}
    for name, proc in unet.attn_processors.items():
        if isinstance(proc, DualLoRACrossAttnProcessor):
            for key, value in proc.state_dict().items():
                lora_state_dict[f"{name}.{key}"] = value
    return lora_state_dict


def load_dual_lora_attn_state_dict(unet, state_dict, strict=True):
    loaded = 0
    for name, proc in unet.attn_processors.items():
        if not isinstance(proc, DualLoRACrossAttnProcessor):
            continue

        prefix = f"{name}."
        proc_state = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not proc_state:
            continue

        proc.load_state_dict(proc_state, strict=strict)
        loaded += 1

    return loaded


def freeze_dual_lora_attn_processors(unet):
    unet_param = next(unet.parameters())
    for proc in unet.attn_processors.values():
        if isinstance(proc, DualLoRACrossAttnProcessor):
            proc.to(device=unet_param.device, dtype=unet_param.dtype)
            proc.requires_grad_(False)


def attach_tlora_sigma_mask_hook(unet, rank, min_rank, alpha_rank_scale, max_timestep):
    if getattr(unet, "tlora_sigma_hook_attached", False):
        return

    original_forward = unet.forward

    def forward_with_sigma_mask(*args, **kwargs):
        sample = args[0] if len(args) > 0 else kwargs.get("sample")
        timestep = args[1] if len(args) > 1 else kwargs.get("timestep")

        cross_attention_kwargs = kwargs.get("cross_attention_kwargs")
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        else:
            cross_attention_kwargs = dict(cross_attention_kwargs)

        if "sigma_mask" not in cross_attention_kwargs and timestep is not None:
            if torch.is_tensor(timestep):
                timestep_value = float(timestep.flatten()[0].item())
            else:
                timestep_value = float(timestep)

            sigma_mask = get_mask_by_timestep(
                timestep=timestep_value,
                max_timestep=max_timestep,
                max_rank=rank,
                min_rank=min_rank,
                alpha=alpha_rank_scale,
            )

            mask_device = sample.device if sample is not None else unet.device
            cross_attention_kwargs["sigma_mask"] = sigma_mask.to(mask_device)

        kwargs["cross_attention_kwargs"] = cross_attention_kwargs
        return original_forward(*args, **kwargs)

    unet.forward = forward_with_sigma_mask
    unet.tlora_sigma_hook_attached = True


def set_text_encoder_sigma_for_generation(pipeline, num_inference_steps):
    config = getattr(pipeline, "tlora_text_encoder_config", None)
    if config is None:
        return False

    rank = int(config["rank"])
    min_rank = int(config["min_rank"])
    alpha_rank_scale = float(config["alpha_rank_scale"])
    max_timestep = int(config["max_timestep"])

    try:
        device = next(pipeline.unet.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        device = "cpu"

    timestep_value = float(max_timestep - 1)
    try:
        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipeline.scheduler.timesteps
        if len(timesteps) > 0:
            first_t = timesteps[0]
            timestep_value = float(first_t.item()) if torch.is_tensor(first_t) else float(first_t)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass

    sigma_mask = get_mask_by_timestep(
        timestep=timestep_value,
        max_timestep=max_timestep,
        max_rank=rank,
        min_rank=min_rank,
        alpha=alpha_rank_scale,
    ).to(device)

    set_text_encoder_sigma_mask(sigma_mask)
    return True
