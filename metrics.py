import math
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from tlora_module import clear_text_encoder_sigma_mask, set_text_encoder_sigma_mask


def _sdp_math_only_context():
    """Force non-flash SDPA kernels for higher-order gradients when available."""
    if not torch.cuda.is_available():
        return nullcontext()

    sdp_kernel = getattr(torch.backends.cuda, "sdp_kernel", None)
    if sdp_kernel is None:
        return nullcontext()

    return sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)


def evaluate_phase1_robustness(
    unet,
    text_encoder,
    text_encoder_2,
    vae,
    noise_scheduler,
    accelerator,
    cp_dataloader,
    encode_batch_fn,
    resolution,
    rank,
    min_rank,
    alpha_rank_scale,
    max_timestep,
    eval_batches=1,
    hessian_power_iters=2,
    perturb_magnitudes=None,
):
    """Evaluate robustness proxies at end of Phase 1.

    Metrics:
      - ScS_c: ||∇_c ||eps_theta(z_t, c)||^2||
      - Curvature proxy: top Hessian eigenvalue estimate via power iteration
      - Perturbation recovery: cosine similarity of noise prediction under
        parameter perturbations (trainable LoRA params only)
    """
    if perturb_magnitudes is None:
        perturb_magnitudes = [1e-4, 1e-3, 1e-2]

    eval_batches = max(1, int(eval_batches))
    hessian_power_iters = max(1, int(hessian_power_iters))
    perturb_magnitudes = [float(m) for m in perturb_magnitudes]

    sensitivity_vals = []
    curvature_vals = []
    perturb_vals = {m: [] for m in perturb_magnitudes}

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    trainable_params += [p for p in text_encoder.parameters() if p.requires_grad]
    trainable_params += [p for p in text_encoder_2.parameters() if p.requires_grad]

    for _ in range(eval_batches):
        batch = next(cp_dataloader)
        input_ids_1 = batch["input_ids"].to(device=next(text_encoder.parameters()).device)
        input_ids_2 = batch["input_ids_2"].to(device=next(text_encoder_2.parameters()).device)

        (
            noisy_latents,
            timesteps,
            _noise,
            prompt_embeds,
            pooled_prompt_embeds,
            add_time_ids,
            sigma_mask,
        ) = encode_batch_fn(
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
        )

        c = prompt_embeds.detach().clone().requires_grad_(True)

        with _sdp_math_only_context():
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=c,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds.detach(),
                    "time_ids": add_time_ids,
                },
                cross_attention_kwargs={"sigma_mask": sigma_mask},
            ).sample

            objective = noise_pred.float().pow(2).mean()
            grad_c = torch.autograd.grad(
                objective,
                c,
                create_graph=True,
                retain_graph=True,
            )[0]

            sensitivity = grad_c.norm() / math.sqrt(c.numel())
            sensitivity_vals.append(sensitivity.detach())

            v = torch.randn_like(c)
            v = v / (v.norm() + 1e-12)
            hvp = None
            for i in range(hessian_power_iters):
                gv = torch.sum(grad_c * v)
                hvp = torch.autograd.grad(
                    gv,
                    c,
                    retain_graph=i < (hessian_power_iters - 1),
                    create_graph=False,
                )[0]
                v = hvp / (hvp.norm() + 1e-12)

            top_eigen = torch.sum(v * hvp)

        curvature_vals.append(top_eigen.detach())

        with torch.no_grad():
            base_flat = noise_pred.detach().float().reshape(noise_pred.shape[0], -1)

        for magnitude in perturb_magnitudes:
            perturb_noises = []
            with torch.no_grad():
                for p in trainable_params:
                    noise_delta = torch.randn_like(p) * magnitude
                    p.add_(noise_delta)
                    perturb_noises.append(noise_delta)

                set_text_encoder_sigma_mask(sigma_mask)
                pert_prompt_output_1 = text_encoder(
                    input_ids_1,
                    output_hidden_states=True,
                )
                pert_prompt_1 = pert_prompt_output_1.hidden_states[-2]

                pert_prompt_output_2 = text_encoder_2(
                    input_ids_2,
                    output_hidden_states=True,
                )
                pert_pooled = pert_prompt_output_2.text_embeds
                pert_prompt_2 = pert_prompt_output_2.hidden_states[-2]
                clear_text_encoder_sigma_mask()

                pert_prompt = torch.cat([pert_prompt_1, pert_prompt_2], dim=-1)
                pert_prompt = pert_prompt.to(device=noisy_latents.device)
                pert_pooled = pert_pooled.to(device=noisy_latents.device)

                perturbed_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=pert_prompt,
                    added_cond_kwargs={
                        "text_embeds": pert_pooled,
                        "time_ids": add_time_ids,
                    },
                    cross_attention_kwargs={"sigma_mask": sigma_mask},
                ).sample

                pert_flat = perturbed_pred.float().reshape(perturbed_pred.shape[0], -1)
                similarity = F.cosine_similarity(base_flat, pert_flat, dim=1).mean()
                perturb_vals[magnitude].append(similarity.detach())

                for p, noise_delta in zip(trainable_params, perturb_noises):
                    p.sub_(noise_delta)

        del noise_pred, objective, grad_c, c

    local_sensitivity = torch.stack(sensitivity_vals).mean().unsqueeze(0)
    local_curvature = torch.stack(curvature_vals).mean().unsqueeze(0)

    gathered_sensitivity = accelerator.gather(local_sensitivity)
    gathered_curvature = accelerator.gather(local_curvature)

    results = {
        "scs_c": gathered_sensitivity.mean().item(),
        "hessian_top_eig": gathered_curvature.mean().item(),
        "perturbation_curve": {},
        "perturbation_auc": None,
    }

    perturb_curve = {}
    for magnitude in perturb_magnitudes:
        local_val = torch.stack(perturb_vals[magnitude]).mean().unsqueeze(0)
        gathered_val = accelerator.gather(local_val)
        perturb_curve[magnitude] = gathered_val.mean().item()

    if len(perturb_curve) > 0:
        sorted_mags = sorted(perturb_curve.keys())
        sorted_scores = [perturb_curve[m] for m in sorted_mags]
        if len(sorted_mags) > 1:
            x = np.log10(np.array(sorted_mags, dtype=np.float64))
            y = np.array(sorted_scores, dtype=np.float64)
            denom = max(float(x[-1] - x[0]), 1e-12)
            auc = float(np.trapz(y, x) / denom)
        else:
            auc = float(sorted_scores[0])
        results["perturbation_curve"] = perturb_curve
        results["perturbation_auc"] = auc

    return results
