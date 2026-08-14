"""Flow-matching Euler step used by the official Music 3 scheduler recipe."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .config import DIT_CFG_SCALE


def make_sigma_schedule(num_inference_steps: int) -> np.ndarray:
    """Official FlowMatchEuler schedule with ``invert_sigmas=True``.

    The pipeline passes ``linspace(1.0, 1/N, N)`` into the scheduler, which then
    does ``sigmas = 1 - sigmas`` and appends ``1.0``. Flow-matching time is
    therefore ``0`` (noise) → ``1`` (data), matching the DiT docstring.
    """
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")
    raw = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=np.float32)
    return np.concatenate([1.0 - raw, np.array([1.0], dtype=np.float32)])


def guided_velocity(cond: mx.array, uncond: mx.array, scale: float = DIT_CFG_SCALE) -> mx.array:
    return uncond + scale * (cond - uncond)


def euler_step(sample: mx.array, velocity: mx.array, sigma: float, sigma_next: float) -> mx.array:
    """Official FlowMatchEuler update: ``x + (σ_next - σ) * v``."""
    return sample + (sigma_next - sigma) * velocity


def denoise_chunk(
    transformer,
    latents: mx.array,
    condition: mx.array,
    num_inference_steps: int = 30,
    guidance_scale: float = DIT_CFG_SCALE,
    previous_latent: mx.array | None = None,
    previous_condition: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Run the Euler loop on one latent window. Unconditional condition is zeros.

    Returns ``(latents, condition)`` so the caller can carry the official overlap
    prompt into the next 200-frame window.
    """
    overlap = 0
    if previous_latent is not None and previous_condition is not None:
        overlap = min(int(previous_latent.shape[-1]), int(condition.shape[1]))
        if overlap > 0:
            condition = mx.concatenate([previous_condition[:, :overlap], condition[:, overlap:]], axis=1)
    sigmas = make_sigma_schedule(num_inference_steps)
    x = latents
    noise_prompt = x[..., :overlap] if overlap > 0 else None
    zeros = mx.zeros_like(condition)
    for i in range(num_inference_steps):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        if overlap > 0:
            # Official blend: at t=0 keep this window's noise; at t=1 lock to the previous window.
            blend = (1.0 - (1.0 - 1e-6) * sigma) * noise_prompt + sigma * previous_latent[..., :overlap]
            x = mx.concatenate([blend, x[..., overlap:]], axis=-1)
        t = mx.full((x.shape[0],), sigma, dtype=x.dtype)
        cond_v = transformer(x, t, condition)
        if guidance_scale == 1.0:
            velocity = cond_v
        else:
            uncond_v = transformer(x, t, zeros)
            velocity = guided_velocity(cond_v, uncond_v, guidance_scale)
        x = euler_step(x, velocity, sigma, sigma_next)
        mx.eval(x)
    if overlap > 0:
        x = mx.concatenate([previous_latent[..., :overlap], x[..., overlap:]], axis=-1)
        mx.eval(x)
    return x, condition
