"""Tests for the shipped flow-matching Euler step and DiT forward."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from minimax_music3_mlx.euler import denoise_chunk, euler_step, guided_velocity, make_sigma_schedule
from minimax_music3_mlx.tiny import make_tiny_modules


def test_euler_step_matches_sigma_delta_times_velocity():
    sample = mx.ones((1, 4, 6))
    velocity = mx.full((1, 4, 6), 2.0)
    out = euler_step(sample, velocity, sigma=1.0, sigma_next=0.5)
    expected = np.ones((1, 4, 6), dtype=np.float32) + (0.5 - 1.0) * 2.0
    assert np.allclose(np.asarray(out), expected)


def test_guided_velocity_is_uncond_plus_scale_delta():
    cond = mx.array([3.0])
    uncond = mx.array([1.0])
    out = guided_velocity(cond, uncond, scale=1.7)
    assert np.allclose(np.asarray(out), np.array([1.0 + 1.7 * 2.0], dtype=np.float32))


def test_dit_euler_one_step_preserves_shape():
    modules = make_tiny_modules(seed=5)
    config = modules.config
    frames = 3
    h = mx.random.normal((1, frames, config.num_codebooks * config.hidden_size))
    condition = modules.condition_encoder(h)
    latents = mx.random.normal((1, config.dit_in_channels, condition.shape[1]))
    out, _ = denoise_chunk(modules.transformer, latents, condition, num_inference_steps=1, guidance_scale=1.7)
    assert out.shape == latents.shape
    mx.eval(out)
    assert mx.isfinite(out).all()
    # A real Euler step must move the sample.
    assert float(mx.abs(out - latents).sum()) > 0.0


def test_sigma_schedule_starts_at_zero_and_ends_at_one():
    sigmas = make_sigma_schedule(4)
    assert sigmas[0] == 0.0
    assert sigmas[-1] == 1.0
    assert np.allclose(sigmas, np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32))
