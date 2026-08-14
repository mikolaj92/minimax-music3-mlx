"""Tests for the shipped vocoder decode."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from minimax_music3_mlx.tiny import make_tiny_modules


def test_vocoder_decode_is_stereo_and_not_silent():
    modules = make_tiny_modules(seed=6)
    config = modules.config
    length = 4
    latents = mx.random.normal((1, config.dit_in_channels, length))
    audio = modules.vocoder(latents)
    mx.eval(audio)
    wave = np.asarray(audio.astype(mx.float32))
    assert wave.shape[0] == 1
    assert wave.shape[1] == 2
    assert wave.shape[2] > 0
    assert np.isfinite(wave).all()
    assert float(np.max(np.abs(wave))) > 0.0
