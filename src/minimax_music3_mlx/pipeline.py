"""Full lyrics + caption → stereo waveform on MLX."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from .ar import generate_frame_hiddens
from .config import (
    CHUNK_FRAMES,
    CHUNK_HOP,
    CROP_LEFT_LATENT,
    CROP_RIGHT_LATENT,
    DIT_CFG_SCALE,
    LATENT_HOP_LENGTH,
    OVERLAP_LATENT_LENGTH,
    Music3Config,
)
from .euler import denoise_chunk
from .prompt import assemble_prompt
from .tiny import Music3Modules, encode_official_text_pair, encode_text_pair, make_tiny_modules


def chunk_starts(num_frames: int) -> list[int]:
    if num_frames <= CHUNK_FRAMES:
        return [0]
    return list(range(0, num_frames - CHUNK_HOP, CHUNK_HOP))


def _crop_waveform(wave: mx.array, chunk_index: int, num_chunks: int) -> mx.array:
    """Official vocoder stitch: drop 86 latent frames on the left and 258 on the right."""
    hop = LATENT_HOP_LENGTH
    left = 0 if chunk_index == 0 else CROP_LEFT_LATENT * hop
    right = 0 if chunk_index == num_chunks - 1 else CROP_RIGHT_LATENT * hop
    end = int(wave.shape[-1]) - right
    if end <= left:
        return wave
    return wave[..., left:end]


def _run_flow(
    modules: Music3Modules,
    frame_hiddens: mx.array,
    num_inference_steps: int,
    seed: int,
) -> mx.array:
    """Denoise overlapping 200-frame windows and return a stitched stereo waveform ``[B, 2, S]``."""
    config = modules.config
    starts = chunk_starts(frame_hiddens.shape[1])
    waves: list[mx.array] = []
    previous_latent = None
    previous_condition = None
    mx.random.seed(seed + 7)
    for start in starts:
        end = min(start + CHUNK_FRAMES, frame_hiddens.shape[1])
        condition = modules.condition_encoder(frame_hiddens[:, start:end])
        length = condition.shape[1]
        noise = mx.random.normal((1, config.dit_in_channels, length)).astype(condition.dtype)
        latents, condition = denoise_chunk(
            modules.transformer,
            noise,
            condition,
            num_inference_steps=num_inference_steps,
            guidance_scale=DIT_CFG_SCALE,
            previous_latent=previous_latent,
            previous_condition=previous_condition,
        )
        carry_start = max(0, int(latents.shape[-1]) - 2 * OVERLAP_LATENT_LENGTH)
        carry_end = max(carry_start, int(latents.shape[-1]) - OVERLAP_LATENT_LENGTH)
        previous_latent = latents[..., carry_start:carry_end]
        previous_condition = condition[:, carry_start:carry_end]
        waves.append(modules.vocoder(latents))
    cropped = [_crop_waveform(wave, index, len(waves)) for index, wave in enumerate(waves)]
    return mx.concatenate(cropped, axis=-1)


def generate_audio(
    modules: Music3Modules,
    lyrics: str,
    prompt: str,
    audio_duration: float = 0.2,
    num_inference_steps: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """Return ``(waveform [samples, 2], sample_rate)``."""
    if audio_duration <= 0:
        raise ValueError("audio_duration must be positive")
    config = modules.config
    text = assemble_prompt(prompt, lyrics)
    if modules.tokenizer_dir:
        text_ids = encode_official_text_pair(text, config, modules.tokenizer_dir)
    else:
        text_ids = encode_text_pair(text, config)
    max_frames = max(1, int(audio_duration * config.frame_rate))
    print(f"AR decode ({max_frames} frames)...", flush=True)
    frame_hiddens = generate_frame_hiddens(
        modules.language_model,
        modules.depth_decoder,
        config,
        text_ids,
        max_frames=max_frames,
        seed=seed,
    )
    mx.eval(frame_hiddens)
    print(f"flow-matching DiT ({num_inference_steps} steps, {len(chunk_starts(frame_hiddens.shape[1]))} windows)...", flush=True)
    audio = _run_flow(modules, frame_hiddens, num_inference_steps, seed)
    mx.eval(audio)
    wave = np.asarray(audio[0].astype(mx.float32))  # [2, samples]
    wave = np.clip(wave, -1.0, 1.0).T  # [samples, 2]
    if not np.isfinite(wave).all():
        raise RuntimeError("vocoder produced non-finite samples")
    return wave, config.sampling_rate


def load_modules(weights_dir: str | Path | None, tiny: bool = False, seed: int = 0) -> Music3Modules:
    if tiny or weights_dir is None:
        return make_tiny_modules(seed=seed)
    path = Path(weights_dir)
    if not path.exists():
        raise FileNotFoundError(path)
    # Full-weight load is implemented via the same module classes + remapped safetensors.
    from .load_weights import load_converted_modules

    return load_converted_modules(path)
