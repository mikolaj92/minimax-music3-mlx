"""Fuse per-codebook AR hidden states onto the Flow-VAE latent timeline."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import Music3Config


def _conv1d_ncl(x_ncl: mx.array, conv: nn.Conv1d) -> mx.array:
    # MLX Conv1d wants NLC.
    return conv(x_ncl.transpose(0, 2, 1)).transpose(0, 2, 1)


def nearest_interpolate_1d(x_ncl: mx.array, size: int) -> mx.array:
    length = x_ncl.shape[-1]
    if size == length:
        return x_ncl
    idx = (mx.arange(size) * (length / size)).astype(mx.int32)
    idx = mx.clip(idx, 0, length - 1)
    return x_ncl[:, :, idx]


def latent_length_from_frames(num_frames: int, config: Music3Config) -> int:
    return max(
        1,
        int(
            num_frames
            * config.output_sampling_rate
            / config.input_sampling_rate
            * config.input_hop_length
            / config.output_hop_length
        ),
    )


class ConditionEncoder(nn.Module):
    def __init__(self, config: Music3Config):
        super().__init__()
        self.config = config
        self.layer_weight_logits = mx.zeros((config.num_condition_layers,))
        self.layer_scale = mx.ones((1,))
        self.proj = nn.Conv1d(config.hidden_size, config.condition_out_dim, kernel_size=3, padding=1)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """``hidden_states``: ``[B, frames, num_layers * hidden]`` → ``[B, latent_len, out_dim]``."""
        batch, num_frames, _ = hidden_states.shape
        layers = self.config.num_condition_layers
        hidden = hidden_states.reshape(batch, num_frames, layers, self.config.hidden_size)
        hidden = hidden.transpose(0, 2, 3, 1)  # B, L, H, F
        weights = mx.softmax(self.layer_weight_logits.astype(mx.float32), axis=0).astype(hidden.dtype)
        hidden = (hidden * weights.reshape(1, layers, 1, 1)).sum(axis=1)  # B, H, F
        hidden = self.layer_scale.astype(hidden.dtype) * hidden
        hidden = _conv1d_ncl(hidden, self.proj)
        target = latent_length_from_frames(num_frames, self.config)
        hidden = nearest_interpolate_1d(hidden, target)
        return hidden.transpose(0, 2, 1)


def fuse_frame_hiddens(global_hidden: mx.array, depth_hiddens: mx.array) -> mx.array:
    """Concat last global hidden with residual-step hiddens: ``[1, H]`` + ``[1, 7H]`` → ``[1, 8H]``."""
    return mx.concatenate([global_hidden, depth_hiddens], axis=-1)
