"""1D flow-matching transformer (2.4B-class in the official checkpoint)."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import Music3Config


def _apply_partial_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    # x: [B, S, H, D]; cos/sin: [S, rotary]
    rotary_dim = cos.shape[-1]
    cos = cos[:, None, :].astype(x.dtype)
    sin = sin[:, None, :].astype(x.dtype)
    rotated = x[..., :rotary_dim]
    half = rotary_dim // 2
    first, second = rotated[..., :half], rotated[..., half:]
    rotate_half = mx.concatenate([-second, first], axis=-1)
    rotated = rotated * cos + rotate_half * sin
    return mx.concatenate([rotated, x[..., rotary_dim:]], axis=-1)


class FourierEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.weight = mx.random.normal((embedding_dim // 2, 1))

    def __call__(self, timestep: mx.array) -> mx.array:
        angles = 2.0 * math.pi * timestep.reshape(-1, 1) @ self.weight.T
        return mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.linear_2 = nn.Linear(out_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(x)))


class DiTAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        # Official Diffusers stores this as ModuleList(Linear, Dropout) → to_out.0.weight
        self.to_out = [nn.Linear(inner, dim, bias=False), nn.Dropout(0.0)]

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        b, s, _ = x.shape
        q = self.to_q(x).reshape(b, s, self.heads, self.head_dim)
        k = self.to_k(x).reshape(b, s, self.heads, self.head_dim)
        v = self.to_v(x).reshape(b, s, self.heads, self.head_dim)
        q = _apply_partial_rotary(q, cos, sin)
        k = _apply_partial_rotary(k, cos, sin)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(self.head_dim)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        y = y.transpose(0, 2, 1, 3).reshape(b, s, -1)
        return self.to_out[1](self.to_out[0](y))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, ff_inner: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DiTAttention(dim, heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff_in = nn.Linear(dim, ff_inner * 2)
        self.ff_out = nn.Linear(ff_inner, dim)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x), cos, sin)
        gate_states, gate = mx.split(self.ff_in(self.norm2(x)), 2, axis=-1)
        return x + self.ff_out(gate_states * nn.silu(gate))


class FlowMatchingTransformer(nn.Module):
    def __init__(self, config: Music3Config):
        super().__init__()
        self.config = config
        inner = config.dit_num_heads * config.dit_head_dim
        concat = 2 * config.dit_in_channels + config.condition_out_dim
        self.time_proj = FourierEmbedding(config.dit_fourier_dim)
        self.time_embed = TimestepEmbedding(config.dit_fourier_dim, inner)
        self.preprocess_conv = nn.Conv1d(concat, concat, kernel_size=1, bias=False)
        self.proj_in = nn.Linear(concat, inner, bias=False)
        self.rotary_dim = config.dit_rotary_dim
        self.transformer_blocks = [
            DiTBlock(inner, config.dit_num_heads, config.dit_head_dim, config.dit_ff_inner_dim)
            for _ in range(config.dit_num_layers)
        ]
        self.proj_out = nn.Linear(inner, config.dit_in_channels, bias=False)
        self.postprocess_conv = nn.Conv1d(
            config.dit_in_channels, config.dit_in_channels, kernel_size=1, bias=False
        )

    def _rotary(self, seq_len: int) -> tuple[mx.array, mx.array]:
        dim = self.rotary_dim
        inv_freq = 1.0 / (10000.0 ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        steps = mx.arange(seq_len).astype(mx.float32)
        freqs = mx.outer(steps, inv_freq)
        freqs = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(freqs), mx.sin(freqs)

    def __call__(self, hidden_states: mx.array, timestep: mx.array, encoder_hidden_states: mx.array) -> mx.array:
        """``hidden_states`` / return: ``[B, C, T]``. ``encoder_hidden_states``: ``[B, T, cond]``."""
        zeros = mx.zeros_like(hidden_states)
        cond = encoder_hidden_states.transpose(0, 2, 1)
        x = mx.concatenate([hidden_states, zeros, cond], axis=1)
        x_nlc = x.transpose(0, 2, 1)
        x_nlc = self.preprocess_conv(x_nlc) + x_nlc
        temb = self.time_embed(self.time_proj(timestep))
        x = self.proj_in(x_nlc)
        x = mx.concatenate([temb[:, None, :], x], axis=1)
        cos, sin = self._rotary(x.shape[1])
        for block in self.transformer_blocks:
            x = block(x, cos, sin)
        x = self.proj_out(x[:, 1:])
        x = self.postprocess_conv(x) + x
        return x.transpose(0, 2, 1)
