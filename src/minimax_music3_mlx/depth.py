"""Local RVQ depth decoder (0.6B-class residual codebook model)."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import Music3Config


def _rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    return mx.fast.rms_norm(x, weight, eps)


def _causal_sdpa(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    # q,k,v: [B, H, L, D]
    scale = 1.0 / math.sqrt(q.shape[-1])
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")


class DepthAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        b, s, _ = x.shape
        q = self.to_q(x).reshape(b, s, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.to_k(x).reshape(b, s, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.to_v(x).reshape(b, s, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        y = _causal_sdpa(q, k, v).transpose(0, 2, 1, 3).reshape(b, s, -1)
        return self.to_out(y)


class DepthBlock(nn.Module):
    def __init__(self, dim: int, heads: int, intermediate: int, eps: float):
        super().__init__()
        self.eps = eps
        self.input_layernorm = nn.RMSNorm(dim, eps=eps)
        self.attn = DepthAttention(dim, heads)
        self.post_attention_layernorm = nn.RMSNorm(dim, eps=eps)
        self.gate_proj = nn.Linear(dim, intermediate, bias=False)
        self.up_proj = nn.Linear(dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.input_layernorm(x))
        n = self.post_attention_layernorm(x)
        return x + self.down_proj(nn.silu(self.gate_proj(n)) * self.up_proj(n))


class RVQDepthDecoder(nn.Module):
    def __init__(self, config: Music3Config):
        super().__init__()
        self.config = config
        residual = config.residual_codebooks
        self.audio_embeddings = nn.Embedding(config.audio_vocab_size * residual, config.hidden_size)
        self.projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.pos_embedding = nn.Embedding(config.depth_max_position_embeddings, config.hidden_size)
        self.layers = [
            DepthBlock(
                config.hidden_size,
                config.depth_num_heads,
                config.depth_intermediate_size,
                config.rms_norm_eps,
            )
            for _ in range(config.depth_num_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.audio_heads = [
            nn.Linear(config.hidden_size, config.audio_vocab_size, bias=False) for _ in range(residual)
        ]

    def __call__(self, inputs_embeds: mx.array) -> mx.array:
        positions = mx.arange(inputs_embeds.shape[1])
        hidden = inputs_embeds + self.pos_embedding(positions)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)
