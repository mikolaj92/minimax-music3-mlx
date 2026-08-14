"""Top-k sampling used by the official Music 3 AR recipe."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .config import AR_SAMPLING_TOP_K


def sample_top_k(logits: mx.array, key: mx.array, top_k: int = AR_SAMPLING_TOP_K) -> tuple[mx.array, mx.array]:
    """Sample one index per row. Returns ``(tokens [B], next_key)``."""
    values = mx.where(mx.isnan(logits), mx.array(-1e9, dtype=logits.dtype), logits).astype(mx.float32)
    k = min(top_k, values.shape[-1])
    # mx.topk is unsorted; the k-th largest is the min of those k values.
    # (torch.topk is descending, so official code can use values[..., -1].)
    threshold = mx.min(mx.topk(values, k, axis=-1), axis=-1, keepdims=True)
    masked = mx.where(values < threshold, mx.array(-1e9, dtype=mx.float32), values)
    # mlx.core.random.categorical is the shipped sampler.
    next_key = mx.random.split(key)[1] if key is not None else mx.random.key(0)
    tokens = mx.random.categorical(masked, axis=-1, key=key)
    return tokens, next_key


def numpy_multinomial(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """CPU fallback used only when an explicit numpy Generator is provided."""
    out = [rng.choice(probs.shape[-1], p=row / max(row.sum(), 1e-12)) for row in probs]
    return np.asarray(out, dtype=np.int32)
