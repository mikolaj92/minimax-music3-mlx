"""Tests that call the shipped AR-one-frame path on mlx-lm Qwen3."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from minimax_music3_mlx.ar import ar_one_frame, generate_depth_codes, qwen3_hidden
from minimax_music3_mlx.sampling import sample_top_k
from minimax_music3_mlx.tiny import encode_text_pair, make_tiny_modules


def test_sample_top_k_keeps_k_candidates_not_just_argmax():
    # Descending logits: argmax is 9. Using the last mx.topk slot as a threshold
    # would collapse to a single token because mlx.topk is unsorted.
    logits = mx.arange(10, dtype=mx.float32)[None, :]
    seen = set()
    for seed in range(80):
        token, _ = sample_top_k(logits, mx.random.key(seed), top_k=5)
        seen.add(int(token.item()))
    assert min(seen) >= 5
    assert 9 in seen
    # Argmax-only threshold would produce {9} only.
    assert seen != {9}


def test_ar_one_frame_returns_semantic_and_hidden():
    modules = make_tiny_modules(seed=1)
    config = modules.config
    text_ids = encode_text_pair("caption and lyrics", config)
    embeds = modules.language_model.model.embed_tokens(text_ids)
    hidden, cache = qwen3_hidden(modules.language_model, embeds)
    last = hidden[:, -1]
    assert last.shape == (2, config.hidden_size)

    result = ar_one_frame(
        modules.language_model,
        modules.depth_decoder,
        config,
        last,
        cache,
        mx.random.key(3),
        emit_frame=True,
    )
    if result.ended:
        pytest.skip("tiny random LM sampled EOS on this seed")
    assert result.semantic_code.shape == (2,)
    assert result.residual_codes.shape == (2, config.residual_codebooks)
    assert result.frame_hidden.shape == (1, config.num_codebooks * config.hidden_size)
    assert result.last_hidden.shape == (2, config.hidden_size)
    mx.eval(result.frame_hidden)
    assert mx.isfinite(result.frame_hidden).all()


def test_depth_decoder_fills_seven_residual_codes():
    modules = make_tiny_modules(seed=2)
    config = modules.config
    last = mx.random.normal((2, config.hidden_size))
    semantic = mx.array([3, 3], dtype=mx.int32)
    codes, depth_h, _ = generate_depth_codes(
        modules.language_model,
        modules.depth_decoder,
        config,
        last,
        semantic,
        mx.random.key(4),
    )
    assert codes.shape == (2, config.num_codebooks)
    assert int(codes[0, 0].item()) == 3
    assert depth_h.shape == (1, config.residual_codebooks * config.hidden_size)
    mx.eval(depth_h)
    assert mx.isfinite(depth_h).all()
