"""Qwen3 global AR: one-frame decode returning hidden states + semantic token."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen3 import Model as Qwen3Model

from .config import AR_CFG_SCALE, AR_CFG_TOP_K, AR_SAMPLING_TOP_K, Music3Config
from .depth import RVQDepthDecoder
from .fusion import fuse_frame_hiddens
from .sampling import sample_top_k


@dataclass
class ARFrame:
    semantic_code: mx.array  # [2]
    residual_codes: mx.array  # [2, 7]
    frame_hidden: mx.array  # [1, num_codebooks * H]
    last_hidden: mx.array  # [2, H] after this frame (for next step)
    cache: list
    ended: bool


def qwen3_hidden(
    language_model: Qwen3Model,
    input_embeddings: mx.array,
    cache=None,
) -> tuple[mx.array, list]:
    """Forward the mlx-lm Qwen3 backbone and return last-layer hidden states."""
    if cache is None:
        cache = make_prompt_cache(language_model)
    hidden = language_model.model(
        mx.zeros(input_embeddings.shape[:2], dtype=mx.int32),
        cache,
        input_embeddings=input_embeddings,
    )
    return hidden, cache


def lm_logits(language_model: Qwen3Model, last_hidden: mx.array) -> mx.array:
    if language_model.args.tie_word_embeddings:
        return language_model.model.embed_tokens.as_linear(last_hidden)
    return language_model.lm_head(last_hidden)


def _embed_audio_frame(
    language_model: Qwen3Model,
    depth: RVQDepthDecoder,
    config: Music3Config,
    frame_codes: mx.array,
) -> mx.array:
    # frame_codes: [2, num_codebooks]
    embed_tokens = language_model.model.embed_tokens
    embeds = embed_tokens(frame_codes[:, :1] + config.audio_code_offset)
    offsets = mx.arange(config.residual_codebooks) * config.audio_vocab_size
    extra = depth.audio_embeddings(frame_codes[:, 1:] + offsets[None, :]).sum(axis=1, keepdims=True)
    embeds = embeds + extra.astype(embeds.dtype)
    return embeds * (config.num_codebooks**-0.5)


def generate_depth_codes(
    language_model: Qwen3Model,
    depth: RVQDepthDecoder,
    config: Music3Config,
    last_hidden: mx.array,
    semantic_code: mx.array,
    rng_key: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Autoregressively sample residual codes c1..c7 and collect their hidden states."""
    sequence = [depth.projection(last_hidden)[:, None, :]]
    code_embed = language_model.model.embed_tokens(semantic_code + config.audio_code_offset)
    sequence.append(depth.projection(code_embed)[:, None, :])
    codes = [semantic_code]
    hidden_parts = []
    key = rng_key
    for index in range(1, config.num_codebooks):
        hidden = depth(mx.concatenate(sequence, axis=1))[:, -1]
        hidden_parts.append(hidden[:1])
        logits = depth.audio_heads[index - 1](hidden)
        conditional, unconditional = logits[:1].astype(mx.float32), logits[1:2].astype(mx.float32)
        guided = unconditional + (conditional - unconditional) * AR_CFG_SCALE
        sampled, key = sample_top_k(guided, key, AR_SAMPLING_TOP_K)
        code = mx.concatenate([sampled, sampled], axis=0)
        codes.append(code)
        if index < config.num_codebooks - 1:
            embed = depth.audio_embeddings(code + (index - 1) * config.audio_vocab_size)
            sequence.append(depth.projection(embed)[:, None, :])
    stacked = mx.stack(codes, axis=1)
    return stacked, mx.concatenate(hidden_parts, axis=-1), key


def ar_one_frame(
    language_model: Qwen3Model,
    depth: RVQDepthDecoder,
    config: Music3Config,
    last_hidden: mx.array,
    cache: list,
    rng_key: mx.array,
    emit_frame: bool = True,
) -> ARFrame:
    """One global-LM step: sample the semantic code, fill residual RVQ, feed the frame back.

    ``last_hidden`` is ``[2, H]`` (conditional / unconditional CFG pair).
    """
    logits = lm_logits(language_model, last_hidden).astype(mx.float32)
    vocab = logits.shape[-1]
    mask = mx.ones((vocab,), dtype=mx.bool_)
    start = config.audio_code_offset
    end = start + config.semantic_vocab_size
    allow = mx.logical_or(
        mx.logical_and(mx.arange(vocab) >= start, mx.arange(vocab) < end),
        mx.arange(vocab) == config.audio_end_token_id,
    )
    logits = mx.where(allow[None, :], logits, mx.array(-1e9, dtype=mx.float32))
    conditional, unconditional = logits[0:1], logits[1:2]
    guided = unconditional + (conditional - unconditional) * AR_CFG_SCALE
    k = min(AR_CFG_TOP_K, conditional.shape[-1])
    # mx.topk is unsorted; min of the k largest is the official k-th threshold.
    threshold = mx.min(mx.topk(conditional, k, axis=-1), axis=-1, keepdims=True)
    guided = mx.where(conditional < threshold, mx.array(-1e9, dtype=mx.float32), guided)
    guided = mx.where(allow[None, :], guided, mx.array(-1e9, dtype=mx.float32))
    sampled, key = sample_top_k(guided, rng_key, AR_SAMPLING_TOP_K)
    token = int(sampled.item())
    if token == config.audio_end_token_id:
        return ARFrame(
            semantic_code=sampled,
            residual_codes=mx.zeros((2, config.residual_codebooks), dtype=mx.int32),
            frame_hidden=mx.zeros((1, config.num_codebooks * config.hidden_size)),
            last_hidden=last_hidden,
            cache=cache,
            ended=True,
        )

    semantic_code = mx.concatenate([sampled, sampled], axis=0) - config.audio_code_offset
    frame_codes, depth_hidden, key = generate_depth_codes(
        language_model, depth, config, last_hidden, semantic_code, key
    )
    frame_hidden = (
        fuse_frame_hiddens(last_hidden[:1], depth_hidden) if emit_frame else mx.zeros((1, config.num_codebooks * config.hidden_size))
    )
    feedback = _embed_audio_frame(language_model, depth, config, frame_codes)
    hidden, cache = qwen3_hidden(language_model, feedback, cache)
    return ARFrame(
        semantic_code=semantic_code,
        residual_codes=frame_codes[:, 1:],
        frame_hidden=frame_hidden,
        last_hidden=hidden[:, -1],
        cache=cache,
        ended=False,
    )


def generate_frame_hiddens(
    language_model: Qwen3Model,
    depth: RVQDepthDecoder,
    config: Music3Config,
    text_ids: mx.array,
    max_frames: int,
    seed: int = 0,
) -> mx.array:
    """Run the official AR loop and return ``[1, frames, num_codebooks * H]``."""
    mx.random.seed(seed)
    key = mx.random.key(seed)
    embeds = language_model.model.embed_tokens(text_ids)
    hidden, cache = qwen3_hidden(language_model, embeds, cache=None)
    last_hidden = hidden[:, -1]
    frames: list[mx.array] = []
    # First decode only consumes <|audio_start|> and is not an emitted frame.
    for frame_index in range(max_frames + 1):
        key, sub = mx.random.split(key)
        result = ar_one_frame(
            language_model,
            depth,
            config,
            last_hidden,
            cache,
            sub,
            emit_frame=frame_index > 0,
        )
        last_hidden, cache = result.last_hidden, result.cache
        if result.ended:
            break
        if frame_index > 0:
            frames.append(result.frame_hidden)
            if frame_index == 1 or frame_index % 10 == 0 or len(frames) >= max_frames:
                print(f"  AR frame {len(frames)}/{max_frames}", flush=True)
            if len(frames) >= max_frames:
                break
    if not frames:
        raise ValueError("MiniMax Music 3 generated zero audio frames")
    return mx.stack(frames, axis=1)
