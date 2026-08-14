"""Random tiny Music 3 stack for tests and the no-weights generate path."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models.qwen3 import Model as Qwen3Model
from mlx_lm.models.qwen3 import ModelArgs

from .config import Music3Config
from .depth import RVQDepthDecoder
from .dit import FlowMatchingTransformer
from .fusion import ConditionEncoder
from .vocoder import Vocoder


def make_qwen3(config: Music3Config) -> Qwen3Model:
    args = ModelArgs(
        model_type="qwen3",
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        rms_norm_eps=config.rms_norm_eps,
        vocab_size=config.vocab_size,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=config.rope_theta,
        head_dim=config.head_dim,
        tie_word_embeddings=config.tie_word_embeddings,
    )
    return Qwen3Model(args)


@dataclass
class Music3Modules:
    config: Music3Config
    language_model: Qwen3Model
    depth_decoder: RVQDepthDecoder
    condition_encoder: ConditionEncoder
    transformer: FlowMatchingTransformer
    vocoder: Vocoder
    tokenizer_dir: str | None = None


def make_tiny_modules(seed: int = 0) -> Music3Modules:
    mx.random.seed(seed)
    config = Music3Config.tiny()
    return Music3Modules(
        config=config,
        language_model=make_qwen3(config),
        depth_decoder=RVQDepthDecoder(config),
        condition_encoder=ConditionEncoder(config),
        transformer=FlowMatchingTransformer(config),
        vocoder=Vocoder(config),
    )


def encode_official_text_pair(text: str, config: Music3Config, tokenizer_dir: str) -> mx.array:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    input_ids = tokenizer(text, return_tensors="np")["input_ids"]
    cond = mx.array(input_ids.astype("int32"))
    uncond = mx.array(input_ids.astype("int32"))
    if uncond.shape[1] > 3:
        mid = mx.full((1, uncond.shape[1] - 3), config.audio_cfg_token_id, dtype=mx.int32)
        uncond = mx.concatenate([uncond[:, :1], mid, uncond[:, -2:]], axis=1)
    return mx.concatenate([cond, uncond], axis=0)


def encode_text_pair(text: str, config: Music3Config, max_len: int = 32) -> mx.array:
    """Deterministic tiny tokenizer + official CFG clone of the prompt."""
    # Keep ids away from the audio-code range so the first AR step is well-defined.
    ids = [((ord(ch) * 17 + i * 3) % 180) + 1 for i, ch in enumerate(text[: max_len - 3])]
    if not ids:
        ids = [1]
    # [bos-like, tokens..., pad, audio_start-like]
    ids = [1] + ids + [2]
    cond = mx.array(ids, dtype=mx.int32)[None, :]
    uncond = mx.array(ids, dtype=mx.int32)[None, :]
    # Official recipe: every token except first and last two becomes the audio-CFG token.
    if uncond.shape[1] > 3:
        mid = mx.full((1, uncond.shape[1] - 3), config.audio_cfg_token_id, dtype=mx.int32)
        uncond = mx.concatenate([uncond[:, :1], mid, uncond[:, -2:]], axis=1)
    return mx.concatenate([cond, uncond], axis=0)
