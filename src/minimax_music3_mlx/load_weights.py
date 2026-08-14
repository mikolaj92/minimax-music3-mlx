"""Load a converted MLX weight tree into the shipped modules."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from .config import Music3Config
from .depth import RVQDepthDecoder
from .dit import FlowMatchingTransformer
from .fusion import ConditionEncoder
from .tiny import Music3Modules, make_qwen3
from .vocoder import Vocoder


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def config_from_converted(root: Path) -> Music3Config:
    lm = _read_json(root / "language_model" / "config.json")
    depth = _read_json(root / "rvq_depth_decoder" / "config.json")
    cond = _read_json(root / "condition_encoder" / "config.json")
    dit = _read_json(root / "transformer" / "config.json")
    voc = _read_json(root / "vocoder" / "config.json")
    cfg = Music3Config()
    if lm:
        cfg.hidden_size = int(lm.get("hidden_size", cfg.hidden_size))
        cfg.vocab_size = int(lm.get("vocab_size", cfg.vocab_size))
        cfg.num_hidden_layers = int(lm.get("num_hidden_layers", cfg.num_hidden_layers))
        cfg.intermediate_size = int(lm.get("intermediate_size", cfg.intermediate_size))
        cfg.num_attention_heads = int(lm.get("num_attention_heads", cfg.num_attention_heads))
        cfg.num_key_value_heads = int(lm.get("num_key_value_heads", cfg.num_key_value_heads))
        cfg.head_dim = int(lm.get("head_dim", cfg.head_dim))
        cfg.max_position_embeddings = int(lm.get("max_position_embeddings", cfg.max_position_embeddings))
        cfg.rms_norm_eps = float(lm.get("rms_norm_eps", cfg.rms_norm_eps))
        rope = lm.get("rope_parameters") or {}
        cfg.rope_theta = float(rope.get("rope_theta", cfg.rope_theta))
        cfg.tie_word_embeddings = bool(lm.get("tie_word_embeddings", cfg.tie_word_embeddings))
    if depth:
        cfg.audio_vocab_size = int(depth.get("audio_vocab_size", cfg.audio_vocab_size))
        cfg.num_codebooks = int(depth.get("num_codebooks", cfg.num_codebooks))
        cfg.depth_num_layers = int(depth.get("num_layers", cfg.depth_num_layers))
        cfg.depth_num_heads = int(depth.get("num_attention_heads", cfg.depth_num_heads))
        cfg.depth_intermediate_size = int(depth.get("intermediate_size", cfg.depth_intermediate_size))
    if cond:
        cfg.condition_out_dim = int(cond.get("out_dim", cfg.condition_out_dim))
        cfg.num_condition_layers = int(cond.get("num_condition_layers", cfg.num_condition_layers))
    if dit:
        cfg.dit_in_channels = int(dit.get("in_channels", cfg.dit_in_channels))
        cfg.dit_num_layers = int(dit.get("num_layers", cfg.dit_num_layers))
        cfg.dit_num_heads = int(dit.get("num_attention_heads", cfg.dit_num_heads))
        cfg.dit_head_dim = int(dit.get("attention_head_dim", cfg.dit_head_dim))
        cfg.dit_ff_inner_dim = int(dit.get("ff_inner_dim", cfg.dit_ff_inner_dim))
        cfg.dit_rotary_dim = int(dit.get("rotary_dim", cfg.dit_rotary_dim))
        cfg.dit_fourier_dim = int(dit.get("fourier_embedding_dim", cfg.dit_fourier_dim))
    if voc:
        cfg.vocoder_input_dim = int(voc.get("decoder_input_dim", cfg.vocoder_input_dim))
        cfg.vocoder_hidden_dim = int(voc.get("decoder_hidden_dim", cfg.vocoder_hidden_dim))
        ratios = voc.get("upsampling_ratios")
        if ratios:
            cfg.vocoder_upsampling_ratios = tuple(int(x) for x in ratios)
        cfg.sampling_rate = int(voc.get("sampling_rate", cfg.sampling_rate))
    return cfg


def _load_component(module, path: Path) -> None:
    """Load remapped MLX weights shard-by-shard. Missing or extra keys raise."""
    from mlx.utils import tree_flatten

    from .convert import sanitize_diffusers_keys

    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"Converted weights missing in {path}")
    expected = {key for key, _ in tree_flatten(module.parameters())}
    loaded: set[str] = set()
    extras: set[str] = set()
    module_sanitize = getattr(module, "sanitize", None)
    for weight_file in files:
        raw = dict(mx.load(str(weight_file)))
        weights = sanitize_diffusers_keys(raw)
        if callable(module_sanitize):
            weights = module_sanitize(weights)
        extras.update(key for key in weights if key not in expected)
        batch = {key: value for key, value in weights.items() if key in expected}
        if batch:
            module.load_weights(list(batch.items()), strict=False)
            loaded.update(batch)
        del raw, weights, batch
    missing = expected - loaded
    if missing or extras:
        raise ValueError(
            f"{path.name}: missing={sorted(missing)[:8]} extras={sorted(extras)[:8]}"
        )


def load_converted_modules(root: str | Path) -> Music3Modules:
    root = Path(root)
    config = config_from_converted(root)

    def _build(name: str, factory):
        folder = root / name
        module = factory()
        if folder.exists():
            print(f"loading {name}...")
            _load_component(module, folder)
        return module

    modules = Music3Modules(
        config=config,
        language_model=_build("language_model", lambda: make_qwen3(config)),
        depth_decoder=_build("rvq_depth_decoder", lambda: RVQDepthDecoder(config)),
        condition_encoder=_build("condition_encoder", lambda: ConditionEncoder(config)),
        transformer=_build("transformer", lambda: FlowMatchingTransformer(config)),
        vocoder=_build("vocoder", lambda: Vocoder(config)),
    )
    tok = root / "tokenizer"
    if tok.exists():
        modules.tokenizer_dir = str(tok)
    return modules
