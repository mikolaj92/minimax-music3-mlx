"""Dump tiny (+ optional 4-bit) Qwen3 tensors for the Swift parity tests."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from minimax_music3_mlx.ar import lm_logits, qwen3_hidden
from minimax_music3_mlx.tiny import encode_official_text_pair, encode_text_pair, make_tiny_modules

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT.parent / "minimax-music3-swift/Tests/MiniMaxMusic3MLXTests/Fixtures"


def _save(path: Path, tensors: dict[str, mx.array]) -> None:
    packed = {}
    for key, value in tensors.items():
        packed[key] = value if mx.issubdtype(value.dtype, mx.integer) else value.astype(mx.float32)
    mx.save_safetensors(str(path), packed)
    print(f"wrote {path} ({len(tensors)} tensors)")


def dump_tiny() -> None:
    modules = make_tiny_modules(seed=0)
    weights = dict(tree_flatten(modules.language_model.parameters()))
    print("tiny keys", len(weights))
    for key in list(sorted(weights))[:8]:
        print(" ", key)
    _save(FIX / "tiny_qwen3.safetensors", weights)

    ids = encode_text_pair("caption and lyrics", modules.config)
    embeds = modules.language_model.model.embed_tokens(ids)
    hidden, cache = qwen3_hidden(modules.language_model, embeds)
    last = hidden[:, -1]
    logits = lm_logits(modules.language_model, last)
    feedback = modules.language_model.model.embed_tokens(ids[:, -1:])
    hidden2, _ = qwen3_hidden(modules.language_model, feedback, cache)
    mx.eval(hidden, last, logits, hidden2)
    _save(
        FIX / "tiny_qwen3_ref.safetensors",
        {
            "input_ids": ids,
            "hidden": hidden,
            "last_hidden": last,
            "logits": logits,
            "step2_hidden": hidden2,
        },
    )


def dump_official_4bit() -> None:
    root = ROOT / "weights/mlx-4bit"
    if not (root / "language_model/config.json").exists():
        print("skip official 4-bit dump: missing", root)
        return
    from minimax_music3_mlx.load_weights import load_converted_modules

    prompt = json.loads((FIX / "prompt_ref.json").read_text())["prompt"]
    modules = load_converted_modules(root)
    ids = encode_official_text_pair(prompt, modules.config, modules.tokenizer_dir)
    embeds = modules.language_model.model.embed_tokens(ids)
    hidden, _ = qwen3_hidden(modules.language_model, embeds)
    last = hidden[:, -1]
    mx.eval(last)
    _save(
        FIX / "official_qwen3_4bit_ref.safetensors",
        {
            "input_ids": ids,
            "last_hidden": last,
        },
    )


if __name__ == "__main__":
    FIX.mkdir(parents=True, exist_ok=True)
    dump_tiny()
    dump_official_4bit()
