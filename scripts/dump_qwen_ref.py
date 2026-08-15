"""Dump tiny (and optional official) Qwen3 hidden states for Swift parity tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from minimax_music3_mlx.ar import qwen3_hidden
from minimax_music3_mlx.tiny import encode_text_pair, make_tiny_modules


def _save(path: Path, tensors: dict[str, mx.array]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.eval(*tensors.values())
    mx.save_safetensors(str(path), {key: value.astype(mx.float32) if value.dtype == mx.bfloat16 else value for key, value in tensors.items()})
    print(f"wrote {path} keys={list(tensors)}")


def dump_tiny(fixtures: Path, seed: int = 0) -> None:
    modules = make_tiny_modules(seed=seed)
    lm = modules.language_model
    weights = {key: value for key, value in tree_flatten(lm.parameters())}
    mx.eval(*weights.values())
    mx.save_safetensors(str(fixtures / "tiny_qwen.safetensors"), weights)
    print(f"wrote {fixtures / 'tiny_qwen.safetensors'} n={len(weights)}")

    text_ids = encode_text_pair("caption and lyrics", modules.config)
    embeds = lm.model.embed_tokens(text_ids)
    hidden, _ = qwen3_hidden(lm, embeds)
    _save(
        fixtures / "tiny_qwen_ref.safetensors",
        {
            "input_ids": text_ids,
            "embeds": embeds,
            "hidden": hidden,
            "last_hidden": hidden[:, -1],
        },
    )
    print("tiny hidden", tuple(hidden.shape), "last", tuple(hidden[:, -1].shape))


def dump_official(fixtures: Path, weights_dir: Path) -> None:
    from minimax_music3_mlx.load_weights import load_converted_modules
    from minimax_music3_mlx.tiny import encode_official_text_pair
    from minimax_music3_mlx.prompt import assemble_prompt

    modules = load_converted_modules(weights_dir)
    text = assemble_prompt("Genre: funk. BPM: 112. Warm vocal.", "[verse]\nHello from MLX\n[chorus]\nSing it back")
    assert modules.tokenizer_dir
    text_ids = encode_official_text_pair(text, modules.config, modules.tokenizer_dir)
    # Keep the official dump short enough for a unit test.
    text_ids = text_ids[:, :8]
    embeds = modules.language_model.model.embed_tokens(text_ids)
    hidden, _ = qwen3_hidden(modules.language_model, embeds)
    last = hidden[:, -1]
    mx.eval(last)
    _save(
        fixtures / "official_qwen_ref.safetensors",
        {
            "input_ids": text_ids,
            "last_hidden": last.astype(mx.float32),
        },
    )
    print("official last", tuple(last.shape), float(mx.abs(last).mean()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures",
        default="../minimax-music3-swift/Tests/MiniMaxMusic3MLXTests/Fixtures",
    )
    parser.add_argument("--official", default="")
    args = parser.parse_args()
    fixtures = Path(args.fixtures)
    dump_tiny(fixtures)
    if args.official:
        dump_official(fixtures, Path(args.official))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
