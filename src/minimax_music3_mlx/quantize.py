"""Quantize converted MiniMax Music 3 MLX weights (Linear layers, group_size=64)."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import mlx.core as mx
from mlx.utils import tree_flatten

from .load_weights import (
    QUANTIZED_COMPONENT_NAMES,
    _load_component,
    apply_linear_quantize,
    config_from_converted,
)
from .depth import RVQDepthDecoder
from .dit import FlowMatchingTransformer
from .tiny import make_qwen3

# Keep shards under typical safetensors / mlx-lm comfort size.
_MAX_SHARD_BYTES = 5 << 30

_SYMLINK_DIRS = ("tokenizer", "scheduler", "vocoder", "condition_encoder")
_COPY_ROOT_FILES = ("modular_model_index.json",)
_SHARD_PREFIX = {
    "language_model": "model",
    "transformer": "diffusion_pytorch_model",
    "rvq_depth_decoder": "diffusion_pytorch_model",
}


def _free_gb(path: Path) -> float:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize / 1e9


def _require_disk(path: Path, min_free_gb: float = 15.0) -> float:
    free = _free_gb(path)
    if free < min_free_gb:
        raise RuntimeError(f"free disk {free:.1f}GB is under {min_free_gb}GB; stopping")
    return free


def _clear_mlx() -> None:
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    else:
        metal = getattr(mx, "metal", None)
        if metal is not None and hasattr(metal, "clear_cache"):
            metal.clear_cache()


def _component_factory(name: str, config):
    if name == "language_model":
        return make_qwen3(config)
    if name == "transformer":
        return FlowMatchingTransformer(config)
    if name == "rvq_depth_decoder":
        return RVQDepthDecoder(config)
    raise ValueError(f"not a quantized component: {name}")


def _make_shards(weights: dict[str, mx.array], max_bytes: int = _MAX_SHARD_BYTES) -> list[dict[str, mx.array]]:
    shards: list[dict[str, mx.array]] = []
    shard: dict[str, mx.array] = {}
    size = 0
    for key, value in weights.items():
        nbytes = int(value.nbytes)
        if shard and size + nbytes > max_bytes:
            shards.append(shard)
            shard, size = {}, 0
        shard[key] = value
        size += nbytes
    if shard:
        shards.append(shard)
    return shards


def save_quantized_module(module, dest: Path, prefix: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    mx.eval(module.parameters())
    weights = dict(tree_flatten(module.parameters()))
    shards = _make_shards(weights)
    total_size = sum(int(value.nbytes) for value in weights.values())
    weights.clear()
    del weights
    n = len(shards)
    name_fmt = f"{prefix}-{{:05d}}-of-{{:05d}}.safetensors" if n > 1 else f"{prefix}.safetensors"
    index = {"metadata": {"total_size": total_size}, "weight_map": {}}
    for i, shard in enumerate(shards):
        filename = name_fmt.format(i + 1, n)
        mx.save_safetensors(str(dest / filename), shard, metadata={"format": "mlx"})
        for key in shard:
            index["weight_map"][key] = filename
        shards[i] = None
        del shard
    index["weight_map"] = {key: index["weight_map"][key] for key in sorted(index["weight_map"])}
    (dest / f"{prefix}.safetensors.index.json").write_text(json.dumps(index, indent=4))


def quantize_component(src_root: Path, dst_root: Path, name: str, bits: int, group_size: int = 64) -> Path:
    config = config_from_converted(src_root)
    src = src_root / name
    dst = dst_root / name
    if dst.exists():
        shutil.rmtree(dst)
    print(f"quantize {name} bits={bits} (free {_free_gb(dst_root):.1f}GB)...", flush=True)
    module = _component_factory(name, config)
    _load_component(module, src)
    apply_linear_quantize(module, bits=bits, group_size=group_size)
    save_quantized_module(module, dst, _SHARD_PREFIX[name])
    for cfg in src.glob("*.json"):
        if cfg.name.endswith(".safetensors.index.json"):
            continue
        (dst / cfg.name).write_text(cfg.read_text())
    del module
    _clear_mlx()
    return dst


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.symlink_to(os.path.relpath(src, start=dst.parent), target_is_directory=src.is_dir())


def install_small_assets(src_root: Path, dst_root: Path) -> None:
    for name in _SYMLINK_DIRS:
        src = src_root / name
        if src.exists():
            _link_or_copy(src, dst_root / name)
    for name in _COPY_ROOT_FILES:
        src = src_root / name
        if src.is_file():
            shutil.copy2(src, dst_root / name)


def write_mlx_config(dst_root: Path, src_root: Path, bits: int, group_size: int = 64) -> Path:
    meta = {
        "format": "mlx",
        "source": str(src_root),
        "layout": "minimax-music3-diffusers",
        "quantized": True,
        "bits": bits,
        "group_size": group_size,
        "mode": "affine",
        "quantize_modules": list(QUANTIZED_COMPONENT_NAMES),
        "load": (
            "construct module, nn.quantize Linear (group_size) so keys include "
            "scales/biases, then Module.load_weights"
        ),
    }
    path = dst_root / "mlx_config.json"
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path


def quantize_tree(
    src: str | Path,
    dst: str | Path,
    bits: int,
    group_size: int = 64,
    min_free_gb: float = 15.0,
) -> Path:
    src_root = Path(src).resolve()
    dst_root = Path(dst).resolve()
    if src_root == dst_root:
        raise ValueError("refusing to write quantized weights over the source tree")
    dst_root.mkdir(parents=True, exist_ok=True)
    _require_disk(dst_root, min_free_gb)
    for name in QUANTIZED_COMPONENT_NAMES:
        _require_disk(dst_root, min_free_gb)
        quantize_component(src_root, dst_root, name, bits=bits, group_size=group_size)
    install_small_assets(src_root, dst_root)
    write_mlx_config(dst_root, src_root, bits=bits, group_size=group_size)
    print(f"wrote {dst_root} bits={bits} free={_free_gb(dst_root):.1f}GB", flush=True)
    return dst_root


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantize MiniMax Music 3 MLX weights")
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--bits", type=int, required=True)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args(list(argv) if argv is not None else None)
    quantize_tree(args.src, args.dst, bits=args.bits, group_size=args.group_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
