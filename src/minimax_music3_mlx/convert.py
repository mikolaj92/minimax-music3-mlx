"""Convert official Diffusers MiniMax Music 3 weights into MLX arrays.

PyTorch Conv1d is ``[out, in, K]``; MLX Conv1d is ``[out, K, in]``.
PyTorch ConvTranspose1d is ``[in, out, K]``; MLX is ``[out, K, in]``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

# Name fragments that identify ConvTranspose1d (DAC vocoder upsamplers).
_CONV_TRANSPOSE_RE = re.compile(
    r"(conv_t\d+|conv_transpose|convtr[^a-z]|deconv)",
    re.IGNORECASE,
)


def remap_conv1d_weight(weight: np.ndarray) -> np.ndarray:
    """``[out, in, K]`` → ``[out, K, in]``."""
    if weight.ndim != 3:
        raise ValueError(f"Conv1d weight must be 3D, got {weight.shape}")
    return np.swapaxes(weight, 1, 2)


def remap_conv_transpose_weight(weight: np.ndarray) -> np.ndarray:
    """``[in, out, K]`` → ``[out, K, in]``."""
    if weight.ndim != 3:
        raise ValueError(f"ConvTranspose1d weight must be 3D, got {weight.shape}")
    return np.transpose(weight, (1, 2, 0))


def fuse_weight_norm(weight_v: np.ndarray, weight_g: np.ndarray) -> np.ndarray:
    """Reconstruct a weight-normed kernel: ``g * v / ||v||``."""
    axes = tuple(range(1, weight_v.ndim))
    v_norm = np.sqrt(np.sum(weight_v.astype(np.float64) ** 2, axis=axes, keepdims=True))
    v_norm = np.maximum(v_norm, 1e-12)
    g = weight_g.astype(np.float64)
    while g.ndim < weight_v.ndim:
        g = np.expand_dims(g, axis=-1)
    return (g * weight_v.astype(np.float64) / v_norm).astype(np.float32)


def is_conv_transpose_key(key: str) -> bool:
    return bool(_CONV_TRANSPOSE_RE.search(key))


def sanitize_diffusers_keys(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Map official Diffusers parameter names onto the shipped MLX modules.

    - ``attn.to_out.weight`` → ``attn.to_out.0.weight`` (ModuleList Linear)
    - drop Dropout ``to_out.1.*`` and cached RoPE buffers
    """
    out: dict[str, np.ndarray] = {}
    for key, value in state.items():
        if "rotary_emb" in key or key.endswith(".inv_freq"):
            continue
        if "transformer_blocks" in key and (".to_out.1." in key or key.endswith(".to_out.1")):
            continue
        # Only the DiT attention uses ModuleList(Linear, Dropout). The RVQ depth
        # decoder's to_out is a bare Linear (official: layers.N.attn.to_out.weight).
        if "transformer_blocks" in key and key.endswith(".to_out.weight"):
            key = key[: -len(".weight")] + ".0.weight"
        elif "transformer_blocks" in key and key.endswith(".to_out.bias"):
            key = key[: -len(".bias")] + ".0.bias"
        out[key] = value
    return out


def remap_tensor(key: str, value: np.ndarray) -> np.ndarray:
    """Apply the layout remap that the key's module type requires."""
    if key.endswith(".alpha"):
        return value
    if value.ndim != 3 or key.endswith((".bias", ".weight_g")):
        return value
    if not key.endswith(".weight") and not key.endswith("weight"):
        return value
    if is_conv_transpose_key(key):
        return remap_conv_transpose_weight(value)
    # 1x1 / dilated residual / preprocess Conv1d kernels.
    return remap_conv1d_weight(value)


def fuse_weight_norm_pairs(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Collapse ``*.weight_g`` + ``*.weight_v`` pairs into a single ``*.weight``."""
    out: dict[str, np.ndarray] = {}
    consumed: set[str] = set()
    for key, value in state.items():
        if key in consumed:
            continue
        if key.endswith(".weight_v"):
            g_key = key[: -len(".weight_v")] + ".weight_g"
            if g_key in state:
                fused_key = key[: -len(".weight_v")] + ".weight"
                out[fused_key] = fuse_weight_norm(value, state[g_key])
                consumed.add(key)
                consumed.add(g_key)
                continue
        if key.endswith(".weight_g"):
            v_key = key[: -len(".weight_g")] + ".weight_v"
            if v_key in state:
                continue
        out[key] = value
    return out


def remap_state_dict(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fuse weight-norm, remap conv kernels, then rename Diffusers keys."""
    fused = fuse_weight_norm_pairs(state)
    remapped = {key: np.ascontiguousarray(remap_tensor(key, value)) for key, value in fused.items()}
    return sanitize_diffusers_keys(remapped)


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        try:
            value = value.numpy()
        except Exception:
            pass
    return np.asarray(value)


def load_safetensors_dir(path: Path) -> dict[str, np.ndarray]:
    import mlx.core as mx

    tensors: dict[str, np.ndarray] = {}
    files = sorted(path.glob("*.safetensors"))
    if not files and path.is_file() and path.suffix == ".safetensors":
        files = [path]
    if not files:
        raise FileNotFoundError(f"No safetensors in {path}")
    for file in files:
        # mlx.load understands bfloat16; numpy's safetensors backend does not.
        loaded = mx.load(str(file))
        for name, value in loaded.items():
            tensors[name] = np.array(value.astype(mx.float32))
    return tensors


def save_mlx_safetensors(path: Path, state: Mapping[str, np.ndarray], dtype: str = "bfloat16") -> None:
    import mlx.core as mx

    path.parent.mkdir(parents=True, exist_ok=True)
    mlx_dtype = getattr(mx, dtype)
    mx.save_safetensors(str(path), {k: mx.array(v).astype(mlx_dtype) for k, v in state.items()})


def _needs_layout_remap(key: str, ndim: int) -> bool:
    if key.endswith((".weight_g", ".weight_v")):
        return True
    if key.endswith(".alpha") or key.endswith((".bias", ".weight_g")):
        return False
    if ndim != 3:
        return False
    return key.endswith(".weight") or key.endswith("weight")


# Smallest first so a remap bug fails cheaply and peak disk stays lower.
_COMPONENT_FOLDERS = (
    "condition_encoder",
    "vocoder",
    "rvq_depth_decoder",
    "transformer",
    "language_model",
)


def convert_shard(src_file: Path, dst_file: Path) -> Path:
    """Remap one safetensors shard without loading sibling shards."""
    import gc

    import mlx.core as mx

    loaded = mx.load(str(src_file))
    if any(_needs_layout_remap(name, int(value.ndim)) for name, value in loaded.items()):
        state = {name: np.array(value.astype(mx.float32)) for name, value in loaded.items()}
        del loaded
        remapped = remap_state_dict(state)
        del state
        save_mlx_safetensors(dst_file, remapped)
        del remapped
    else:
        remapped = sanitize_diffusers_keys(loaded)
        del loaded
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(dst_file), {k: v.astype(mx.bfloat16) for k, v in remapped.items()})
        del remapped
    gc.collect()
    return dst_file


def convert_component(src_folder: Path, dst_folder: Path, delete_source: bool = False) -> Path:
    """Convert one Diffusers component, shard by shard, keeping original filenames."""
    files = sorted(src_folder.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors in {src_folder}")
    dst_folder.mkdir(parents=True, exist_ok=True)
    for file in files:
        size_gb = file.stat().st_size / 1e9
        print(f"  {file.name} ({size_gb:.1f}G)")
        convert_shard(file, dst_folder / file.name)
        if delete_source:
            file.unlink()
            print(f"    deleted source {file.name}")
    for cfg in src_folder.glob("*.json"):
        (dst_folder / cfg.name).write_text(cfg.read_text())
    return dst_folder


def convert_diffusers_dir(src: str | Path, dst: str | Path, delete_source: bool = False) -> Path:
    """Convert a Diffusers MiniMax-Music3 checkout into an MLX weight tree."""
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.mkdir(parents=True, exist_ok=True)

    for name in _COMPONENT_FOLDERS:
        folder = src_path / name
        if not folder.exists():
            continue
        print(f"converting {name}...")
        convert_component(folder, dst_path / name, delete_source=delete_source)

    for extra in ("tokenizer", "scheduler", "modular_model_index.json"):
        item = src_path / extra
        if item.is_file():
            (dst_path / extra).write_bytes(item.read_bytes())
        elif item.is_dir():
            dest = dst_path / extra
            dest.mkdir(parents=True, exist_ok=True)
            for file in item.rglob("*"):
                if file.is_file():
                    rel = file.relative_to(item)
                    (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                    (dest / rel).write_bytes(file.read_bytes())

    meta = {"format": "mlx", "source": str(src_path), "layout": "minimax-music3-diffusers"}
    (dst_path / "mlx_config.json").write_text(json.dumps(meta, indent=2))
    return dst_path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert MiniMax Music 3 Diffusers weights to MLX")
    parser.add_argument("--src", required=True, help="Path to MiniMaxAI/MiniMax-Music3 checkout")
    parser.add_argument("--dst", required=True, help="Output directory for MLX weights")
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete each source shard after its remapped copy is written",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    out = convert_diffusers_dir(args.src, args.dst, delete_source=args.delete_source)
    print(f"Wrote MLX weights to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
