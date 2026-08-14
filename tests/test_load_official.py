"""Official Diffusers-layout dummy weights must convert and populate shipped modules."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from safetensors.numpy import save_file

from minimax_music3_mlx.config import Music3Config
from minimax_music3_mlx.convert import convert_diffusers_dir, is_conv_transpose_key
from minimax_music3_mlx.load_weights import load_converted_modules
from minimax_music3_mlx.tiny import make_tiny_modules


def _tiny_component_configs(config: Music3Config) -> dict[str, dict]:
    return {
        "language_model": {
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "num_hidden_layers": config.num_hidden_layers,
            "intermediate_size": config.intermediate_size,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "max_position_embeddings": config.max_position_embeddings,
            "rms_norm_eps": config.rms_norm_eps,
            "tie_word_embeddings": config.tie_word_embeddings,
            "rope_parameters": {"rope_theta": config.rope_theta},
        },
        "rvq_depth_decoder": {
            "hidden_size": config.hidden_size,
            "num_layers": config.depth_num_layers,
            "num_attention_heads": config.depth_num_heads,
            "intermediate_size": config.depth_intermediate_size,
            "audio_vocab_size": config.audio_vocab_size,
            "num_codebooks": config.num_codebooks,
        },
        "condition_encoder": {
            "condition_hidden_dim": config.hidden_size,
            "num_condition_layers": config.num_condition_layers,
            "out_dim": config.condition_out_dim,
        },
        "transformer": {
            "in_channels": config.dit_in_channels,
            "condition_dim": config.condition_out_dim,
            "num_layers": config.dit_num_layers,
            "num_attention_heads": config.dit_num_heads,
            "attention_head_dim": config.dit_head_dim,
            "ff_inner_dim": config.dit_ff_inner_dim,
            "rotary_dim": config.dit_rotary_dim,
            "fourier_embedding_dim": config.dit_fourier_dim,
        },
        "vocoder": {
            "latent_channels": config.dit_in_channels,
            "decoder_input_dim": config.vocoder_input_dim,
            "decoder_hidden_dim": config.vocoder_hidden_dim,
            "upsampling_ratios": list(config.vocoder_upsampling_ratios),
            "sampling_rate": config.sampling_rate,
        },
    }


def _mlx_to_official_tensor(key: str, value: np.ndarray) -> np.ndarray:
    if key.endswith(".alpha"):
        assert value.shape[0] == 1 and value.shape[-1] == 1, value.shape
        return value
    if value.ndim == 3 and key.endswith(".weight"):
        if is_conv_transpose_key(key):
            # MLX [out, K, in] → PyTorch ConvTranspose [in, out, K]
            return np.transpose(value, (2, 0, 1))
        # MLX [out, K, in] → PyTorch Conv1d [out, in, K]
        return np.swapaxes(value, 1, 2)
    return value


def _split_weight_norm(pt_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode a PyTorch Conv1d kernel as weight_v / weight_g so fuse reconstructs it."""
    g = np.sqrt(np.sum(pt_weight.astype(np.float64) ** 2, axis=(1, 2), keepdims=True))
    g = np.maximum(g, 1e-6)
    return pt_weight.astype(np.float32), g.astype(np.float32)


def _write_official_tree(src: Path, modules) -> dict[str, np.ndarray]:
    """Write a Diffusers-layout tree whose tensors are the reverse of convert."""
    config = modules.config
    for name, payload in _tiny_component_configs(config).items():
        folder = src / name
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps(payload))

    components = {
        "language_model": modules.language_model,
        "rvq_depth_decoder": modules.depth_decoder,
        "condition_encoder": modules.condition_encoder,
        "transformer": modules.transformer,
        "vocoder": modules.vocoder,
    }
    expected_mlx: dict[str, np.ndarray] = {}
    for name, module in components.items():
        official: dict[str, np.ndarray] = {}
        for key, tensor in tree_flatten(module.parameters()):
            mlx_arr = np.array(tensor, dtype=np.float32)
            expected_mlx[f"{name}.{key}"] = mlx_arr
            official[key] = np.ascontiguousarray(_mlx_to_official_tensor(key, mlx_arr))

        # Official vocoder stores weight-norm pairs, not a fused kernel.
        if "conv_in.weight" in official:
            weight_v, weight_g = _split_weight_norm(official.pop("conv_in.weight"))
            official["conv_in.weight_v"] = np.ascontiguousarray(weight_v)
            official["conv_in.weight_g"] = np.ascontiguousarray(weight_g)

        # Bare Linear name that sanitize must rewrite to to_out.0.
        if "transformer_blocks.0.attn.to_out.0.weight" in official:
            official["transformer_blocks.0.attn.to_out.weight"] = official.pop(
                "transformer_blocks.0.attn.to_out.0.weight"
            )

        save_file(official, src / name / "diffusion_pytorch_model.safetensors")
    return expected_mlx


def test_convert_and_load_official_layout_dummy(tmp_path: Path):
    mx.random.seed(21)
    reference = make_tiny_modules(seed=21)
    src = tmp_path / "MiniMax-Music3"
    expected = _write_official_tree(src, reference)
    dst = tmp_path / "mlx"
    convert_diffusers_dir(src, dst)

    loaded = load_converted_modules(dst)

    # Out-proj from official to_out.weight must land on to_out.0
    loaded_out = dict(tree_flatten(loaded.transformer.parameters()))
    assert "transformer_blocks.0.attn.to_out.0.weight" in loaded_out
    assert "transformer_blocks.0.attn.to_out.weight" not in loaded_out
    assert "preprocess_conv.bias" not in loaded_out
    assert "postprocess_conv.bias" not in loaded_out

    def _np(x):
        import mlx.core as mx

        if hasattr(x, "astype"):
            try:
                x = x.astype(mx.float32)
            except Exception:
                pass
        return np.array(x, dtype=np.float32)

    ref_to_out = expected["transformer.transformer_blocks.0.attn.to_out.0.weight"]
    got_to_out = _np(loaded_out["transformer_blocks.0.attn.to_out.0.weight"])
    # Converted checkpoints are bfloat16; ~1e-3 is one ULP around these magnitudes.
    assert np.allclose(got_to_out, ref_to_out, atol=2e-3, rtol=1e-2)

    # Snake alpha stays official NCL [1, C, 1] and is applied, not swapped.
    loaded_voc = dict(tree_flatten(loaded.vocoder.parameters()))
    alpha = _np(loaded_voc["snake_out.alpha"])
    assert alpha.shape == expected["vocoder.snake_out.alpha"].shape
    assert alpha.shape[0] == 1 and alpha.shape[-1] == 1
    assert np.allclose(alpha, expected["vocoder.snake_out.alpha"], atol=2e-3, rtol=1e-2)

    # weight_g/v fused and remapped onto conv_in.weight
    assert np.allclose(
        _np(loaded_voc["conv_in.weight"]),
        expected["vocoder.conv_in.weight"],
        atol=2e-3,
        rtol=1e-2,
    )

    # A DiT 1x1 conv also remapped and loaded (not leftover random init).
    assert np.allclose(
        _np(loaded_out["preprocess_conv.weight"]),
        expected["transformer.preprocess_conv.weight"],
        atol=2e-3,
        rtol=1e-2,
    )

    # Forward still works after official load.
    frames = 2
    h = mx.random.normal((1, frames, loaded.config.num_codebooks * loaded.config.hidden_size))
    cond = loaded.condition_encoder(h)
    latents = mx.random.normal((1, loaded.config.dit_in_channels, cond.shape[1]))
    vel = loaded.transformer(latents, mx.array([0.5]), cond)
    audio = loaded.vocoder(vel)
    mx.eval(audio)
    assert audio.shape[1] == 2
    assert np.isfinite(np.array(audio)).all()


def test_load_raises_on_missing_official_keys(tmp_path: Path):
    config = Music3Config.tiny()
    src = tmp_path / "MiniMax-Music3" / "transformer"
    src.mkdir(parents=True)
    (src / "config.json").write_text(
        json.dumps(_tiny_component_configs(config)["transformer"])
    )
    save_file(
        {"preprocess_conv.weight": np.ones((8, 8, 1), dtype=np.float32)},
        src / "diffusion_pytorch_model.safetensors",
    )
    dst = tmp_path / "mlx"
    convert_diffusers_dir(tmp_path / "MiniMax-Music3", dst)
    # Incomplete transformer weights must not be swallowed.
    try:
        load_converted_modules(dst)
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError("expected load_converted_modules to raise on missing keys")
