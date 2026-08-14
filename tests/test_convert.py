"""Tests for the shipped Diffusers → MLX conv remap."""

from __future__ import annotations

import numpy as np

from minimax_music3_mlx.convert import (
    fuse_weight_norm,
    is_conv_transpose_key,
    remap_conv1d_weight,
    remap_conv_transpose_weight,
    remap_state_dict,
)


def test_conv1d_remap_swaps_in_and_kernel():
    weight = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    remapped = remap_conv1d_weight(weight)
    assert remapped.shape == (2, 5, 3)
    assert np.array_equal(remapped, np.swapaxes(weight, 1, 2))


def test_conv_transpose_remap_moves_in_out_kernel():
    weight = np.arange(3 * 4 * 6, dtype=np.float32).reshape(3, 4, 6)
    remapped = remap_conv_transpose_weight(weight)
    assert remapped.shape == (4, 6, 3)
    assert np.array_equal(remapped, np.transpose(weight, (1, 2, 0)))


def test_remap_state_dict_classifies_vocoder_keys():
    conv = np.arange(2 * 3 * 1, dtype=np.float32).reshape(2, 3, 1)
    conv_t = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)
    state = {
        "preprocess_conv.weight": conv.copy(),
        "blocks.0.conv_t1.weight": conv_t.copy(),
        "proj.bias": np.ones(2, dtype=np.float32),
    }
    out = remap_state_dict(state)
    assert out["preprocess_conv.weight"].shape == (2, 1, 3)
    assert out["blocks.0.conv_t1.weight"].shape == (2, 4, 3)
    assert np.array_equal(out["proj.bias"], state["proj.bias"])


def test_weight_norm_fuse_then_remap():
    v = np.ones((2, 3, 5), dtype=np.float32)
    g = np.array([2.0, 3.0], dtype=np.float32).reshape(2, 1, 1)
    fused = fuse_weight_norm(v, g)
    assert fused.shape == v.shape
    # ||v|| = sqrt(15); reconstructed scale is g / ||v||
    expected = g * v / np.sqrt(15.0)
    assert np.allclose(fused, expected, atol=1e-6)
    remapped = remap_state_dict({"conv1.weight_v": v, "conv1.weight_g": g})
    assert "conv1.weight" in remapped
    assert remapped["conv1.weight"].shape == (2, 5, 3)


def test_convert_diffusers_dir_roundtrip(tmp_path):
    from safetensors.numpy import save_file

    from minimax_music3_mlx.convert import convert_diffusers_dir, load_safetensors_dir

    src = tmp_path / "MiniMax-Music3"
    voc = src / "vocoder"
    voc.mkdir(parents=True)
    save_file(
        {
            "dec_in_proj.weight": np.arange(4 * 2 * 1, dtype=np.float32).reshape(4, 2, 1),
            "blocks.0.conv_t1.weight": np.arange(3 * 4 * 8, dtype=np.float32).reshape(3, 4, 8),
        },
        voc / "diffusion_pytorch_model.safetensors",
    )
    (voc / "config.json").write_text('{"sampling_rate": 44100}')
    (src / "modular_model_index.json").write_text("{}")

    dst = tmp_path / "mlx"
    convert_diffusers_dir(src, dst)
    loaded = load_safetensors_dir(dst / "vocoder")
    assert loaded["dec_in_proj.weight"].shape == (4, 1, 2)
    assert loaded["blocks.0.conv_t1.weight"].shape == (4, 8, 3)
    assert (dst / "modular_model_index.json").exists()


def test_conv_transpose_key_detection():
    assert is_conv_transpose_key("blocks.0.conv_t1.weight")
    assert not is_conv_transpose_key("preprocess_conv.weight")
    assert not is_conv_transpose_key("dec_in_proj.weight")
