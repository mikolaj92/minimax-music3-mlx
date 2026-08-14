"""DAC-style Flow-VAE decoder: latents ``[B, C, T]`` → stereo ``[B, 2, samples]``."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import Music3Config


def _ncl_to_nlc(x: mx.array) -> mx.array:
    return x.transpose(0, 2, 1)


def _nlc_to_ncl(x: mx.array) -> mx.array:
    return x.transpose(0, 2, 1)


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # Official MiniMaxMusic3Snake1d: Parameter[1, C, 1] in NCL.
        self.alpha = mx.ones((1, channels, 1))

    def __call__(self, x_ncl: mx.array) -> mx.array:
        alpha = self.alpha.astype(x_ncl.dtype)
        return x_ncl + (1.0 / (alpha + 1e-9)) * (mx.sin(alpha * x_ncl) ** 2)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        pad = (7 - 1) * dilation // 2
        self.snake1 = Snake1d(dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad)
        self.snake2 = Snake1d(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)

    def __call__(self, x_ncl: mx.array) -> mx.array:
        y = _ncl_to_nlc(self.snake1(x_ncl))
        y = self.conv1(y)
        y = _ncl_to_nlc(self.snake2(_nlc_to_ncl(y)))
        y = self.conv2(y)
        return x_ncl + _nlc_to_ncl(y)


class VocoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.snake1 = Snake1d(input_dim)
        padding = math.ceil(stride / 2)
        self.conv_t1 = nn.ConvTranspose1d(
            input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=padding
        )
        self.res_unit1 = ResidualUnit(output_dim, dilation=1)
        self.res_unit2 = ResidualUnit(output_dim, dilation=3)
        self.res_unit3 = ResidualUnit(output_dim, dilation=9)

    def __call__(self, x_ncl: mx.array) -> mx.array:
        y = self.conv_t1(_ncl_to_nlc(self.snake1(x_ncl)))
        y = _nlc_to_ncl(y)
        y = self.res_unit1(y)
        y = self.res_unit2(y)
        return self.res_unit3(y)


class Vocoder(nn.Module):
    def __init__(self, config: Music3Config):
        super().__init__()
        self.config = config
        latent_half = config.dit_in_channels // 2
        self.dec_in_proj = nn.Conv1d(latent_half, config.vocoder_input_dim, kernel_size=1)
        self.conv_in = nn.Conv1d(config.vocoder_input_dim, config.vocoder_hidden_dim, kernel_size=7, padding=3)
        blocks = []
        output_dim = config.vocoder_hidden_dim
        for index, stride in enumerate(config.vocoder_upsampling_ratios):
            input_dim = config.vocoder_hidden_dim // (2**index)
            output_dim = config.vocoder_hidden_dim // (2 ** (index + 1))
            blocks.append(VocoderBlock(input_dim, output_dim, stride))
        self.blocks = blocks
        self.snake_out = Snake1d(output_dim)
        self.conv_out = nn.Conv1d(output_dim, 1, kernel_size=7, padding=3)

    def __call__(self, latents: mx.array) -> mx.array:
        """``latents`` ``[B, C, T]`` → stereo ``[B, 2, samples]`` in ``[-1, 1]``."""
        batch, channels, length = latents.shape
        half = self.config.dit_in_channels // 2
        hidden = latents.reshape(batch * 2, half, length)
        hidden = _nlc_to_ncl(self.dec_in_proj(_ncl_to_nlc(hidden)))
        hidden = _nlc_to_ncl(self.conv_in(_ncl_to_nlc(hidden)))
        for block in self.blocks:
            hidden = block(hidden)
        wave = mx.tanh(self.conv_out(_ncl_to_nlc(self.snake_out(hidden))))
        return _nlc_to_ncl(wave).reshape(batch, 2, -1)
