"""End-to-end generate through the shipped CLI entry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from minimax_music3_mlx.generate import generate_song, main


LYRICS = "[verse]\nMorning light filtering through the pine\n[chorus]\nSoftly the world begins to breathe"
PROMPT = "Genre: acoustic pop. BPM: 96. Warm female vocal."


def test_generate_song_writes_stereo_wav(tmp_path: Path):
    out = tmp_path / "clip.wav"
    path = generate_song(
        lyrics=LYRICS,
        prompt=PROMPT,
        output=out,
        audio_duration=0.12,
        num_inference_steps=1,
        seed=8,
        tiny=True,
    )
    assert path.exists()
    wave, rate = sf.read(str(path), always_2d=True)
    assert wave.shape[1] == 2
    assert rate == 44100
    assert wave.shape[0] > 0
    assert np.isfinite(wave).all()
    assert float(np.max(np.abs(wave))) > 0.0


def test_multi_window_clip_is_not_naive_concat(tmp_path: Path):
    """Overlapping 200-frame windows must crop, not append whole vocoder outputs."""
    from minimax_music3_mlx.pipeline import _crop_waveform, chunk_starts
    import mlx.core as mx

    assert chunk_starts(200) == [0]
    assert chunk_starts(250) == [0, 100]
    fake = mx.ones((1, 2, 200_000))
    first = _crop_waveform(fake, 0, 2)
    last = _crop_waveform(fake, 1, 2)
    assert int(first.shape[-1]) < 200_000
    assert int(last.shape[-1]) < 200_000
    assert int(first.shape[-1]) != int(last.shape[-1])


def test_cli_main_exit_zero(tmp_path: Path):
    out = tmp_path / "cli.wav"
    code = main(
        [
            "--tiny",
            "--lyrics",
            LYRICS,
            "--prompt",
            PROMPT,
            "--duration",
            "0.12",
            "--steps",
            "1",
            "--seed",
            "9",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    assert out.exists()
