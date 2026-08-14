"""Launchable MiniMax Music 3 generate: lyrics + caption → stereo WAV on MLX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from .pipeline import generate_audio, load_modules


def write_wav(path: str | Path, waveform: np.ndarray, sample_rate: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if waveform.ndim != 2 or waveform.shape[1] != 2:
        raise ValueError(f"expected [samples, 2] stereo, got {waveform.shape}")
    sf.write(str(path), waveform, sample_rate, subtype="PCM_16")
    return path


def generate_song(
    lyrics: str,
    prompt: str,
    output: str | Path,
    audio_duration: float = 0.2,
    num_inference_steps: int = 2,
    seed: int = 0,
    weights: str | Path | None = None,
    tiny: bool = False,
) -> Path:
    modules = load_modules(weights, tiny=tiny or weights is None, seed=seed)
    wave, rate = generate_audio(
        modules,
        lyrics=lyrics,
        prompt=prompt,
        audio_duration=audio_duration,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )
    return write_wav(output, wave, rate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a MiniMax Music 3 song on Apple MLX")
    parser.add_argument("--lyrics", required=True, help="Lyrics with [verse]/[chorus] tags on their own lines")
    parser.add_argument("--prompt", required=True, help="Music description / structured caption")
    parser.add_argument("--output", default="song.wav")
    parser.add_argument("--duration", type=float, default=0.2, dest="audio_duration")
    parser.add_argument("--steps", type=int, default=2, dest="num_inference_steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", default=None, help="Converted MLX weight directory")
    parser.add_argument("--tiny", action="store_true", help="Use random tiny weights (no 25GB download)")
    args = parser.parse_args(argv)

    # Import mlx here so the entry point is auditable as an MLX path.
    import mlx.core as mx  # noqa: F401

    try:
        path = generate_song(
            lyrics=args.lyrics,
            prompt=args.prompt,
            output=args.output,
            audio_duration=args.audio_duration,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            weights=args.weights,
            tiny=args.tiny,
        )
    except Exception as exc:
        print(f"generate failed: {exc}", file=sys.stderr)
        return 1

    info = sf.info(str(path))
    print(
        json.dumps(
            {
                "output": str(path),
                "frames": int(info.frames),
                "channels": int(info.channels),
                "samplerate": int(info.samplerate),
                "backend": "mlx",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
