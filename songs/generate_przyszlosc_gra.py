#!/usr/bin/env python3
"""Generate Przyszłość Gra via MiniMax Music 3 MLX."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path("/Users/mikomac/Developer/OSS/minimusic")
SONGS = ROOT / "songs"
LYRICS = (SONGS / "przyszlosc-gra.lyrics.txt").read_text(encoding="utf-8")
PROMPT = (SONGS / "przyszlosc-gra.prompt.txt").read_text(encoding="utf-8").strip()
OUTPUT = SONGS / "przyszlosc-gra.wav"
WEIGHTS = ROOT / "weights" / "mlx-8bit"

sys.path.insert(0, str(ROOT / "src"))

from minimax_music3_mlx.generate import generate_song  # noqa: E402


def main() -> int:
    print(f"lyrics chars: {len(LYRICS)}", flush=True)
    print(f"prompt: {PROMPT[:120]}...", flush=True)
    print(f"weights: {WEIGHTS}", flush=True)
    print(f"output: {OUTPUT}", flush=True)
    started = time.time()
    path = generate_song(
        lyrics=LYRICS,
        prompt=PROMPT,
        output=OUTPUT,
        audio_duration=90.0,
        num_inference_steps=30,
        seed=42,
        weights=WEIGHTS,
        tiny=False,
    )
    elapsed = time.time() - started
    print(json.dumps({"output": str(path), "elapsed_s": round(elapsed, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
