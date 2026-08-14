# MiniMax Music 3 on MLX

Local Apple Silicon runtime for [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3). Not ComfyUI, not CUDA, not mlx-vlm.

Pipeline: Qwen3 global AR → local RVQ depth decoder → fused hidden-state condition → flow-matching Euler DiT → DAC vocoder → stereo WAV (44.1 kHz).

## Setup

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Generate (tiny random weights)

```bash
python -m minimax_music3_mlx.generate \
  --tiny \
  --lyrics $'[verse]\nMorning light filtering through the pine\n[chorus]\nSoftly the world begins to breathe' \
  --prompt "Genre: acoustic pop. BPM: 96. Warm female vocal." \
  --duration 0.2 \
  --output song.wav
```

## Official weights

Diffusers-only subset (~25 GB). Do not download `qwen_7B/`, `dav.pth`, or `flowmatching_vae.pth`.

```bash
hf download MiniMaxAI/MiniMax-Music3 \
  --include "modular_model_index.json" \
  --include "language_model/*" "transformer/*" "vocoder/*" \
  --include "rvq_depth_decoder/*" "condition_encoder/*" \
  --include "tokenizer/*" "scheduler/*" \
  --local-dir ./MiniMax-Music3

python -m minimax_music3_mlx.convert --src ./MiniMax-Music3 --dst ./weights/mlx

# Tight disk: convert shard-by-shard and delete each source file after it is written
# python -m minimax_music3_mlx.convert --src ./MiniMax-Music3 --dst ./weights/mlx --delete-source

python -m minimax_music3_mlx.generate \
  --weights ./weights/mlx \
  --lyrics $'[verse]\n...\n[chorus]\n...' \
  --prompt "..." \
  --duration 8 \
  --steps 30 \
  --output song.wav
```

`--duration` is an upper bound in seconds (25 AR frames/s). Structure tags such as `[verse]` must be on their own line.

## Tests

```bash
pytest -q
```

## License

Inference code in this repository is Apache-2.0. MiniMax-Music3 **weights** remain under the [MiniMax-Music3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-Music3). See `NOTICE`.

Commercial products that ship the model must display “MiniMax-Music3”. Yearly revenue above USD 20M needs a separate MiniMax authorization.
