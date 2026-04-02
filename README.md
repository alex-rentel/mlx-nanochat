# mlx-nanochat

**Train your own ChatGPT on Apple Silicon. Full pipeline: data → pretrain → SFT → RL → chat.**

A proper MLX port of [Karpathy's nanochat](https://github.com/karpathy/nanochat) — the minimal, end-to-end ChatGPT training pipeline.

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> *"The goal of nanochat is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000."* — Andrej Karpathy

## What this is

A single-file-ish, readable, hackable codebase that trains a ChatGPT-class model from scratch on your Mac. No PyTorch. No CUDA. Just MLX and Apple Silicon.

```
Raw data → Tokenizer → Pretrain → SFT → GRPO (RL) → Chat
   All on your Mac. All in one repo. ~$0 in cloud costs.
```

## One complexity dial

```bash
python train.py --depth=4    # Tiny model, 10 minutes, fits 8GB Mac
python train.py --depth=12   # Medium model, 2 hours, needs 16GB
python train.py --depth=24   # GPT-2 class, 8 hours, needs 64GB
python train.py --depth=32   # Best quality, 24 hours, needs 64GB+
```

`--depth` automatically sets: model width, attention heads, learning rate, batch size, training duration. You don't configure anything else.

## What's different from the original

| | karpathy/nanochat | mlx-nanochat |
|---|---|---|
| Backend | PyTorch + CUDA | **MLX** (Apple native) |
| Hardware | 8x H100 for GPT-2 class | **1x M1 Max** (slower but works) |
| Flash Attention | FA3 kernels | **MLX additive masks** (functionally equivalent) |
| Optimizer | Muon + AdamW | **AdamW** (Muon MLX port in progress) |
| Training cost | ~$100 (8x H100 rental) | **$0** (your Mac) |
| Full pipeline | data → pretrain → SFT → GRPO → chat | **Same** |

## Full Pipeline

```bash
# 1. Download training data (FineWeb subset)
python -m nanochat_mlx.dataset -n 8     # 8 shards, ~800MB

# 2. Train BPE tokenizer
python -m scripts.tok_train              # vocab_size=32768

# 3. Pretrain base model
python -m scripts.train --depth=12       # ~2 hours on M1 Max

# 4. Supervised fine-tuning (SmolTalk)
python -m scripts.sft --depth=12         # ~30 minutes

# 5. Chat with your model
python -m scripts.chat --depth=12 --interactive

# 6. Evaluate
python -m scripts.chat_eval --depth=12   # ARC, MMLU, GSM8K
```

Or use the web GUI:
```bash
python -m scripts.quickstart
# Open http://127.0.0.1:8000
```

## Why this matters for Eden

mlx-nanochat is the **training backbone** for [eden-models](https://github.com/alex-rentel/eden-models). We modify the architecture to specialize for tool-calling:

```
nanochat architecture (general chat)
    + tool-calling training data (50K examples)
    + Eden's tool format (<tool_call> tags)
    = Eden-1B (1-bit tool-calling specialist)
```

## Training Performance

| Mac | depth=4 | depth=12 | depth=24 |
|---|---|---|---|
| M1 8GB | ~10 min | ~4 hrs | OOM |
| M1 Max 64GB | ~5 min | ~2 hrs | ~8 hrs |
| M4 16GB | ~4 min | ~1.5 hrs | ~10 hrs |
| M4 Max 128GB | ~3 min | ~45 min | ~4 hrs |

## Credits

**Original concept and implementation:** [Andrej Karpathy](https://github.com/karpathy/nanochat) (MIT License)

The nanochat architecture, scaling laws, training methodology, and the "one complexity dial" design are all Karpathy's work. This is a community MLX port adapted for Apple Silicon training.

**Also referenced:**
- [scasella/nanochat-mlx](https://github.com/scasella/nanochat-mlx) — Earlier MLX port
- [NeuroArchitect/nanochat-mlx](https://github.com/NeuroArchitect/nanochat-mlx) — Another MLX port
- [Doriandarko/MLX-GRPO](https://github.com/Doriandarko/MLX-GRPO) — GRPO training on MLX
- Apple MLX team — framework and examples

## Related

- [mlx-autoresearch](https://github.com/alex-rentel/mlx-autoresearch) — Autonomous research loops on Mac
- [mlx-turboquant](https://github.com/alex-rentel/mlx-turboquant) — KV cache quantization for MLX
- [eden-models](https://github.com/alex-rentel/eden-models) — Training pipeline using nanochat for tool-calling LLMs

## License

MIT — matching the original nanochat license.
