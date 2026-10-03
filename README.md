# Nuri

Nuri is an **encoder-only language model** that can also reason in a hidden
"scratchpad" and write text, without an autoregressive decoder.

- It **understands** text like a BERT-style encoder: classification, MLM and
  sentence embeddings.
- It **thinks** by updating a small set of hidden vectors (latent slots) several
  times with the same block. You choose how many times (`T`) at inference.
- It **writes** by filling a masked answer in parallel and then fixing the
  least confident words over a few rounds (`S`), instead of writing one token
  after another.

Languages: Korean and English first, some Japanese. License: MIT.

> Status: the model code, training loop and tests are implemented and run on
> CPU. No model has been trained yet. Numbers in the design documents are
> estimates until the GPU benchmark is done.

## How it works

```
prompt tokens ──► Input Encoder (bidirectional, L layers) ──► memory X
                                                               │
learned slots R0 ──► Reasoning Core: one shared block × T ◄────┘  (reads X every step)
                                     │
                                     ▼ R
answer canvas [MASK MASK ...] ──► Readout (2 layers, reads [X; R]) ──► words
        ▲                                                              │
        └──────── re-mask the least confident words, repeat S times ◄──┘
```

1. **Input encoder.** A modern bidirectional Transformer (RMSNorm, RoPE,
   SwiGLU, QK-norm, no bias). Its output `X` is the read-only memory.
2. **Reasoning core.** `K` learned slots attend to `X`, to each other, and go
   through an FFN. The same block runs `T` times, so thinking longer costs no
   extra parameters. The input memory's keys and values are computed once and
   reused for every step.
3. **Readout.** The answer starts as all `[MASK]`. Each round predicts every
   masked word at once, keeps the confident ones and re-masks the rest
   (Mask-Predict). `X` and `R` are computed once per request.
4. **Heads.** MLM on `X`, sentence embeddings (`fast` = from `X`,
   `reasoning` = from `X` and `R`), an answer length head, and an optional
   classification head.

The full design, with every decision and its reason, is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Models we plan to release

| Name | What it is |
|---|---|
| `nuri-tokenizer` | byte-level BPE tokenizer for Korean, English and Japanese |
| `nuri-base` | ~150M dense model, pretraining only |
| `nuri-base-moe` | ~242M MoE model (~150M active per token), pretraining only |
| `nuri-base-it` / `nuri-base-moe-it` | instruction-tuned versions |
| `nuri-base-embed` | sentence embedding model built from `nuri-base` |

All of them are trained on one A100 80GB or H100 in about one GPU-day in total.
Intermediate checkpoints are published as revisions.

## Install

```bash
pip install -e ".[dev]"
python -m pytest          # ~80 tests, a few seconds on CPU
```

Python 3.10+ and PyTorch 2.4+ are required. Training uses bf16 on CUDA and
falls back to fp32 on CPU.

## Train

All data paths are passed as flags or environment variables.

```bash
# 1. Train tokenizer variants and compare bytes per token per language
python tools/train_tokenizer.py --input-file $NURI_TOKENIZER_INPUT --format jsonl \
    --output-dir runs/tokenizer --vocab-sizes 32768 49152 65536

# 2. Tokenize raw text into a corpus folder (tokens.bin + offsets.npy)
python scripts/tokenize_corpus.py --tokenizer runs/tokenizer/vocab-65536 --out corpus/ text.jsonl

# 3. Base stage: MLM + text span denoising
python scripts/train.py --stage base --preset release-dense --corpus corpus/ \
    --out runs/nuri-base --steps 20000 --batch-size 64 --save-every 2000

# 4. Instruction tuning (adds synthetic reasoning tasks automatically)
python scripts/train.py --stage it --init runs/nuri-base/final --tokenizer runs/tokenizer/vocab-65536 \
    --instructions it.jsonl --corpus corpus/ --out runs/nuri-base-it --steps 3000

# 5. Embedding training
python scripts/train.py --stage embed --init runs/nuri-base/final --tokenizer runs/tokenizer/vocab-65536 \
    --pairs pairs.jsonl --out runs/nuri-base-embed --steps 3000
```

Use `--preset release-moe` in step 3 for the MoE model. Data formats:

- instructions: `{"messages": [{"role": "user", "content": "..."}], "response": "..."}`
- pairs: `{"query": "...", "positive": "..."}`

Each run writes `train_log.jsonl` (loss, LR, tokens, MoE expert load, ...),
public checkpoints (`step-N/`, `pre-decay/`, `final/`) and private resume
state (`training_state/`).

## Use a checkpoint

```python
import torch
from nuri.checkpoint import load_pretrained
from nuri.generation import generate

model = load_pretrained("runs/nuri-base-it/final").eval()
prompt = torch.tensor([[...]])        # token ids from the tokenizer

# Generate: T = thinking steps, S = refinement rounds
result = generate(model, prompt, steps=4, rounds=4)
print(result.tokens)                  # answer ids, cut at the first EOS

# Base models have no trained length head: give the answer length yourself
result = generate(base_model, prompt, canvas_len=32)

# Sentence embeddings (unit length); `dim` picks a Matryoshka prefix
z = model.embed(prompt, mode="reasoning", steps=4, dim=128)
```

## Repository layout

| Path | Contents |
|---|---|
| `nuri/config.py` | `NuriConfig` and the presets (`reference`, `release-dense`, `release-moe`, `tiny`) |
| `nuri/layers.py` | RMSNorm, RoPE, attention, SwiGLU |
| `nuri/moe.py` | sparse MoE FFN and the dense/MoE factory |
| `nuri/encoder.py`, `nuri/core.py`, `nuri/readout.py` | the three stages of the model |
| `nuri/model.py` | `NuriModel`: assembles everything and the heads |
| `nuri/generation.py` | the refinement sampler |
| `nuri/corruption.py`, `nuri/data.py`, `nuri/tasks.py` | masking, corpus format, batch builders, synthetic reasoning tasks |
| `nuri/losses.py`, `nuri/trainer.py`, `nuri/pipeline.py`, `nuri/schedule.py`, `nuri/checkpoint.py` | training |
| `scripts/` | `train.py`, `tokenize_corpus.py` |
| `tools/` | tokenizer training, benchmark, the shared-file watcher |
| `docs/` | design spec, decision log, agent discussion |

## How this project is built

Nuri is designed and implemented by two AI agents (Claude and Codex) working
in the same repository under the owner's direction. They coordinate through
[docs/DISCUSSION.md](docs/DISCUSSION.md), review each other's commits, and
follow the rules in [AGENTS.md](AGENTS.md).

## License

MIT. See [LICENSE](LICENSE).
