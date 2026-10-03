# Implementation and local usage

Nuri now has an executable PyTorch implementation of the agreed architecture.
It supports bidirectional encoding, recurrent latent reasoning, masked text
generation, and normalized sentence embeddings. Dense and sparse MoE presets
share the same public interfaces.

CPU tests and small synthetic training runs check correctness. They do not
establish language quality or A100/H100 throughput. The one-day training
budget in [ARCHITECTURE.md](ARCHITECTURE.md) remains a plan to measure on the
target GPU. There are no trained release weights in this repository.

## Set up the environment

From the repository root:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

Use Python 3.10 or newer. Install a CUDA-enabled PyTorch build on the training
machine. The Windows development environment used for the initial checks has
CPU-only PyTorch, so CUDA memory and fused attention performance are unmeasured.

## Train and compare tokenizer variants

The tokenizer tool reads UTF-8 text or JSONL objects with a `text` field and
an optional `lang` field. Plain text treats each nonempty line as a document.
It preserves case, accents, spacing, and Unicode with byte-level BPE. Its
special IDs are fixed: `[PAD]=0`, `[MASK]=1`, `[EOS]=2`, then `[SEP]` and the
system/user/assistant role tokens.

Set private input locations through environment variables:

```powershell
python tools/train_tokenizer.py `
  --format jsonl --eval-format jsonl `
  --output-dir runs/tokenizer --vocab-sizes 32768 49152 65536
```

`NURI_TOKENIZER_INPUT` supplies the training input and `NURI_TOKENIZER_EVAL`
supplies an optional held-out input. The explicit equivalents are
`--input-file` and `--eval-file`.

Each `vocab-N` folder contains `tokenizer.json`, the fast-tokenizer metadata,
special-token metadata, an instruction chat template, and `metrics.json`.
The report records the requested and actual vocabulary sizes and bytes per
token overall and by language. Larger bytes per token means fewer tokens for
the same bytes; evaluate each language separately. A tiny input may produce
fewer merges than requested, so use the **actual** vocabulary size in the model.
Metrics from the training input are labeled as such and must not be presented
as held-out results. Input filenames and sources are not saved in the report.

Create a token corpus with the selected tokenizer:

```powershell
python scripts/tokenize_corpus.py `
  --tokenizer $env:NURI_TOKENIZER `
  --out $env:NURI_TEXT_CORPUS $env:NURI_RAW_TEXT
```

The corpus consists of `tokens.bin` (`uint32` token IDs) and `offsets.npy`
(`int64` document boundaries). Prefix/span splitting uses these boundaries;
documents are not packed together for attention. Keep local corpora and their
provenance under `.private/` or outside this repository.

## Benchmark before choosing a training batch size

Use synthetic inputs to measure real work on the target GPU:

```powershell
python tools/benchmark_nuri.py `
  --preset release-dense --training --task denoise `
  --device cuda --dtype bfloat16 `
  --batch-size 8 --input-length 1024 --output-length 256 `
  --steps 4 --warmup 5 --iterations 20 `
  --output runs/benchmarks/dense-denoise.json

python tools/benchmark_nuri.py `
  --preset release-moe --task generate `
  --device cuda --dtype bfloat16 `
  --batch-size 1 --input-length 1024 --output-length 256 `
  --steps 4 --rounds 4 --warmup 5 --iterations 20 `
  --output runs/benchmarks/moe-generation.json
```

`--config` accepts an executable `NuriConfig` JSON such as a saved checkpoint's
`config.json`. The architecture declaration `configs/nuri_v0.design.json` has
a different purpose and is not accepted as a model config.

Available tasks are `mlm`, `denoise`, `embed-fast`, `embed-reasoning`, and
`generate`. Training benchmarks support MLM and denoising and include backward,
gradient clipping, and an AdamW step. Generation computes X/R and fixed memory
K/V once per request, then runs S canvas rounds. GPU timings synchronize CUDA;
reports include latency, input throughput, output sequence throughput, peak
allocated/reserved memory, T/S, dtype, parameter count, and sequence lengths.

Input throughput counts input tokens once per request. Output sequence
throughput counts a generated sequence once, rather than multiplying it by S.
These definitions differ from a count of all intermediate canvas positions.
Compare the same task, lengths, batch size, dtype, T, and S. The current MoE
backend loops over experts in Python and is labeled `portable_expert_loop`;
the matching active FFN width does not imply matching wall time.

A CPU smoke benchmark uses much smaller settings:

```powershell
python tools/benchmark_nuri.py --preset tiny --training `
  --input-length 8 --output-length 4 --warmup 1 --iterations 2
```

## Train the three stages

Base training combines input MLM and conditional span denoising. IT combines
natural instruction answers, generated structured reasoning examples, and
span denoising when a corpus is supplied. Embedding training starts from the
dense base and trains matched query/document modes with contrastive losses.

```powershell
python scripts/train.py --stage base --preset release-dense `
  --tokenizer $env:NURI_TOKENIZER --corpus $env:NURI_TEXT_CORPUS `
  --out runs/nuri-base --steps 20000 --batch-size 32 --save-every 2000

python scripts/train.py --stage it --init runs/nuri-base/final `
  --tokenizer $env:NURI_TOKENIZER --instructions $env:NURI_IT_DATA `
  --corpus $env:NURI_TEXT_CORPUS `
  --out runs/nuri-base-it --steps 3000 --batch-size 32

python scripts/train.py --stage embed --init runs/nuri-base/final `
  --tokenizer $env:NURI_TOKENIZER --pairs $env:NURI_PAIR_DATA `
  --out runs/nuri-base-embed --steps 3000 --batch-size 32
```

Use `release-moe` for the MoE base. Step counts and batch sizes above are
examples; choose them using the measured budget. IT JSONL objects contain
`messages` (role/content objects) and `response`. Pair JSONL objects contain
`query` and `positive`. No corpus sources are embedded in code or tracked files.

The target answer never enters the prompt's bidirectional memory. IT appends
EOS only to complete natural answers and excludes overlong answers. Base
spans receive EOS only when they reach the real document end. Random span
lengths do not train the length head.

The base stage uses the selected tokenizer's actual vocabulary size. Every
saved checkpoint receives a copy of its tokenizer files. IT/embed can use
that copy when `--tokenizer` is omitted. A mismatch in vocabulary size or
PAD/MASK/EOS IDs fails before training.

## Resume an interrupted stage

Use the private optimizer checkpoint to continue the same run:

```powershell
python scripts/train.py --stage base --preset release-dense `
  --tokenizer $env:NURI_TOKENIZER --corpus $env:NURI_TEXT_CORPUS `
  --out runs/nuri-base --steps 20000 --batch-size 32 --save-every 2000 `
  --resume runs/nuri-base/training_state/step-2000.pt
```

Keep the original data, stage, model, total step target, batch size, schedule,
and seed. `--steps` is the original total target, not a number of extra steps.
For an IT/embed continuation, supply the original stage and data flags.
`--init` starts a new stage from public weights; `--resume` restores the private
full-precision weights, AdamW moments, next step, cumulative token counts, and
the random states used for task/depth selection and data corruption. CPU and
CUDA random state are restored where applicable.

Resume reconstructs the model from the saved config; it does not guess the
current preset, and IT/embed resume does not require `--init`. A changed
optimization schedule, seed, task mix, or loss setting is rejected. Output
location and log/save cadence may change. Resuming a state that already reached
the total target reports completion without taking another optimizer step.

Public `model.safetensors` is exported in BF16 by default. It is suitable for
inference or a new training stage, while an exact continuation uses the private
training-state file. Keep `training_state/` private when creating release
packages. Tests compare an interrupted/resumed CPU run with an uninterrupted
run; this does not establish bitwise reproducibility across GPU backends.

## Generate text from a checkpoint

The inference CLI loads `config.json`, `model.safetensors`, and a compatible
`tokenizer.json`. The tokenizer defaults to the checkpoint folder; pass
`--tokenizer` if it is stored separately. Vocabulary size and special IDs must
match the checkpoint.

```powershell
# A base checkpoint requires an explicit canvas length.
python tools/infer_nuri.py --checkpoint runs/nuri-base/final `
  --tokenizer $env:NURI_TOKENIZER --text "Continue this short sentence:" `
  --canvas-len 32 --steps 4 --rounds 4

# An IT checkpoint can use its trained length head.
python tools/infer_nuri.py --checkpoint runs/nuri-base-it/final `
  --tokenizer $env:NURI_TOKENIZER --chat `
  --text "Explain why the sky looks blue." --max-len 128
```

`--chat` formats the input as a user turn ending with the assistant role tag.
Use it with an instruction-tuned checkpoint. Repeat `--text` to batch inputs.
Inputs that exceed the checkpoint's limit fail clearly; `--truncate` explicitly
keeps the last tokens for chat or first tokens for plain input.

The result contains decoded answers, token IDs, canvas lengths, and whether
each canvas contains EOS. EOS is trimmed only after the final round. A result
without EOS uses its full canvas; its canvas length is a generation limit,
not a guarantee of a complete sentence. The sampler never emits MASK/PAD.

`--steps` controls latent depth T and `--rounds` controls output refinement S.
They are separate settings. Higher values increase compute and are not a
guarantee of better answers.

## Create sentence embeddings

```powershell
python tools/infer_nuri.py --checkpoint runs/nuri-base-embed/final `
  --tokenizer $env:NURI_TOKENIZER --task embed --mode reasoning `
  --steps 4 --dim 128 --text "A first sentence." --text "Another sentence." `
  --output runs/embeddings.json
```

`fast` pools X directly. `reasoning` combines pooled X with the recurrent
representation. Both return unit-length vectors, including after selecting
a prefix with `--dim`. Release embedding training supervises 64/128/256
dimensions. Use the same mode for queries and indexed documents; similarity
across modes needs its own alignment evaluation.

## Build and verify a local release folder

The export tool needs Transformers in addition to the core dependencies:

```powershell
python -m pip install transformers
python scripts/export_release.py --checkpoint runs/nuri-base-it/final `
  --out runs/release/nuri-base-it --name nuri-base-it --stage it
```

Supply `--tokens`, `--gpu`, and `--gpu-hours` when those facts are known.
Use `--stage base`, `it`, or `embed` to identify the trained checkpoint.
An IT export requires the trained-length marker. Tokenizer files come from
the checkpoint unless `--tokenizer` selects another compatible location.

The folder includes BF16 safetensors, the model config and Python files, the
tokenizer, the MIT license, and an English model card. A verification JSON
report sits beside the output folder. The tool reloads with Nuri and
Transformers, compares logits and token predictions, and checks generation
and normalized embeddings. A failed check exits with an error. Export creates
local files; publishing is a separate operation.

```python
from transformers import AutoModel, AutoTokenizer

folder = "runs/release/nuri-base-it"
tokenizer = AutoTokenizer.from_pretrained(folder)
model = AutoModel.from_pretrained(folder, trust_remote_code=True).eval()
batch = tokenizer(["A short prompt."], return_tensors="pt", padding=True)
result = model.generate_refine(batch.input_ids, batch.attention_mask, T=4, S=4)
answer = tokenizer.decode(result.tokens[0], skip_special_tokens=True)
vectors = model.encode(batch.input_ids, batch.attention_mask, mode="reasoning", dim=128)
```

The exported classes also support `AutoModelForMaskedLM` and
`AutoModelForSequenceClassification`. Classification creates a downstream
head when one is absent from the base checkpoint; train it before using its
predictions. The separate `structure_logits(r)` head reads only the latent
state for structure-check experiments. The ordinary classifier can read X/R.
The embedding and generation helpers above accept token IDs; the tokenizer
converts text at the call site. Standard autoregressive `generate()` is not
the masked refinement API.

AutoClass configuration and relative-code packaging follow the
[Transformers custom-model interface](https://huggingface.co/docs/transformers/custom_models).
The actual exported-folder load was also tested in a fresh Python process
with its tokenizer, embeddings, and generation helper.

## Evaluate a checkpoint

Run the generation diagnostics on an IT checkpoint:

```powershell
python scripts/evaluate.py --checkpoint runs/nuri-base-it/final `
  --suites reasoning curves interventions --examples 300 `
  --output runs/eval/nuri-base-it.json

python scripts/evaluate.py --checkpoint runs/nuri-base/final `
  --suites mlm diagnostics --corpus $env:NURI_EVAL_CORPUS `
  --output runs/eval/nuri-base.json

python scripts/evaluate.py --checkpoint runs/nuri-base-embed/final `
  --suites retrieval --pairs $env:NURI_EVAL_PAIRS `
  --output runs/eval/nuri-base-embed.json
```

Use a separate evaluation corpus/pair file. The generated reasoning tasks use
the task generator's evaluation split. Tokenizer files default to the checkpoint
folder and must match its vocabulary and special IDs.

The generation report includes exact token match, missing EOS, and canvas
length errors. It compares predicted and oracle lengths for IT; base models
use oracle length because their length head is untrained. The T curve uses
one checkpoint at T=1/2/4/8/16. The S curve uses S=1/2/4/8. The R interventions
zero the latent state or swap it between samples before length prediction and
canvas readout. Use oracle length when you need to separate readout behavior
from the effect on the length head.

Retrieval reports Recall@k and MRR for matched query/document modes and supported
embedding prefixes. Core diagnostics report state/update norms, slot cosine,
and effective rank per recurrent step. Diagnostics characterize behavior;
changing states, higher confidence, or unchanged floor-level accuracy do not
establish reasoning. Depth curves also do not replace an independently trained
comparison of recurrence. Mark inference depths outside the trained set when
interpreting the report.

## What is implemented and what still needs experiments

Implemented: pre-norm RMSNorm, bias-free projections, SwiGLU, RoPE on input
and canvas self-attention, optional QK RMS normalization, shared gated latent
recurrence, differentiable fixed-memory caches, masked-only vocabulary
projection, conditional canvas denoising, greedy confidence remasking, trained
length prediction, dense/MoE presets, and fast/reasoning Matryoshka embeddings.
Release presets use QK normalization and random training depth `{2,4,8}`.

MoE routing is dropless, top-k probabilities are renormalized, shared experts
run for all valid tokens, and shared experts are excluded from router losses.
Router math stays FP32 inside an autocast-disabled scope. Padding is excluded
from pooling, memory attention, and expert routing.

Adaptive halting, GQA/MLA, local/global schedules, grouped hypothesis slots,
draft feedback, sequence packing/unpadding, grouped-GEMM dispatch, and FP8
remain separate experiments. Their effectiveness is not established by having
the surrounding architecture in place.

Tests cover padding invariance, reused-cache gradients, masked targets,
generation behavior, tokenizer round trips, checkpoint reload, and synthetic
training. The actual text CLI pipeline has also been exercised from tokenizer
training through base/IT/embed training and checkpoint-based inference. The
generated text in this short smoke run is only a pipeline check.
An A100/H100 run is still needed to establish the real batch size,
throughput, language quality, and whether changing T/S improves task results.
