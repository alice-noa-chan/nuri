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

### Optional training memory and compilation settings

`--corpus` accepts one folder or several `folder=weight` arguments. Keep
private folder locations in environment variables; the weights control which
corpus supplies each sampled document. A single unweighted folder preserves
the original sampling behavior. Release plans can supply a list of these
arguments under a stage's `corpus` key.

`--pack-mlm` optionally fills base-stage MLM rows with multiple documents.
Segment masks keep attention within each document, including local attention
layers; span batches keep their original format. Progress token counts include
valid positions rather than padding. Packing is opt-in, and its GPU throughput
gain is unmeasured. Resume checks the saved batch construction, including the
pack flag, when this metadata is present in the private state.

`--grad-accum N` averages N microbatch losses before one optimizer update.
`--batch-size` remains the size of a microbatch; the progress target and
`--steps` count optimizer updates. This reduces the memory needed to accumulate
more examples, but contrastive negatives still come from each individual
microbatch. Accumulating embedding batches does not create the negative pool
of one larger contrastive batch. Keep the accumulation count on resume because
it changes the gradients, random draws and number of processed examples.

`--gradient-checkpointing` recomputes input/readout block activations during
backward instead of retaining all of them. It trades extra computation for
lower activation memory; the shared recurrent core is not checkpointed.
MoE auxiliary records are collected only on the original forward, so
recomputation does not duplicate the router loss or expert-load counts.

`--compile` asks PyTorch to compile the encoder blocks, recurrent core and
readout blocks in place, preserving parameter and checkpoint names. It is
opt-in. The first forward can take longer while compilation runs. This path
has not been validated on the target A100/H100, and no speed gain is claimed;
measure it with the actual sequence lengths and task mix before using it for
the one-day run. Activation checkpointing and compilation are runtime options
that may change on resume; numerical equivalence across GPU backends is not
guaranteed.

## Monitor training and view TensorBoard

The CLI prints which preparation step is running: loading the tokenizer,
creating/loading the model, opening the corpus, and preparing the task mix.
Separate tqdm bars count instruction, structured-reasoning and pair examples
as they are tokenized. The optimizer loop then shows the release stage
(`base`, `it`, `embed`), sampled task, active reasoning depth T, WSD phase
(`warmup`, `stable`, `decay`), and current operation (`batch`, `forward`,
`backward`, `optimizer`, or `saving <checkpoint>`).

The training bar counts **completed optimizer updates**, with percentage,
elapsed time, speed and estimated time remaining. It starts at the saved
number of completed updates when resuming. The postfix shows the most recent
sampled loss, learning rate, gradient norm before clipping, cumulative
processed tokens, and tokens/second. Task-specific denoising, length, fast/
reasoning contrastive losses and MoE balance/z terms appear when available.
Terminal refreshes are throttled; initial-step operations and checkpoint saves
are displayed immediately. Metrics refresh every 10 updates by default, with
the first and last update always recorded. Use `--log-every 1` to sample every
update. Bars go to stderr, while setup/completion messages go to stdout.

TensorBoard is optional. Install its extra and enable it on any stage:

```powershell
python -m pip install -e ".[tensorboard]"

python scripts/train.py --stage base --preset release-dense `
  --tokenizer $env:NURI_TOKENIZER --corpus $env:NURI_TEXT_CORPUS `
  --out runs/nuri-base --steps 20000 --batch-size 32 `
  --save-every 2000 --log-every 10 --tensorboard

tensorboard --logdir runs
```

Open the local address printed by TensorBoard, usually `http://localhost:6006`.
The default enabled event directory is `<out>/tensorboard`. To choose another
directory, pass `--tensorboard-dir runs/metrics/base-dense`; this flag enables
events even without `--tensorboard`. Use a separate directory for each run or
stage. `--no-progress` hides all tqdm bars and keeps setup messages, JSON logs
and any enabled TensorBoard events. TensorBoard is never imported when events
are disabled; an enabled run with a missing dependency gives the install command.

| TensorBoard tag | Meaning |
|---|---|
| `loss/total`, `loss/<task>/total` | The optimized loss, including active weighted length and MoE terms; compare individual task curves because objectives have different scales. |
| `loss/<task>/denoise`, `/length`, `/fast`, `/reasoning` | Available unweighted objective components for the sampled task. |
| `optimizer/lr`, `optimizer/grad_norm` | Current learning rate and gradient norm before clipping. |
| `data/processed_tokens` | Cumulative valid input/canvas positions, or query/document positions for pairs, preserved on resume. Padding and repeated recurrent steps do not add tokens. |
| `batch/<kind>/targets` | Targets in the sampled batch: masked tokens for MLM/canvas tasks, positive pairs for embedding tasks. These units stay separate. |
| `performance/tokens_per_second`, `/steps_per_second` | Work since the previous metric sample in this process divided by monotonic elapsed time. Includes batch building, logging and intervening saves. The first resumed sample counts only new work. |
| `train/reasoning_depth`, `train/schedule_phase` | Active T (0 for MLM, which skips the core), and phase code 0=warmup, 1=stable, 2=decay. |
| `train/reasoning_depth_mean`, `/min`, `/max` | Numeric depth summaries when an accumulated update samples several depths, rather than claiming one common T. |
| `moe/balance`, `moe/z` | Router auxiliary terms when MoE is active. |
| `moe/layer_<n>/expert_<e>_assignments` | Routed assignments for each expert in each auxiliary record of the sampled batch. Compare these within the same task. |

JSON and TensorBoard use the same sampling cadence. The existing JSON `step`
stays zero-based; TensorBoard's horizontal step is `step + 1`, the number of
completed optimizer updates. Events contain explicit numeric measurements;
raw input text, token IDs, data paths and full training settings are not copied
into events. The writer flushes periodically and closes on completion or an
exception, so the final available measurements remain readable.

With gradient accumulation, loss components are means over the microbatches
that contain that component, while target and expert-assignment counts are
summed over the update. A mixed-task update gets a `loss/mixed/total` curve
rather than assigning its mean to one individual task. The bar also shows the
current microbatch index/count and its task/depth; it advances only after the
optimizer update.

To continue one graph after interruption, keep the original event directory
and add the usual `--resume` flag. Events after the restored checkpoint are
hidden and replaced by resumed measurements; older points remain. A new event
directory starts at the restored step without copying historical points.
Changing progress visibility, event destination or log cadence is allowed on
resume. An incompatible optimization setting is rejected before opening the
writer, so it cannot purge valid event history. A finished checkpoint produces
no extra optimizer steps or metric points.

The implementation uses [tqdm](https://tqdm.github.io/docs/tqdm/) and PyTorch's
[SummaryWriter](https://docs.pytorch.org/docs/stable/tensorboard.html).

The multi-stage release runner accepts boolean `tensorboard` and `no_progress`
options under the plan's `train` object, with per-stage overrides. Value options
such as `log_every`, `tensorboard_dir` and `grad_accum` belong to each stage.
Use a private plan based on `configs/release_plan.example.json`:

```powershell
python scripts/run_release.py .private/release-plan.json --dry-run
python scripts/run_release.py .private/release-plan.json
```

With plan-wide `"train": {"tensorboard": true}`, each stage uses its own
`<stage output>/tensorboard` directory. The runner invokes training, requested
evaluations and local export in sequence and continues interrupted stages.

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
location, progress/event options and log/save cadence may change. Resuming a state that already reached
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
  --suites mlm denoise diagnostics --corpus $env:NURI_EVAL_CORPUS `
  --output runs/eval/nuri-base.json

python scripts/evaluate.py --checkpoint runs/nuri-base-embed/final `
  --suites retrieval --pairs $env:NURI_EVAL_PAIRS `
  --output runs/eval/nuri-base-embed.json
```

Use a separate evaluation corpus/pair file. The generated reasoning tasks use
two reported splits: `shared` reuses the training name/node pools with newly
sampled chains, graphs and expressions; `eval` uses disjoint names as a stress
test. Curves and R interventions use `shared` so an unseen answer vocabulary
does not dominate the reasoning comparison. A distant seed creates new samples,
but it does not guarantee that every sampled problem is absent from training.
Tokenizer files default to the checkpoint folder and must match its vocabulary
and special IDs.

The `denoise` suite measures held-out span reconstruction through the recurrent
core and readout: per-sample cross-entropy and masked-token accuracy at several
mask ratios and reasoning depths. MLM accuracy measures the input encoder and
vocabulary head, so report both when evaluating a base checkpoint.

The generation report includes exact token match, missing EOS, and canvas
length errors. It compares predicted and oracle lengths for IT; base models
use oracle length because their length head is untrained. Over-limit answers
are excluded and counted. Oracle reports include a length-only answer baseline,
since the supplied answer length can reveal the label. The T curve uses
one checkpoint at T=1/2/4/8/16. The S curve uses S=1/2/4/8. The R interventions
zero the latent state or swap it between samples before length prediction and
canvas readout. Use oracle length when you need to separate readout behavior
from the effect on the length head.

Retrieval reports Recall@k and MRR for matched query/document modes and supported
embedding prefixes, with stable document order used to break score ties.
R-shuffle batches contain at least two eligible samples; a singleton tail is
merged so every sample receives another sample's state. Core diagnostics report state/update norms, slot cosine,
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

Optional local/global input schedules use `attention_schedule` and
`local_radius`; the implementation uses a dense attention mask, so no speedup
is claimed. Greedy refinement remains the default, with optional temperature/
top-k sampling available through the generation API. These options, activation
checkpointing and compilation need measurements before release training.

Adaptive halting, GQA/MLA, grouped hypothesis slots,
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
