# Performance expectations

Date: October 4, 2026. Release training has not started. The completed GPU
comparison and batch-capacity follow-up measure synthetic training computation,
not model quality or response latency. See [GPU_COMPARISON.md](GPU_COMPARISON.md)
and [GPU_CAPACITY.md](GPU_CAPACITY.md) for measurements.

## What to expect first

Nuri is initially a small experimental encoder with dedicated embedding
training and a bounded masked generator. Task-specific retrieval, useful
features for a labeled classifier, and short structured answers are plausible
first uses to test. Broad conversation, extensive factual knowledge, long
coherent answers and reliable multi-step reasoning are uncertain goals.
There is no measured accuracy, F1, retrieval score or answer-quality percentage.

| Use | Reasonable first goal | What must be checked |
|---|---|---|
| Understanding | Learn useful text features and reduce held-out masked-token loss | A classification head needs a labeled probe or fine-tune; the default release has no trained classification head |
| Embeddings | Improve same-mode retrieval on a specified task | Candidate difficulty, valid positives, fast/reasoning modes and 64/128/256 dimensions |
| Generation | Produce short answers with constrained formats or answers grounded in the input | Predicted length, wording, correctness, EOS and instruction compliance |
| Reasoning | Test whether latent state helps bounded multi-step tasks | Task baselines, normal/zero/shuffled state, depth and difficulty curves |

The response canvas contains at most 256 tokenizer positions, including EOS.
An EOS-terminated answer therefore has at most 255 content tokens. Token
counts are not word or character counts. A base checkpoint requires a supplied
canvas length; automatic length is trained in the instruction stage.

The release presets contain 150,652,931 dense parameters and 242,739,203 total
MoE parameters. Some weights serve the recurrent core and generator, so these
totals are not equivalent to a pretrained encoder of the same headline size.
Matching active FFN projection work also does not match router/dispatch cost.
More MoE weights do not establish better quality than dense.

## What the measured budget suggests

For the measured A100 batch-64 candidates, the stage-budget calculation gives:

| Current D-017 base window | Input positions | Canvas positions | Combined positions | Status |
|---|---:|---:|---:|---|
| Dense, 9 hours | about 3.024B | about 0.378B | about 3.402B | Synthetic estimate |
| Portable loop MoE, 7 hours | about 1.934B | about 0.242B | about 2.176B | Synthetic estimate |

These estimates use microbatch 64 without checkpointing, input length 1,024,
canvas length 256, equal MLM/span task weights, equal weights on measured
T=2/4/8 mean latency, and an assumed 0.9 usable-time factor.
They count repeated processing exposures. They are not unique text tokens,
loss-bearing target counts, generated output or a quality prediction.
The benchmark excludes data loading and host-side batch building, and uses
approximate corruption workloads.
The historical eight-hour dense proxy exceeded 3B combined positions by only
0.798%. D-017 keeps the dense size, starts from nine/seven hours and moves time
from MoE to dense if the first real-data valid-token projection falls short.
Real-data overhead and padding mean even the new synthetic projection does
not establish 3B valid training positions within the shared day.
Fresh-process batch fit does not establish mixed-stage fit; embedding pairs
at T=8 allocate 74.13 GiB on a device reporting 79.25 GiB. Checkpointing was
not measured because no OOM triggered the conditional cases. Real data,
variable lengths and gradient accumulation still need matching calibration
before fixing release optimizer steps. The earlier batch-8 comparison remains
historical evidence, not the current intended-batch forecast.

A small budget can still support useful encoder learning.
[Cramming](https://proceedings.mlr.press/v202/geiping23a.html) demonstrated
an optimized scratch MLM trained on one consumer GPU in one day. Nuri splits
the day between two bases, their instruction stages, an embedding stage and
overhead, and adds generation/recurrent components. That result supports
feasibility, not a prediction of matching quality.

[ModernBERT](https://arxiv.org/html/2412.13663v2) reports training on 2T tokens.
Its scale, recipe and topology differ from Nuri. Similar parameter counts do
not establish comparable quality, and the training-scale gap is not a quality
ratio. Quality must be measured on the intended task and language.

## How to judge the first trained checkpoints

Freeze held-out groups, candidate pools, tokenizer, mask seeds, metric
definitions and task baselines before final evaluation. Use development data
to select checkpoint and T/S settings. Report Korean, English and Japanese
separately where sample counts support it, together with exclusions and
uncertainty. Training source records stay private.

| Area | First report | Current implementation |
|---|---|---|
| MLM and span reconstruction | Held-out CE/accuracy, mask rate and target counts; distinguish partial targets from a fully masked canvas | Masked-token and denoising suites exist |
| Retrieval | Recall@1/5/10 and MRR, identical candidate pools, fast/reasoning modes and each embedding dimension | Retrieval scoring exists; it assumes one row-aligned gold candidate |
| Short generation | Predicted-length exact match for IT; oracle-length diagnostic; EOS, coverage and format checks | Synthetic conditions and an optional full-file held-out `instructions` CLI suite exist; both report missing/early EOS, canvas and returned-answer length MAE, eligible coverage and over-cap counts. Oracle reports include `length_only_accuracy` (the best exact match from answer length alone); exact match does not measure open-ended semantic quality |
| Latent reasoning | Same-checkpoint T/S curves and normal/zero/shuffled state above frozen task baselines | Curves, state interventions and a halting curve (accuracy vs mean core steps) exist on a `shared` split (training name pools, new combinations) plus an `eval` stress split (unseen names); `scripts/ablate.py --seeds` compares variants with mean/std; none of these is an independently trained release-scale comparison |
| Classification | A labeled probe with fixed splits, majority baseline, accuracy and macro-F1 | Additional analysis is needed; there is no classification release stage |
| Serving speed | Complete embedding/generation call, representative lengths and batches, actual T/S, mean/median/p95 and memory | No trained-model inference latency has been measured |

Additional reporting should include frozen majority/chance controls where
meaningful, grouped uncertainty and blinded review
of short natural instructions. These are proposed analysis additions, not
existing results. Exact match alone misses legitimate alternative wording.

The first useful quality claim should identify a specific task, a frozen
task-relevant baseline, its improvement and inference cost. Beating random
initialization demonstrates learning but does not establish product quality.
Increasing T or S must show a measured benefit; weight sharing alone does not
prove that more computation improves answers.
