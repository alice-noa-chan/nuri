# Performance expectations

Date: October 4, 2026. Release training has not started. The completed GPU
comparison measures synthetic training computation, not model quality or
response latency. See [GPU_COMPARISON.md](GPU_COMPARISON.md) for measurements.

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

For the tested A100 shapes, the existing stage-budget calculation gives:

| Eight-hour base window | Input positions | Canvas positions | Status |
|---|---:|---:|---|
| Dense | about 1.995B | about 0.249B | Estimate |
| Portable loop MoE | about 1.267B | about 0.158B | Estimate |

These estimates use microbatch 8, input length 1,024, canvas length 256,
equal weights on measured T=2/4/8 mean latency, and a 0.9 usable-time factor.
They count repeated processing exposures. They are not unique text tokens,
loss-bearing target counts, generated output or a quality prediction.
The benchmark excludes data loading and host-side batch building, and uses
approximate corruption workloads.
Intended microbatch, packing, checkpointing, variable lengths and gradient
accumulation still need matching calibration before fixing release steps.

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
| Short generation | Predicted-length exact match for IT; oracle-length diagnostic; EOS, coverage and format checks | Synthetic generation conditions, missing EOS, chosen-length errors and over-cap counts exist |
| Latent reasoning | Same-checkpoint T/S curves and normal/zero/shuffled state above frozen task baselines | Curves and state interventions exist; they are not an independently trained architecture comparison |
| Classification | A labeled probe with fixed splits, majority baseline, accuracy and macro-F1 | Additional analysis is needed; there is no classification release stage |
| Serving speed | Complete embedding/generation call, representative lengths and batches, actual T/S, mean/median/p95 and memory | No trained-model inference latency has been measured |

Additional reporting should include frozen majority/chance controls where
meaningful, early-EOS rate, length MAE, grouped uncertainty and blinded review
of short natural instructions. These are proposed analysis additions, not
existing results. Exact match alone misses legitimate alternative wording.

The first useful quality claim should identify a specific task, a frozen
task-relevant baseline, its improvement and inference cost. Beating random
initialization demonstrates learning but does not establish product quality.
Increasing T or S must show a measured benefit; weight sharing alone does not
prove that more computation improves answers.
