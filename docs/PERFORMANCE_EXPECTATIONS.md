# Performance expectations

Original forecast: October 4, 2026, before release training. The completed GPU
comparison and batch-capacity follow-up measure synthetic training computation,
not model quality or response latency. See [GPU_COMPARISON.md](GPU_COMPARISON.md)
and [GPU_CAPACITY.md](GPU_CAPACITY.md) for measurements.

Update, October 5: the first Dense Base has been trained and directly assessed
in [DENSE_BASE_EVALUATION.md](DENSE_BASE_EVALUATION.md). It learned MLM patterns,
but coherent generation failed. D-019 adds canvas positions to correct the
confirmed all-MASK symmetry. The repaired model's language quality is unmeasured.
Keep the forecasts below separate from those actual legacy-checkpoint results.

## What to expect first

Nuri is initially a small experimental encoder with dedicated embedding
training and a bounded masked generator. Task-specific retrieval, useful
features for a labeled classifier, and short structured answers are plausible
first uses to test. Broad conversation, extensive factual knowledge, long
coherent answers and reliable multi-step reasoning are uncertain goals.
These were first-use hypotheses, not measured accuracy or quality predictions.

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

The original measured geometry contained 150,652,931 dense parameters and
242,739,203 total MoE parameters. Corrected D-019 presets add 196,608 position
weights, giving totals of 150,849,539 and 242,935,811. Some weights serve the recurrent core and generator, so these
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

## Forecast before release training

The current forecast is a useful experimental encoder with limited conditional
generation. General classification features and narrow retrieval are more
plausible than broad semantic search, fluent instruction following or strong
reasoning. This is a judgment about the current shared-day scratch plan, not
a measured result. Assume stable optimization and the reviewed recipe.

| Capability | Codex's central forecast | Independent Claude assessment |
|---|---|---|
| Understanding | After a separate, comparable English GLUE fine-tune, plan around 65–75 points. A favorable run could reach the high 70s; a poor run could fall below 60. Task-specific classification can still be useful. | About 60% subjective probability of reaching 70, and 30% of reaching 75. Generally at or below the crammed-BERT tier. |
| Retrieval | Near-duplicate and narrow-domain matching are plausible; broad semantic retrieval is likely below multilingual E5-small. Removing easy pairs alone does not create semantic training quality. | About 35% subjective probability of clearly improving on matched TF-IDF for low-overlap queries; below 5% of competing with multilingual E5-small on public benchmarks. |
| Short generation | Simple labels, fixed formats and answers copied or extracted from context are the strongest candidates. Expect repetition, unsupported content and length errors on broader instructions. | About 5–10% subjective probability that at least half of short natural answers are acceptable to a blind reviewer. Claude is more pessimistic about generation overall; Codex's optimism concerns tightly constrained answers. |
| Latent reasoning | Some bounded tasks may benefit, but a flat or worse depth curve is a credible outcome. Increasing T is not a substitute for language learning. | About 25% subjective probability of a measurable benefit above T=4 on synthetic suites; below 10% on natural tasks. |
| Broad chat | Likely substantially below SmolLM2-135M-Instruct or Qwen2.5-0.5B-Instruct; this day is not expected to produce a dependable general assistant. | Below 2% subjective probability of useful broad chat. |

These ranges and probabilities are uncalibrated planning judgments. They are
not confidence intervals, expected accuracy percentages or scores calculated
from FLOPs. The GLUE forecast requires downstream labeled fine-tuning beyond
the planned release stages; it is not a zero-shot score or a Korean/Japanese
forecast. Classification has no trained default release head. Claude's full
independent assessment is recorded in [DISCUSSION.md](DISCUSSION.md).

The short-budget evidence gives reasons for both optimism and restraint:

- [Cramming](https://arxiv.org/html/2212.14034v1) reports GLUE-dev 78.3–78.6
  after one day of scratch MLM, versus 80.9 for its fully trained BERT-base
  control. Downstream fine-tuning is outside that day; inputs are 128 tokens.
  [MosaicBERT](https://arxiv.org/html/2312.17482v2) reports 79.6 after 1.13 hours
  on eight A100s. Its score aggregation and training recipe differ, so these
  numbers are reference points, not a shared leaderboard or Nuri predictions.
- [ModernBERT](https://arxiv.org/html/2412.13663v2) and
  [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
  each report 2T pretraining tokens. The latter has a similar headline size
  to Nuri dense, but radically more training exposure. Parameter count alone
  does not predict comparable language quality.
- [Multilingual E5](https://arxiv.org/html/2402.05672v1) begins with pretrained
  encoders and adds roughly 1B contrastive pairs, then supervised fine-tuning
  with hard negatives and distillation. Nuri's brief embedding stage is not
  equivalent to this training. The comparison motivates better relevant
  positives and evaluation, not merely more repeated easy pairs.
- [Mask-Predict](https://aclanthology.org/D19-1633/) demonstrates that masked
  conditional generation can work. Its translation results also show the
  importance of length prediction and distilled targets. They support the
  generation mechanism without guaranteeing a general chat capability.
  [Recurrent-depth research](https://arxiv.org/html/2502.05171v1) reports a
  3.5B-parameter, 800B-token model; its gains cannot be assumed for Nuri.

Do not convert a CPU valid/padded ratio into a new GPU token forecast by
multiplying the synthetic estimate. Bucketing changes rows, widths, targets
and time per step; equal batch-task probabilities also do not imply equal
token exposures. Actual rate calibration still controls D-017. Likewise, the
roughly 16-times result in a particular
[masked-diffusion study](https://arxiv.org/abs/2410.18514) is not a universal
compute penalty for Nuri's conditional denoising and refinement algorithm.

Within this day, dense is the more likely first quality candidate: it receives
more base time and the tested portable MoE is slower. This is a forecast, not
a finding that dense always outperforms MoE. The research release is worth
pursuing if narrow learning evidence and architecture experiments are the goal.
For an immediately useful general chat model, the current plan is insufficient.
Keep the agreed five stages and existing first-hour real-task gate; this
assessment changes no architecture, budget, spending hold or release decision.

## What if the shared budget becomes 48 hours?

This is a hypothetical forecast for one A100 80GB, the same scratch architecture
and all five release stages. It does not amend the 24-hour decision or authorize
spending. Assume stable training and enough varied inputs for the added steps.

| Capability | 24-hour judgment | 48-hour judgment |
|---|---|---|
| Understanding | English GLUE-style fine-tuning around 65–75 | Around 66–77 with unchanged inputs/mix; around 68–78 if limited-language coverage improves and more additional base time goes to dense |
| Retrieval | Narrow and near-duplicate matching are plausible; broad semantics are weak | More consistent narrow retrieval is plausible; multilingual E5-small remains an unlikely broad-benchmark target without better semantic supervision |
| Short generation | Constrained labels/formats and input-grounded answers are the strongest candidates | A better chance of useful constrained workflows and more consistent wording; broad natural instructions remain unreliable |
| Broad chat | Not expected to be a dependable assistant | Still likely below extensively pretrained small instruct models such as SmolLM2-135M-Instruct |
| Latent reasoning | Benefit from greater depth may be flat or inconsistent | More opportunity for the core to learn, but extra time alone does not establish a depth benefit |

The GLUE ranges are subjective planning judgments after a separate downstream
fine-tune. They are not expected zero-shot accuracies, calibrated intervals or
Korean/Japanese predictions. In the improved-coverage case, plan for a few
points of improvement, roughly 2–4, with substantial overlap between forecasts.
Gains can be negligible if diversity, semantic supervision or optimization is
the bottleneck. Repeating a finite corpus still offers fresh corruptions, but
does not add new facts or relevant retrieval pairs.

Claude's independent assessment is only modestly more optimistic for 48 hours.
In a dense-priority scenario with reviewed language sampling, his subjective
probability of reaching English GLUE-style 70 rises from about 60% to 70%, and
of reaching 75 from 30% to 40%. These are chances of reaching the stated score,
not classification accuracies. He expects the largest understanding benefit in
Korean, little broad retrieval benefit without better pairs and unreliable
general generation/chat. Codex independently checked the private aggregate
counts underlying the limited-language repetition concern; source information,
counts and the detailed check remain private. Improving coverage or changing
language weights is a proposal, not an adopted change to D-002.

[Research on data-constrained language models](https://arxiv.org/abs/2305.16264)
supports diminishing value from many repeats. It studies autoregressive models,
so its repetition thresholds are not established limits for dynamically masked
Nuri. Document weights, raw average lengths and synthetic batch rates do not
establish language-token shares or epoch counts for packed/bucketed training.

The [MosaicBERT paper](https://arxiv.org/html/2312.17482v2) reports GLUE-dev
79.6 after 1.13 hours, 82.2 after 2.81 hours and 83.2 after 4.6 hours, always on
eight A100s. This supports modest rather than proportional score improvements;
its curve does not calibrate Nuri. The longer
[Cramming experiment](https://arxiv.org/html/2212.14034v1) uses 48 hours on eight
A6000s: sixteen times the original compute, not a one-GPU two-day comparison.
It reaches 79.8 or 80.6 versus 78.6, with uneven gains across tasks.

For a useful dense model, Codex favors giving most additional base time to dense
while retaining MoE. One illustrative split is 22 hours dense base, 12 hours MoE
base, three hours for each instruction stage, three hours embedding and five
hours preparation/evaluation: 48 hours total. This is an allocation proposal,
not a launch plan. Doubling all existing stage windows instead is a separate
comparison case; neither choice guarantees twice as many valid tokens. Measure
actual rates and development quality before deciding where extra hours help.
Claude favors a more concentrated alternative, about 30 hours dense with
roughly 8–9 hours MoE and shorter tuning windows. Neither proposal is finalized.

Extending an existing checkpoint after its learning rate has decayed to zero is
not equivalent to planning a 48-hour run from the start. The warmup/decay horizon
and continuation policy must be reviewed first. Exact-resume protections remain
in force; this document does not change any executable plan or checkpoint.

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
