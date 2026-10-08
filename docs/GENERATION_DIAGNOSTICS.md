# Generation diagnostics on synthetic tasks (Q-17)

Recorded on 2026-10-07. These are measured CPU results from small models,
not release-scale results. They test four candidate causes of the failed
Dense Base generation listed in DISCUSSION.md ("Q-17 candidate causes").
No architecture, recipe or default changed.

## Summary

| Candidate cause | Result on synthetic tasks |
|---|---|
| Input position linkage | **Confirmed as the main failure.** With the unchanged model, copy and reverse fail (token accuracy ≈0.24, adjacent-repeat rate ≈0.40 at S=4 and ≈0.93 at S=1). Adding one learned absolute position vector to the input embeddings solves both (1.000 in all 12 runs). |
| Readout depth | **Ruled out.** One, two and four readout layers give the same failure on copy and reverse. |
| Many valid answers (no distillation) | **Confirmed.** The same inputs with one answer per input are solved exactly; with two valid successors per step, S=1 breaks the graph at about half the steps and repeats tokens. |
| Sparse EOS targets | **Confirmed.** EOS in 10% of training canvases drops correct EOS placement from 1.000 to 0.38-0.43 while content stays mostly right. |
| Untrained draft revision | **Small effect.** Revision training plus a revising decoder raises valid transitions from 0.866 to 0.901 on the many-answer task. A revising decoder without that training is worse and unstable (0.612 ± 0.198). |

S=4 token accuracy was below S=1 only on tasks that already failed. On the
many-answer task, S=4 lowers single-reference accuracy (0.271 → 0.245)
while it raises valid transitions (0.485 → 0.866). A single reference
answer therefore penalizes legal alternatives, so "S=4 below S=1" in the
real evaluation is not by itself evidence that refinement harms answers.

## Why position linkage matters

The input encoder uses RoPE, which puts only relative position into the
attention scores and nothing into the token vectors. The readout reads X
through cross-attention with no position signal. A canvas position can
therefore ask "where is token A?" but not "what is input token 5?". Tasks
that need only content (count, branch-single) work; tasks that need ordered
reads from the input (copy, reverse) fail and fall back to repeating likely
tokens. The repeated tokens look like the repeated words of the Dense Base.

The real model has the same structure. D-019 added learned positions to the
canvas, but X still has none. Answers that reuse words from the prompt in
order need the missing link. These synthetic results do not measure how
much of the real failure this explains; that needs a real-data test.

## Setup

Code: `scripts/generation_diagnostics.py` and `nuri/synthetic_generation.py`.
Each run trains a new model with the production model, `corrupt_canvas`,
`canvas_loss` and `generate` code.

- Model: hidden 128, 4 heads, FFN 352, 3 input layers, 2 readout layers
  (unless varied), 8 latent slots, learned canvas positions, QK-norm, train
  depths {2, 4}, inference depth 4; about 1.4M parameters.
- Vocabulary: PAD, MASK, EOS and 32 content tokens. Inputs have 6-12 tokens;
  continuation targets have 6-12 tokens plus EOS.
- Training: 2,000 AdamW updates, batch 64, peak LR 1e-3, 100 warmup steps,
  cosine decay to 10%, gradient clipping 1.0, fresh examples each step.
  Canvas corruption is the base recipe (10% full mask, otherwise 15-95%).
- Evaluation: 512 held-out examples per run, oracle canvas length, greedy
  decoding, S=1 and S=4. Three seeds (0, 1, 2) per row; values are
  mean ± sample standard deviation over seeds.
- Host: one CPU process per run with one thread, six runs in parallel,
  about 10-16 minutes per run; 57 runs in total.

Tasks: **copy** and **reverse** repeat or reverse the input; **count**
continues an arithmetic progression; **branch** continues a walk on a fixed
graph where each token has two allowed successors; **branch-single** uses
the same inputs but always takes the first successor, imitating training
on a teacher's single output (sequence-level distillation).

Metrics: token accuracy over the canvas including the EOS slot (the EOS
slot alone gives about 0.1 on copy); exact match; correct EOS placement;
adjacent-repeat rate in the content part (the reference rate is 0.030 for
copy and reverse, 0 for the others); and, for branch tasks, the share of
steps that follow the graph ("validity"). The **standard** decoder is
`generate` (kept tokens are never rescored). The **revise** decoder
re-predicts every position each round with the same re-mask schedule.

## Results

### Unchanged recipe on each task

| Task | Acc S=1 | Acc S=4 | Acc S=4 revise | Exact S=4 | Repeat S=1 | Repeat S=4 | Final loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| copy | 0.280 ± 0.002 | 0.236 ± 0.003 | 0.245 ± 0.009 | 0.000 | 0.928 | 0.396 | 1.329 |
| reverse | 0.280 ± 0.002 | 0.238 ± 0.001 | 0.251 ± 0.003 | 0.000 | 0.936 | 0.399 | 1.328 |
| count | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.998 ± 0.002 | 1.000 | 0.000 | 0.000 | 0.001 |
| branch | 0.271 ± 0.008 | 0.245 ± 0.004 | 0.201 ± 0.032 | 0.002 | 0.337 | 0.010 | 0.708 |
| branch-single | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.992 ± 0.010 | 1.000 | 0.000 | 0.000 | 0.000 |

### Readout depth and input positions (copy and reverse)

| Task | Variant | Acc S=1 | Acc S=4 | Exact S=4 | Repeat S=4 |
|---|---|---:|---:|---:|---:|
| copy | readout 1 | 0.279 ± 0.003 | 0.235 ± 0.004 | 0.000 | 0.353 |
| copy | readout 2 (unchanged) | 0.280 ± 0.002 | 0.236 ± 0.003 | 0.000 | 0.396 |
| copy | readout 4 | 0.283 ± 0.002 | 0.240 ± 0.004 | 0.000 | 0.442 |
| copy | input positions from start | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 | 0.030 |
| copy | input positions from end | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 | 0.030 |
| reverse | readout 1 | 0.278 ± 0.003 | 0.234 ± 0.005 | 0.000 | 0.363 |
| reverse | readout 2 (unchanged) | 0.280 ± 0.002 | 0.238 ± 0.001 | 0.000 | 0.399 |
| reverse | readout 4 | 0.283 ± 0.000 | 0.238 ± 0.010 | 0.000 | 0.438 |
| reverse | input positions from start | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 | 0.030 |
| reverse | input positions from end | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 | 0.030 |

With input positions, the repeat rate equals the reference rate (0.030):
the remaining repeats are in the inputs themselves.

### Sparse EOS targets

| Task | EOS in training canvases | Acc S=4 | Exact S=4 | Correct EOS S=4 |
|---|---:|---:|---:|---:|
| count | 100% | 1.000 ± 0.000 | 1.000 | 1.000 |
| count | 10% | 0.933 ± 0.004 | 0.430 ± 0.035 | 0.432 ± 0.035 |
| branch-single | 100% | 1.000 ± 0.000 | 1.000 | 1.000 |
| branch-single | 10% | 0.928 ± 0.002 | 0.381 ± 0.026 | 0.381 ± 0.026 |

The lost accuracy is almost entirely the EOS slot: branch-single validity
stays 1.000. At the oracle length the model still leaves out EOS in most
answers, because training canvases usually ended without one.

### Many valid answers and revision training (branch)

| Canvas training | Validity S=1 | Validity S=4 standard | Validity S=4 revise | Repeat S=1 |
|---|---:|---:|---:|---:|
| mask (unchanged) | 0.485 ± 0.022 | 0.866 ± 0.016 | 0.612 ± 0.198 | 0.337 |
| random-noise | 0.534 ± 0.010 | 0.821 ± 0.019 | **0.901 ± 0.011** | 0.231 |
| self-draft | 0.393 ± 0.015 | 0.757 ± 0.021 | 0.877 ± 0.005 | 0.367 |

On branch-single, every training and decoder combination keeps validity at
0.995 or higher (1.000 for both revision trainings), so neither revision
training nor the revising decoder harms a one-answer task. Revision training helps only together with the revising
decoder, and only by about 0.035 over the unchanged pair. It is a minor
lever next to positions, single-answer targets and EOS.

## Real-text check of the fixes (CPU)

`scripts/readout_probe.py` starts from the corrected Dense initializer
(the retained Dense Base with D-019 canvas positions), freezes the encoder,
core and token embedding, caches X and R (T=4) and trains only the readout
for 600 steps (batch 8, LR 3e-4, inputs cut to 128 tokens, spans to 32).
Training uses the training corpora (Korean/English/Japanese weights
0.6/0.3/0.1); evaluation uses 192 validation examples per set, never test,
with oracle canvas lengths. "mid" spans are cut at random points; "end"
spans end at the document end and must finish with EOS. One seed; treat
small differences as noise.

| Arm | Set | Full-mask CE | Acc S=1 | Acc S=4 | Token F1 S=4 | Repeat S=1 | Correct EOS S=4 |
|---|---|---:|---:|---:|---:|---:|---:|
| initial (no readout training) | end | 6.836 | 0.070 | 0.054 | 0.214 | 0.794 | 0.130 |
| baseline | end | 6.850 | 0.085 | 0.070 | 0.223 | 0.783 | 0.198 |
| memory positions | end | 6.852 | 0.085 | 0.069 | 0.224 | 0.789 | 0.193 |
| memory positions + span_end_prob 0.5 | end | **6.475** | **0.131** | **0.132** | **0.250** | 0.798 | **0.745** |
| baseline | mid | 6.794 | 0.097 | 0.080 | 0.238 | 0.787 | - |
| memory positions | mid | 6.786 | 0.099 | 0.081 | 0.239 | 0.781 | - |
| memory positions + span_end_prob 0.5 | mid | 6.933 | 0.083 | 0.070 | 0.226 | 0.817 | - |

- EOS-bearing spans work on real text: correct EOS placement rises from
  0.198 to 0.745 and document-end token accuracy nearly doubles. Half of
  the training spans moved to document ends, which costs some mid-document
  accuracy in this short probe. This supports D-023.
- Memory positions are indistinguishable from the baseline after 600
  steps: the tables start at zero and barely move in so few updates. A
  longer probe decides whether they stay (see DISCUSSION).
- All arms still repeat heavily at S=1 (about 0.79) and reach under 15%
  token accuracy: a short readout-only repair on a frozen encoder does not
  make the Dense Base a good generator.

The same probe was repeated for 3,000 steps on a Colab T4 GPU
(`--device cuda`; pool of 6,000 cached training prefixes drawn from a
10,000-document training subset, 192 validation examples per set from a
1,200-document validation subset; one seed):

| Arm | Set | Full-mask CE | Acc S=1 | Acc S=4 | Token F1 S=4 | Repeat S=1 | Correct EOS S=4 | Final train loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| initial | end | 6.913 | 0.064 | 0.060 | 0.214 | 0.763 | 0.141 | - |
| baseline | end | 6.902 | 0.084 | 0.079 | 0.228 | 0.789 | 0.224 | 3.875 |
| memory positions | end | 6.906 | 0.086 | 0.076 | 0.232 | 0.800 | 0.177 | 3.875 |
| memory positions + span_end_prob 0.5 | end | **6.349** | **0.169** | **0.160** | **0.253** | **0.671** | **0.979** | 3.499 |
| baseline | mid | 7.003 | 0.092 | 0.070 | 0.221 | 0.801 | - | |
| memory positions | mid | 7.007 | 0.093 | 0.070 | 0.224 | 0.801 | - | |
| memory positions + span_end_prob 0.5 | mid | 7.075 | 0.097 | 0.074 | 0.239 | 0.744 | - | |

- EOS-bearing spans: correct EOS placement 0.224 -> 0.979 and document-end
  accuracy doubles; with more steps the mid-document cost seen at 600
  steps disappears (S=4 accuracy 0.070 -> 0.074). D-023 stands.
- Memory positions added to an already trained readout show no effect:
  the final training loss matches the baseline to three decimals. Zero-
  initialized tables on a readout that learned to work without positions
  barely train in a short warm start, so this does not test training from
  scratch, which the release run does.

### From-scratch real-text A/B of memory positions

Two models built from the same random initialization (reference geometry
with the 65,536-token vocabulary: 66.1M weights, 66.6M with the memory
tables; the tables start at zero and every other weight is identical)
were trained from scratch with `scripts/train.py --stage base` on a Colab
T4 for 2,000 steps (batch 32, MLM/span 50/50, packed MLM, bucketing, seed
0) on a 68,000-document training subset (Korean/English/Japanese weights
0.72/0.27/0.01, about 42M tokens). Evaluation used the probe's validation
sets above. One seed, early in training (span loss about 7.3).

| Metric | none | learned |
|---|---:|---:|
| span training loss, last quarter | 7.303 | 7.257 |
| MLM training loss, last quarter | 6.081 | 6.034 |
| mid: full-mask CE | 7.929 | 7.933 |
| mid: S=4 token accuracy | 0.055 | 0.055 |
| mid: S=4 token F1 | 0.135 | **0.157** |
| mid: S=4 repeat rate | 0.515 | **0.412** |
| end: full-mask CE | 7.979 | 7.966 |
| end: S=4 token accuracy | 0.051 | 0.052 |
| end: S=4 token F1 | 0.128 | **0.151** |
| end: S=4 repeat rate | 0.479 | **0.382** |

Memory positions reduce repetition by about a fifth and raise token F1 in
both sets, with unchanged cross-entropy and accuracy. The MLM loss, which
never uses the readout, also moved by a similar amount, so part of the
difference may be run-to-run noise. The evidence is weak but consistent
with the synthetic result and shows no harm; D-024 enables the tables in
new release presets.

## The readout bypasses X through R (and R dropout)

Checks on the corrected Dense initializer (CPU, 30-64 validation spans,
full-mask single fill) found a further cause. Readout cross-attention puts
91% (layer 1) and 40% (layer 2) of its weight on the 32 R slots and 1.4% /
5.7% on the last 8 input tokens. Hiding X from the readout memory barely
changes full-mask accuracy (0.059 -> 0.052) and pushes predictions of the
three most frequent tokens ('.', newline, ',') to 97% (68% normally, 8% in
the references). The readout reads a 32-slot summary, so it can name the
topic but not copy specifics; the first token after the prompt is right
3.3% of the time although the same encoder reaches 65% MLM accuracy.

R dropout (recipe `r_dropout_prob`) hides R from the readout memory for a
share of span samples during training. A from-scratch A/B on a Colab T4
used the 66M model of the D-024 test with memory positions and
span_end_prob 0.5 in both arms (2,000 steps, batch 32, one seed):

| Metric | r_dropout 0 | r_dropout 0.5 |
|---|---:|---:|
| attention on R, layer 1 / 2 | 0.23 / 0.32 | **0.12 / 0.10** |
| full-mask accuracy, normal -> X hidden | 0.059 -> 0.033 | 0.066 -> 0.031 |
| span training loss, last quarter | 6.949 | 6.948 |
| mid: full-mask CE | 8.130 | **8.090** |
| mid: S=4 token accuracy | 0.048 | **0.054** |
| mid: S=4 repeat rate | 0.458 | **0.443** |
| end: full-mask CE | 7.469 | **7.421** |
| end: S=4 token accuracy | 0.120 | **0.123** |
| end: S=4 token F1 | 0.165 | 0.161 |
| correct EOS, S=4 | 1.000 | 1.000 |

R dropout halves the attention on R and gives small gains in
cross-entropy and accuracy, with a slightly lower document-end F1. The
evidence is weak (one seed) but matches the intended mechanism; D-025
adopts it for base span training. Both arms also show that a readout
trained from scratch with memory positions already reads X (accuracy
falls by about half when X is hidden), unlike the retained Dense readout;
the runs differ in many ways, so this supports D-024 without being a
controlled comparison.

## Further checks: optimizer, clipping and decoding order

- **Weight decay on memory positions (fixed).** The optimizer exempted only
  parameter names containing "embedding", so the D-024 tables were decayed
  while the canvas position table was not. All position tables are now
  exempt. The D-024 A/B ran with the decay.
- **Span gradients are always clipped (tested, not adopted).** In the
  retained Dense Base log, span-step gradient norms rise from 1.6 to 3.2
  while MLM steps stay near 0.8; with grad_clip 1.0 every span step is
  scaled down 2-3x, which could weaken the generation signal into the
  shared encoder. A from-scratch A/B (Colab T4, 66M model, 2,000 steps,
  one seed, D-023/D-024/D-025 in both arms) raised the threshold to 3.0:

  | grad_clip | Clipped updates | Train span / MLM loss (last quarter) | Full-mask CE mid / end | S=4 accuracy mid / end | S=4 F1 mid / end | S=4 repeat mid / end |
  |---|---:|---:|---:|---:|---:|---:|
  | 1.0 (current) | 61% | 6.908 / 5.890 | 8.467 / 7.554 | 0.048 / 0.123 | 0.163 / 0.168 | 0.408 / 0.439 |
  | 3.0 | 2% | 7.077 / 6.175 | 8.039 / 7.436 | 0.056 / 0.121 | 0.147 / 0.150 | 0.483 / 0.464 |

  The larger updates made both training losses worse (MLM too, so it is
  not a span-only trade) and the generation metrics split: lower
  held-out CE but lower F1 and more repetition. No clear gain, so
  grad_clip stays 1.0 and no code was added. The 1.0 arm repeats the
  D-025 r50 arm with the weight-decay fix; its held-out CE differs from
  that run by 0.38 (mid) and 0.13 (end) while training losses agree,
  which gives a rough idea of how noisy one-seed held-out CE is here.
- **Confidence order commits punctuation first (amplifier).** After round
  1 of S=4, 85% of kept tokens are '.', ',' or newline, with 5% accuracy;
  the readout hidden states share a direction whose top logits are exactly
  those tokens. Choosing kept tokens by PMI instead of confidence:

  | Keep score | Accuracy mid / end | Token F1 mid / end | Repeat mid | Frequent-token share mid |
  |---|---:|---:|---:|---:|
  | log p (current) | 0.054 / 0.051 | 0.217 / 0.217 | 0.121 | 0.247 |
  | log p - 0.5 log prior | 0.052 / 0.057 | 0.213 / 0.214 | 0.109 | 0.159 |
  | log p - 1.0 log prior | 0.050 / 0.058 | 0.211 / 0.210 | 0.110 | 0.158 |

  Fewer punctuation tokens but no better answers, so the decoder stays as
  it is; the weak predictions come from the causes above.
- **Ruled out:** tied-embedding norms do not favor frequent tokens
  (frequent tokens have smaller norms; correlation with log frequency
  -0.41).

## Limits

- Small synthetic models, 2,000 updates and three seeds. Large effects
  (positions, EOS, one vs. many answers) are far outside the seed spread;
  the revision-training effect is small and should be treated as tentative.
- Synthetic tasks isolate mechanisms. They do not show how much each
  mechanism contributes to the real Korean, English and Japanese failure.
- The position variant adds learned absolute positions to the input
  embeddings only for this test. It is not a proposed final design; for
  example, it does not cover packed MLM rows or inputs longer than the
  learned table.
- Oracle canvas lengths are used throughout; the length head is not tested.

## Reproduce

```text
python scripts/generation_diagnostics.py --out runs/generation_diagnostics --shard 0 6 --threads 1
...                                                                        --shard 5 6 --threads 1
python scripts/generation_diagnostics.py --out runs/generation_diagnostics --summarize-only
```

Run the six shard commands in parallel. Results stay in the ignored `runs/`
directory.
