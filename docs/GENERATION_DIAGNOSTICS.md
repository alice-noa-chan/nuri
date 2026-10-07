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
