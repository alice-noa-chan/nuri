# Retained Dense encoder: wider frozen-head comparison

The revised CPU assessment narrows the English sentiment deficit, but Nuri
still scores below all five requested English reference families. Its accuracy
gap is **6.25 to 16.41 percentage points**. The Korean topic result remains
useful within the assessed panel. These results do not establish general
BERT-family quality and do not authorize full production training.

This assessment changes head selection, **not encoder weights**. The retained
checkpoint, cached features and original [v1 reports](ENCODER_BENCHMARK_RESULTS.md)
are preserved. No GPU, cloud compute or completion TEST is used.

## What changed

V1 selected the upper regularization boundary in 19 of 21 jobs. Before this
rescore, a separate protocol froze **17 half-decade values from 0.01 through
1,000,000**, applied identically to all 21 existing model/task jobs. The rule
flags a selected endpoint for later review without extending the grid during
the run. **All 21 selected candidates are interior to this grid.** That removes
this particular boundary problem; it does not establish a global optimum.

The head remains training-standardized centered ridge with an unpenalized
intercept. Validation macro-F1 selects the candidate; accuracy and fixed grid
order break ties. Every job uses the same 1,024 fitting rows and 256 validation
rows as v1, unchanged content-token mean features and the original input
assessment binding. Sentence pairs use separately encoded means with
concatenation, absolute difference and product, rather than joint attention.

Claude's exploratory wider-grid results and the original validation results
were consulted before declaring this follow-up. Reusing those validation rows
adds selection optimism. The scores and conditional row intervals are
diagnostics, **not blind confirmation, completion-test results or new learning**.
The finer declared grid can select a different candidate from Claude's
exploration; macro-F1, rather than accuracy alone, governs both selections.

## Measured validation scores

Cells show **accuracy / macro-F1**, both as percentages, on 256 rows per task.

| Model or control | Korean topic classification (7 classes) |
|---|---:|
| Nuri retained Dense input encoder | 79.69 / 79.94 |
| Multilingual BERT Base cased | 70.31 / 70.45 |
| Korean ELECTRA Base v3 | 63.67 / 63.81 |
| Random Nuri encoder (seed 0) | 20.70 / 20.62 |
| Hashed character features | 42.58 / 42.35 |

| Model or control | English sentiment (2 classes) | English sentence-pair inference (3 classes) |
|---|---:|---:|
| Nuri retained Dense input encoder | 72.27 / 72.27 | 51.95 / 51.88 |
| BERT Base uncased | 84.77 / 84.77 | 53.12 / 53.07 |
| ModernBERT Base | 78.52 / 78.51 | 56.64 / 56.36 |
| RoBERTa Base | 85.94 / 85.94 | 58.20 / 58.18 |
| DeBERTa Base | 85.94 / 85.94 | 58.59 / 58.57 |
| ELECTRA Base discriminator | 88.67 / 88.67 | 53.12 / 53.00 |
| Random Nuri encoder (seed 0) | 55.47 / 55.46 | 41.02 / 40.83 |
| Hashed character features | 66.41 / 66.37 | 48.44 / 47.64 |

The label-only controls remain unchanged: training-majority accuracy is
14.45%, 50.00% and 33.59% for the three tasks; deterministic training-prior
random accuracy is 17.19%, 52.34% and 41.41%, respectively. These controls
provide context, not acceptance thresholds.

## Interpretation and next evidence

All five English sentiment comparisons have negative Nuri-minus-reference
95% paired row intervals for accuracy and macro-F1. For example, the accuracy
differences are -6.25 points against ModernBERT (interval -12.11 to -0.78)
and -16.41 against ELECTRA (-22.27 to -10.16). Every requested-family
sentence-pair difference still has an interval containing zero. This small
sentence-pair diagnostic remains inconclusive about model-family differences.

The bootstrap uses 2,000 deterministic resamples, paired on validation row
IDs and labels. Intervals condition on the selected heads, assume independent
rows, and omit search choice, training-seed and grouped-sampling uncertainty.
The sentence-pair inputs remain an internal split of isolated training-pool
components rather than an official publisher leaderboard split. No pristine
pretraining holdout is established for any encoder.

A stronger Korean reference needs its own pinned panel revision and common
visible-text audit. Matched downstream encoder adaptation is separate from
this frozen-head comparison. Japanese, token labeling, extractive QA, semantic
retrieval, generation and broader domain performance remain unassessed by
these three tasks. The preparation work in
[QUALITY_IMPROVEMENT.md](QUALITY_IMPROVEMENT.md) addresses these gaps without
treating one score or lower training loss as universal quality.

Detailed protocols, input bindings, predictions, bootstrap results and
numeric TRAIN/VALID TensorBoard candidate curves remain in ignored private
storage. No source identities or example text are published here.

An independent CPU reviewer replayed all 357 candidate fits, every selected
prediction/ID/label set, 24 paired comparisons, six controls and all 21 selected
bootstrap results. All matched. The reviewer also checked 21 actual TensorBoard
files and 1,848 numeric scalar values, including every candidate metric and
boundary flag; no completion-test or non-scalar payload was found. This verifies
the computation and bindings, not model quality beyond the assessed tasks.
