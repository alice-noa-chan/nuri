# Retained Dense encoder: first frozen comparison

The retained encoder provides useful Korean classification features in this
diagnostic. Its English sentiment score is below every requested reference:
the accuracy gap is 13.28 to 23.05 percentage points.
These results do not establish the requested general BERT-family quality.
Keep full production training on hold while choosing a targeted improvement
or a stronger matched adaptation assessment. No additional encoder learning
was performed in this comparison.

## Assessment conditions

Each task has 1,024 fitting rows and 256 validation rows. The validation
rows select one regularization value from the declared five-value grid.
Completion test remains reserved: it was not transferred, extracted or scored.
Consequently the scores below are preliminary validation selection scores.

Encoders are frozen. Every model uses content-token masked means, training-only
feature standardization, and the same centered ridge classification head.
Sentence-pair features concatenate two separately encoded means, their absolute
difference and product. They do not use joint attention between the sentences.
The declared grid is `0.01, 0.1, 1, 10, 100`, selected by validation macro-F1,
then accuracy, then grid order. Exact source and protocol bindings stay private.

**Follow-up limitation (2026-10-06):** 19 of these 21 saved reports select
the grid's upper value, 100; two select 10. The search therefore does not
bracket every model's useful regularization range. These immutable results
remain valid under the declared five-candidate protocol, but their score gaps
are not estimates of each model's best frozen-probe performance. Claude
reports gains in a wider-grid diagnostic; those measurements are a separate
validation exploration, not new encoder learning or a reviewed replacement
of this report. A versioned equal-budget grid and boundary flags are required
before rescoring the same caches. Reusing the consulted validation rows adds
selection optimism; completion TEST remains reserved. See
[the broad improvement work plan](QUALITY_IMPROVEMENT.md).

All models receive the same visible strings, IDs and labels within a task.
Every field fits all applicable tokenizers within 128 encoded tokens; no input
was clipped and extraction reports show zero truncated fields. Runs use CPU
FP32, two threads, batches of eight, eager reference attention and no reference
compilation. These conditions do not measure optimized GPU inference.

Fitting and validation groups are isolated by exact text and known families.
The sentence-pair task uses predeclared independent internal splits drawn from
isolated original training-pool components. It is not an official publisher
development/test or leaderboard result. No pristine pretraining holdout is
claimed for any encoder.

## Measured validation scores

Cells show **accuracy / macro-F1**, both as percentages. Each cell assesses
256 rows. All declared classes are included in macro-F1.

| Model or control | Korean topic classification (7 classes) |
|---|---:|
| Nuri retained Dense input encoder | 76.56 / 76.83 |
| Multilingual BERT Base cased | 68.36 / 68.41 |
| Korean ELECTRA Base v3 | 61.72 / 62.05 |
| Random Nuri encoder (seed 0) | 18.36 / 18.32 |
| Hashed character features | 33.98 / 34.25 |

| Model or control | English sentiment (2 classes) | English sentence-pair inference (3 classes) |
|---|---:|---:|
| Nuri retained Dense input encoder | 61.72 / 61.72 | 46.88 / 46.45 |
| BERT Base uncased | 81.25 / 81.25 | 49.22 / 49.16 |
| ModernBERT Base | 75.00 / 75.00 | 47.66 / 47.53 |
| RoBERTa Base | 83.59 / 83.59 | 51.95 / 51.90 |
| DeBERTa Base (original) | 81.64 / 81.64 | 51.95 / 51.92 |
| ELECTRA Base discriminator | 84.77 / 84.76 | 46.88 / 46.87 |
| Random Nuri encoder (seed 0) | 52.73 / 52.68 | 35.94 / 35.80 |
| Hashed character features | 66.41 / 66.37 | 41.41 / 41.27 |

| Label-only control | Korean topic | English sentiment | English sentence-pair inference |
|---|---:|---:|---:|
| Training-majority label | 14.45 / 3.61 | 50.00 / 33.33 | 33.59 / 16.76 |
| Training-prior random labels (seed 0) | 17.19 / 17.08 | 52.34 / 52.24 | 41.41 / 41.17 |

The random-label control samples from fitting-set class frequencies once per
task with seed zero. The same predictions are reused in all paired comparisons.
The majority/prior-argmax rule is a separate deterministic control.

## Paired differences and uncertainty

Differences below are **Nuri minus reference**, in percentage points, with
95% percentile intervals from 2,000 paired row resamples and seed zero.
Positive point values favor Nuri under this assessment. A crossing-zero interval
is inconclusive and does not establish equivalence.

The intervals are conditional on the fixed fitting rows and validation-selected
heads. They omit head-selection, fitting-sample and initialization uncertainty,
assume independent rows, and do not account for remaining within-family
dependence or multiple comparisons. Interpret them as descriptive uncertainty
on these assessed predictions, not broad model-family superiority tests.

| Task | Reference | Accuracy difference [95% interval] | Macro-F1 difference [95% interval] |
|---|---|---:|---:|
| Korean topic classification | Multilingual BERT Base cased | +8.20 [+2.34, +14.84] | +8.42 [+2.42, +14.88] |
| Korean topic classification | Korean ELECTRA Base v3 | +14.84 [+8.20, +21.09] | +14.78 [+8.22, +21.20] |
| English sentiment classification | BERT Base uncased | -19.53 [-26.17, -12.50] | -19.53 [-26.23, -12.56] |
| English sentiment classification | ModernBERT Base | -13.28 [-20.31, -5.86] | -13.28 [-20.31, -5.88] |
| English sentiment classification | RoBERTa Base | -21.88 [-28.12, -15.62] | -21.88 [-28.16, -15.59] |
| English sentiment classification | DeBERTa Base (original) | -19.92 [-26.95, -12.89] | -19.92 [-26.93, -12.89] |
| English sentiment classification | ELECTRA Base discriminator | -23.05 [-29.69, -16.02] | -23.04 [-29.72, -16.08] |
| English sentence-pair inference | BERT Base uncased | -2.34 [-10.16, +5.47] | -2.72 [-10.43, +5.20] |
| English sentence-pair inference | ModernBERT Base | -0.78 [-8.98, +7.03] | -1.08 [-9.20, +6.83] |
| English sentence-pair inference | RoBERTa Base | -5.08 [-12.50, +2.73] | -5.45 [-12.93, +2.39] |
| English sentence-pair inference | DeBERTa Base (original) | -5.08 [-12.89, +3.12] | -5.47 [-13.11, +2.57] |
| English sentence-pair inference | ELECTRA Base discriminator | +0.00 [-7.81, +7.81] | -0.42 [-8.17, +7.24] |

## Parameters and verification

Nuri uses its input embedding and input encoder only. The reasoning core,
readout and untrained contrastive projections are excluded. Reference poolers
and unused pretraining/discriminator heads are also excluded. Counts below are
the actual parameters present in each loaded encoder, not an active MoE count.

| Encoder | Used parameters (millions) |
|---|---:|
| Nuri retained Dense input encoder | 121.13 |
| BERT Base uncased | 108.89 |
| ModernBERT Base | 149.01 |
| RoBERTa Base | 124.06 |
| DeBERTa Base (original) | 138.60 |
| ELECTRA Base discriminator | 108.89 |
| Multilingual BERT Base cased | 177.26 |
| Korean ELECTRA Base v3 | 112.33 |

The 21 completed jobs took about 43.3 minutes of
combined active attempt time, including the preserved failed first attempt.
This excludes transfer, preparation and the gap between attempts. The owned
runtime had two CPU cores; no paid GPU rental or production training occurred.

The first attempt rejected a bin-only reference. It was preserved. Three
already pinned binary checkpoints were then converted with a weights-only CPU
loader into safetensors. Every named tensor was reread and verified for exact
shape, dtype and bytes, with all values finite and original files unchanged.
The continuation reused three verified completed jobs and wrote the remaining
18 into fresh folders under the unchanged assessment protocol.

All result-transport members and hashes were replayed. Parent verification
joined raw input, model, tokenizer, feature, protocol and completion bindings;
checked fitting-only standardization and fixed-grid selection; and recomputed
all validation metrics and stored bootstrap intervals. The source initializer
was unchanged. Data identities, example text, provenance, detailed predictions
and receipts remain in ignored private storage.

An independent read-only replay also refit all five candidates for all 21 jobs.
Every fitting and validation candidate metric, selected prediction, stored
interval, label-only control and paired difference reproduced exactly. The
complete compressed result transport and all 229 archive members were verified.
The owned CPU session was terminated after retrieval; the provider subsequently
reported zero active assignments and zero ongoing compute-unit usage.

TensorBoard has the actual `train` and `valid` accuracy/macro-F1 values for
each regularization candidate. The horizontal steps are candidate evaluations,
not pretraining epochs. There is no fabricated test curve.

## What follows from this result

There is evidence of useful learned Korean features and a substantial English
sentiment deficit under this frozen-head protocol. A successful Korean task
does not establish general encoder quality. A weak frozen probe also does not
prove that full task-specific fine-tuning would fail.

All five sentence-pair reference difference intervals include zero, so this
task is inconclusive about their relative quality. Every sentence-pair fitting
score is 100%, including random and character controls; fitting accuracy alone
does not establish semantic quality. The assessed validation rows and stated
uncertainty supply the evidence.

Before committing the remaining full-training budget, inspect language
coverage and objective effectiveness privately, then choose a bounded focused
improvement with the same comparisons or matched encoder fine-tuning with equal
adaptation and validation search budgets. Do not assume that longer generator
training alone will close the English gap. Any paid experiment still needs
its reviewed scope and budget; this comparison does not authorize it.

Japanese, token labeling, extractive answers, semantic retrieval, generation
and full encoder fine-tuning remain unassessed. The corrected generator has
not gained a language-quality result from this input-encoder diagnostic.

See [the assessment commands and limits](ENCODER_BENCHMARK.md) and
[the continuation evidence plan](QUALITY_EVIDENCE_PLAN.md).
