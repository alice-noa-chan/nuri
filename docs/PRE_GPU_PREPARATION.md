# Preparation before GPU reevaluation

The 2026-10-06 work completes the local implementation and preparation boundary
requested by the user. It does not start GPU rental, cloud compute, actual
encoder adaptation, production learning, external upload or reserved TEST
scoring. Existing checkpoints, earlier protocols and results remain intact.

## Implemented and verified locally

| Unit | Completed behavior | Evidence |
|---|---|---|
| Explicit training recipes | Private opt-in stage settings, strict validation and resume binding; default computation remains unchanged | [TRAINING_RECIPE.md](TRAINING_RECIPE.md); 111 focused recipe/training/release tests |
| Release runner | Explicit per-stage recipe forwarding; malformed declarations fail before allocation or subprocess launch | `scripts/run_release.py`; 10 focused runner tests |
| Batch-policy audit | Bounded TRAIN-only CPU simulation using actual MLM/span builders; separately counts document draws, valid positions and supervised targets | `tools/audit_batch_policy.py`; 12 synthetic controls and private corpus-bound receipts |
| Corrected frozen probes | A separately declared, equally applied wider grid with boundary flags and immutable reports | [ENCODER_BENCHMARK_RESULTS_V2.md](ENCODER_BENCHMARK_RESULTS_V2.md); independent replay of 357 candidate fits |
| Matched adaptation worker | One explicit LR/seed job updates a copied dense encoder and linear head, selects epochs on VALID and preserves source weights | [ENCODER_FINETUNING.md](ENCODER_FINETUNING.md); 106 synthetic CPU adaptation/extractor/probe tests |
| Language monitoring | Optional fixed Korean/English/Japanese objective groups, separate numeric curves and split/path checks | [HOLDOUT_MONITORING.md](HOLDOUT_MONITORING.md); 63 focused controls from the preceding preparation unit |
| Local reference preparation | Pinned files, required-weight loading, tokenizer limits and artificial-string forward mechanics checked before paid compute | Private receipts for eight references and the native Dense encoder |

Test counts describe overlapping focused suites, not additive independent
evidence. The adaptation controls train tiny synthetic fixtures; no retained
encoder or real reference was adapted in this preparation.

All nine local loader checks use CPU FP32, eager execution and artificial
strings. Each produces finite content-mean features without truncation.
Required encoder weights load completely. Three conversions preserve all
607 named tensors and match the earlier derived file hashes. These checks
reduce known loader risks; they do not verify GPU kernels, backward memory,
mixed precision, sustained throughput or freedom from future failures.

## Prepared comparison and retention paths

The frozen-feature replacement assesses the same retained encoder. Its Korean
topic accuracy is 79.69%. English sentiment accuracy is 72.27%, against
78.52–88.67% for the five requested reference families. All selected ridge
optima are interior. The English shortfall remains, while sentence-pair
differences remain inconclusive. These are reused validation diagnostics,
not newly learned quality or general encoder parity.

A stronger compatible Korean Base reference is available locally. The original
Korean common-visible TRAIN/VALID strings, labels and IDs remain unchanged;
all four panel tokenizers fit them without clipping. Its task score is not
measured here. A single topic task still cannot establish broad Korean quality.

Private matched-adaptation preparation declares equal learning-rate, seed,
epoch and input budgets. Learning-rate selection uses the mean VALID metric
across all declared seeds, with every selected seed reported. Picking the best
seed is prohibited. The single-candidate worker retains metrics and predictions,
not adapted deployment weights. A declared comparison budget is not a measured
runtime or an affordable paid execution plan.

Private multilingual retention inputs distinguish natural instruction
validation, fixed-pool retrieval and locally generated short controls.
Generated grounding/counterfactual and arithmetic controls are narrow template
checks. Oracle length, predicted length when trained, EOS and paired response
changes must be reported separately. Merely changing an answer after an input
change does not demonstrate correct grounding. These controls do not supply
broad chat, multi-hop reasoning, token labeling or extractive QA evidence.

Candidate preparation preserves original TRAIN membership, inherited language
assignments and existing VALID consultation limits. It samples without
replacement, records shortages instead of duplicating rows, and preserves
full answers. Candidate publication and independent split/overlap review must
both pass before a pool can be consumed. Partial or failed attempts are
preserved evidence. Source/language counts and length proxies do not establish
domain diversity or absence of semantic near duplicates.

## What remains for a later measured decision

1. Review a bounded experiment with an actual provider quote and calibration.
   Include transfer, startup, assessment, retrieval and teardown, plus prior
   spending, under the strictly-below-USD-50 aggregate limit.
2. Verify the selected GPU's actual batch fit, forward/backward behavior,
   checkpoint recovery, throughput and cost cutoff. Local CPU mechanics cannot
   provide those measurements.
3. Run only the separately accepted bounded comparison or training candidate.
   Measure language-specific task gains and retention; preserve failed and
   inconclusive results. A better probe or lower loss does not authorize full
   production continuation.
4. Expand missing language/task coverage and agree on practical acceptance
   margins before claiming broad quality. Japanese encoder tasks, broader
   Korean tasks, semantic similarity, token labels, extractive QA and independent
   natural instruction completion coverage remain incomplete.

D-021's comparator, reserved completion TEST policy, architecture defaults and
five-release scope remain unchanged. Real GPU evaluation and broad improvement
are not complete. Prepared code, cached files and passing CPU tests support
the next investigation; they do not guarantee BERT-family quality.

Dataset identities, links, provenance, example text, detailed predictions,
sampling choices and private receipts stay outside Git. Local preparation
artifacts must remain private until a separate authorized transport or release.
