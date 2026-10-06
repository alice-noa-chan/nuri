# Improvement across supported languages and capabilities

Status: preparation and review, following the user's 2026-10-06 instruction.
Improve Korean as well as English, give English additional priority, and keep
Japanese coverage explicit under D-002. The target covers Nuri's understanding,
embedding/retrieval and generation functions. It is not a promise of competence
in every possible domain. No new model learning or quality gain is claimed by
this preparation. Full paid continuation remains on hold.

The [offline preparation record](PRE_GPU_PREPARATION.md) now covers committed
recipe controls, exposure auditing, local reference checks and matched
adaptation mechanics. The separately declared
[wider frozen-probe diagnostic](ENCODER_BENCHMARK_RESULTS_V2.md) is complete.
Its score changes do not represent new learning.

[ARCHITECTURE.md](ARCHITECTURE.md) remains the design authority. This plan does
not change the five releases, agreed Base mix, tokenizer, repair thresholds,
reasoning pilot decision or immutable training package. A changed training
recipe needs its own versioned comparison and review.

## Evidence and missing coverage

The [original frozen comparison](ENCODER_BENCHMARK_RESULTS.md) and
[versioned replacement diagnostic](ENCODER_BENCHMARK_RESULTS_V2.md) are the measured
starting point. Its validation rows selected the ridge regularization; they
are not completion TEST. It establishes limited representation utility, not
full downstream fine-tuning quality.

| Capability | Current evidence | Next useful comparison |
|---|---|---|
| Topic and sentiment classification | Useful Korean topic features; a clear English sentiment deficit against all five requested Base references | Broader Korean/English tasks, unchanged frozen probes, then matched task-specific adaptation |
| Sentence relationships | English pair differences are inconclusive | Korean/English relationship tasks; matched pair heads and adaptation, with fixed visible inputs |
| Semantic similarity and retrieval | Trained semantic embedding quality is unassessed; Base projection weights are not trained for this purpose | Actual contrastive adaptation, fixed query/document candidates, Recall@k/MRR and lexical controls |
| Grounded short answers | Corrected natural-language behavior is unassessed | Korean/English context-question answers, input counterfactuals, answer review and correct length/EOS accounting |
| Instruction-conditioned generation | Legacy failure is measured; the position correction has structural evidence | Actual corrected Base/IT updates, decoded validation answers, coherence/repetition and refinement cost |
| Structured reasoning | Synthetic mechanisms and controls exist | New combinations, exact match, genuine input interventions and T/S curves; Japanese templates are not implemented |
| Token labeling and extractive QA | Dedicated heads/alignment/evaluators are absent | Prepare aligned token labels and start/end supervision as distinct work units before claiming these abilities |
| Robustness and usable context | Padding/cache/packing mechanics have tests | Language/task validation by length, domain and input-order perturbation, with truncation and latency reported |

Use Korean and English as separate primary assessment columns. Include genuine
Japanese inputs where prepared and label other Japanese cells unassessed.
The synthetic reasoning generator currently defaults to Korean/English;
passing Japanese selects the English template branch. Do not advertise that
as Japanese reasoning. Generated grounded answers and extractive start/end
prediction are different assessment paths.

## Improvement and retention decisions

Keep every required language/task cell visible. A weighted mean may summarize
results, but it cannot hide a failed or missing cell. English improvement does
not compensate for a Korean regression, and preserving Korean alone does not
fulfill the user's request to improve it. Final improvement claims need
positive evidence on the declared primary metrics in each required
language/capability cell.

Freeze the task packet, splits, reference checkpoints, adaptation/selection
budget, seeds, input limits, primary metrics, practical improvement margins
and tolerated regressions before scoring a candidate. Numeric margins are
not selected here or inferred from the previous result. Distinguish early
diagnostic eligibility from final acceptance: a retention check permits
further investigation, while a claimed broad improvement requires measured
gains. A missing metric stays unassessed, rather than becoming zero or passing.

For the encoder, retain the five requested Base families with compatible
language coverage and matched protocols. Classification uses accuracy/macro-F1;
token labeling needs aligned token metrics; similarity needs declared
correlation; extractive answers need matched answer metrics. Compare model
parameters actually used and adaptation cost. Never rank different-tokenizer
models by raw MLM cross-entropy.

Retrieval fixes the full document candidate pool and relevance labels across
checkpoints. Report fast/reasoning modes, Matryoshka dimensions, lexical
overlap strata, false-negative handling and measured latency. In-batch
contrastive validation loss is supplementary rather than a retrieval score.

Generation uses grounded short tasks plus human-read open answers. Keep
predicted length separate from oracle length, and report the length-only
baseline, exclusions, early/missing EOS, repeated text and source sensitivity.
Fix decoding inputs/policy and report actual readout calls and latency for
S/block comparisons. A Base explicit-length result does not establish trained
automatic length prediction or IT instruction following.

Use TRAIN for learning and exposure audits, VALID for selecting candidates,
and completion TEST only for the final frozen assessment. If repeated
development on the same validation packet overfits its scores, prepare a new
privately audited development holdout; never spend reserved TEST on tuning.
Report conditional interval limits and seed variability where measured.

## Work sequence

1. **Separate evidence by language and function.** Optional `language_groups`
   now prepare separate stage-objective curves in
   [HOLDOUT_MONITORING.md](HOLDOUT_MONITORING.md#separate-language-objectives).
   Keep a separate cross-capability validation packet across Base, IT and Embed
   checkpoints: these stages have different objectives, so one stage's loss
   curves cannot check retention of all the others. Expand the unassessed cells
   with privately reviewed inputs and matched evaluators.
   Correct the probe search before treating the current gap as a model ceiling:
   19 of the 21 v1 reports select the upper regularization boundary. The
   separately versioned equal-budget wider grid, explicit boundary flags
   and immutable candidate reports are now complete; all v2 selections are
   interior. Preserve the originals and record prior
   diagnostic consultation of these validation rows. Add a compatible Korean
   RoBERTa-class reference and broader Korean tasks under a declared panel
   revision; do not overwrite v1 or claim new encoder learning from rescoring.
2. **Audit actual training exposure privately.** Count accepted/excluded rows,
   unique and repeated presentations, input/output tokens and loss-bearing
   targets by language and objective using TRAIN only. Inspect quality, lengths,
   duplicates and domain diversity. Corpus weights select documents; task
   weights select microbatches. Neither equals token, target, gradient or compute
   share. A weak English score alone does not establish English undersampling.
3. **Prepare controlled recipe candidates.** Preserve the D-021 comparator.
   Use measured exposure to propose extra English coverage with explicit
   Korean/Japanese coverage. Assess dynamic masking and document isolation
   already present in the code, rather than calling them missing features.
   Test one changed cause at a time: exposure, supervised target distribution,
   warm-start schedule or task adaptation. Keep actual compute/targets and
   cross-capability retention visible; do not invent a beneficial ratio.
4. **Distinguish pooling from encoder limits.** Prepare equal-budget downstream
   adaptation against the requested references. A frozen mean-pooled deficit
   does not prove that fine-tuning cannot help. Use repeated seeds and identical
   validation search before making a broader encoder-family claim.
5. **Train capability-specific paths with retention checks.** The span path
   trains core/readout, IT trains natural response length and instruction
   behavior, and contrastive pairs train embeddings. More MLM alone cannot
   supply all three. IT has span replay but no MLM; Embed uses pairs only.
   Forgetting is a hypothesis to assess before/after each stage, not an already
   measured cause. Review replay or objective changes separately if supported.
6. **Choose the next bounded learning experiment from evidence.** Freeze its
   outputs, stop boundary, recovery state, all-in cost and validation decisions
   before renting. Include prior spending under the aggregate below-USD-50
   limit. No score trend or supplementary curve automatically resumes full
   production training.

The current 50/50 Base recipe uses differently normalized MLM/span losses.
Equal batch probabilities are not equal gradient contributions. Full-mask
corruption differs from the generated-context trajectory; a different mask
rate is a candidate, not an established correction. Accumulation increases
optimizer batch size without enlarging a microbatch's contrastive candidate
pool. Keep these mechanisms and measurements distinct.

## Research basis and limits

[RoBERTa](https://arxiv.org/abs/1907.11692) demonstrates the importance of the
training recipe and examines dynamic masking under controlled comparisons.
[Cramming](https://proceedings.mlr.press/v202/geiping23a.html) provides positive
small-budget encoder evidence after a focused MLM GPU day, with preprocessing
and downstream fine-tuning outside that time. These findings motivate recipe
and exposure audits; they do not forecast Nuri's score across languages or its
additional generation/embedding stages.

[Don't Stop Pretraining](https://arxiv.org/abs/2004.10964) finds benefits from
domain/task adaptation in its assessed settings. Applying that principle to
Nuri is a hypothesis requiring broader retention checks.
The [existing literature review](LITERATURE_REVIEW.md) supplies separate
generation and recurrent-core precedents with their transfer limits.

Data identities, links, permissions, prepared examples, sampling decisions
and exposure ledgers stay in ignored `.private/`; raw/prepared inputs stay in
ignored `data/`. Tracked files contain generic methods and measured aggregate
model results only.
