# Comparing frozen encoder features

This diagnostic asks whether the retained input encoder has useful sentence
representations before more production training. It fits a small supervised
head on frozen features. It does not update an encoder, evaluate the generator,
or establish performance after full task-specific fine-tuning.

The feature extractor and ridge-probe runner are implemented. The first
comparison is running on CPU. Exact inputs, model revisions, protocol receipts
and detailed results stay in ignored private storage.
Never put data names, sources or example text into this document or Git.

## What is compared

Use the input encoder's content-token masked mean. For Nuri this is X, before
the reasoning core, readout and contrastive projections. The latter projections
have not been trained by Base pretraining. Loading the input embedding and
encoder alone also avoids treating unused parameters as evaluation compute.

For a sentence pair, encode each sentence separately and concatenate its two
means, their absolute difference and their elementwise product. Use the same
features for every encoder. This is a separate-encoding probe; it does not
measure joint sentence attention or a fully fine-tuned pair classifier.

The English panel contains Base checkpoints from all five requested reference
families. Korean tasks use Korean-compatible references in a separate panel.
Predeclare the versions and languages. A multilingual model can have a larger
embedding table, so report the actual used parameters rather than assuming
equal size. Do not compare incompatible pretraining losses across tokenizers.

Freeze a shared visible text prefix that fits every applicable tokenizer before
extracting features. Exclude padding and tokenizer-added special tokens from
pooling, use evaluation mode, and record encoded lengths, any truncation and
actual extraction time. Do not silently give one model more original text.

## Splits and adaptation

Prepare integer-labeled rows with stable IDs. Separate original text/family
groups before selection, then audit exact duplicates again after clipping.
Keep publisher-held-out groups outside head fitting and record selection
privately. If native holdouts fail isolation, a privately declared diagnostic
may instead assign wholly isolated components of the original training pool
to independent internal train/valid/reserved-test splits before any scores.
Label that task as train-pool-derived, not an official publisher dev/test score.
Keep the original all-label closure and cross-native exclusions. Neither policy
proves absence from the encoders' earlier pretraining corpora.

The preliminary protocol uses up to 1,024 training rows and 256 validation
rows per task. Completion test remains reserved. All models receive the same
rows, labels and representation recipe for a task.

Standardization uses training features only. Fit centered ridge regression
with an unpenalized intercept and one-hot class targets. Select from the fixed
regularization grid `0.01, 0.1, 1, 10, 100` using validation macro-F1, then
accuracy, then the declared grid order. Report both accuracy and macro-F1,
the class universe and assessed counts. A deterministic head has no optimizer
seed; a random encoder control must declare its initialization seed.

Include training-majority, class-prior, hashed character-feature and random
encoder controls. A useful encoder should add evidence beyond these controls.
They help identify a weak task or lexical shortcuts; they do not turn a narrow
task panel into a complete measure of understanding.

## Running the diagnostic

Install the project's `hf` extra for reference encoders and `tensorboard` extra
for events. The native path also works with the ordinary project dependencies.
Keep the manifest, assessment protocol, output caches and reports under
`.private/`; raw/prepared rows stay under `data/`. Paths in a manifest are
resolved relative to that manifest's directory.

The feature manifest has `schema_version: 1`, a fixed `label_count`, a `model`
object, `inputs.train` and `inputs.valid`, and an `assessment_protocol` object.
Each input and the assessment protocol declare `path` and `sha256`. The model
declares `backend`, `path_or_id` and `revision`. Use `nuri` for a native
checkpoint or `transformers` for a compatible Base encoder. Reference revisions
must be immutable 40-character commit hashes; local weight/tokenizer bytes are
also hashed. An optional native `random_initialization_seed` creates a control
using the same geometry and tokenizer, without loading learned weights.

Input JSONL rows contain only `id`, `text`, integer `label`, `split`, and optional
`text_pair`. A task cannot mix single-text and pair geometry. The declared class
universe is fixed even if a class is absent from a particular split.

```powershell
python scripts/encoder_features.py --manifest .private/benchmark/features.json --output .private/benchmark/features-v1 --splits train valid --threads 2
python scripts/encoder_probe.py --train-features .private/benchmark/features-v1/train.npz --valid-features .private/benchmark/features-v1/valid.npz --protocol .private/benchmark/assessment.json --output .private/benchmark/probe-v1.json --bootstrap-samples 2000 --tensorboard-dir .private/benchmark/events
```

These are generic paths; create the private inputs and bindings before running
them. Outputs must be fresh. Feature extraction uses FP32 and at most two CPU
threads, with per-field batches of eight and at most 128 encoded tokens. Reference
encoders use eager attention and disable reference compilation when exposed by
their configuration. Required missing encoder weights are rejected; an unused
pooler and surplus pretraining heads are recorded separately. This is a declared
CPU diagnostic rather than a comparison of optimized GPU inference kernels.

Some pinned upstream checkpoints only provide binary weights. Before using
them, verify the original file hash, load with `torch.load(weights_only=True)`
on CPU, and convert every named tensor to safetensors without casting or
removing aliases. Clone shared storage into separate contiguous tensors while
preserving both key names. Reread the derivative and verify every shape, dtype
and raw tensor byte sequence, finite values, and the unchanged original file.
Record upstream revision, original and derivative hashes privately. The
upstream revision describes lineage; the actual loaded derivative is also
hash-bound. Keep strict required-weight loading checks.

Use a separate whole-process deadline that remains active for the entire
owned process group. Preserve failed attempts and their logs. A continuation
can reuse only verified completed results under the unchanged assessment
protocol; remaining jobs use fresh artifacts. Bind any weight-format
conversion separately. Report active processing time separately from transfer,
preparation and gaps between attempts.

Progress identifies loading, the split/text field, feature batches and probe
regularization candidates. TensorBoard contains real `train` and `valid` probe
metrics; no test curve is fabricated for a reserved split. Add
`--reference-report` to the probe for matched paired score differences. The
probe checks exact assessed row order, labels, class universe and assessment
protocol across models. Its memory guard defaults to 512 MiB for array workspace;
an explicit larger allowance may be needed for larger feature tables.

The extractor writes compressed NPZ arrays, a detailed report and a bound
completion receipt after successful extraction and source rechecks. Partial
folders are not complete caches. The probe refuses TEST features. An extractor
test opt-in alone does not authorize changing a completion protocol.

## Reading results

Validation selects the regularization and supplies the preliminary scores.
Consequently these are selection scores, not an independent final test.
Paired row bootstrap intervals can describe uncertainty on these fixed rows
and the selected head. They do not include training-sample, initialization or
hyperparameter-selection uncertainty. Do not call a small score difference a
general advantage when the interval or assessed scope does not support it.

Report every reference and every assessed task, including failures. Distinguish
proximity to an older reference from proximity to the stronger references.
Do not average unrelated tasks into an unexplained quality percentage.

Matching these probes is preliminary evidence. Missing their scores does not
prove that full fine-tuning would fail. A stronger comparison needs matched
encoder fine-tuning, equal validation search and seed budgets, and a frozen
completion test. Token labeling, extractive answers, retrieval, Japanese and
generation also require their own assessments.

Keep all adapted heads and caches separate from source checkpoints. Bind them
to input, model, tokenizer and protocol hashes. Refuse stale caches, invalid
features, mismatched class universes and train/validation overlap. A failed
run must not leave a successful completion receipt.

This diagnostic authorizes no paid server or automatic production training.
The user reviews the actual evidence before committing the remaining budget.
