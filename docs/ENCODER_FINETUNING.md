# Matched encoder-adaptation worker

`scripts/encoder_finetune.py` prepares one declared learning-rate/seed candidate
for a supervised encoder comparison. It updates a fresh encoder copy and a
linear softmax head on TRAIN, then selects an epoch on VALID. This is separate
from the frozen ridge probes in [ENCODER_BENCHMARK.md](ENCODER_BENCHMARK.md).
Synthetic CPU checks establish the worker's mechanics; no real adaptation or
quality improvement is established by this implementation.

The worker reuses the unchanged strict row reader, tokenizer helpers,
content-mean pooling and selective encoder loaders from `encoder_features.py`.
Nuri uses only its input embedding and input encoder. Its recurrent core,
readout, embedding projections and vocabulary head are excluded. Reference
encoders keep their existing positions and architecture, excluding an unused
pooler. All copied encoder parameters receive gradients. This initial worker
accepts dense encoders only, without changing production training defaults.

## Private manifests and protocol

The manifest and protocol must be inside an ignored `.private` directory.
Inputs and checkpoints must already exist locally; this worker does not fetch
models or data. Manifest paths resolve relative to the manifest. JSONL rows use
exactly `id`, `text`, `label`, `split` and optional `text_pair`, as in the
extractor. Splits are nonempty, have disjoint row IDs and share a fixed class
universe. Exactly TRAIN and VALID bindings are accepted. A TEST manifest entry
or TEST row is refused, with no completion override.

```json
{
  "schema_version": 1,
  "model": {
    "backend": "nuri",
    "path_or_id": "local-encoder",
    "revision": "explicit-immutable-identity"
  },
  "inputs": {
    "train": {"path": "train.jsonl", "sha256": "exact-file-sha256"},
    "valid": {"path": "valid.jsonl", "sha256": "exact-file-sha256"}
  },
  "label_count": 2,
  "protocol": {"max_length": 128},
  "assessment_protocol": {"path": "assessment.json", "sha256": "exact-file-sha256"}
}
```

`model` also accepts the extractor's optional tokenizer path/revision and
Nuri random-initialization seed. The `transformers` backend requires its pinned
40-character revision and local model/tokenizer directories. The optional
`assessment_protocol` binds the broader frozen comparison. Optional
`model_artifacts_sha256` and `tokenizer_artifacts_sha256` bind the loader's
artifact inventory using `digest_json(records_identity(records))` from the
extractor. Unknown fields are rejected.

The manifest's optional extractor `protocol` accepts its existing geometry
fields. Pooling, pair features and dtype remain content mean,
`concat_u_v_absdiff_product` and FP32. The extractor's `device` field, if present,
remains `cpu`; the adaptation execution device is chosen separately by CLI.
The manifest's visible-token limit must match the adaptation protocol. The
extractor batch size does not override the adaptation batch size.

```json
{
  "schema_version": 1,
  "kind": "encoder_finetune",
  "learning_rates": [0.00001, 0.00003],
  "seeds": [0, 1],
  "epochs": 3,
  "batch_size": 8,
  "max_length": 128,
  "label_count": 2,
  "task_type": "single",
  "optimizer": {"weight_decay": 0.01, "grad_clip": 1.0}
}
```

Declare 1–8 strictly increasing positive finite learning rates, 1–8 distinct
integer seeds in `0..2**32-1`, 1–20 epochs, batch size 1–64, visible length
1–128 per field and 2–1,024 classes. `task_type` is `single` or `pair`.
`optimizer` is optional; its defaults are shown above. Weight decay must be
finite and nonnegative; the gradient clip must be finite and positive. These
bounds limit this worker, rather than identify an optimal training recipe.
Each split is limited to 20,000 rows and 32 MiB, and the loaded encoder to
250 million parameters. These limits do not prove that every allowed shape
fits available GPU memory.

## Computation and selection

Prepare common-visible raw strings before running candidates. The worker
tokenizes all TRAIN/VALID fields in bounded batches before the first update
and rejects any tokenizer truncation or field without content. It checks
truncation again in each batch. Reference wrappers and padding are excluded
from content pooling; unknown content tokens remain included. Nuri excludes
its padding, mask and EOS IDs from pooling. Encoders keep the contextualization
mask supplied by the shared tokenizer helper.

Pair fields are encoded separately with the same encoder. The head receives
`concat(u, v, abs(u-v), u*v)`. It receives the single pooled vector for a
single-text task. Head dimensions and parameter counts are reported; differing
encoder widths naturally yield different head sizes.

The worker uses FP32, constant-rate AdamW with PyTorch's default betas/epsilon,
one update per minibatch, mean cross entropy and combined encoder/head gradient
clipping. There is no accumulation, scheduler, AMP or architecture modification.
A seeded generator shuffles TRAIN rows each epoch. Finite loss and pre-clip
gradient norm are required before every update; model, head and optimizer
tensors are checked for finiteness before publishing completion.

Full supplied TRAIN and VALID rows are assessed after each epoch in eval mode
with gradients disabled. Their loss uses the same row-weighted cross-entropy
reduction. `train_observed_loss` separately summarizes training-mode losses
observed during updates; it is not the final model's eval-mode TRAIN loss.
Macro-F1 includes all declared classes, assigning zero F1 to a class with no
true or predicted examples. Selection uses VALID macro-F1, then VALID accuracy,
then the earliest epoch. TRAIN performance does not select the epoch.

A private outer controller must freeze all jobs, use equal LR/seed/epoch
budgets for each model and declare cross-candidate selection and seed
aggregation before scores exist. It must not silently pick the luckiest seed.
Repeated VALID selection is not independent completion evidence. Lexical and
majority controls remain the separate frozen probes; they are not adapted
encoder architectures.

## Invocation and evidence

```text
python scripts/encoder_finetune.py --manifest <private-manifest.json> --protocol <private-protocol.json> --learning-rate 0.00001 --seed 0 --output <new-private-folder> --device cpu --threads 2 --tensorboard-dir <new-private-events>
```

`--device` accepts explicit `cpu` or available `cuda`; CPU is the default.
Threads are bounded to 1–2. `--no-progress` hides tqdm. Optional TensorBoard
emits numeric TRAIN observed loss and pre-clip norm per optimizer update, plus
TRAIN/VALID eval loss, accuracy and macro-F1 per epoch. There are no TEST events,
paths or example strings in event tags.

Outputs and event directories must be fresh, private and outside source
checkpoint/tokenizer folders. `IN_PROGRESS.json` records the candidate;
`REPORT.json` holds epoch metrics/predictions, selection, row identities,
tokenization coverage, actual optimizer updates/presentations/positions and
observed timings. `COMPLETE.json` binds the report's SHA256. Exceptions retain
an incomplete directory and `FAILED.json`; they do not publish completion.
No prior output is replaced.

The receipt binds the manifest, adaptation/assessment protocols, input bytes,
model/tokenizer artifacts, initial encoder tensors, runtime versions and
worker/extractor/metric/native source files. Runtime source hashes are captured
at worker entry and rechecked after the job, as are the bound input/artifact
bytes. A frozen outer controller should additionally verify the worker bytes
before launching the process. Checkpoints are copied, including tensors loaded
through memory maps, before learning. Source checkpoint bytes stay unchanged.

The worker retains predictions and metrics, **not adapted weights**. Its final
encoder tensor hash describes the last epoch, explicitly identified by
`final_tensor_epoch`; it does not identify a saved selected-epoch checkpoint.
This result supports a declared diagnostic comparison, not deployment,
resume, broader language/capability quality or reference leaderboard parity.
Any real adaptation and later completion assessment require their own reviewed
inputs, budget and execution. Existing immutable runs and D-021's frozen
comparator remain unchanged.
