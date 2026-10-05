# Training, validation and test monitoring

Training can optionally assess explicit validation and test inputs and write
separate numeric TensorBoard curves. This is bounded objective monitoring;
the full release evaluation remains a separate task. Existing runs without
holdout settings keep their training computation and checkpoint behavior.

Use `--holdout-config` or `NURI_HOLDOUT_CONFIG` to select a private JSON file.
Both `valid` and `test` blocks are required. Base needs a separate corpus in
each block, IT needs separate instructions and a separate span corpus when
the training stage includes spans, and Embed needs separate positive pairs.
No missing holdout input is replaced with training data.

The private schema is:

```json
{
  "schema_version": 1,
  "valid_every": 2000,
  "batch_size": 32,
  "input_manifest": "runtime-inputs.json",
  "input_manifest_sha256": "<SHA256 of the manifest>",
  "valid": {
    "corpus": ["valid-token-corpus"],
    "instructions": "valid-instructions.jsonl",
    "pairs": "valid-pairs.jsonl",
    "batches_per_task": 8,
    "seed": 0,
    "depth": 4,
    "exposure_caveats": ["Record known prior consultation here."]
  },
  "test": {
    "corpus": ["test-token-corpus"],
    "instructions": "test-instructions.jsonl",
    "pairs": "test-pairs.jsonl",
    "batches_per_task": 8,
    "seed": 0,
    "depth": 4
  }
}
```

Paths are relative to the configuration file unless absolute. Corpus entries
also accept `folder=weight`. A configuration can supply resources for several
stages; each stage uses its own objectives. Instruction and pair JSONL formats
match the training CLI. Optional `reasoning` in either block is an explicit
instruction-format JSONL resource. The CLI does not generate test reasoning
examples implicitly. An unassessed training task is named in the report.

`--valid-every` overrides the private interval. Zero means completion-only
validation; the default has no periodic validation. Test runs only when all
optimizer updates finish. Use validation, never test, for a repair gate or
hyperparameter selection. Previously consulted data do not become pristine
test data simply by receiving a new split label.

Before allocating the model, the CLI rejects shared resolved paths across
training, validation and test. The optional manifest contains
`files: [{"path": "...", "sha256": "..."}]`, with paths relative to that manifest.
It must cover all configured resources, including corpus `tokens.bin` and
`offsets.npy`. Exact duplicate resource hashes across splits are rejected;
equal offsets alone are allowed. These are preparation metadata checks.
Deployment preflight must still verify actual input bytes and grouped split
hygiene. A different path alone does not prove that its content is unseen.

Each assessment repeats the same batches and corruptions, uses eval mode and
no gradients, and restores module modes, Python/Torch/CUDA random states,
task generators and MoE telemetry. Text corpora remain memory mapped and only
one batch is evaluated at a time. Instruction and pair sources use a fixed
uniform reservoir with bounded retained memory; source, eligible, excluded
and selected-pool row counts are reported. Ineligible rows follow the existing
training batch rules, including response capacity. No response is truncated
to create a held-out target. Sampling can repeat rows; presentation counts
are not unique coverage or exhaustive epochs.

`holdout_log.jsonl` records `coverage: bounded_sample`, batch and selected-target
SHA256 digests, assessment policy binding, actual rows/batches/targets,
separate input/canvas positions, loss numerators and denominators, and exposure
caveats. MLM CE is weighted by supervised tokens. Canvas denoising CE is
weighted by target-bearing rows; natural length CE is weighted by all rows.
Pair objectives average fixed-batch contrastive losses by pair rows and state
each in-batch candidate count; this is not full-pool retrieval evaluation.

TensorBoard provides `train/<task>/objective_loss`,
`valid/<task>/objective_loss` and `test/<task>/objective_loss`, plus objective
components and numeric coverage counts. The training curves are observed
pre-update microbatch losses, averaged over sampled microbatches for that task.
Held-out curves pool their exact objective denominators over fixed batches.
Neither curve includes MoE auxiliary penalties. Existing `loss/total` and
routing tags still describe the optimized training total. TensorBoard's only
holdout text tags are the fixed coverage label and policy/batch/target SHA256
digests. It receives no raw input text, source names, arguments or paths.

Final assessment finishes before `final/model.safetensors` appears. A stop
before or between assessment batches restores training state, writes the
latest full checkpoint, and exits with code 75. Resume at the completed final
optimizer boundary finishes pending assessment before publishing final
weights. Atomic `holdout/valid-stepN.json` and `holdout/test-stepN.json` receipts
are written after successful logging and TensorBoard flush. Reuse requires
the exact streamed model-weight/config digest, objective settings, holdout
policy, split/step and complete finite task counts, metrics and target digests.
An arbitrary or interrupted JSONL history entry cannot skip final assessment.
A matching receipt in the same output directory is reused at that boundary
rather than consulting test again. Existing final weights and configuration
must match the resumed model in the export's stored precision. Mixed-run
output reuse fails before assessment and requires a fresh output directory;
it preserves the earlier final artifacts. A same-model final boundary with
a missing or damaged receipt is reassessed and saved again after success.
Non-finite or failed assessment retains a
full recovery state and publishes no final weights. If an append was interrupted,
its incomplete history line is preserved; atomic receipts are the authoritative
completed records.
