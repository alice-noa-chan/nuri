# Release runner holdout settings

Keep real release plans and input configuration files in `.private/`. Dataset
identities and provenance belong only in private records. Paths in the release
plan are used as given, relative to the process working directory when needed.

Each stage can explicitly forward the training CLI's holdout controls:

```json
{
  "stages": {
    "base-dense": {
      "stage": "base",
      "holdout_config": ".private/holdouts/base-dense.json",
      "valid_every": 2000
    },
    "it-dense": {
      "stage": "it",
      "holdout_config": ".private/holdouts/it-dense.json",
      "valid_every": 0
    }
  }
}
```

The runner rejects blank configuration paths, negative or noninteger intervals,
and an interval without an explicit stage configuration before launching any
stage. These two keys belong inside each stage, rather than the plan-wide
`train` block. Other runner settings retain their existing behavior.

Zero means validation at completion only. With a holdout configuration, test
also runs only after all optimizer updates finish for that distinct model
stage. Omitting both keys preserves the training CLI's default behavior; its
environment-variable input remains available to direct CLI users. The training
CLI validates the private holdout schema, stage resources and input-manifest
binding. See [HOLDOUT_MONITORING.md](HOLDOUT_MONITORING.md) for the schema,
loss denominators, fixed corruption and interruption/recovery protocol.

The corrected retry uses a distinct private configuration for each model stage,
derived from the new grouped split manifest. Use periodic validation and
validation-only generation probes for repair gates, checkpoint selection and
setting comparisons. Reserve test for the frozen final assessment protocol.
Consulted data do not become pristine test data by changing their split label.

The optional post-stage `evaluate` suites are separate from training holdout
curves. Their legacy `data.eval_corpus`, `data.eval_instructions` and
`data.eval_pairs` inputs must be chosen explicitly. Use validation resources
for repair or setting comparisons; do not point iterative comparisons at final
test resources.

## Reusing a completed post-stage evaluation

The runner stores a successful evaluation receipt in the ignored output
folder's run summary. It reuses the result only when all of the following
match: final model/config and tokenizer bytes; the complete evaluation
command; evaluator and model runtime source bytes; the Python/platform and
installed package versions; an explicit allowlist of evaluation-input and
device environment settings; exact selected evaluation-input hashes; any
explicit private input manifest; and the output artifact hash. The output
must also remain a valid JSON object. Streaming input hashes cover corpus
tokens and offsets, instruction JSONL or pair JSONL only when their suites
use those resources. Training corpora are not read for this purpose.

An optional private manifest can be supplied as `data.eval_input_manifest`.
Explicit `input_manifest` or `eval_input_manifest` fields at the plan or
stage level, and a stage holdout configuration's input manifest, are also
bound. These receipt hashes identify evidence; they do not replace manifest
coverage verification or grouped split preparation. No full environment,
credentials or raw data are included in receipts.

An existing result without a matching receipt is assessed again. A missing
or damaged result, changed input bytes at the same path, new final weights,
or changed evaluation settings invalidate reuse. Before replacement, the
runner preserves previous result bytes and receipt metadata under
`<out>/eval/history/<stage>/`. It removes the canonical stale output so a
successful child must produce a fresh result. Interrupted or failed
evaluation receives no success receipt; input, checkpoint or source changes
during assessment also fail certification. Use an ignored output directory
and retain this evidence with the private run artifacts.

Receipt reuse prevents routine runner reruns from consulting the same final
test protocol again. It does not make setting comparisons on test valid, or
prove an assessment completed if the runner died before it saved its receipt.
Select checkpoints and decoding settings using validation only. Each distinct
final model stage may receive the agreed end-only test assessment; its frozen
settings must not be selected from those test results.
