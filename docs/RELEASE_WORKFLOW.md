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
