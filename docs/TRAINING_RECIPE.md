# Opt-in training recipe controls

`scripts/train.py --recipe-config <private JSON>` exposes a small set of existing
training controls for separately reviewed experiments. `NURI_RECIPE_CONFIG`
supplies the default path; an explicit CLI path takes precedence. Real recipe
files and their input/exposure notes belong in ignored `.private/`.

Without a recipe, task construction, defaults, RNG draws and legacy resume
behavior remain unchanged. The D-021 Dense repair comparator still uses its
frozen 50/50 MLM/span mix and explicit peak LR 1e-4. Adding this interface does
not change that comparator, the architecture, five intended releases, immutable
packages or permission to perform paid learning. No candidate settings are a
measured optimum or evidence of quality improvement.

## Schema version 1

A recipe declares exactly one `stage`: `base`, `it` or `embed`, matching the CLI.
Unknown fields, duplicate JSON keys, unsupported stage settings and invalid
numeric values are rejected. The only top-level fields are `schema_version`,
`stage`, `task_mix`, `loss` and `corruption`. The last three are optional.

This example explicitly declares the current Base controls; it is not a new
recommended ratio:

```json
{
  "schema_version": 1,
  "stage": "base",
  "task_mix": {"mlm": 0.5, "span": 0.5},
  "corruption": {"mlm_rate": 0.3, "min_span": 16}
}
```

| Section | Base | IT | Embed |
|---|---|---|---|
| `task_mix` keys | `mlm`, `span` | `instruction`, `reasoning`, `span` | `pairs` |
| `loss` keys | None | `length_weight` | `embed_dims`, `temperature`, `cross_mode_weight` |
| `corruption` keys | `mlm_rate`, `min_span` | `min_span` | None |

Task weights are positive finite relative weights, and their sum must be
finite. An explicit mix may select a subset of the available stage tasks; an
omitted task is not sampled. Zero weights are rejected: omit a task explicitly
when preparing that comparison. Requested tasks must have their training
inputs. For example, IT `reasoning` requires a positive `--reasoning-examples`,
and IT `span` requires `--corpus`. Missing requested tasks never receive silent
weight redistribution. Without `task_mix`, the existing stage mix and its
existing missing-input redistribution are retained. All available task
objects remain present for resume and assessment.

Weights select accumulation microbatches, not tokens or gradient contributions.
Corpus weights still select documents. Use the private exposure audit and
language/capability validation to assess actual coverage; increasing an English
weight is a hypothesis and must not hide Korean or Japanese regression.

`mlm_rate` is in `(0, 1]`, default 0.3. It controls both unpacked and packed MLM;
the existing special-token exclusions and at-least-one-target rule remain.
`min_span` is a positive integer no greater than the model's response limit,
default 16. It sets the minimum random continuation span when enough document
tokens remain; shorter document tails still produce shorter spans. EOS and
prefix/target separation are unchanged.

`length_weight` and `cross_mode_weight` are finite and nonnegative. Their
defaults remain 0.1 and 0 respectively. `temperature` is finite and positive,
default 0.05. `embed_dims` contains 1 to 8 strictly increasing positive integer
prefixes, each no greater than the model's embedding dimension. Omission keeps
the existing `(64, 128, 256)` setting. Explicit incompatible dimensions are
rejected rather than silently clipped. These controls do not change embedding
geometry, candidate pools or the model architecture.

An explicit `--cross-mode-weight` conflicting with the recipe is rejected.
Otherwise, the recipe value is used when declared, or the CLI/default value
is retained. Opt-in IT that disables length supervision, or selects only span,
does not newly set `length_head_trained`; a previously trained length head
retains its flag. No-recipe IT keeps its legacy stage behavior.

Learning-rate schedules, batch size, accumulation, packing, bucketing and
runtime backend settings remain separate CLI controls. Canvas full-mask
probability/range, new objectives, MLM replay during IT, auxiliary loss weights,
architecture changes and decoding policies are not part of this schema.

## Assessment and resume binding

Recipe parsing, static task availability and model-geometry checks happen
before constructing a model. An invalid recipe does not allocate model weights
or create a training output. Data eligibility remains checked by the existing
preparation and batch-building paths.

Aggregate and optional language holdouts use the same recipe corruption and
loss controls as training. Assessment visits every available holdout objective;
it does not draw tasks from the training mix or require an identical holdout
task set. Fixed corruption seeds, bounded coverage and completion-only TEST
rules remain in effect. A changed mask rate or span-length policy changes the
assessed target distribution: compare candidates additionally under an
independent frozen policy. Changed-policy losses alone are not improvement
evidence, and existing repair-gate policies must remain frozen.

Each recipe task records a normalized, path-free snapshot and its SHA256 in
`Task.options`. Full training states persist those options; holdout receipts
also include the recipe digest. JSON key order and file location do not change
the normalized binding. Explicit defaults are still a declared recipe and
cannot be silently substituted for a no-recipe state.

Recipe-enabled CLI resume requires matching saved recipe settings and hashes
before model construction. States without recipe evidence, including very old
states without task options, are rejected. The trainer then applies its
existing complete settings and task-options checks. To change a recipe, use
`--init` and a fresh output directory with a new optimizer/schedule, preserving
the old state and evidence. Removing a recipe from a recipe-bound resume also
fails the existing task-options check.

Programmatic callers use `load_recipe_config(path, stage)` or
`TrainingRecipe.from_dict(data, stage)`, then pass `recipe=` to `base_stage`,
`instruction_stage` or `embedding_stage`. Pass `recipe_loss_settings(recipe)`
to `TrainSettings` as keyword overrides. Call `validate_recipe_resume` on saved
state extras before a programmatic recipe resume: old-state validation cannot
be inferred by the pipeline. Holdout construction may set
`apply_recipe_mix=False` to retain every available assessment task.

The release runner forwards `recipe_config` only when explicitly declared in
an individual stage. Plan-wide recipe settings are rejected so a Base recipe
cannot be inherited by IT or Embed. The training CLI validates the file's
stage and settings; forwarding a path alone does not establish input eligibility
or quality. No command in this document launches training.
