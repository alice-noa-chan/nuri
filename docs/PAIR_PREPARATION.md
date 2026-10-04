# Private pair auditing and preparation

`tools/prepare_pair_views.py` measures lexical overlap and can make an immutable
training-only selection. It reads existing positives; it never creates labels,
changes an original file or assigns a new train/evaluation split. A low overlap
score alone does not establish that a positive is relevant.

The default mode only audits. Keep the JSON plan and all real paths in ignored
`.private/`, and run from the repository root:

```powershell
python tools/prepare_pair_views.py --plan .private/pair-plan.json
```

The private plan needs `job`, `tokenizer`, `train`, `eval` and `max_input_len`.
The job is a fresh opaque name. Inputs are canonical UTF-8 JSONL objects with
only string `query` and `positive` fields. Source metadata is rejected instead
of copied into runtime examples. The tokenizer may be a `tokenizer.json` file
or its folder. Set `max_input_len` to the evaluated checkpoint's exact limit.
Optional `expected_sha256` binds any of the three inputs to an expected hash.
Inputs are read only; symlinks and redirected paths are rejected.

Without a plan, audit paths come from `NURI_PAIR_DATA`, `NURI_EVAL_PAIRS` and
`NURI_TOKENIZER`; `NURI_PAIR_VIEW_JOB` and `NURI_PAIR_MAX_INPUT_LEN` provide the
job and input limit. `NURI_PAIR_VIEW_PLAN` can supply the private plan path.
Explicit `--plan` overrides that environment variable. Optional `--job` and
`--max-input-len` override the plan's corresponding settings.

Audit tokenization uses `add_special_tokens=False`, exactly as retrieval does.
Each side keeps its first N tokens before removing tokenizer-marked special
IDs for the overlap calculation. Optional `special_token_ids` adds explicit
IDs to exclude. These exclusions affect the overlap diagnostic, not saved text
or model inputs. The score is token-set Jaccard. Zero overlap is separate from
an undefined overlap caused by an empty lexical side; undefined rows never
count toward lower-overlap support. Counts include retrieval eligibility and
truncation. Configurable `overlap_bins` defaults to edges
`[0, 0.25, 0.5, 0.75, 0.8, 0.9, 1]`. These are diagnostic intervals, not an
optimal training cutoff. Zero and empty lexical views have separate counts.

To prepare a derived training view, explicitly add `--prepare` and configure
all four fields in the private plan's `selection` object:

| Field | Meaning |
|---|---|
| `threshold` | A pair is high overlap when its defined score is at least this value. |
| `max_high_overlap_pairs` | Maximum high-overlap training pairs to retain; zero drops that category. |
| `min_low_overlap_pairs` | Required count below the threshold; a positive integer, excluding undefined scores. |
| `seed` | Explicit integer controlling deterministic high-overlap membership. |

All lower-overlap training rows remain. Undefined-overlap rows also remain,
but provide no lower-overlap support credit. High-overlap membership uses the
smallest SHA256 ranks of `seed:original_row_ordinal`; ties use original order.
The output preserves original training order. The support check must pass and
at least two retrieval-eligible training pairs must remain. A larger cap may retain every original
high-overlap row. No threshold or cap is silently chosen for preparation.

```powershell
# Use a new job for each audit or preparation attempt.
python tools/prepare_pair_views.py --plan .private/pair-plan.json `
  --job pair-view-v2 --prepare
```

Audit records, source paths/hashes, the effective plan, row-level overlap and
selection ledgers, explicit retained/excluded/undefined counts, output hashes and full verification live only in
`.private/pair-views/<job>/`. Row IDs are one-based nonblank-record ordinals
bound to the source hash; physical line numbers are also retained privately.
The tool streams records and hashes rather than storing full tokenized inputs;
selection memory grows with the configured high-overlap cap.

Preparation writes only canonical examples to
`data/prepared/pair-views/<job>/{train,eval}.jsonl`. Evaluation is a byte-for-byte
copy, including all original candidate documents. Training content and order
are independently replayed from the original rows and private selection ledger.
Original hashes and file metadata are rechecked before completion. A generic
runtime `COMPLETE.json` contains no sources or audit details; the final private
completion marker binds the private verification report. Audit-only mode writes
no runtime examples and uses only the private completion marker.

Both destination trees and the plan must be ignored and untracked. Jobs never
overwrite a previous success or failure; a failed attempt keeps private failure
evidence, removes any successful completion markers, and needs a new job name.
Keyboard interruption inside the active job follows the same cleanup and stores
a private cancellation record before returning a generic cancelled error.
Original group validation remains the input
producer's responsibility because canonical files contain no group metadata.
This tool preserves their existing split membership; it does not detect semantic
overlap between different rows, verify positive relevance or measure retrieval
quality. Review those properties before choosing a policy or adding new inputs.

Progress bars show generic phases and byte counts. `--no-progress` disables
them. Console output contains only completion mode or generic errors; detailed
audit results remain private.
