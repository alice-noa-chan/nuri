# Training, validation and test preparation

Use `tools/prepare_data.py --plan <private-plan>` to prepare a new immutable job.
Plans, input identities, group assignments, counts and hashes stay in `.private/`.
Prepared examples stay in ignored `data/`. Original files are read only.

Older plans continue to produce `train/eval` with the same grouping, ordering and
deduplication. Three-way preparation requires an explicit `split_policy`:

- `three_way`: specify positive `valid_fraction` and `test_fraction`, whose sum
  is less than one. The remaining original groups are training candidates.
- `preserve_legacy_train`: preserve the existing `eval_fraction` and input
  membership, and specify `valid_share` for the old held-out group pool. Bind
  reconstructed legacy file bytes with private `legacy_hashes` before reusing
  an existing checkpoint or training corpus.

The deterministic group hash uses the original row's `group_field`, or its full
normalized content, before text chunking. Identical normalized content, including
shared chunks, connects whole groups. Duplicate-connected held-out groups receive
one validation/test destination. Conflicting explicit `valid/test` assignments
fail. Duplicate records appear once. This checks exact content and configured
group identities; it does not detect semantic paraphrases or unprovided family IDs.

Known consulted held-out rows can be supplied in a private `exposed_eval_rows`
JSON list. Each entry binds `kind`, `lang`, zero-based legacy held-out `ordinal`
and canonical record `digest`. The complete connected group moves to validation.
Bad indices or hashes fail. Reconstruct token-document ordinals using the original
tokenizer and minimum-length rule before supplying prepared-row ordinals.

A retrospective split does not erase prior training or evaluation exposure. Never
move old training documents into a supposedly unseen test. Record earlier probes,
incomplete exposure records and already-consulted evaluation pools privately, and
carry that limitation into reported results. Use validation for repeated checks;
run test only at completion. Bounded monitoring is not exhaustive evaluation.

Each completed three-way job writes a private consumption manifest containing
explicit split paths, counts, file hashes and group/row evidence. Runtime files
contain only normalized examples and a source-free completion marker. An existing
job is never overwritten. Failed or cancelled jobs retain partial evidence without
a successful completion marker. Immutable raw imports can be reused only with an
explicit `reuse_raw: true`, the exact imported path and an expected SHA256.

`scripts/tokenize_corpus.py --plan <private-plan>` consumes this completed manifest
and binds corpus output to the exact tokenizer hash and loaded vocabulary size.
Its private plan contains `schema_version: 1`, `job`, `tokenizer`,
`tokenizer_sha256`, `vocab_size`, `preparation_manifest` and its SHA256; batch size
and minimum tokens are optional. An optional `reuse_train` mapping binds existing
training corpora to identical prepared input and token/offset hashes. This avoids
retokenizing unchanged training input. Reuse also requires the original job's
completion marker to bind the same loaded tokenizer hash and vocabulary size;
older vocabulary corpora cannot silently pass as final inputs. Validation/test
corpora are fresh. The resulting private manifest includes all canonical and
token/offset runtime files.
The original positional-input tokenization command remains available.

`tools/prepare_pair_views.py` accepts either legacy `train/eval` or explicit
`train/valid/test` paths. Three-way inputs reject exact duplicate pairs, including
cross-split duplicates. Audit remains the default. An explicit preparation cap
affects training only; validation and test bytes are copied unchanged. The tool
consumes prepared splits and never invents a new held-out partition.
Environment users can provide `NURI_PAIR_DATA`, `NURI_VALID_PAIRS` and
`NURI_TEST_PAIRS` instead of the legacy `NURI_EVAL_PAIRS` input.
