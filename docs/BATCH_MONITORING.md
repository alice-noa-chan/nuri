# Batch geometry and memory monitoring

Training writes numeric batch information to each sampled `train_log.jsonl`
record and to TensorBoard when it is enabled. The progress bar shows actual
row presentations, the largest input/canvas widths and the valid-token share.
CUDA runs also show peak allocated memory in GiB.

`batch_shapes` describes each microbatch in the logged optimizer update:

- Task kind, actual rows and the depth the core runs (zero for MLM).
- Padded and valid input/canvas positions, and loss-bearing targets.
- Input/canvas widths; pair batches also record separate query/document widths.

`batch_summary` sums row presentations and token counts across those
microbatches, and records their largest row count and widths. Pairs count both
input views as processed input tokens, but one pair as one row. Packed MLM
rows may contain several isolated documents: rows are not unique documents.
The valid fraction divides valid input plus canvas positions by their padded
total. These counts do not change the equal-microbatch loss reduction.

TensorBoard writes these values under `batch/`. Existing cumulative
`data/processed_tokens` and target totals retain their meaning. Batch shapes
describe the logged update, not every update between log points. Use
`--log-every 1` during actual-batch calibration to observe every update.
No input tokens, text, source paths or dataset identities are logged here.

On CUDA, `cuda_peak_allocated_bytes` and `cuda_peak_reserved_bytes` report
allocator peaks since training began or the previous sampled record. They
include intervening updates and may include checkpoint operations between
records. TensorBoard writes them under `memory/`. CPU runs omit these fields;
zero would falsely suggest a GPU measurement. Reserved memory includes cached
allocator blocks. These are allocator measurements, not all device memory or
a guarantee that another batch will fit.

For packed or bucketed training, validate actual rows, canvas targets and
memory together. A bound on combined padded tokens alone does not bound
latent slots or vocabulary logits. Exercise the planned task mix, complete
canvas masking and the largest trained depth before fixing a release batch.
CPU correctness checks cannot replace that GPU measurement.

When resuming into the same TensorBoard directory, the monitor keeps existing
event files and purges steps after the checkpoint. TensorBoard reads filenames
in lexical order, so a new writer may wait until the next wall-clock second
before opening its file. This avoids reading same-second writer counters,
process IDs or host names in the wrong order. If existing events are dated in
the future, correct the system clock or use a new `tensorboard_dir`; the monitor
raises an error before writing. It waits at most once, for about one second;
a clock that does not advance or an unparseable event filename also raises a
clear error without changing previous event files.
