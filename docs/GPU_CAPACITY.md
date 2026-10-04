# A100 batch-capacity measurements

Measured on October 4, 2026 (UTC). All 18 synthetic BF16 training cases
completed on an A100-SXM4-80GB. Batch 64 was the fastest of the tested 32/64
choices in the T=4 selection screen. This is a measured candidate for later
training checks, not a change to trainer defaults or a model-quality result.

The earlier [A100/H100 comparison](GPU_COMPARISON.md) remains a separate
measurement at smaller base batches. Only A100 was measured in this follow-up;
there is no controlled H100 batch-64 comparison.

## Workload and verification

- Benchmark source: commit `13b6170998628c0d61c3b825cb4bfd8820018d7e`.
- Vocabulary 65,536; dense 150,652,931 parameters; loop MoE 242,739,203 total.
- Same release topology as the earlier comparison: width 768, 10 input layers,
  2 readout layers, 32 latent slots and QK normalization.
- MoE uses 16 routed experts, top-3 routing, one shared expert and a dense core.
- Input length 1,024; denoising canvas length 256; five warmup iterations and
  20 measured iterations per case, seed 0.
- Each timed iteration includes forward, backward, clipping and an AdamW step.
  Each case runs in a fresh process. Host batch construction is excluded.
- MLM has T=0. The initial denoise/pairs screen uses T=4. Selected shapes also
  measure T=2 and T=8. Native GQA, compilation and grouped MoE remain disabled.
- Packed MLM uses four 256-token documents per row and the actual document
  isolation mask. It still processes a full row with dense attention kernels.

Runtime: Python 3.12.14, PyTorch 2.8.0+cu126, CUDA build 12.6, driver
580.173.02, BF16 and four OMP/MKL threads. The pinned image is
`vastai/pytorch:2.8.0-cuda-12.6.3-py312-24.04-2026-09-08`.
CUDA reports 85,093,777,408 bytes, or 79.25 GiB, of device memory.

The runner completed in 427.76 seconds, excluding provider setup and retrieval.
No OOM occurred. All six conditional checkpointed fallback slots were therefore
not applicable; this run provides no GPU measurement of checkpointing.
Archive CRC and SHA-256, 46 artifact hashes and all 18 report/summary/config
matches were independently checked before resource destruction. The archive
has 47 members when its retrieval manifest is included.

## Batch screen

Input throughput counts each input position once. Pairs count both sequences.
Canvas positions are excluded from the input-throughput column. Memory uses
GiB (2^30 bytes); reserved memory includes the CUDA allocator's cached blocks.

| Workload | Batch | Input tokens/s | Mean seconds/iteration | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|
| Dense MLM | 32 | 104,409 | 0.3138 | 27.41 | 50.75 |
| Dense denoise, T=4 | 32 | 93,911 | 0.3489 | 24.76 | 46.43 |
| Loop MoE MLM | 32 | 84,829 | 0.3863 | 32.48 | 55.40 |
| Loop MoE denoise, T=4 | 32 | 74,025 | 0.4427 | 30.28 | 52.36 |
| Dense MLM | 64 | 108,947 | 0.6015 | 52.96 | 71.48 |
| Dense denoise, T=4 | 64 | 99,518 | 0.6585 | 47.50 | 61.58 |
| Loop MoE MLM | 64 | 89,024 | 0.7362 | 62.03 | 75.37 |
| Loop MoE denoise, T=4 | 64 | 82,153 | 0.7977 | 57.23 | 71.03 |
| Dense packed MLM | 32 | 103,437 | 0.3168 | 28.04 | 51.41 |
| Dense packed MLM | 64 | 108,035 | 0.6066 | 54.21 | 72.60 |
| Dense pairs, T=4 | 32 | 108,790 | 0.6024 | 37.38 | 37.71 |
| Dense pairs, T=4 | 64 | 115,143 | 1.1383 | 72.95 | 73.35 |

Dense and MoE batch selection minimizes equal-weight MLM/denoise T=4 mean
seconds per sample. Pair selection maximizes its measured input throughput.
Batch 64 wins each screen, but larger batches were not tested. The screen
does not establish an optimum for every task mix or for the real training loop.

Packed MLM at batch 64 reaches about 99.2% of unpacked full-row throughput.
These synthetic full rows cannot measure possible padding savings on short
real documents. Packing does not establish a sparse-kernel speedup.

## Depth checks at the selected batch

All rows below use batch 64 without activation checkpointing.

| Workload | T | Input tokens/s | Mean seconds/iteration | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|
| Dense denoise | 2 | 100,917 | 0.6494 | 47.20 | 61.27 |
| Dense denoise | 8 | 96,496 | 0.6792 | 48.10 | 62.19 |
| Loop MoE denoise | 2 | 83,250 | 0.7872 | 56.94 | 70.72 |
| Loop MoE denoise | 8 | 80,220 | 0.8170 | 57.81 | 71.66 |
| Dense pairs | 2 | 116,930 | 1.1209 | 72.35 | 72.56 |
| Dense pairs | 8 | 110,890 | 1.1820 | 74.13 | 74.36 |

Pairs T=8 leaves only about 5.12 GiB beyond allocated tensors on this device.
MoE MLM reserves 75.37 GiB. Fresh-process successes do not establish memory
behavior when a training process mixes tasks, accumulates gradients, writes
checkpoints or evaluates. Confirm actual stage memory before using batch 64.

## Stage-budget estimate

The eight-hour calculation below preserves the original D-009 comparison.
The current D-017 plan uses nine hours for dense and seven for MoE; its derived
projection is shown separately after the historical calculation.

Use mean latency, not the mean of throughputs. For each base family:

```text
denoise_seconds = (seconds_T2 + seconds_T4 + seconds_T8) / 3
mixed_seconds = 0.5 * mlm_seconds + 0.5 * denoise_seconds
timed_iterations = floor(8 * 3600 * 0.9 / mixed_seconds)
input_positions = timed_iterations * 64 * 1024
canvas_positions = timed_iterations * 0.5 * 64 * 256
```

The 0.9 usable-time factor is an assumption. Equal task shares and uniform
depth weights are expectations over a random mix, not exact observed counts.
The iteration count is a timing proxy: the benchmark steps its optimizer every
microbatch, whereas release training may accumulate several microbatches per
optimizer update. It does not directly prescribe release optimizer steps.

| Eight-hour base window | Mean mixed seconds | Timed iteration proxy | Input positions | Canvas positions | Combined positions |
|---|---:|---:|---:|---:|---:|
| Dense | 0.631952 | 41,015 | 2,687,959,040 | 335,994,880 | 3,023,953,920 |
| Portable loop MoE | 0.768396 | 33,732 | 2,210,660,352 | 276,332,544 | 2,486,992,896 |

The historical D-009 dense size rule counted input plus canvas positions. The
dense estimate passes 3B by only 23,953,920 positions, or **0.798%**. Input
alone is 2.688B. Extra unavailable time of about 0.7 percentage points beyond
the assumed 10% can erase the combined-position margin. This is a provisional
arithmetic pass, not a robust real-data budget or an automatic architecture
decision. D-017 now keeps the dense size and reallocates base time instead
of applying that historical shrink rule. A hypothetical packed-MLM substitution
lowers the estimate to
3.012B combined; it is not selected as the default here.

For the current D-017 split, using the same unrounded latencies and factor:

| Current base window | Timed iteration proxy | Input positions | Canvas positions | Combined positions |
|---|---:|---:|---:|---:|
| Dense, 9 hours | 46,142 | 3,023,962,112 | 377,995,264 | 3,401,957,376 |
| Portable loop MoE, 7 hours | 29,516 | 1,934,360,576 | 241,795,072 | 2,176,155,648 |

These are synthetic projections, not additional measured training runs.
The first real-data hour checks valid-token throughput; if needed, D-017
moves time from MoE to dense. That adjustment must preserve all five stages,
evaluation/retrieval time and the shared 24-hour ceiling. It does not guarantee
that 3B actual valid positions fit. Record the resulting unequal time/token
budgets and report any remaining feasibility shortfall.

For a 1.5-hour embedding window with the same factor and uniform T=2/4/8
latency, the proxy is 4,236 iterations, 271,104 repeated pair presentations
and 555,220,992 input positions across both sequences. These are not unique
pairs or actual accumulated optimizer updates.

All synthetic rows are fully valid. Real document batches can contain padding;
padded tensor positions must not be reported as observed training tokens.
Recheck the size rule using actual valid input/canvas throughput and the real
task mix before fixing release steps.

All forecasts count processing exposures. They are not unique text tokens,
supervised-target counts, generated output or quality scores. Real data
loading, corruption, padding, the IT length-head loss, task switching,
accumulation and sustained stability still need matching measurements.
The five releases continue to share one 24-hour budget.

## Reproduce an individual case

Use the recorded source and runtime. For example:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python tools/benchmark_nuri.py \
  --preset release-dense --training --task denoise --device cuda \
  --dtype bfloat16 --batch-size 64 --input-length 1024 --output-length 256 \
  --steps 4 --warmup 5 --iterations 20 --seed 0 --output dense-b64-t4.json
```

For MLM, use `--task mlm --steps 0`; for packed MLM also use
`--pack-mlm --packed-document-length 256`. For pairs use `--task pairs`;
each sample contains two input sequences. For MoE use
`--preset release-moe --moe-backend loop`. Repeat denoise and pairs at
`--steps 2` and `--steps 8` to reproduce the depth checks. These commands
measure training computation; they do not train or evaluate a release model.

## Results, cost and cleanup

JSON, logs, hashes and runtime evidence remain private. TensorBoard events
were generated locally from the verified reports after retrieval: 90 scalar
tags and 19 text tags. Event values were checked against the JSON within
float32 precision. These are postprocessed synthetic calibration events,
not live GPU-training curves or model-quality metrics.

The booked rate was $0.936/hour for compute plus $0.055556/hour for 40 GB
disk; traffic was $40/TB in each direction. The provider's current display
shows $0.22 for this additional instance and $1.92 aggregate usage including
the initial comparison, within the user's $10 testing limit. Rounded
itemized rows sum to $1.93 while the aggregate is $1.92; these displays are
not a settled invoice, and recent activity may be delayed.

After verified local retrieval, the instance and its disk were destroyed.
The console shows zero instances, zero storage volumes and $0/hour. The
temporary account registration of the approved SSH public key was removed;
local key files were preserved. No real training input was uploaded and
no release training started. Full training requires a separate budget that
accounts for its larger storage, transfer, calibration and retrieval needs.
