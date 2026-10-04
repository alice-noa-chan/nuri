# A100 and H100 calibration

Measured on October 4, 2026 (UTC). This is a small training-throughput
comparison using synthetic random token IDs. It is not a trained-model
evaluation. No release training has started.

## Same workload on both hosts

- Final vocabulary: 65,536 entries.
- Dense model: 150,652,931 parameters. MoE model: 242,739,203 parameters.
- Release presets: hidden size 768, 10 input layers, 2 readout layers,
  32 latent slots, QK normalization and one slot group.
- MoE: 16 routed experts, top-3 routing and one shared expert. The common
  comparison uses the portable loop backend. The recurrent core stays dense.
- BF16 training with forward, backward, gradient clipping and an AdamW step.
- Input length 1,024. Denoising canvas length 256. MLM and denoising use
  microbatch 8; embedding training uses 32 query/document pairs.
- Five warmup iterations, followed by 20 measured iterations, seed 0.
- MLM skips the reasoning core. Denoising and pairs measure T=2, 4 and 8.
- All optional model experiments and native GQA remain disabled.

Both hosts used Python 3.12.14, PyTorch 2.8.0+cu126 and CUDA 12.6 from the same
pinned image:

```text
vastai/pytorch:2.8.0-cuda-12.6.3-py312-24.04-2026-09-08
```

All 15 recorded package versions matched, including NumPy 2.5.3,
safetensors 0.6.2, tqdm 4.70.0 and TensorBoard 2.20.0. The drivers differed:
H100 used 560.35.03; A100 used 580.173.02. CPU allocations also differed.
Results describe these rented hosts and this software stack. They do not
establish a general performance rule for every A100 or H100.

## Baseline measurements

Throughput counts observed input tokens once per training iteration. A pair
contains two input sequences, so both the query and document are counted.
The denoising canvas is reported separately by the benchmark; its positions
are not added to the input-throughput column.

| Workload | H100 input tokens/s | A100 input tokens/s | H100 mean seconds/iteration | A100 mean seconds/iteration | H100 peak allocated GiB | A100 peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| Dense MLM | 145,062 | 87,807 | 0.0565 | 0.0933 | 8.22 | 8.17 |
| Dense denoise, T=4 | 105,349 | 70,198 | 0.0778 | 0.1167 | 7.80 | 7.75 |
| Dense pairs, T=4 | 185,813 | 108,180 | 0.3527 | 0.6058 | 37.46 | 37.38 |
| Loop MoE MLM | 14,642 | 58,692 | 0.5595 | 0.1396 | 10.30 | 10.22 |
| Loop MoE denoise, T=4 | 13,243 | 42,924 | 0.6186 | 0.1908 | 10.10 | 10.05 |

Both devices completed all 11 common baseline/depth cases. Dense workloads
were faster on the tested H100. Loop MoE was much faster on the tested A100.
The benchmark does not isolate why the hosts differ; driver, CPU, dispatch
overhead and host conditions are possible factors that need separate tests.

Loop MoE timings on H100 varied substantially. For example, denoise T=4 had
median 0.502 seconds, mean 0.619 seconds and p95 0.936 seconds. A lower mean
in a deeper case does not demonstrate that more reasoning is faster.

Peak allocated memory is the memory occupied by live tensors. Reports also
record peak reserved memory, which includes the CUDA allocator's cached
blocks. Neither value establishes the largest usable training microbatch.

## Additional grouped-backend check

The H100 supported BF16, had compute capability 9.0 and exposed
`torch._grouped_mm`. Two separate opt-in grouped MoE cases were attempted
after the common loop measurements: MLM and denoise T=4.

Both reached their five-minute per-case timeout without producing a measured
report. The owned subprocesses were terminated and all common results were
retained. This is a timeout result, not a successful grouped measurement or a
measured fallback. Keep the loop backend as the supported default for this
tested build. Availability of a grouped operation alone does not prove that
the complete forward/backward training path is usable.

H100's complete runner took about 756 seconds, including both optional
timeouts. A100's common runner took about 155 seconds. These durations
exclude provider startup, dependency preparation and result retrieval.

## Cost and interpretation

The booked compute rates were $2.133333/hour for H100 and $0.936/hour for
A100. With 40 GB of container disk, running rates were approximately
$2.144444/hour and $0.991556/hour. Bandwidth is billed separately.

For compute-only efficiency, H100 must be about 2.28 times faster to match
A100's input tokens per dollar at these rates. None of the five baseline
workloads reached that threshold. The measured A100 host is therefore the
better starting point for a private training budget at the tested shapes.
This is a calibration recommendation, not approval to start release training.

After local artifact verification, both comparison instances were destroyed.
The provider console confirmed zero instances, zero storage volumes and a
current spending rate of $0/hour. The account registration of the existing
SSH public key was removed; the local key files were kept.

The provider's itemized display shows $1.57 for H100 and $0.14 for A100,
or $1.71 together, within the approved $10 comparison limit. Displayed credit
changed from $12.16 to $10.45, matching that total. These are rounded provider
figures, including displayed storage and small transfer charges; the console
warns that recent activity may be delayed. No full-training spend is included.

## Reproduce a baseline

Use the same release preset, runtime, shape and dtype. For example:

```bash
python tools/benchmark_nuri.py --preset release-dense --task mlm --training \
  --device cuda --dtype bfloat16 --batch-size 8 --input-length 1024 \
  --output-length 256 --steps 0 --warmup 5 --iterations 20 --seed 0 \
  --output dense-mlm.json

python tools/benchmark_nuri.py --preset release-dense --task denoise --training \
  --device cuda --dtype bfloat16 --batch-size 8 --input-length 1024 \
  --output-length 256 --steps 4 --warmup 5 --iterations 20 --seed 0 \
  --output dense-denoise-t4.json

python tools/benchmark_nuri.py --preset release-dense --task pairs --training \
  --device cuda --dtype bfloat16 --batch-size 32 --input-length 1024 \
  --steps 4 --warmup 5 --iterations 20 --seed 0 --output dense-pairs-t4.json
```

For the common MoE tasks, use `--preset release-moe --moe-backend loop`.
The same denoise and pairs commands can measure T=2 and T=8. This comparison
does not include MoE pairs training. Optional grouped checks require an
eligible H100/runtime and an external process timeout.

## Practical limits

These short fixed-shape measurements exclude host data loading, the
instruction length-head loss, checkpoint I/O and sustained stability tests.
They do not measure a maximum microbatch, answer quality or embedding quality.
A one-day release plan still needs task weights, actual training inputs and
final-vocabulary preparation checks before execution.

MLM and denoising used only about 8-10 GiB of peak allocated tensor memory
at microbatch 8. That does not measure utilization or establish how a larger
batch will scale. The release size rule needs measurements at the intended
training batch, including packing or checkpointing when those are used.
Random-depth training also needs the measured T=2/4/8 costs and its task mix;
a T=4-only forecast is provisional. Further rentals require a separate
spending authorization.

JSON reports, configuration hashes, runtime manifests, logs and TensorBoard
events are retained privately. Both result bundles are independently checked
after transfer. No training-input identities or private source records belong
in this public calibration report.
