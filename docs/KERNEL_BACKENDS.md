# Optional training kernels

Nuri can reduce temporary activation and vocabulary-logit memory with opt-in
runtime kernels. Model topology, checkpoint parameter names, tied embeddings,
MLM token weighting and canvas sample weighting stay unchanged. Defaults remain
SDPA, ordinary PyTorch layers and full-logit CE until GPU calibration supports a
different choice. These options also apply to training holdouts.

| Option | Implementation | Hardware and limits |
|---|---|---|
| `--attention-backend sdpa` | PyTorch dispatch | Portable default; PyTorch can already choose fused attention |
| `--attention-backend flash2` | External FlashAttention 2 | NVIDIA Ampere or newer; BF16/FP16 queries; A100 supported |
| `--attention-backend flash3` | External FlashAttention 3 | Hopper, such as H100; this adapter accepts head sizes 64, 128 and 256 |
| `--liger-kernels` | Liger RMSNorm and dense SwiGLU | NVIDIA CUDA and compatible Triton; routed MoE experts retain their existing implementation |
| `--vocab-loss-backend torch` | Full selected-token logits plus FP32 CE | Portable reference |
| `--vocab-loss-backend chunked` | Checkpointed linear projection and PyTorch CE | Portable; bounds each transient logit matrix to `vocab_chunk_size × vocab_size` |
| `--vocab-loss-backend liger` | Fused linear MLM CE; checkpointed linear plus Liger canvas CE | NVIDIA CUDA; preserves unequal canvas row weighting |

Use FlashAttention 2 on A100. FlashAttention 3 is a Hopper implementation, not an
A100 upgrade. Install the matching CUDA PyTorch runtime and kernel packages on
the GPU host. Installation instructions and compatibility requirements are in
the [FlashAttention repository](https://github.com/Dao-AILab/flash-attention) and
the [Liger repository](https://github.com/linkedin/Liger-Kernel). The Liger adapter
uses the public functional API available since v0.6.4.

For a Linux CUDA environment, install PyTorch first, then the optional extras:

```bash
python -m pip install packaging psutil ninja
python -m pip install -e '.[flash2,liger]' --no-build-isolation
```

FlashAttention 3 needs the separate upstream Hopper package, including its
CUDA requirements. Follow the upstream instructions; it is not installed by
the `flash2` extra. Missing packages or unsupported hardware fail explicitly.
Ordinary CPU loading, training and exported HF models need neither package.

## Attention masks and numerical behavior

External FlashAttention handles non-causal self/cross attention and compact GQA
heads. The padding adapter removes masked keys and values, retaining all query
positions to preserve existing padded-query outputs. It does not unpad the
whole encoder or its FFNs. With many padded queries, full unpadding remains a
separate optimization.

Document isolation in packed MLM, local attention and grouped reasoning slots
use query-specific masks. They retain the exact SDPA path. A run can therefore
request FlashAttention and execute SDPA exclusively, or use both. Benchmark JSON
records `attention_backend_requested`, `attention_backend`,
`attention_paths_used` and `gqa_paths_used`. `sdpa` identifies the PyTorch API
path, not its internal CUDA kernel. An external kernel is not automatically
faster than PyTorch's own fused dispatch.

RMSNorm uses FP32 normalization and weight multiplication with out-of-place
backward to protect shared residual gradients. Liger loss uses scalar fused
MLM mean. Canvas CE instead applies the original per-token row weights after
non-fused CE within checkpointed chunks. This avoids weighted `reduction=none`
fused linear CE gradients in older Liger releases.

Different kernels and projection chunk sizes can change floating-point rounding.
Tests compare outputs and gradients with tolerances. An exact resume requires
the original `attention_backend`, `liger_kernels`, `vocab_loss_backend` and
`vocab_chunk_size`; legacy states imply their original defaults. To change these
choices deliberately, start a fresh run from checkpoint weights with `--init`.
That starts new optimizer and schedule state. Opt-in numerical choices also bind
holdout receipts. Public model configs do not enable optional kernels on load.

## Calibrate before extending training

Start with matched tasks, model, dtype, lengths, reasoning depth, corruption,
batch size and warmup. Measure at least these configurations on A100:

| Comparison | Attention | Vocabulary loss | Layer kernels |
|---|---|---|---|
| Reference | SDPA | torch | PyTorch |
| Bound vocabulary memory | SDPA | chunked | PyTorch |
| External attention | flash2 | chunked | PyTorch |
| Fused layers and loss | flash2 | liger | Liger |

The synthetic benchmark includes forward, backward, gradient clipping and AdamW.
For example, the reference measurement is:

```bash
python tools/benchmark_nuri.py --preset release-dense --task mlm --training \
  --device cuda --dtype bfloat16 --batch-size 32 --input-length 512 \
  --warmup 5 --iterations 20 --output .private/calibration/mlm-reference.json
```

Repeat with `--vocab-loss-backend chunked --vocab-chunk-size 256`, then add
`--attention-backend flash2`, then replace the loss with
`--vocab-loss-backend liger --liger-kernels`. Write each JSON to a different
private output. Repeat for `--task denoise --output-length 128 --steps 4`, and
for `--pack-mlm --packed-document-length 128` when packing is actually used.
For embedding training use `--task pairs`; vocabulary CE options do not apply.
Run each candidate in a fresh process. Confirm the real CUDA tests first:

```bash
python -m pytest -q tests/test_attention_backend.py tests/test_liger_backend.py
```

Increase the micro-batch only after correctness and memory checks pass. Compare
`peak_cuda_allocated_bytes`, `peak_cuda_reserved_bytes` and valid tokens per
second. Synthetic rows are all valid; use training batch monitoring to measure
real padding. Gradient checkpointing can reduce memory while slowing steps.
Smaller vocabulary chunks also introduce recomputation and launch overhead.
Lower peak VRAM alone does not prove higher throughput.

For a fixed time budget, the additional token exposure is approximately
`(new valid tokens/second − old valid tokens/second) × training seconds`.
This is an estimate, not unique text coverage or a count of supervised targets.
Use the actual training task mixture and reserve time for validation, saving and
evaluation. No VRAM percentage, A100 speedup or quality improvement has been
measured on the current CPU-only development host.

When comparing quality at a fixed effective batch, reduce `--grad-accum` as the
micro-batch grows and verify the schedule. Gradient accumulation does not create
a larger in-batch negative pool for contrastive learning. If the effective batch
changes, learning-rate and update-count choices need their own assessment.
Use VALID for selecting settings and preserve TEST for final evaluation. Better
throughput can fund more training; it does not by itself establish parity with
ModernBERT, RoBERTa, DeBERTa, BERT or ELECTRA.

The same flags are accepted by `scripts/train.py`. Release plans can set kernel
choices in their `train` object, with per-stage overrides. Keep training input
paths and private plans under the existing environment-variable/private-config
policy.
