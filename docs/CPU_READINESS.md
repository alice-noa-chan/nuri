# CPU readiness before A100 experiments

Recorded on 2026-10-07. This audit covers implementation correctness and local
CPU execution. It does not establish competitive model quality or an A100
training budget. Architecture decisions and default training recipes remain
unchanged. No GPU was used, no retained model was trained, and reserved TEST
inputs were not read.

## Corrections

Each unit was formatted, linted and tested before its separate commit.

| Area | Correction | Regression evidence |
|---|---|---|
| Synthetic reasoning tasks | Reject unsupported language labels and invalid batch requests before drawing examples. | Supported-language replay and unchanged RNG state after rejected requests. |
| Token corpora and mixtures | Validate complete token storage, index type and shape, exact boundary coverage, document indices, nonempty sampling sources and finite positive weights. | Malformed-file rejection and unchanged valid sampling. |
| Corruption | Exclude arbitrary configured special IDs from random MLM replacements; validate finite corruption probabilities. | Replacement coverage and exact legacy RNG replay for the default contiguous special prefix. |
| Batch exposure audit | Attribute mixed-language span inputs and targets after the production bucket selection. Add optional source indices without changing ordinary batch keys or RNG draws. | Audit counts match the actual sampler for packed and unpacked MLM and bucketed and unbucketed span batches. |
| Model configuration | Reject fractional dimensions, nonfinite constants and invalid MoE auxiliary weights before tensor allocation. | Invalid-boundary checks alongside model, checkpoint, training and export tests. |
| Local tooling | Add repository-local Ruff settings, clean affected imports and variable names, and annotate intentional standalone-script imports. | Repository core lint and relevant utility, export and CLI tests. |
| Windows path safety test | Use a temporary directory junction when symlink creation requires unavailable privileges. | The redirected destination is rejected and its external target remains empty. |

[Source-conditioning guidance](SOURCE_CONDITIONING.md) now requires inspection
of tokenizer eligibility, complete references and actual partner coverage
before interpreting scores. Earlier diagnostic packets and results are retained.
All input identities, exposure measurements, selection records, generated
samples and evaluation results remain in `.private/` under D-008.

## CPU evidence

The local runtime was Python 3.11 with `torch 2.13.0+cpu`; CUDA was unavailable.

| Check | Measured result | Scope |
|---|---|---|
| Frozen-runtime full suite | 1,233 passed, eight conditional skips | Zero failures or errors; includes exact resume, release CLI, export and source-binding checks. |
| Subsequent pair-preparation suite | 49 passed, zero skipped | Covers the Windows path check that the full suite skipped. These tests overlap the full suite and are not added to its count. |
| Private supplementary-input preparation checks | 10 passed | Input selection and readback only; no score-driven selection. |
| Repository lint | Passed | `python -m ruff check nuri scripts tools tests`, using the checked-in core rule selection. |
| Formatting | Passed for changed Python files | Unrelated files were not reformatted merely to change their style. |
| Syntax compilation and privacy checks | Passed | Compile the runtime and utilities; check tracked and staged files against the private-file policy. |
| Full release geometry | Five forward/backward and optimizer cases passed | Fresh random dense and MoE weights, artificial token IDs, FP32 CPU execution. |

The full-size checks used 150,849,539 dense parameters and 242,935,811 MoE
parameters, with two rows per batch. Dense MLM, denoising and pair losses, plus
MoE MLM and denoising losses, produced finite gradients and parameters after a
fresh AdamW step. Full-logit and checkpointed chunked losses and every produced
parameter gradient matched within the declared tolerances (`rtol=2e-4`,
`atol=2e-5`). These small synthetic batches do not establish CUDA memory fit,
BF16 parity, real-data throughput or useful learning.

Private CPU receipts include full current-byte checks of permitted token
corpora, vocabulary bounds, tokenizer and retained-weight bindings, actual
mixed-batch exposure, and bounded VALID evaluator replays. Legacy assessment
counts and cross-entropies replayed exactly on the current runtime. Replays of
the corrected initializer contain zero optimizer updates; they are diagnostic
evidence before repair learning. They do not demonstrate a quality improvement.
Supplementary conditioning inputs were selected before scoring and evaluated
with complete, untruncated references and eligible comparison partners.

## Remaining GPU checks

Seven CUDA-specific test cases remain unverified. Six are applicable to A100:
FlashAttention 2 parity, CUDA allocator reporting, two Liger loss cases, Liger
layer gradients, and native GQA BF16 parity. The seventh checks FlashAttention 3
and requires Hopper hardware; its A100 skip is expected.

On the CUDA host, run:

```bash
python -m pytest -q -rs tests/test_attention_backend.py \
  tests/test_liger_backend.py tests/test_batch_monitoring.py \
  tests/test_native_gqa.py
```

Inspect the skip reasons. Missing optional packages must not be counted as
successful A100 kernel validation. CPU adapter mocks test dispatch behavior;
the actual CUDA kernels still need their own output and gradient comparisons.

Then follow the matched measurements in [KERNEL_BACKENDS.md](KERNEL_BACKENDS.md):

1. Compare SDPA/full CE, SDPA/chunked CE, FlashAttention 2/chunked CE, and
   FlashAttention 2/Liger at equal shapes, dtype and training settings, each in
   a fresh process. Include MLM, packed MLM, denoising and pair tasks as used by
   the planned run. Pair losses do not use vocabulary CE.
2. Record actual attention paths, peak allocated and reserved VRAM, valid
   tokens per second and supervised targets. A requested backend may use an
   SDPA fallback for packed or query-specific masks.
3. Run a bounded real-input calibration with the planned task mixture, saving
   and resuming a checkpoint. Include data loading, padding, validation and
   saving when estimating available training time.
4. Compare short learning runs that change one factor at a time. Report both
   update counts and wall time, assess the fixed VALID protocol, inspect
   generated outputs, and check encoder quality retention. Preserve TEST for
   final evaluation.

Use current source and initializer hashes when preparing the GPU package.
Reconcile the remaining budget and current GPU cost before paid execution.
Passing these mechanical checks alone is insufficient to justify a full
48–72-hour run: short learning evidence must support the selected recipe.
Matching ModernBERT, RoBERTa, DeBERTa, BERT or ELECTRA still requires defined
downstream tasks, comparable evaluation and measured scores.
