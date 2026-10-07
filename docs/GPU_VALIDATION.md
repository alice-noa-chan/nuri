# Bounded A100 validation

All six applicable A100 kernel checks passed and all three short warm starts
completed. More full-canvas corruption did not restore useful generation.
Changing language weights improved Japanese reconstruction loss while
worsening Korean MLM loss; encoder probes did not establish a consistent
improvement. None of the models produced an exact natural answer in the
bounded diagnostic, and inspected outputs remained repetitive and incoherent.

Recorded on 2026-10-07. This experiment checks real CUDA execution and three
short Dense Base warm starts. It does not establish parity with ModernBERT,
RoBERTa, DeBERTa, BERT or ELECTRA. Architecture decisions and default recipes
remain unchanged. All input identities, provenance, generated samples and
checkpoint receipts stay private under D-008. Reserved TEST was not read.

## Runtime and implementation evidence

The owned host used one NVIDIA A100 80GB PCIe, Python 3.12.14, PyTorch
2.8.0+cu126, CUDA 12.6, FlashAttention 2.8.3 and Liger 0.6.4. The Dense model
has 150,849,539 parameters. This diagnostic did not train the MoE or IT stages.

The selected kernel suite passed 48 cases with one expected skip. All six
A100-specific cases ran successfully: FlashAttention 2 output/gradient parity,
CUDA allocation reporting, two Liger loss checks, Liger layer gradients and
native GQA BF16 parity. FlashAttention 3 requires Hopper and was skipped.
The earlier CPU adapter mocks are not counted as real kernel execution.

Actual training exposed a process-supervisor defect: `--no-progress` kept
stdout quiet while the structured optimizer-step log advanced. The supervisor
incorrectly classified that as a 900-second idle period. Commit `3eda390`
counts completed step advances as progress while retaining hard deadlines,
nonfinite checks and bounded shutdown. Repeated/decreasing steps, changed loss
values and incomplete rows do not refresh the idle timer. All 17 supervisor
regressions passed on both the local CPU and the Linux GPU host; Ruff lint,
formatting and syntax compilation passed before the separate commit.

The stopped first arm saved its full state after 1,622 completed updates and
resumed with the same optimizer, schedule, random state and original deadline.
Earlier failure receipts were retained. Preparation also required recovering
a quiet setup process after its idle timeout. These interruptions were control
failures; the saved training tensors were finite. Startup and repair time count
against the rental budget.

## Measured execution choices

Each synthetic training benchmark ran in a fresh process at BF16 autocast,
batch 16, input length 1,024 and, for denoising, canvas length 256 with T=4.
Three warmup iterations preceded 12 measured iterations. Timing includes
forward, backward, gradient clipping and AdamW; it excludes real input loading
and host packing. GQA and gradient checkpointing were off. Reported memory is
peak allocated GiB, not total device capacity or peak reserved memory.

| Requested runtime | MLM seconds / GiB | Packed MLM seconds / GiB | Denoising seconds / GiB |
|---|---:|---:|---:|
| SDPA, full PyTorch CE | 0.1734 / 13.943 | 0.1745 / 14.262 | 0.1955 / 13.430 |
| SDPA, chunked PyTorch CE | 0.1935 / 10.578 | 0.1956 / 10.896 | 0.2056 / 12.147 |
| FlashAttention 2, chunked PyTorch CE | 0.1937 / 10.811 | 0.1939 / 10.896 | 0.2245 / 12.648 |
| FlashAttention 2, Liger CE and layers | 0.2061 / 7.974 | 0.2063 / 8.055 | 0.1924 / 9.078 |

Packed FlashAttention requests actually executed SDPA because document
isolation needs a query-specific mask. Unpacked FlashAttention cases executed
FlashAttention 2. The chosen runtime was SDPA with full PyTorch CE and ordinary
layers: it had the lowest sum of these three task latencies at this batch size.
This is a local execution choice, not evidence that fused kernels are generally
slower. Larger microbatches and maximum useful device utilization were not
searched.

Chosen-runtime input throughput was 94,467 tokens/s for MLM, 93,869 for packed
MLM and 83,822 for denoising. Input throughput counts observed input once and
excludes canvas positions. T=8 denoising took 0.2069 seconds and 13.572 GiB.
The pair benchmark took 0.3432 seconds and 19.600 GiB, counting both query and
document inputs. Synthetic throughput is not a real-input training projection.
The full benchmark receipts also record peak reserved memory and actual paths.

## Frozen comparison

The plan, assets, retained corrected initializer and assessment packets were
bound before learning. Each arm starts from that same initializer with a new
optimizer and schedule. Each completes 4,000 optimizer updates, seed 41, peak
LR 1e-4, 80 warmup updates and WSD decay over the final 15%. MLM/span sampling
remains 50/50, with packed MLM and bucketed spans. CUDA training uses BF16
autocast; span microbatches draw reasoning depth from {2, 4, 8}, while MLM
skips the recurrent core. Requested microbatch 16 with
four accumulation steps gives nominal batch 64; bucketing changes actual rows
per update. These are matched-update comparisons, not equal-FLOP comparisons.

| Arm | Full-canvas corruption probability | Korean / English / Japanese document sampling weights |
|---|---:|---:|
| Baseline | 0.10 | 0.72 / 0.27 / 0.01 |
| More full canvases | 0.30 | 0.72 / 0.27 / 0.01 |
| Language weighting | 0.10 | 0.45 / 0.50 / 0.05 |

Sampling weights are not measured language-token shares: packing, document
lengths and bucket selection affect actual exposure. Short real-input runs
included intentional full-state save/resume before the matched schedule was
selected. The three initial calibration estimates were 0.567, 0.567 and 0.633
seconds per update. Full-run times and processed positions follow below.

| Arm | Updates | Valid input + canvas positions | Supervised targets | Training command seconds |
| --- | --- | --- | --- | --- |
| Baseline | 4,000 | 176,976,893 | 46,968,334 | 2235.9 |
| More full canvases | 4,000 | 176,976,893 | 48,142,853 | 2220.8 |
| Language weighting | 4,000 | 177,025,439 | 46,926,485 | 2221.0 |

Positions are processed presentations, not unique corpus tokens. Training command
seconds include process startup, saving and the first arm's stopped/resumed
commands; they exclude the intervening repair pause. Its final log's seconds
field resets at resume and must not be mistaken for its total training time.
Rental time includes that pause and all other session work.

## Fixed language diagnostics

The assessment uses the same frozen VALID packet for every model, FP32 CUDA
inference from BF16 exported weights and T=4. Each language contributes 64 MLM
rows, 24 spans at corruption 0.15, 24 at 0.50 and eight fully masked spans.
Fully masked target counts are 256 Korean, 232 English and 256 Japanese tokens.
Cross-entropy below is the mean of row means; token accuracy uses target-token
counts. These bounded legacy diagnostics are not the formal CPU repair gate
at its prescribed checkpoint boundary.

| Model | Language | MLM CE | Full-span CE | Full-span token accuracy | Swapped-input CE |
| --- | --- | --- | --- | --- | --- |
| Initializer | Korean | 1.6184 | 7.5259 | 7.03% | 9.4348 |
| Initializer | English | 0.6783 | 5.8424 | 6.90% | 7.2868 |
| Initializer | Japanese | 3.5830 | 7.4059 | 8.98% | 8.5448 |
| Baseline | Korean | 1.6702 | 7.4437 | 7.42% | 9.5008 |
| Baseline | English | 0.7015 | 5.3736 | 8.62% | 7.2202 |
| Baseline | Japanese | 3.7041 | 7.4051 | 12.89% | 9.1040 |
| More full canvases | Korean | 1.6767 | 7.3871 | 7.81% | 9.4933 |
| More full canvases | English | 0.7090 | 5.4153 | 10.78% | 7.2890 |
| More full canvases | Japanese | 3.7619 | 7.4030 | 9.77% | 9.2139 |
| Language weighting | Korean | 1.7147 | 7.4616 | 8.20% | 9.5250 |
| Language weighting | English | 0.6732 | 5.3231 | 9.05% | 7.3370 |
| Language weighting | Japanese | 2.5382 | 6.5495 | 12.89% | 8.6352 |

| Model | Language | Partial-span CE (0.15) | Partial-span CE (0.50) |
| --- | --- | --- | --- |
| Initializer | Korean | 3.9103 | 5.2280 |
| Initializer | English | 2.2329 | 2.9848 |
| Initializer | Japanese | 6.6450 | 6.5104 |
| Baseline | Korean | 3.2930 | 4.4909 |
| Baseline | English | 1.6306 | 2.5995 |
| Baseline | Japanese | 5.8282 | 5.8132 |
| More full canvases | Korean | 3.1170 | 4.4621 |
| More full canvases | English | 1.7522 | 2.6525 |
| More full canvases | Japanese | 5.9140 | 5.7384 |
| Language weighting | Korean | 3.3192 | 4.5720 |
| Language weighting | English | 1.5884 | 2.4908 |
| Language weighting | Japanese | 4.1580 | 4.5078 |

| Model | Language | S=1 token accuracy | S=4 token accuracy | S=1 exact spans | S=4 exact spans |
| --- | --- | --- | --- | --- | --- |
| Initializer | Korean | 7.03% | 4.30% | 0/8 | 0/8 |
| Initializer | English | 6.90% | 4.74% | 0/8 | 0/8 |
| Initializer | Japanese | 8.98% | 4.30% | 0/8 | 0/8 |
| Baseline | Korean | 7.42% | 4.69% | 0/8 | 0/8 |
| Baseline | English | 8.62% | 5.60% | 0/8 | 0/8 |
| Baseline | Japanese | 12.89% | 7.42% | 0/8 | 0/8 |
| More full canvases | Korean | 7.81% | 4.30% | 0/8 | 0/8 |
| More full canvases | English | 10.78% | 6.90% | 0/8 | 0/8 |
| More full canvases | Japanese | 9.77% | 6.25% | 0/8 | 0/8 |
| Language weighting | Korean | 8.20% | 4.30% | 0/8 | 0/8 |
| Language weighting | English | 9.05% | 3.88% | 0/8 | 0/8 |
| Language weighting | Japanese | 12.89% | 11.72% | 0/8 | 0/8 |

The input-swap diagnostic uses the same target with mismatched context.
Lower correct-context CE indicates context sensitivity; it does not establish
usable generation. All complete-span exact-match results must be read beside
the low target-token accuracy and refinement behavior.

Every trained arm improved partial-span CE from the initializer in all three
languages. Fully masked token accuracy remained below 13% in every language,
and S=4 had lower token accuracy than S=1 in every assessed model/language
cell. The language-weighting arm reduced Japanese MLM CE from 3.5830 to
2.5382 but raised Korean CE from 1.6184 to 1.7147; English moved from 0.6783
to 0.6732. One pretraining seed and this small packet cannot establish a
general language tradeoff or identify the cause of generation failure.

| Model | Declared group | EOS-aware exact match | Conditioning gap | Missing EOS | Early EOS | Length-only exact match |
| --- | --- | --- | --- | --- | --- | --- |
| Initializer | Korean | 0.00% | 0.00 pp | 99.22% | 0.00% | 46.88% |
| Initializer | English | 0.00% | 0.00 pp | 100.00% | 0.00% | 41.41% |
| Initializer | Japanese | 0.00% | 0.00 pp | 54.69% | 12.50% | 21.88% |
| Baseline | Korean | 0.00% | 0.00 pp | 100.00% | 0.00% | 46.88% |
| Baseline | English | 0.00% | 0.00 pp | 97.66% | 0.00% | 41.41% |
| Baseline | Japanese | 0.00% | 0.00 pp | 35.16% | 0.00% | 21.88% |
| More full canvases | Korean | 0.00% | 0.00 pp | 100.00% | 0.00% | 46.88% |
| More full canvases | English | 0.00% | 0.00 pp | 100.00% | 0.00% | 41.41% |
| More full canvases | Japanese | 0.00% | 0.00 pp | 28.12% | 0.78% | 21.88% |
| Language weighting | Korean | 0.00% | 0.00 pp | 98.44% | 0.00% | 46.88% |
| Language weighting | English | 0.00% | 0.00 pp | 97.66% | 0.78% | 41.41% |
| Language weighting | Japanese | 0.00% | 0.00 pp | 75.78% | 0.00% | 21.88% |

Natural conditioning uses 128 complete, eligible VALID examples per declared group,
each with a different prompt/answer partner of equal reference token length.
Partners are reused where needed, especially in Japanese; these are not 128
independent paired comparisons. Decoding uses T=4/S=4, temperature zero and
oracle canvas length. The Base length head is untrained. Oracle length MAE is
zero by construction and says nothing about learned length prediction.
Response changes after an input swap do not imply correctness. The length-only
baseline uses the consulted validation answers and is a diagnostic, not a
training-fitted generalization baseline.

Language labels in this natural diagnostic denote the declared input groups.
Inspection found an out-of-language prompt and answer in the English group;
these groups must not be presented as clean monolingual benchmarks. The
frozen inputs were preserved for the matched comparison. Reference quality,
language eligibility and partner dependence need review before a new quality
benchmark. Exact match against one reference also penalizes legitimate
alternative wording, especially for long answers.

Bounded inspection of saved examples from every model and group found common
word repetition, punctuation/newline repetition, unrelated fragments and mixed
or invalid decoded text. This is qualitative evidence alongside the numeric
diagnostics, not exhaustive or blinded human grading. The current Base models
have not received instruction tuning, so this diagnostic cannot substitute for
evaluating a trained IT model or isolate architectural causes.

## Encoder retention

The frozen CPU protocol pools the masked mean of input memory X in FP32;
it excludes the recurrent core, readout and untrained embedding projection.
Each task has 1,024 TRAIN and 256 VALID rows, max length 128, batch eight and
two CPU threads. Ridge classification uses TRAIN standardization, an
unpenalized intercept and 17 alphas from 0.01 to 1e6. VALID macro-F1 selects
alpha, with accuracy as the tie breaker. No selected alpha is at a grid edge.

| Model | Task | Accuracy | Macro-F1 | Selected alpha |
| --- | --- | --- | --- | --- |
| Initializer | Korean topic (7 classes) | 79.69% | 79.94% | 316 |
| Initializer | English sentiment (2 classes) | 72.27% | 72.27% | 1e+04 |
| Initializer | English inference (3 classes) | 51.95% | 51.88% | 3.16e+03 |
| Baseline | Korean topic (7 classes) | 78.52% | 78.40% | 3.16e+03 |
| Baseline | English sentiment (2 classes) | 69.14% | 69.14% | 3.16e+03 |
| Baseline | English inference (3 classes) | 50.78% | 50.78% | 3.16e+04 |
| More full canvases | Korean topic (7 classes) | 79.69% | 79.85% | 3.16e+03 |
| More full canvases | English sentiment (2 classes) | 72.27% | 72.26% | 1e+04 |
| More full canvases | English inference (3 classes) | 51.17% | 51.20% | 1e+05 |
| Language weighting | Korean topic (7 classes) | 77.73% | 78.00% | 316 |
| Language weighting | English sentiment (2 classes) | 72.66% | 72.65% | 1e+04 |
| Language weighting | English inference (3 classes) | 50.78% | 50.87% | 3.16e+04 |

| Arm | Task | Accuracy change from initializer | Conditional paired 95% interval |
| --- | --- | --- | --- |
| Baseline | Korean topic (7 classes) | -1.17 pp | [-4.69, +2.34] pp |
| Baseline | English sentiment (2 classes) | -3.12 pp | [-7.42, +1.17] pp |
| Baseline | English inference (3 classes) | -1.17 pp | [-6.25, +3.91] pp |
| More full canvases | Korean topic (7 classes) | +0.00 pp | [-3.52, +3.12] pp |
| More full canvases | English sentiment (2 classes) | +0.00 pp | [-3.91, +3.91] pp |
| More full canvases | English inference (3 classes) | -0.78 pp | [-6.64, +4.70] pp |
| Language weighting | Korean topic (7 classes) | -1.95 pp | [-4.69, +0.39] pp |
| Language weighting | English sentiment (2 classes) | +0.39 pp | [-3.52, +4.30] pp |
| Language weighting | English inference (3 classes) | -1.17 pp | [-6.64, +4.69] pp |

Paired intervals use 2,000 row bootstrap samples with seed zero. They are
conditional on the VALID-selected alpha and assume independent rows; they
exclude alpha-selection and pretraining-seed uncertainty. Prior validation
consultation and retained pretraining exposure are preserved in the private
lineage. Point changes alone do not establish a regression or improvement.
No retention acceptance margin was specified. Japanese classification,
retrieval, instruction tuning and matched full downstream fine-tuning were
not assessed by this experiment.

## Retrieval, budget and conclusion

The 4,177,827,156-byte input archive was downloaded on the owned GPU host
from the user-provided endpoint, verified by SHA-256, and checked against all
20 member bindings. The endpoint and asset identities remain private.
The 6,477,184,845-byte result archive and all 208 listed members were verified
locally. Three final weight exports and their exact 4,000-update full states
were also retrieved separately; optimizer/RNG state was present, all model
and optimizer tensors were finite, and FP32 master weights cast exactly to
the exported BF16 tensors. Earlier interruption receipts remain available.

The session kept its original independent local/host guards, USD 12 cap and
7.25-hour maximum. Conservative rental intent-to-absence time was 3 hours,
2 minutes, from 01:58:58 to 05:01:10 UTC. The combined running quote was
USD 1.20/hour: GPU 1.0667 plus reserved disk 0.1333. Provider-posted charges
for this owned instance at completion were:

| Component | Posted USD |
|---|---:|
| GPU | 3.181 |
| Reserved disk | 0.404 |
| Download traffic | 0.000 |
| Upload traffic | 0.173 |
| Total | **3.758** |

This total uses instance-specific invoice rows, not the account balance delta
or charges from another project. The provider's billed GPU and disk durations
are slightly shorter than the conservative intent-to-absence clock. Compute,
storage and bandwidth are separate charges; storage continues for stopped
instances, which is why deletion is required. See [Vast.ai billing](https://docs.vast.ai/guides/reference/billing).

After verified retrieval, the owned instance was destroyed and its absence
was confirmed through the provider API. No persistent volume was created.
Other project instances were not changed. No additional learning or rental
was started after this diagnostic.

These results support the tested implementation and identify a supervision
bug. They do not justify a 48–72-hour run of the unchanged recipe or establish
competitive encoder quality. Stronger full-canvas supervision and changed
language weights must be judged from their individual language and encoder
results; a lower denoising loss does not by itself accept the generator.
Q-17 remains open. Architecture or production-recipe changes require a
separate evidence-based decision in the design discussion.
