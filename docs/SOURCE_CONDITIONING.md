# Bounded decoded source-conditioning checks

`scripts/evaluate.py --suites conditioning` supplements loss monitoring with
actual decoded input-replacement checks. Use privately reviewed VALID inputs
for development. This suite does not change the frozen repair gate, consume
reserved TEST for tuning, or authorize another training stage.

## What the comparison does

The evaluator reads canonical instruction rows (`messages`, `response`) with
the existing training prompt format. It selects at most 128 eligible rows by
default with a declared seed. `--conditioning-examples` accepts 1 to 1,024.
Selection is a bounded diagnostic, not exhaustive coverage of the file.

For each selected row, a deterministic partner must have a different reference
answer, the same reference token length and a different visible prompt after
the existing last-N clipping. The model generates once from the original
prompt and again from the partner's actual prompt. Each call re-encodes its
input and starts with an all-MASK canvas. The reference response is not appended
as teacher memory; a grounded prompt can legitimately contain answer tokens.
This tests complete-input sensitivity rather than only replacing latent slots.

Both passes use greedy T=4/S=4 generation. Base models use the reference length
including EOS. IT models with a trained length head additionally use predicted
lengths. Equal reference lengths reduce the oracle-length shortcut; they do
not make oracle and predicted generation equivalent. This is a substitution
among existing examples, not a dataset of human-edited context counterfactuals.
The length-only diagnostic uses the same selected VALID answers and is an
optimistic target-majority bound, not a baseline fitted on TRAIN.

The report distinguishes:

- Original content exact match from exact match that also requires EOS.
- Swapped output matching the original reference from matching its own
  partner reference.
- Both targets being correct from merely changing the answer.
- Missing/early EOS, canvas-length errors and the original-minus-swapped
  original-reference correctness gap.
- Actual generated rows, readout calls, elapsed seconds and coverage.

Free-response exact match is strict: a valid paraphrase can fail. Human-read
samples and broader task metrics remain necessary. A high response-change rate
with wrong answers is a failed correctness diagnostic. No score in this suite
is an automatic broad-quality acceptance threshold.

## Coverage and preservation

Empty tokenized prompts and answers too long for the canvas are counted and
excluded; answers are never shortened. Empty responses remain legitimate
one-EOS references, but normally lack a different equal-length answer partner.
Truncated eligible prompts are counted. Unpaired selected rows, distinct
partners and reused partner presentations are explicit. A file can have no
usable pairs: that result is **unassessed**, with null quality metrics.

Freeze tokenizer eligibility and partner coverage before scoring. A file with
128 rows can contain far fewer assessable pairs: references must fit the
response limit including EOS, and selected rows need distinct visible prompts
and distinct equal-length reference answers. Report the supplied, eligible and
paired counts separately. When preparing a supplementary packet, retain full
reference answers, record prompt truncation or exclude it explicitly, preserve
the original packet, and publish a new private selection receipt. Selecting
equal-length partners changes the answer-length distribution; this diagnostic
does not establish representative instruction quality.

The selection digest binds selected indices and both effective token sequences.
Partners can repeat, so pair observations are not independent statistical
samples. Freeze file content/order, checkpoint, tokenizer, selection budget,
seed and decoding policy when comparing checkpoints. Record prior validation
consultation. Do not pool languages to hide a regression or fill an unassessed
language with another language's templates.

Evaluation runs without gradients and restores module training modes and Torch
RNG on success or failure. These controls make a later bounded check safer to
insert between learning segments; they do not recover optimizer state or
demonstrate GPU stability. This command accepts exported checkpoints, not a
private interrupted training-state file. Preserve any full state and make a
separate bound weights snapshot before a later early check.

## Private samples and numeric curves

For a reviewed English VALID file, a separately accepted execution can use:

```bash
python scripts/evaluate.py --checkpoint "$NURI_CHECKPOINT" --suites conditioning \
  --instructions "$NURI_VALID_INSTRUCTIONS" --conditioning-examples 128 --seed 0 \
  --conditioning-language en --conditioning-step 1000 \
  --output .private/checks/aggregate.json \
  --conditioning-samples-output .private/checks/samples.jsonl \
  --conditioning-tensorboard-dir .private/checks/events
```

The step is the actual completed-update count of that snapshot; 1000 above is
only an example. Checkpoint, input and runtime receipt binding belongs to the
private command producer. Use separate reviewed Korean, English and Japanese
files and independent outputs. Language tags are declared metadata, not
automatic language identification or proof of coverage.

Decoded samples contain up to eight pairs per mode, effective prompts,
references, returned tokens, raw canvases, chosen lengths and EOS flags. The
sample path must resolve inside the project's ignored `.private/` directory.
Raw text stays out of aggregate reports, stdout and TensorBoard. Aggregate
reports bind private samples by count and SHA256. Conditioning reports,
sample files and event directories must use fresh paths; existing artifacts
are preserved. Failed publication remains a failure, not reusable success.

Optional TensorBoard tags use
`valid/<language>/conditioning/<mode>/<metric>` at the supplied completed step.
They contain numeric scalars only, including coverage and an `unassessed` flag.
Null quality metrics do not become zero-valued passing curves.

For corruption changes during training, use the independent fixed corruption
recipe described in [TRAINING_RECIPE.md](TRAINING_RECIPE.md). Raising training
full-mask frequency tests exposure to the first generation fill. It does not
solve the later difference between visible gold context during training and
generated context during refinement.
