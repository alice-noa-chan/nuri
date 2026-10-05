# Dense Base: measured evaluation

Date: 2026-10-05. This assessment uses the completed Dense Base checkpoint,
before instruction tuning and embedding training.

The encoder has learned masked-token patterns. Its current generation path does
not produce reliable, coherent text in these direct probes. A structural
symmetry in the initial masked output canvas explains the repeated first-fill
tokens and requires architecture review. Treat this checkpoint as an experimental
base for further work; these results do not establish general chat quality.

## Checkpoint and execution

- Parameters: 150,652,931.
- Planned Dense Base updates completed: 143,797.
- Recorded training time: approximately 18.73 hours.
- Frozen production source: `a6c0a04b2abff2e82d5c14cb35df998ebbb7f30f`.
- Native weight SHA-256:
  `493beb389d13c844031b150b87128d7f952f26c9263a42318554701b5b2c1c5e`.
- Evaluation: CPU, float32, at most four threads, low scheduling priority and
  fixed process deadlines on the existing host. No GPU inference was added to
  the concurrent production training process.

Model hashes before and after each completed evaluation match. The inference
source hashes match the frozen production archive. Complete probes, raw results,
sample selections and operational receipts stay in ignored private storage.
CPU timings under concurrent training are not GPU serving benchmarks.

## Masked-token learning

For each language, select 64 held-out documents with a recorded seed and score
only their first 256 tokens. Use the same documents and corruption for the
trained checkpoint and a seed-0 random model with the same geometry. The table
counts only selected positions actually replaced by MASK; it excludes unchanged
and random-token targets from the usual 80/10/10 corruption.

| Language | True MASK targets | Trained top-1 accuracy | Trained loss | Random top-1 accuracy |
|---|---:|---:|---:|---:|
| Korean | 1,933 | 64.77% | 1.6192 | 0% |
| English | 1,683 | 81.46% | 0.6875 | 0% |
| Japanese | 1,505 | 49.77% | 3.6732 | 0% |

The random-model losses range from 11.2243 to 11.3031. The checkpoint therefore
learned predictive token patterns on these samples. This is partial coverage of
document leads, with one corruption seed. Differences between languages also
reflect different evaluation text distributions; they do not rank general
language ability.

A separate set of 24 handwritten masked-word probes gives six exact reference
matches, versus zero for the random model. Some references are not uniquely
correct: for example, both "cats" and "people" can fit a sentence about three
subjects sleeping on a sofa. Exact reference matching is a strict diagnostic,
not a comprehension benchmark. All subtokens intersecting the chosen word are
hidden; tokenizer boundaries can include a neighboring suffix or space.

Direct examples show both strengths and errors:

- The model restores the Korean word for "melt" in a sentence about ice in a
  warm place, and restores "book" with its particle in a library sentence.
- It restores both named recipients in a pair of Korean book-giving examples.
  Their token lengths differ, so this pair does not establish a controlled
  reasoning result.
- In a Korean capital sentence, it predicts "Republic of Korea" where the
  reference is "Seoul".
- An English key-location passage produces "check the lock" for both bag and
  drawer variants. This does not recover the supplied location.

## Retrieval before embedding training

The handwritten test has 24 queries, 24 designated positive documents and eight
distractors. All methods use the same 32-document pool and stable tie handling.
Model retrieval uses the full 256-dimensional output, the same mode on both
sides, and T=4 for reasoning mode.

| Method | Designated-document Recall@1 | Recall@5 | MRR |
|---|---:|---:|---:|
| Dense Base, fast | 29.17% | 91.67% | 0.5520 |
| Dense Base, reasoning | 29.17% | 91.67% | 0.5520 |
| Random model, fast | 4.17% | 41.67% | 0.2321 |
| Random model, reasoning | 4.17% | 50.00% | 0.2667 |
| Matched token TF-IDF | 4.17% | 54.17% | 0.2918 |

The trained model finds some useful topic relations: a Korean query about a
phone losing power retrieves a smartphone battery explanation. However, the
first result reverses several giver/recipient and bag/drawer relations.
Fast and reasoning rankings are identical on this small set; that observation
does not prove the recurrent core is generally unnecessary.

These are designated-document recovery scores. The pool contains cross-language
equivalents, and some weather queries permit multiple relevant answers.
An off-diagonal result is therefore not automatically a semantic failure.
The embedding projections have not received contrastive training. This small
test is also a different candidate pool from earlier retrieval controls, so its
numbers must not be compared directly with those controls or published model
benchmarks.

## Actual text generation

Use ten handwritten raw prompts across Korean, English and Japanese: five
narrative prefixes and five questions. Run each at a fixed canvas length of 16
and 32 tokens, T=4, S=4 and greedy decoding. The lengths are chosen before
observing outputs and are not derived from references. No chat template is
applied. Base has no trained automatic length head; questions are pre-IT stress
tests, while raw prefix continuation matches its training purpose more closely.

The 20 outputs contain repeated punctuation, duplicated words, broken grammar
and repeated question labels. For example, a Korean capital question at length
16 returns:

```text
답답변:
답변::답변
답답변:
```

An English rainy-day prefix at length 16 continues:

```text
 felt,. I was scared. But I thought was me, and I felt
```

At S=1, both tested Korean and English rainy-day prefixes return only repeated
periods at lengths 16 and 32. S=4 changes the outputs but does not establish
coherent generation. Missing EOS alone is not a failure: Base also trained on
interior spans that do not end a document. Intermediate saved canvases are
post-remasking states, not complete answers from each round.

## Reconstruction with genuine document boundaries

A separate check selects eight Korean documents without cropping them before
choosing the target. Encode only the last at most 256 tokens preceding the cut;
the following target contains at most 32 content tokens. Add EOS only when the
target reaches the original document end. All eight sampled spans are interior
spans and therefore receive no EOS.

| Masked share of target canvas | Single-forward token accuracy | Per-sample mean loss |
|---|---:|---:|
| 15% | 37.50% | 3.1471 |
| 50% | 32.81% | 4.2689 |
| 100% | 6.64% | 7.4996 |

With every target hidden and the exact target canvas length supplied, S=1
reconstructs 6.64% of 256 tokens; S=4 reconstructs 4.30%. Both produce zero exact
span matches out of eight. Known target length conditions these results, and
many natural continuations can be valid. These scores measure recovery of the
chosen span, not open-ended text quality or perplexity.

The earlier diagnostic used cropped document windows and could put EOS at an
artificial endpoint. It is retained privately as history and is superseded by
this original-boundary check.

## Initial canvas symmetry

The current readout receives identical token embeddings at every position of
an all-MASK canvas. Its self-attention applies RoPE to Q and K, but not to V or
the residual stream. Different attention weights still sum identical value
vectors. The memory cross-attention then receives identical queries with no
output-position feature. Pointwise norms and feed-forward layers preserve the
same hidden vector at each position.

This argument holds layer by layer, independently of the learned weights in
the current deterministic topology. It explains why the first greedy fill
cannot assign different tokens to different positions. Later remasking can
introduce visible-token anchors and break the symmetry, but those anchors
start from the repeated first fill.

The actual trained checkpoint reproduces this on two prompts, at lengths 16
and 32. All-MASK logit differences across positions are at most 9.54e-6, and
every position chooses token 20, a period. An EOS anchor at the last position
breaks the hidden-state symmetry and yields four or seven distinct greedy
tokens. That anchor is only a diagnostic intervention; useful generation from
it has not been demonstrated. An independent tiny-model reproduction reaches
the same structural conclusion.

IT with unchanged topology retains this first-fill symmetry. A trained
output-position feature in the canvas residual/value path needs review,
checkpoint compatibility rules, a position-variation regression, and renewed
generation checks. The existing stochastic sampler can introduce token
variation, but it does not make the identical first-pass conditional
distributions position-specific, and its quality benefit is unmeasured.

## Assessment and pending work

Keep the learned encoder checkpoint for further experiments. Masked-token
learning is supported; broad understanding, reliable semantic retrieval and
coherent generation are not established by this evaluation. Address the
readout symmetry before accepting generation quality. Review the proposal in
DISCUSSION before changing the declared architecture or a live training run.

The existing run and its spending cutoff remain active. This assessment changes
no production source, checkpoint weights, optimizer state, stage allocation or
deadline. Final IT and embedding evaluations remain pending.
