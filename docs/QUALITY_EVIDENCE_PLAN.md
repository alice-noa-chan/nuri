# Practical quality evidence before full training

Status: the user authorized the retained-encoder comparison; execution is in
progress. No rental, production training, new allocation or additional spending
cap is approved by this document.
The completed upload remains immutable. The user requires general encoder
quality comparable to BERT, ModernBERT, RoBERTa, DeBERTa and ELECTRA. Exact
checkpoint versions, language coverage, input hashes and preliminary adaptation
budgets are declared privately before scoring. No general parity acceptance
margin or automatic continuation is defined. The named families are a comparison panel,
not one interchangeable score or permission to accept only the weakest model.

Nuri's corrected generator has not been trained and assessed on natural language.
The 716 passing CPU tests establish implementation behavior. Position-sensitive
logits and symbolic learning show that the identified symmetry defect can be
removed; they do not establish useful Korean answers or semantic retrieval.
The retained Dense input encoder is already trained. The position repair
preserves its input embedding and encoder tensors, so its representation
quality can be assessed before paying for further generator training.

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| CPU, migration and recovery controls | Correct mechanics and preserved initialization/state | Real GPU fit, speed or language quality |
| Frozen encoder features with supervised linear probes | Representation utility on the assessed tasks | Performance after full task-specific fine-tuning, or parity across a whole encoder family |
| Matched task-specific encoder fine-tuning | A direct downstream comparison under the declared adaptation protocol | Equal performance on unassessed tasks or languages |
| Existing first-quarter Dense Base repair gate | Matched span improvement, encoder retention and input sensitivity | Instruction following, automatic response length or trained semantic embeddings |
| Actual IT validation and decoded answers | Behavior on the assessed instructions, with predicted lengths and EOS | General chatbot quality or correctness on every prompt |
| Actual embedding validation with relevant candidates | Retrieval on the assessed query/candidate population in the stated mode | Broad semantic understanding or performance on another retrieval domain |

The repair gate remains frozen. Its Korean full-mask minimum is 34 correct
targets out of 256, or about 13.3%, across eight already consulted spans. That is
a minimum improvement over the inadequate legacy checkpoint, not a practical
chat-quality target. Recovery of one recorded continuation is also different
from the quality of other valid natural continuations. See
[the measured legacy evaluation](DENSE_BASE_EVALUATION.md) and
[the stage objectives and evaluation rules](ARCHITECTURE.md).

First compare the existing encoder. Cache masked-mean X and fit small heads
using training labels for sentence classification, sentence-pair inference or
similarity. Token-label tasks instead need aligned per-token X features.
Use identically specified pooling for frozen
references, and majority, lexical and random-initialization controls. Nuri's
untrained embedding projections are not the encoder-quality representation.
This is an inexpensive diagnostic, not a substitute for matched fine-tuning;
a weak probe alone does not establish that adapted performance must be weak.
The separate feature extractor and supervised probe now implement this bounded
diagnostic. See [the commands and result limits](ENCODER_BENCHMARK.md). Matched
encoder fine-tuning remains a separate work unit. The training runtime and
completed upload remain unchanged.

The stronger comparison uses same-language Base checkpoints from the requested
families, identical raw examples and split policy, comparable task heads,
controlled visible-text limits, equal validation search/early-stop budgets and
repeated seeds. Report per-task accuracy/F1 or correlation, uncertainty,
truncation, latency and the parameters actually used. Nuri's full parameter
count includes core/readout weights absent from an X-only probe. Do not rank
models by raw MLM losses across different vocabularies or compare discriminator
losses with token-prediction CE. Do not use English-focused references on Korean
inputs to claim general superiority. Semantic retrieval needs its own matched
adaptation and query/candidate protocol.

There is positive evidence that a modest budget can learn useful encoders:
[Cramming](https://arxiv.org/abs/2212.14034) reports often near-BERT downstream
results after one GPU day of focused MLM. Preprocessing and downstream
fine-tuning were excluded from that time. This does not transfer a score to
Nuri's multilingual mixed-objective five-stage recipe.
[ModernBERT's author card](https://huggingface.co/answerdotai/ModernBERT-base)
reports 149M Base parameters, 2T training tokens and primarily English coverage;
its [paper](https://arxiv.org/html/2412.13663v2) uses downstream adaptation.
Similar parameter counts alone do not establish comparable quality.

Separately assess short answers grounded in supplied text and retrieval of
relevant text from a candidate set when claiming the additional generation
and embedding capabilities. Freeze expected answers or relevance labels, examples,
failure rules and comparison settings privately before evaluating candidates.
Include mistakes, repetitions, missing or early EOS, input-counterfactual
responses, query/candidate counts and actual latency. Use the same inputs,
length limits and scoring rules for the legacy model and each candidate. A
general-chat claim needs a broader task suite than a few successful examples.

If the user accepts a bounded paid experiment, start with the existing exact
quarter of the calibrated Dense Base repair schedule. Its training portion is
estimated at about one hour; transfer, startup, calibration, CPU replay,
assessment and retrieval add billed time. The exact boundary and full optimizer,
RNG and schedule state must be preserved. A wall-time stop before the required
boundary cannot authorize the gate or continuation. A numeric pass still needs
decoded-output review, and the user's current concern prevents automatic full
continuation based on that pass alone.

Useful instruction answers require evidence from actual IT updates, and semantic
retrieval requires actual contrastive updates. If that evidence is needed before
committing the remaining budget, prepare separately reviewed stage boundaries
and a capped continuation proposal. These controls are not implemented or
approved here. Do not pretend that the existing Base gate tests those abilities,
silently shorten stages, discard the MoE releases or add an independent T=1
versus T=4 training pilot. Preserve D-014 and the agreed architecture unless
the user explicitly changes them.

Before any paid experiment, quote its complete setup, transfer, compute,
assessment and teardown cost. Choose a small independent spending/time cap and
verify the provider cutoff. Include previous spending in the strictly-below-$50
aggregate limit. No hourly price or final quality is assumed. Stop and retrieve
the evidence when the cap or required boundary is reached; failed or inconclusive
results do not authorize more spending.

If the user requires demonstrated practical quality before paying for any
learning experiment, do not recommend the custom generator's full training yet.
Evaluate an already trained candidate against the chosen tasks first. Reusing
its encoder is an architecture option in §8-B, but attaching new generation or
embedding heads still requires separate evidence and does not inherit a quality
guarantee.

Validation supplies these decisions. Completion test remains reserved under
D-020, with existing exposure limits reported. A later improvement claim must
name the evaluated tasks, measured results and failures; it must not substitute
test count, falling training loss or the proposed budget for practical quality.
