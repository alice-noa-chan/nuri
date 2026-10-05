# Literature review for the corrected Nuri retry

Reviewed on 2026-10-05 by Codex, with independent generation and encoder/core agent reviews. This research memo supports Q-17; [ARCHITECTURE.md](ARCHITECTURE.md) remains the design authority. Recommendations below are experiments, not changes to defaults or acceptance thresholds.

## Main assessment

Conditional masked generation has a sound research precedent. Nuri's learned output positions are a well-supported correction to the specific all-MASK symmetry defect. The next useful question is whether the corrected model learns input-conditioned language under its actual training budget. More refinement, deeper readout and trajectory training may help, but the cited work does not establish their benefit for Nuri.

Evidence is separated throughout this note:

- **Measured locally:** the old checkpoint's generation failure and position symmetry, recorded in [DENSE_BASE_EVALUATION.md](DENSE_BASE_EVALUATION.md).
- **Structural reasoning:** why identical canvas values survive Q/K-only positional rotation in the old topology.
- **Published results:** behavior measured by paper authors in their own models and settings.
- **Nuri hypotheses:** proposed transfers that still need controlled validation. Corrected natural-language quality has not yet been measured.

## What the primary sources establish

| Source | Evidence and relevance to Nuri |
|---|---|
| [Mask-Predict, EMNLP 2019](https://aclanthology.org/D19-1633/) | A bidirectional target stack repeatedly fills masked positions using source cross-attention. Training selects a masked count uniformly from `1..L`. The [official target architecture](https://github.com/facebookresearch/Mask-Predict/blob/main/fairseq/models/bert_seq2seq.py) adds learned positions to token embeddings before attention. This is the closest precedent for Nuri's conditional readout and directly supports D-019's residual/value-path positions. Its translation results do not establish general conversational quality. |
| [MDLM, NeurIPS 2024](https://arxiv.org/abs/2406.07524) | Derives a weighted masked objective and reverse sampler from a specified diffusion process. Its substitution parameterization forbids predicting MASK and carries visible tokens forward. Explicit timestep conditioning is optional. Nuri's conditional loss is valid on its own terms, but its current corruption and sampler have no demonstrated equivalence to this likelihood bound. See the [author implementation](https://github.com/kuleshov-group/mdlm). |
| [LLaDA, NeurIPS 2025](https://arxiv.org/abs/2502.09992) | Uses independent Bernoulli response masking at a sampled ratio, an inverse-ratio weighted objective and a bidirectional model. Low-confidence selection worked better than random selection in reported settings. Its 8B model used 2.3 trillion pretraining tokens and approximately 130,000 H800 hours. These results establish large-scale feasibility, not Nuri quality at a small budget. Its EOS-padding recipe differs from Nuri's target policy. |
| [ReMDM, NeurIPS 2025](https://arxiv.org/abs/2503.00307) | Provides a derived remasking process that can revise earlier predictions. This supports investigating reversible sampling; it does not turn Nuri's Mask-Predict heuristic into the same process. The [author code](https://github.com/kuleshov-group/remdm) is a reference for a separate sampler experiment. |
| [Beyond Confidence, ICML 2026](https://proceedings.mlr.press/v306/chen26ci.html) | Studies historical context and consistency beyond a single confidence score. This motivates measuring sequence coherence and intermediate corrections. A different [2026 preprint](https://arxiv.org/abs/2606.12232) reports limited remasking gains at short block sizes and possible diversity loss. The two findings make settings and matched comparisons essential; remasking is not a universal improvement. |
| [Masks Can Be Distracting, ICML 2026](https://proceedings.mlr.press/v306/piskorz26a.html) | Finds context-position sensitivity and distractor effects from appended masks in the studied diffusion models. Nuri's encoder never reads its output canvas, so that canvas cannot directly distract its cached input encoding through the studied mechanism. Readout sensitivity to canvas length remains an untested transfer hypothesis. |
| [Perceiver IO, 2021](https://arxiv.org/abs/2107.14795) | Uses a latent bottleneck with explicit output queries. Sequence outputs have position-dependent queries; the language configuration also supplies input position features. This supports testing explicit source-position features if needed. It does not show that Nuri's contextual input memory lacks order, or that input tokens and abstract reasoning slots should share cross-attention RoPE coordinates. |
| [Recurrent-depth reasoning / Huginn, 2025](https://arxiv.org/abs/2502.05171) | Demonstrates test-time recurrent computation after training with variable recurrence. The reported model has 3.5B parameters and 800B training tokens. Its recurrent state has one vector per input token, unlike Nuri's 32 latent slots. Shared weights and extra computation are promising principles; improved Nuri reasoning at larger `T` remains unmeasured. See the [author implementation](https://github.com/seal-rg/recurrent-pretraining). |
| [ModernBERT, 2024](https://arxiv.org/abs/2412.13663) | Combines RoPE, gated feed-forward layers, alternating local/global attention, unpadding and efficient kernels. Base has 149M parameters and was trained on 2T tokens. Nuri already shares several architectural ideas but uses RMSNorm/SwiGLU rather than the paper's LayerNorm/GeGLU. Nuri's dense local attention mask is not an implementation of sparse window kernels or unpadding. See the [author code](https://github.com/AnswerDotAI/ModernBERT). |
| [Cramming, ICML 2023](https://proceedings.mlr.press/v202/geiping23a.html) | Reports useful, often BERT-like downstream encoder performance after one GPU day of focused MLM pretraining. Preprocessing and downstream fine-tuning are outside that time limit. This is positive evidence that a modest budget can learn useful representations. It does not transfer a score to Nuri's combined generation, MoE, instruction and embedding stages. See the [author recipe](https://github.com/JonasGeiping/cramming). |

Two recent **preprints**, whose findings should remain provisional, address training alignment:

- [Reducing Pretraining-Generation Mismatch, August 2026](https://arxiv.org/abs/2608.09424) studies clean-prefix and masked-suffix training, including an autoregressive prefix objective. Nuri already uses a clean, separately encoded prefix for span denoising. Adding the paper's prefix objective would change supervision and attention; it is not a missing switch in the current recipe.
- [Trajectory-Aware Training / PUMBA, September 2026](https://arxiv.org/abs/2609.37974) studies training on consecutive model-induced generation states with information and gradients spanning steps. It also reports overfitting under some exact-alignment settings. Nuri's single corrupted-target training leaves a real gold-context versus generated-context gap. Its optional draft feedback is not this training procedure and should not be enabled on the strength of the paper alone.

## Comparison with the current implementation

### Output positions and memory

In the old readout, every valid all-MASK position began with the same vector. Q/K-only RoPE could change attention weights, but a normalized weighted sum of identical value vectors was still identical. Shared tokenwise layers preserved that equality. Equal queries in unrotated memory cross-attention retrieved equal memory mixtures. This is a property of that topology, not a general failure of RoPE.

D-019 adds a learned position vector once to the canvas residual before readout. Positions therefore reach both the residual and value paths. Mask-Predict's learned target embeddings and Perceiver IO's position-dependent output queries support this choice. Breaking the invariant permits different tokens at different output positions; it does not ensure coherent or correctly conditioned sentences.

Input memory `X` already comes from a position-aware encoder. Jointly permuting its encoded rows and corresponding mask/K/V does not remove positional information stored inside those vectors. A meaningful input-order test swaps roles or order **before encoding**, while keeping the token multiset and length fixed.

### Corruption, refinement and length

[corruption.py](../nuri/corruption.py) chooses the full-mask branch with probability 0.1; otherwise it samples a ratio in `[0.15, 0.95]` and masks an exact rounded count. Rounding can create additional fully masked short rows. Branch probability is not the full-mask share of target tokens or objective weight. The loss averages masked tokens per row rather than using LLaDA's inverse-ratio objective.

[generation.py](../nuri/generation.py) defaults to four refinement rounds. It stores confidence when a masked token is predicted and may later remask visible tokens. This matches the [official Mask-Predict strategy](https://github.com/facebookresearch/Mask-Predict/blob/main/fairseq/strategies/mask_predict.py): retained confidence is deliberate, not a newly discovered defect. LLaDA's [official sampler](https://github.com/ML-GSAI/LLaDA/blob/main/generate.py) instead selects only currently masked positions for commitment. Nuri already supports revisiting predictions.

Training sees visible gold tokens around masked targets; generation sees its own predictions. More rounds can provide context and also propagate errors. At `S=8`, a late round has approximately 12.5% masked positions, below the nominal 15% training floor for sufficiently long responses. Larger `S` needs quality and actual latency measurements.

[Mask-Predict, Section 5.4](https://aclanthology.org/D19-1633.pdf) uses sequence-level distillation from an autoregressive teacher in its main translation recipe and reports worse results on raw targets, with a setting-dependent gap. Its successful four-round results therefore do not predict raw-text continuation quality for Nuri. Reducing target ambiguity is a possible interpretation of distillation's benefit in that application; adopting it here would be a separate training proposal.

[LLaDA, Appendix B.4](https://arxiv.org/html/2502.09992v3) also compares fixed-length left-to-right blocks with parallel denoising inside each block. This motivates a training-free blockwise Mask-Predict comparison, not a claim that block decoding always improves quality. Nuri can retain bidirectional attention while freezing earlier output blocks. Smaller blocks require more full-canvas readout calls, so matched `S` alone does not match inference compute. Measure actual latency and calls, and keep this supplementary policy outside the frozen acceptance gate.

Base requires an explicit canvas length because its length head is untrained. Instruction tuning trains that head. Nuri supervises EOS only at genuine endings; interior training spans can legitimately omit it. Separate token quality, early EOS, missing EOS and length prediction. Do not transplant an EOS-padding adjustment from another model without matching its target construction.

## Recommended experiment order

All model scoring for diagnosis and selection uses validation. Completion test remains reserved for the final selected run under D-020. [HOLDOUT_MONITORING.md](HOLDOUT_MONITORING.md) defines that separation. Supplementary diagnostics must not change the frozen repair gate.

| Priority | Experiment | Decision it can inform |
|---|---|---|
| 1: prepared retry | Run the unchanged D-019 numeric and qualitative gate at the exact calibrated checkpoint boundary. Preserve the legacy comparison and genuine input-swap controls. | Does the minimal correction produce conditioned language rather than only distinct logits? |
| 2: same-checkpoint diagnosis | Fix `T=4`, inputs and explicit/oracle lengths. Compare `S=1, 2, 4`; use `S=8` as a bounded diagnostic. Separately compare full-canvas and opt-in left-to-right block refinement with explicit block size and readout-call/latency accounting. Inspect intermediate predictions, repetition, corrections gained/lost and latency. At matched masks, compare visible gold context with visible generated context. | Is failure concentrated in first fill, refinement or recovery from generated errors? |
| 3: same-checkpoint diagnosis | Compare lengths on genuine endings; for IT, separate oracle from predicted lengths. Swap complete `X`, `R` and masks together across real batch rows. Separately test raw role/order swaps and `R` shuffling. With length fixed, compare `T=2, 4, 8` at fixed `S` and record latency. | Are length errors, weak input conditioning or latent-state dependence limiting quality? A zeroed `R` is a distribution shift; a small effect at a poor-quality floor is not proof the core is unused. |
| 4: separate training proposal | Compare a different full-mask rate or uniform masked count with the repaired baseline. Keep optimizer steps, task exposure and length distributions controlled; report masked-target counts and compute. | Does corruption coverage help full-mask conditioned generation? Additional masks could not have repaired the old structural symmetry. |
| 5: later controlled proposal | Test readout depth, explicit source-position features or multi-round trajectory training individually, only after the repaired baseline. Compare core `T` separately from refinement `S`. | Which extra capacity or computation produces useful quality per unit of time? No new default is justified yet. |

## Performance expectations

The defensible near-term target is reduced collapse, useful short input-conditioned generation after instruction tuning, and measurable representation learning. This is a project hypothesis, not a measured forecast. Cramming provides encouraging small-budget encoder evidence, while ModernBERT, LLaDA and Huginn demonstrate how much more training supports their stronger published results.

Similar parameter counts, lower training loss, distinct first-fill logits or more recurrent steps do not establish comparable quality. Report actual processed tokens, task exposure, validation objectives, decoded samples and inference latency before claiming a model tier. The first corrected-language gate should determine continuation under the existing budget. This review does not alter the frozen runtime configuration, package or quality thresholds.

## Follow-up after Claude's review

D-021 lowers the fully decayed Dense warm-start peak from `3e-4` to `1e-4`, retaining the 50/50 Base mix and the full-schedule gate boundary. This is a conservative run policy; it is not a measured optimal LR. The earlier interrupted MoE checkpoint is a different case and keeps its separately declared peak.

[Gupta et al. (2023)](https://arxiv.org/abs/2308.04014) observe an initial loss rise under rewarming, followed by longer-run benefits. [Ibrahim et al. (2024)](https://arxiv.org/abs/2403.08763) study rewarming, redecay and replay at much larger scales. These papers motivate caution but do not predict a Nuri-specific loss spike or prove a smaller peak is superior. LR does not directly change no-grad evaluation; it changes the preceding weight updates. A failed numeric gate remains a failed gate, including an MLM-only miss.

Supplementary measurements should report clipping for accumulated optimizer updates, sampled pre-clip module gradients and position-table row norms. A last-microbatch task label cannot assign a combined gradient to one task. Report a training-fitted, input-free position-frequency line beside the full-mask gate using [the frequency tool](../tools/position_frequency_baseline.py). A measured prior failing the gate limits that particular explanation; it does not rule out every input-free strategy or establish conditioning by itself. Genuine source swaps and qualitative assessment remain necessary.

The original package is preserved. Revised code and D-021 policy require a new source-bound package; these follow-up changes do not relax the frozen quality thresholds or consult test for selection.
