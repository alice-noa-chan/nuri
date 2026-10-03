# Nuri Joint Architecture Review

**Status: historical v0 design review.** The three subagents read the same `.dev/idea1.md` through `idea4.md`, developed proposals for their assigned roles, and exchanged messages before the root agent integrated the architecture. This record documents design reasoning; it does not report model training or measured performance.

The choices below describe the v0 review at that time. Later decisions, including release defaults, are recorded in `docs/ARCHITECTURE.md`, the current source of truth. Subsequent reviews belong in `docs/DISCUSSION.md`.

## Roles and Cross-Review

| Participant | Responsibility | Topics reviewed with other participants |
|---|---|---|
| reasoning_core | Latent recurrence, gradients, position encoding, and reasoning validation | Fixed X/R reuse and the risk of bypassing reasoning through direct X access with generation_design; shared K/V projections and preserved gradients with efficiency_review. |
| generation_design | Noisy canvas, leakage, length, and masked decoding | The contract that the canvas does not enter the core with reasoning_core; S-round and vocabulary-projection costs with efficiency_review. |
| efficiency_review | Modern encoder components, MoE, parameters, FLOPs, and the one-day budget | Total generation cost with generation_design; gates, routing, and reuse of source projections with reasoning_core. |
| root | General-purpose heads, budget integration, and experiment order | Alignment between embedding modes when projection P is shared, the scope of model terminology, feature defaults, and evaluation conditions. |

## Agreed Decisions and Rationale

1. **Separate X and R from the output canvas.** Run the input encoder and core once per generation request. This avoids re-encoding the source in every output-refinement round. Output-to-R feedback is a separate experiment.
2. **Read-only memory does not mean detach.** Reuse the fixed X's K/V projections within one forward pass and accumulate gradients from every recurrent step. Freezing the encoder is a separate training strategy.
3. **Disable latent RoPE and step embeddings by default.** Slots have no position in the sentence, and there is no evidence that iterations beyond the training range automatically become stable. Input and canvas self-attention each use RoPE over their own token positions.
4. **Use masked denoising and Mask-Predict-style refinement for v0 generation.** Do not call the method diffusion without fully defining its diffusion process and reverse transitions.
5. **Start with a separate length-prediction head.** State explicitly that gold-length training supplies an implicit length condition. Report predicted-length results by default and use oracle-length results for diagnosis.
6. **Store confidence from the last masked prediction.** Reusing logits produced when an already visible token can read its own embedding may contaminate evaluation through a self-copying shortcut.
7. **Allow direct access to X.** This may help exact copying and preservation of information from long inputs. Check the core's actual contribution through X-only, R-only, and X+R comparisons, plus R removal and shuffling.
8. **Evaluate fast and reasoning embedding modes separately.** A shared projection does not guarantee alignment in the same vector space. Default retrieval uses the same mode for queries and documents.
9. **Establish the first baseline with a dense model.** Introduce MoE in selected input FFNs first, and investigate recurrent-core MoE later. Record computation and dispatch costs when increasing the number of experts or top-k.
10. **Treat T and S as separate computation axes.** For this reference configuration, increasing S from 4 to 8 costs much more than increasing T from 4 to 8. Include the output vocabulary projection in cost estimates.
11. **Do not treat increased answer confidence or state changes alone as evidence of reasoning.** Prioritize final task performance, harder splits, depth curves, and intervention experiments. A correct fixed point that stops changing need not be a failure.
12. **Define the scope of the encoder-family terminology.** The model has no autoregressive decoder but does have a cross-attention output readout. Do not present it as a strict, single-stack BERT topology.

## How the `.dev` Ideas Were Incorporated

| Original idea | Adopted components | Components reserved for separate experiments |
|---|---|---|
| idea1 | Read-only X, learned R, shared recurrence, bidirectional R, residual connections, and SwiGLU | Adaptive halting, intermediate supervision, and random depth. |
| idea2 | Instruction-conditioned output, a masked canvas, and noncausal refinement | A diffusion objective and alternative length or decoding methods. |
| idea3 | An FFN factory and the placement contract for tokenwise sparse MoE | Enabling input/core MoE and selecting expert count, width, and top-k. |
| idea4 | Pre-norm, RMSNorm, SwiGLU, RoPE, and bias-free linear layers | QK normalization, local/global attention, GQA, MLA, and backend optimization. |

Additional proposed experiments were reasoning-conditioned Matryoshka embeddings, grouped hypothesis slots, and a learned step selector. The performance of these combinations had not been established.

## References and Limits of Applicability

- [ModernBERT](https://arxiv.org/abs/2412.13663): A reference for modern bidirectional encoder components. Nuri v0 uses global attention in every input layer; any proposed local/global schedule is a separate Nuri experiment.
- [Perceiver](https://arxiv.org/abs/2103.03206) and [Perceiver IO](https://arxiv.org/abs/2107.14795): References for input-to-latent attention and latent-based output queries. They do not establish that the full Nuri model, including its input encoder, scales linearly with input length.
- [Universal Transformers](https://arxiv.org/abs/1807.03819): A reference for depth recurrence and adaptive computation.
- [Recurrent-depth research](https://arxiv.org/abs/2502.05171): A reference for training and stabilizing shared recurrence and depth scaling. Results from a decoder model do not guarantee Nuri's performance.
- [Mask-Predict](https://aclanthology.org/D19-1633/): A reference for conditional masked prediction and confidence-based refinement. Nuri's combination of a length head and recurrent core is a separate proposal.
- [LLaDA](https://arxiv.org/abs/2502.09992): A reference when defining a masked-diffusion variant. Do not treat its sampler as equivalent to the v0 heuristic.
- [Switch Transformers](https://arxiv.org/abs/2101.03961): A reference for sparse expert routing and stabilization. The Nuri candidate reviewed here uses top-2 and is not compute-equivalent to its dense baseline.
- [SimCSE](https://arxiv.org/abs/2104.08821) and [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147): References for embedding objectives and representation learning at multiple dimensions.

## Questions Unresolved at the Time of This Review

The GPU count and exact H100 variant, training languages and data, tokenizer, reuse of a pretrained checkpoint, model runtime, and optimizer settings had not yet been selected. The design JSON declared architecture reference values rather than an executable configuration. Throughput within one day, batch size, and final language quality were not treated as established before measurement. Consult the current architecture decision log for choices made after this historical review.
