# Nuri release results — 2026-10-24

These are small matched task measurements of the completed checkpoints, not an official full benchmark submission or an overall capability score. Evaluation used one Colab Tesla T4. Korean frozen-feature classification is the strongest measured result. English classification and semantic similarity trail stronger references, and generated explanations remain repetitive and unreliable. Treat these as research checkpoints, not general chat assistants.

## Frozen encoder classification

The encoder stays frozen. A standardized ridge classifier uses content-token means (sentence pairs use both means, absolute difference and product). Each task has 1,024 fitting rows and 256 validation rows; regularization is selected from 17 fixed values by validation macro-F1. English values below are **consulted validation scores**, not independent tests. Korean values are from a **separate 256-row completion test**, with the already-selected classifier. Row/text overlap with the probe fitting and validation pools was excluded; earlier pretraining exposure is not ruled out.

| Model | KO topic: test accuracy % | EN sentiment: validation accuracy % | EN inference: validation accuracy % |
| --- | --- | --- | --- |
| nuri-base | 85.16 | 77.73 | 54.30 |
| nuri-base-it | 77.73 | 80.08 | 54.69 |
| nuri-base-moe | 85.16 | 74.61 | 53.52 |
| nuri-base-moe-it | 79.30 | 71.09 | 51.95 |
| nuri-base-embed | 85.94 | 77.34 | 57.03 |
| KLUE RoBERTa Base | 84.38 | - | - |
| BERT Base uncased | - | 84.77 | 53.12 |
| ELECTRA Base | - | 88.67 | 53.12 |
| RoBERTa Base | - | 85.94 | 58.20 |
| DeBERTa Base | - | 85.94 | 58.59 |
| ModernBERT Base | - | 78.52 | 56.25 |

Nuri Dense Base's Korean test accuracy has a row-bootstrap 95% interval of 80.47–89.45%; KLUE RoBERTa Base's is 79.69–88.67%. The overlapping intervals do not establish that Nuri is better. A fitted classifier's score is not an untrained native classification head's score. Dense and MoE trained for different amounts, so this is not a controlled architecture comparison.

## Semantic sentence similarity

Cosine similarity versus human similarity labels, Spearman correlation x100. Korean has 519 pairs and English 1,379. No labels fit the embedding head in this assessment. Nuri's trained sentence head uses `encode(..., mode="fast"|"reasoning", dim=256)`; plain mean is a different 1,024-dimensional representation. E5 uses query prefixes on both sides. MiniLM uses its documented 128-token limit; the other models use up to 512 tokens. Reference pretraining/benchmark exposure is not ruled out.

| Representation | KO Spearman x100 | EN Spearman x100 |
| --- | --- | --- |
| Nuri Embed /fast 256d | 60.64 | 65.55 |
| Nuri Embed /reasoning 256d | 60.63 | 65.46 |
| Nuri Embed /plain mean 1024d | 62.40 | 68.91 |
| Multilingual E5 small | 75.76 | 84.11 |
| Multilingual E5 base | 78.01 | 85.65 |
| Multilingual MiniLM L12 | 65.89 | 84.42 |

On a separate 2,977-query /2,977-document **near-duplicate recovery** pool, Nuri Embed fast R@1 is 77.49%, versus E5 small 76.79% and E5 base 76.69%. This pool is not broad semantic search; its close recovery scores do not erase the semantic-similarity gap above.

## Grounded short-answer generation

Normalized **decoded-string exact match**, not token-ID equality. Each model receives the same 80 fresh questions (40 KO /40 EN) asking for a value present in a passage. Nuri IT uses predicted output lengths, T=4, S=4, and a 128-token canvas cap; external models use their own chat templates, greedy decoding and up to 128 new tokens. Qwen3 thinking is disabled. Inputs are capped at 512 tokens; tokenizers imply different text budgets. These consulted examples are validation diagnostics, not a claim of clean final-test generalization.

| Model | KO extraction: 40 examples % | EN extraction: 40 examples % | Combined: 80 examples % |
| --- | --- | --- | --- |
| Nuri Dense IT | 37.50 | 50.00 | 43.75 |
| Nuri MoE IT | 25.00 | 27.50 | 26.25 |
| Qwen2.5 0.5B Instruct | 0.00 | 85.00 | 42.50 |
| Qwen3 0.6B (non-thinking) | 90.00 | 100.00 | 95.00 |
| SmolLM2 360M Instruct | 0.00 | 60.00 | 30.00 |

The two Nuri IT models score about 61% on a 300-question disjoint-name synthetic stress pool, but explicitly trained on the task templates, and arithmetic accuracy is only 0–2% in its per-language stress slices. Synthetic template success is not evidence of general reasoning or superiority to instruction-model references. Open explanations contain circular statements, repeated phrases and factual errors. More latent steps are not proven to improve general reasoning.

## Rejected short continuation experiment

The released weights are the original completed exports. An additional one-seed screen warm-started Dense IT for 300 full-model optimizer updates per arm (effective batch 4, fresh AdamW, LR 5e-5, warmup 15), changing only full-mask probability between 0.1 /0.3 /0.5. All source/target stream digests match. Natural generation uses 128 fixed examples, predicted lengths, T=4, S=4 and cap 128. The language reporting audit found 64 Korean-script, 60 mainly English and 4 Japanese-script references; four were initially mislabelled English. The cohort and aggregate scoring were preserved.

| Dense IT condition | Natural token F1 x100 | Repeated 4-grams % | Extraction exact match % |
| --- | --- | --- | --- |
| init | 39.98 | 19.23 | 33.50 |
| p10 | 36.43 | 26.69 | 32.50 |
| p30 | 35.29 | 26.26 | 23.50 |
| p50 | 35.23 | 28.54 | 21.00 |

`init` is the unchanged release; `p10`, `p30`, `p50` are experimental continuations. F1 is reference-token overlap, not factual accuracy; repetition is duplicate output 4-grams. Both higher-mask arms failed the predeclared generation adoption criteria and worsened grounded extraction. S=16 also worsened versus initialization. None of these continuation weights was adopted. This small-batch, bounded-data screen does not show that every longer or better-tuned training run would fail.

## Release boundaries

- Supported languages: Korean and English first, some Japanese; no broad multilingual coverage claim.
- Input limit 1,024 tokens; answer canvas at most 256 including EOS. Automatic length is trained only in IT checkpoints.
- `generate_refine` is a custom masked-refinement API; standard autoregressive `generate()` and text-generation pipelines do not implement this model.
- The trained run predates the sampler retention fix: length bucketing discarded unused candidates and packing discarded tails. Source pass counters did not establish complete trained coverage. Public processed-position counts remain actual work, not unique corpus tokens. The corrected sampler is for future fresh training and has not been evaluated in a complete Base retrain.
- Models are not safety-assessed and are unsuitable for factual assistance or high-stakes decisions without independent validation.
- Training dataset identities, raw/prepared records and optimizer/resume states are not distributed. Published metrics omit private example text and source metadata; they do not constitute an independently reproducible public benchmark.

## Package integrity

The released BF16 safetensors are byte-identical to the completed exports evaluated above. Every model includes inference code, tokenizer files, MIT license, this report, a compact numeric evaluation record and SHA256SUMS. Exact code/weight/export verification records were retained locally. The five original weights were not replaced by the rejected continuations.
