"""Evaluations from ARCHITECTURE.md §12.

All functions take token ids, so they work with any tokenizer, and return
plain dicts that can be written as JSON. Every result reports its settings
(T, S, number of examples) so numbers are never shown without context.

- generation_accuracy: exact match, token F1, repeat rate, EOS errors, length MAE
  and eligible coverage
- depth_curve / rounds_curve: the same metric over several T or S
- r_interventions: normal R vs zeroed R vs R shuffled between samples
- mlm_accuracy: masked token accuracy on held-out text
- span_denoising: held-out span reconstruction loss at fixed mask ratios
  (measures the core + readout path of base models, which MLM skips)
- retrieval_metrics: Recall@k and MRR for an embedding mode
- lexical_retrieval_baseline: the same retrieval scored by shared tokens only (no model)
- retrieval_ranks / lexical_retrieval_ranks / ranking_metrics_from_ranks: per-query ranks and
  the metrics computed from them, for reports that split queries into groups
- core_diagnostics: state norm, update norm, slot cosine and effective rank per step
- halting_curve: exact match vs mean core steps for several halting tolerances
"""

from __future__ import annotations

import hashlib
import json
import random
import time
import warnings
from collections import Counter
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from .corruption import mask_for_mlm, pad_batch, split_span
from .generation import generate
from .model import NuriModel

Example = tuple[list[int], list[int]]  # (prompt ids, answer ids without EOS)


def _batches(items: Sequence, size: int, min_size: int = 1):
    """Consecutive chunks of `size`. A tail shorter than min_size joins the previous chunk."""
    chunks = [items[start:start + size] for start in range(0, len(items), size)]
    if len(chunks) > 1 and len(chunks[-1]) < min_size:
        tail = chunks.pop()
        chunks[-1] = list(chunks[-1]) + list(tail)
    yield from chunks


def _length_only_accuracy(answers: list[list[int]]) -> float:
    """Best exact match a model could get from the answer length alone.

    With the oracle length the canvas size is given, so a model can pick the
    most common answer of that length without reading the input at all
    (for example "yes" and "no" with different token counts). Oracle exact
    match is only meaningful well above this number.
    """
    by_length: dict[int, dict[tuple, int]] = {}
    for answer in answers:
        counts = by_length.setdefault(len(answer), {})
        counts[tuple(answer)] = counts.get(tuple(answer), 0) + 1
    best = sum(max(counts.values()) for counts in by_length.values())
    return best / max(len(answers), 1)


def token_f1(predicted: list[int], reference: list[int]) -> float:
    """Bag-of-tokens F1 between two token lists; 1.0 when both are empty."""
    if not predicted or not reference:
        return float(predicted == reference)
    overlap = sum((Counter(predicted) & Counter(reference)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def adjacent_repeat_rate(tokens: list[int]) -> float:
    """Share of adjacent token pairs that repeat the same token (0 for fewer than two tokens)."""
    pairs = list(zip(tokens, tokens[1:]))
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else 0.0


@torch.no_grad()
def generation_accuracy(model: NuriModel, examples: Sequence[Example], steps: int | None = None,
                        rounds: int | None = None, oracle_length: bool = False, batch_size: int = 32,
                        transform_r: Callable[[torch.Tensor], torch.Tensor] | None = None,
                        halt_tol: float | None = None) -> dict:
    """Exact match of generated answers, with partial-credit and repetition metrics.

    oracle_length=True gives the canvas the gold length (answer + EOS), which
    isolates denoising errors from length errors. Otherwise the length head
    picks the length, so the model must be instruction-tuned. Oracle results
    also report length_only_accuracy: how much the given length alone reveals.

    Examples whose answer plus EOS does not fit M are excluded (never cut)
    and counted in "excluded". Coverage is the eligible fraction of the supplied
    examples. Early EOS means an EOS-terminated answer shorter than the reference.
    Canvas-length MAE includes EOS; answer-length MAE uses returned content tokens,
    including when EOS is missing. Other metrics use eligible examples only.
    Exact match against one reference also fails legitimate alternative
    wordings, so two more metrics are reported: token_f1 is the bag-of-tokens
    F1 between answer and reference (partial credit, as in extractive QA),
    and repeat_rate is the share of adjacent identical tokens in the answer,
    a reference-free sign of degenerate output; reference_repeat_rate gives
    the same share for the references as context.
    With transform_r, every batch has at least two
    samples so that shuffling R always swaps it between different samples.
    """
    model.eval()
    device = next(model.parameters()).device
    config = model.config
    eligible = [(p, list(a)) for p, a in examples if len(a) + 1 <= config.max_response_len]
    if transform_r is not None and len(eligible) < 2:
        raise ValueError("R interventions need at least two eligible examples")
    min_size = 2 if transform_r is not None else 1
    counts = {"correct": 0, "missing_eos": 0, "early_eos": 0, "wrong_length": 0,
              "canvas_length_error": 0, "answer_length_error": 0, "total": 0, "core_steps": 0,
              "token_f1": 0.0, "repeat_rate": 0.0, "reference_repeat_rate": 0.0}
    for chunk in _batches(eligible, max(batch_size, min_size), min_size):
        prompts = [p[-config.max_input_len:] for p, _ in chunk]
        ids, valid = pad_batch(prompts, config.pad_id)
        gold_lengths = torch.tensor([len(a) + 1 for _, a in chunk])
        result = generate(model, ids.to(device), valid.to(device), steps=steps, rounds=rounds,
                          canvas_len=gold_lengths.to(device) if oracle_length else None,
                          transform_r=transform_r, halt_tol=halt_tol)
        if result.reasoning_steps is not None:
            counts["core_steps"] += int(result.reasoning_steps.sum())
        for (_, answer), tokens, ended, length, gold in zip(chunk, result.tokens, result.ended_with_eos,
                                                            result.lengths.tolist(), gold_lengths.tolist()):
            counts["correct"] += tokens == list(answer)
            counts["missing_eos"] += not ended
            counts["early_eos"] += ended and len(tokens) < len(answer)
            counts["wrong_length"] += length != gold
            counts["canvas_length_error"] += abs(length - gold)
            counts["answer_length_error"] += abs(len(tokens) - len(answer))
            counts["token_f1"] += token_f1(tokens, list(answer))
            counts["repeat_rate"] += adjacent_repeat_rate(tokens)
            counts["reference_repeat_rate"] += adjacent_repeat_rate(list(answer))
            counts["total"] += 1
    total = max(counts["total"], 1)
    report = {"exact_match": counts["correct"] / total, "missing_eos_rate": counts["missing_eos"] / total,
              "early_eos_rate": counts["early_eos"] / total,
              "canvas_length_mae": counts["canvas_length_error"] / total,
              "answer_length_mae": counts["answer_length_error"] / total,
              "length_error_rate": counts["wrong_length"] / total,
              "token_f1": counts["token_f1"] / total, "repeat_rate": counts["repeat_rate"] / total,
              "reference_repeat_rate": counts["reference_repeat_rate"] / total,
              "examples": counts["total"],
              "excluded": len(examples) - len(eligible),
              "evaluation_coverage": len(eligible) / max(len(examples), 1),
              "mean_core_steps": counts["core_steps"] / total, "halt_tol": halt_tol,
              "steps": steps if steps is not None else config.default_depth,
              "rounds": rounds if rounds is not None else config.refine_rounds, "oracle_length": oracle_length}
    if oracle_length:
        report["length_only_accuracy"] = _length_only_accuracy([a for _, a in eligible])
    return report


@torch.no_grad()
def source_conditioning(model: NuriModel, examples: Sequence[Example], *, max_examples: int = 128,
                        seed: int = 0, steps: int = 4, rounds: int = 4,
                        oracle_length: bool = True, batch_size: int = 32, sample_count: int = 0) -> dict:
    """Compare real input replacements at equal reference lengths.

    Each selected prompt receives a different visible prompt whose reference
    has the same length and different content. Both generations start from
    MASK; references are not appended as teacher memory. Partners may repeat and are counted.
    Correctness with EOS matters separately from merely changing the output.
    This bounded diagnostic is not an edited-context counterfactual dataset.
    """
    if type(max_examples) is not int or not 1 <= max_examples <= 1024:
        raise ValueError("max_examples must be an integer in [1, 1024]")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(sample_count) is not int or not 0 <= sample_count <= 32:
        raise ValueError("sample_count must be an integer in [0, 32]")
    if type(seed) is not int or type(steps) is not int or steps < 0:
        raise ValueError("seed must be an integer and steps a nonnegative integer")
    if type(rounds) is not int or rounds < 1 or type(oracle_length) is not bool:
        raise ValueError("rounds must be positive and oracle_length a boolean")
    if not oracle_length and not model.config.length_head_trained:
        raise ValueError("predicted-length conditioning requires a trained length head")

    config = model.config
    eligible, empty, over_cap, truncated = [], 0, 0, 0
    for index, (prompt, answer) in enumerate(examples):
        if not prompt:
            empty += 1
        elif len(answer) + 1 > config.max_response_len:
            over_cap += 1
        else:
            truncated += len(prompt) > config.max_input_len
            eligible.append((index, list(prompt[-config.max_input_len:]), list(answer)))
    selected = random.Random(seed).sample(eligible, min(max_examples, len(eligible)))
    selected.sort(key=lambda row: row[0])
    pairs = []
    for index, prompt, answer in selected:
        partner = next((row for row in selected
                        if row[1] != prompt and row[2] != answer and len(row[2]) == len(answer)), None)
        if partner is not None:
            pairs.append(((index, prompt, answer), partner))

    selection = [(a, b) for a, b in pairs]
    report = {"status": "assessed" if pairs else "unassessed_no_length_matched_pairs",
              "input_examples": len(examples), "eligible_examples": len(eligible),
              "excluded_empty_prompt": empty, "excluded_over_cap_answer": over_cap,
              "eligible_truncated_prompts": truncated, "selected_examples": len(selected),
              "unpaired_selected_examples": len(selected) - len(pairs), "paired_examples": len(pairs),
              "unique_partner_examples": len({b[0] for _, b in pairs}),
              "reused_partner_presentations": len(pairs) - len({b[0] for _, b in pairs}),
              "evaluation_coverage": len(pairs) / max(len(examples), 1),
              "max_examples": max_examples, "seed": seed, "steps": steps, "rounds": rounds,
              "oracle_length": oracle_length, "temperature": 0.0,
              "pair_policy": "different visible prompts and answers; equal reference token lengths",
              "prompt_policy": "last max_input_len tokens, matching generation_accuracy",
              "selection_sha256": hashlib.sha256(json.dumps(selection).encode()).hexdigest(),
              "generated_rows": 0, "readout_calls": 0}
    counts = {name: 0 for name in ("original_content_exact_match", "original_token_sequence_exact_match",
                                  "swapped_original_target_token_sequence_exact_match",
                                  "swapped_own_target_token_sequence_exact_match", "response_change_rate",
                                  "both_targets_correct_rate", "original_missing_eos_rate",
                                  "swapped_missing_eos_rate", "original_early_eos_rate",
                                  "swapped_early_eos_rate", "original_canvas_length_mae",
                                  "swapped_canvas_length_mae")}
    samples = []
    if not pairs:
        return {**report, **{key: None for key in counts}, "conditioning_gap": None,
                "length_only_token_sequence_exact_match": None, "seconds": 0.0,
                **({"samples": samples} if sample_count else {})}

    device = next(model.parameters()).device
    modes = [(module, module.training) for module in model.modules()]
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    start = time.perf_counter()
    try:
        model.eval()
        # Keep optional development checks from changing subsequent training RNG.
        with torch.random.fork_rng(devices=cuda_devices):
            for chunk in _batches(pairs, batch_size):
                results = []
                for side in (0, 1):
                    ids, valid = pad_batch([pair[side][1] for pair in chunk], config.pad_id)
                    lengths = torch.tensor([len(pair[side][2]) + 1 for pair in chunk], device=device)
                    result = generate(model, ids.to(device), valid.to(device), steps=steps, rounds=rounds,
                                      canvas_len=lengths if oracle_length else None, temperature=0.0)
                    results.append(result)
                    report["generated_rows"] += len(chunk)
                    report["readout_calls"] += result.readout_calls
                original, swapped = results
                for row, (before, after) in enumerate(chunk):
                    answer, other_answer = before[2], after[2]
                    first, second = original.tokens[row], swapped.tokens[row]
                    ended, other_ended = original.ended_with_eos[row], swapped.ended_with_eos[row]
                    first_ok = first == answer and ended
                    second_ok = second == other_answer and other_ended
                    counts["original_content_exact_match"] += first == answer
                    counts["original_token_sequence_exact_match"] += first_ok
                    counts["swapped_original_target_token_sequence_exact_match"] += second == answer and other_ended
                    counts["swapped_own_target_token_sequence_exact_match"] += second_ok
                    counts["response_change_rate"] += first != second
                    counts["both_targets_correct_rate"] += first_ok and second_ok
                    counts["original_missing_eos_rate"] += not ended
                    counts["swapped_missing_eos_rate"] += not other_ended
                    counts["original_early_eos_rate"] += ended and len(first) < len(answer)
                    counts["swapped_early_eos_rate"] += other_ended and len(second) < len(other_answer)
                    counts["original_canvas_length_mae"] += abs(int(original.lengths[row]) - len(answer) - 1)
                    counts["swapped_canvas_length_mae"] += abs(int(swapped.lengths[row]) - len(other_answer) - 1)
                    if len(samples) < sample_count:
                        sample = {"original_index": before[0], "partner_index": after[0],
                                  "original_prompt": before[1], "swapped_prompt": after[1],
                                  "original_reference": answer, "swapped_reference": other_answer}
                        for prefix, result in (("original", original), ("swapped", swapped)):
                            length = int(result.lengths[row])
                            sample.update({f"{prefix}_tokens": result.tokens[row],
                                           f"{prefix}_ended_with_eos": result.ended_with_eos[row],
                                           f"{prefix}_canvas_length": length,
                                           f"{prefix}_canvas": result.canvas[row, :length].cpu().tolist()})
                        samples.append(sample)
    finally:
        for module, training in modes:
            module.training = training
    report.update({key: value / len(pairs) for key, value in counts.items()})
    report["conditioning_gap"] = (report["original_token_sequence_exact_match"]
                                  - report["swapped_original_target_token_sequence_exact_match"])
    report["length_only_token_sequence_exact_match"] = (
        _length_only_accuracy([a[2] for a, _ in pairs]) if oracle_length else None)
    report["seconds"] = time.perf_counter() - start
    if sample_count:
        report["samples"] = samples
    return report


def depth_curve(model: NuriModel, examples: Sequence[Example], depths=(1, 2, 4, 8, 16), **kwargs) -> list[dict]:
    """Exact match at several reasoning depths T of the same checkpoint (D-014)."""
    return [generation_accuracy(model, examples, steps=t, **kwargs) for t in depths]


def halting_curve(model: NuriModel, examples: Sequence[Example], tolerances=(None, 0.01, 0.03, 0.1),
                  steps: int = 16, **kwargs) -> list[dict]:
    """Exact match and mean core steps at several halting tolerances (None = always `steps`)."""
    return [generation_accuracy(model, examples, steps=steps, halt_tol=tol, **kwargs) for tol in tolerances]


def rounds_curve(model: NuriModel, examples: Sequence[Example], rounds=(1, 2, 4, 8), **kwargs) -> list[dict]:
    """Exact match at several refinement rounds S."""
    return [generation_accuracy(model, examples, rounds=s, **kwargs) for s in rounds]


def zero_r(r: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(r)


def shuffle_r(r: torch.Tensor) -> torch.Tensor:
    """Give every sample the R of the next sample (a fixed derangement for batches > 1)."""
    return r.roll(1, dims=0)


def r_interventions(model: NuriModel, examples: Sequence[Example], **kwargs) -> dict:
    """Accuracy with normal R, zeroed R and R shuffled between samples.

    These are measurements, not conclusions. Only when the normal accuracy is
    clearly above trivial baselines (chance, majority answer, and for oracle
    length the length_only_accuracy) does a small drop suggest that the
    readout bypasses the core. An untrained model stays at the floor under
    every intervention and says nothing about the core.
    """
    return {"normal": generation_accuracy(model, examples, **kwargs),
            "zero_r": generation_accuracy(model, examples, transform_r=zero_r, **kwargs),
            "shuffled_r": generation_accuracy(model, examples, transform_r=shuffle_r, **kwargs)}


@torch.no_grad()
def mlm_accuracy(model: NuriModel, documents: Sequence[list[int]], rate: float = 0.15, batch_size: int = 32,
                 seed: int = 0) -> dict:
    """Top-1 accuracy and loss on masked tokens of held-out documents (first N tokens of each)."""
    model.eval()
    device = next(model.parameters()).device
    config = model.config
    generator = torch.Generator().manual_seed(seed)
    special = (config.pad_id, config.mask_id, config.eos_id)
    correct = targets = 0
    loss_sum = 0.0
    for chunk in _batches(list(documents), batch_size):
        ids, valid = pad_batch([d[:config.max_input_len] for d in chunk], config.pad_id)
        corrupted, mask = mask_for_mlm(ids, valid, config.mask_id, config.vocab_size, special, rate, generator)
        x, _ = model.encode(corrupted.to(device), valid.to(device))
        logits = model.mlm_logits(x, mask.to(device)).float()
        labels = ids[mask].to(device)
        correct += int((logits.argmax(-1) == labels).sum())
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
        targets += int(mask.sum())
    return {"accuracy": correct / max(targets, 1), "loss": loss_sum / max(targets, 1), "targets": targets,
            "mask_rate": rate}


@torch.no_grad()
def span_denoising(model: NuriModel, documents: Sequence[list[int]], ratios=(0.15, 0.5, 1.0),
                   steps: int | None = None, batch_size: int = 32, seed: int = 0) -> list[dict]:
    """Per-sample mean CE and token accuracy of reconstructing held-out spans.

    Each document is split once (fixed seed) into a prefix and the following
    span, exactly like the base-stage objective; then the same spans are
    scored with `ratio` of their positions masked (1.0 = write the whole span).
    Lower loss at the same ratio means a better encoder + core + readout path.
    """
    model.eval()
    device = next(model.parameters()).device
    config = model.config
    rng = random.Random(seed)
    pairs = [split_span(d, config.max_input_len, config.max_response_len, config.eos_id, 16, rng)
             for d in documents if len(d) >= 2]
    rows = []
    for ratio in ratios:
        generator = torch.Generator().manual_seed(seed)
        loss_sum = correct = targets = samples = 0.0
        for chunk in _batches(pairs, batch_size):
            ids, valid = pad_batch([p for p, _ in chunk], config.pad_id)
            spans, canvas_valid = pad_batch([s for _, s in chunk], config.pad_id)
            loss_mask = torch.zeros_like(canvas_valid)
            for row in range(len(chunk)):
                length = int(canvas_valid[row].sum())
                count = max(1, round(ratio * length))
                loss_mask[row, torch.randperm(length, generator=generator)[:count]] = True
            canvas = spans.masked_fill(loss_mask, config.mask_id)
            canvas, canvas_valid = canvas.to(device), canvas_valid.to(device)
            x, valid = model.encode(ids.to(device), valid.to(device))
            if config.draft_feedback:
                r = model.reason(x, valid, steps, draft=(canvas, canvas_valid))
            else:
                r = model.reason(x, valid, steps)
            cache = model.generation_memory(x, valid, r)
            hidden = model.readout_hidden(canvas, canvas_valid, cache)
            mask = loss_mask.to(device)
            logits = model.vocab_logits(hidden[mask]).float()
            labels = spans.to(device)[mask]
            ce = F.cross_entropy(logits, labels, reduction="none")
            row_of = mask.nonzero(as_tuple=True)[0]
            per_sample = torch.zeros(len(chunk), device=device).index_add_(0, row_of, ce) / mask.sum(1)
            loss_sum += float(per_sample.sum())
            samples += len(chunk)
            correct += int((logits.argmax(-1) == labels).sum())
            targets += int(mask.sum())
        rows.append({"mask_ratio": ratio, "loss": loss_sum / max(samples, 1), "accuracy": correct / max(targets, 1),
                     "spans": int(samples), "targets": int(targets),
                     "steps": steps if steps is not None else config.default_depth})
    return rows


@torch.no_grad()
def retrieval_ranks(model: NuriModel, pairs: Sequence[Example], mode: str = "fast", steps: int | None = None,
                    dim: int | None = None, batch_size: int = 64, doc_mode: str | None = None) -> torch.Tensor:
    """Rank (1 = first) of each query's own document among all documents in `pairs`.

    Queries and documents are cut to max_input_len, as the model sees them.
    doc_mode (default: same as mode) embeds the documents with another mode,
    e.g. reasoning queries against fast documents. Read such cross-mode
    numbers only for models trained with cross-mode alignment (§6).
    """
    model.eval()
    device = next(model.parameters()).device
    doc_mode = doc_mode or mode

    def embed(side: int, side_mode: str) -> torch.Tensor:
        out = []
        for chunk in _batches([p[side][:model.config.max_input_len] for p in pairs], batch_size):
            ids, valid = pad_batch(chunk, model.config.pad_id)
            out.append(model.embed(ids.to(device), valid.to(device), mode=side_mode, steps=steps, dim=dim))
        return torch.cat(out)

    return ranks_from_scores(embed(0, mode) @ embed(1, doc_mode).T).cpu()


def retrieval_metrics(model: NuriModel, pairs: Sequence[Example], mode: str = "fast", steps: int | None = None,
                      dim: int | None = None, ks=(1, 5, 10), batch_size: int = 64,
                      doc_mode: str | None = None) -> dict:
    """Recall@k and MRR of retrieval_ranks; each query should find its own document."""
    ranks = retrieval_ranks(model, pairs, mode, steps, dim, batch_size, doc_mode)
    result = ranking_metrics_from_ranks(ranks, ks)
    result.update(pairs=len(pairs), mode=mode, doc_mode=doc_mode or mode, dim=dim, steps=steps)
    return result


def ranks_from_scores(scores: torch.Tensor) -> torch.Tensor:
    """Rank (1 = first) of document i for query i, from a (queries x documents) score matrix.

    Documents are ranked by score with ties broken by document order (stable
    sort), so tied scores never all get rank 1.
    """
    order = scores.float().sort(dim=1, descending=True, stable=True).indices
    gold = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1)
    return (order == gold).float().argmax(dim=1) + 1


def ranking_metrics_from_ranks(ranks: torch.Tensor, ks=(1, 5, 10)) -> dict:
    """Recall@k and MRR from per-query ranks (an empty set gives zeros)."""
    if ranks.numel() == 0:
        return {**{f"recall@{k}": 0.0 for k in ks}, "mrr": 0.0}
    ranks = ranks.float()
    result = {f"recall@{k}": float((ranks <= k).float().mean()) for k in ks}
    result["mrr"] = float((1.0 / ranks).mean())
    return result


def lexical_retrieval_ranks(pairs: Sequence[Example], vocab_size: int, max_len: int | None = None) -> torch.Tensor:
    """retrieval_ranks scored by shared tokens only (TF-IDF cosine), with no model.

    Token counts are kept sparse, so this stays cheap for thousands of pairs
    even with a 64k vocabulary. Pass the model's max_input_len as max_len so
    the baseline sees the same text the model sees; without it the ranks are
    a full-text diagnostic. Empty texts score 0 against everything.
    """
    queries = _token_counts([list(q)[:max_len] for q, _ in pairs], vocab_size)
    documents = _token_counts([list(d)[:max_len] for _, d in pairs], vocab_size)
    # Inverse document frequency over the document side.
    present = torch.zeros(vocab_size).index_add_(0, documents.indices()[1], torch.ones(documents.indices().shape[1]))
    idf = torch.log((1 + len(pairs)) / (1 + present)) + 1

    def weighted(m: torch.Tensor) -> torch.Tensor:
        """Rows scaled by IDF and normalized to length 1."""
        values = m.values() * idf[m.indices()[1]]
        norms = torch.zeros(m.shape[0]).index_add_(0, m.indices()[0], values ** 2).sqrt().clamp_min(1e-12)
        return torch.sparse_coo_tensor(m.indices(), values / norms[m.indices()[0]], m.shape, check_invariants=True)

    with warnings.catch_warnings():  # PyTorch notes that its internal sparse CSR path is "beta"
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
        scores = torch.sparse.mm(weighted(queries), weighted(documents).t()).to_dense()
    return ranks_from_scores(scores)


def lexical_retrieval_baseline(pairs: Sequence[Example], vocab_size: int, max_len: int | None = None,
                               ks=(1, 5, 10)) -> dict:
    """Recall@k and MRR of lexical_retrieval_ranks.

    If the embedding model barely beats this, the pairs mostly test word
    overlap, not meaning.
    """
    result = ranking_metrics_from_ranks(lexical_retrieval_ranks(pairs, vocab_size, max_len), ks)
    result.update(pairs=len(pairs), mode="lexical-tfidf", max_len=max_len)
    return result


def _token_counts(texts: list[list[int]], vocab_size: int) -> torch.Tensor:
    """Sparse (texts x vocabulary) matrix of token counts."""
    rows = [i for i, text in enumerate(texts) for _ in text]
    cols = [token for text in texts for token in text]
    indices = torch.tensor([rows, cols], dtype=torch.long).reshape(2, -1)
    return torch.sparse_coo_tensor(indices, torch.ones(len(cols)), (len(texts), vocab_size),
                                   check_invariants=True).coalesce()


@torch.no_grad()
def core_diagnostics(model: NuriModel, prompts: Sequence[list[int]], steps: int = 8) -> list[dict]:
    """Per-step statistics of the raw recurrent state, averaged over the batch.

    These are diagnostics only: a state that stops changing is not proof of
    collapse, and a changing state is not proof of reasoning (§12).
    """
    model.eval()
    device = next(model.parameters()).device
    ids, valid = pad_batch([p[:model.config.max_input_len] for p in prompts], model.config.pad_id)
    x, valid = model.encode(ids.to(device), valid.to(device))
    _, states = model.reason(x, valid, steps, return_states=True)
    rows = []
    for t, state in enumerate(states):
        state = state.float()
        slots = F.normalize(state, dim=-1)
        cosine = slots @ slots.transpose(1, 2)
        k = state.shape[1]
        off_diagonal = (cosine.sum(dim=(1, 2)) - k) / max(k * (k - 1), 1)
        singular = torch.linalg.svdvals(state - state.mean(dim=1, keepdim=True))
        p = singular / singular.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        effective_rank = torch.exp(-(p * p.clamp_min(1e-12).log()).sum(dim=-1))
        row = {"step": t, "state_norm": float(state.norm(dim=-1).mean()),
               "slot_cosine": float(off_diagonal.mean()), "effective_rank": float(effective_rank.mean())}
        if t > 0:
            update = (state - states[t - 1].float()).norm(dim=-1) / states[t - 1].float().norm(dim=-1).clamp_min(1e-12)
            row["relative_update"] = float(update.mean())
            if model.core.last_step_expert_counts:  # MoE core: which experts each step used
                row["expert_counts"] = model.core.last_step_expert_counts[t - 1].tolist()
        rows.append(row)
    return rows


def sample_examples(examples: Sequence[Example], count: int, seed: int = 0) -> list[Example]:
    """A fixed random subset, so repeated evaluations use the same examples."""
    examples = list(examples)
    return random.Random(seed).sample(examples, min(count, len(examples)))
