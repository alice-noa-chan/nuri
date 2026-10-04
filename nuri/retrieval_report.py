"""Retrieval scores by lexical overlap, with one fixed document candidate pool.

Strata describe the query and its row-aligned positive; they never restrict
which documents that query competes against. These diagnostics alone do not
establish semantic retrieval quality.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from . import evaluation
from .model import NuriModel


DEFAULT_OVERLAP_EDGES = (0.25, 0.5, 0.75, 0.8, 0.9)


def _statistics(values: list[float | int]) -> dict:
    """An empty population has no observed mean or extrema."""
    return {"count": len(values), "mean": sum(values) / len(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def _overlap_profiles(pairs, max_len: int, special_ids: set[int]) -> list[dict]:
    rows = []
    for query, document in pairs:
        # Match the prefix truncation in both shared scoring helpers.
        query, document = query[:max_len], document[:max_len]
        if not query or not document:
            raise ValueError("retrieval rows need non-empty query and document token views")
        query_lexical = [token for token in query if token not in special_ids]
        document_lexical = [token for token in document if token not in special_ids]
        query_set, document_set = set(query_lexical), set(document_lexical)
        overlap = (len(query_set & document_set) / len(query_set | document_set)
                   if query_set and document_set else None)
        rows.append({"query_tokens": len(query), "positive_tokens": len(document),
                     "query_lexical_tokens": len(query_lexical),
                     "positive_lexical_tokens": len(document_lexical), "jaccard": overlap})
    return rows


def _summary(rows: list[dict]) -> dict:
    return {key: _statistics([row[key] for row in rows if row[key] is not None])
            for key in ("query_tokens", "positive_tokens", "query_lexical_tokens",
                        "positive_lexical_tokens", "jaccard")}


def _strata(rows: list[dict], edges: tuple[float, ...]) -> list[tuple[dict, list[int]]]:
    specs = [{"name": "empty_lexical", "lower": None, "upper": None},
             {"name": "zero_overlap", "lower": 0.0, "upper": 0.0}]
    limits = (0.0, *edges, 1.0)
    for index, (lower, upper) in enumerate(zip(limits, limits[1:])):
        left = "(" if index == 0 else "["
        right = "]" if upper == 1.0 else ")"
        specs.append({"name": f"jaccard_{left}{lower:g},{upper:g}{right}",
                      "lower": lower, "upper": upper,
                      "lower_inclusive": index > 0, "upper_inclusive": upper == 1.0})
    indices = [[] for _ in specs]
    for index, row in enumerate(rows):
        overlap = row["jaccard"]
        if overlap is None:
            bucket = 0
        elif overlap == 0.0:
            bucket = 1
        else:
            bucket = 2 + sum(overlap >= edge for edge in edges)
        indices[bucket].append(index)
    return list(zip(specs, indices))


def _checked_ranks(ranks: torch.Tensor, candidates: int) -> torch.Tensor:
    ranks = ranks.detach().cpu()
    if (ranks.shape != (candidates,) or not torch.isfinite(ranks).all()
            or (ranks < 1).any() or (ranks > candidates).any() or (ranks != ranks.round()).any()):
        raise ValueError("retrieval helpers must return one finite integer rank per query")
    return ranks


@torch.no_grad()
def overlap_stratified_retrieval(model: NuriModel, pairs: Sequence[evaluation.Example], *,
                                dims=(64, 128, None), modes=("fast", "reasoning"),
                                ks=(1, 5, 10), special_ids: Sequence[int] | None = None,
                                overlap_edges=DEFAULT_OVERLAP_EDGES, steps: int | None = None,
                                batch_size: int = 64, include_cross_mode: bool = True) -> dict:
    """Absolute Recall@k/MRR for overlap strata against all supplied documents.

    Each row has one gold document at the same index. Token views are the first
    max_input_len tokens, with stable document-order tie breaking. Special IDs
    are removed only from overlap/lexical-length diagnostics; they remain in
    model and TF-IDF inputs. TF-IDF fits IDF on the document side only.

    None selects the full embedding dimension. Unsupported larger prefixes
    are skipped, and duplicate effective dimensions are reported once. Empty
    input returns explicit zero counts/metrics; an actually empty token side
    in a supplied row fails rather than changing the candidate population.
    Empty lexical sets have undefined overlap, separate from nonempty sets
    whose overlap is zero. Reports contain aggregates, not token content.
    include_cross_mode adds the existing full-dimension reasoning-query/fast-
    document diagnostic. It is meaningful only after alignment training.
    """
    config = model.config
    pairs = [(list(query), list(document)) for query, document in pairs]
    modes = tuple(dict.fromkeys(modes))
    if not modes or any(mode not in ("fast", "reasoning") for mode in modes):
        raise ValueError("modes must contain fast and/or reasoning")
    ks = tuple(dict.fromkeys(ks))
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(include_cross_mode, bool):
        raise ValueError("include_cross_mode must be a boolean")
    edges = tuple(float(edge) for edge in overlap_edges)
    if any(not math.isfinite(edge) or not 0 < edge < 1 for edge in edges) or any(
            left >= right for left, right in zip(edges, edges[1:])):
        raise ValueError("overlap_edges must be strictly increasing inside (0, 1)")
    dims = tuple(dict.fromkeys(dims))
    if not dims or any(dim is not None and (isinstance(dim, bool) or not isinstance(dim, int) or dim < 1)
                       for dim in dims):
        raise ValueError("dims must contain positive integers or None for the full embedding")
    selected_dims = [dim for dim in dims if dim is None or dim <= config.embed_dim]
    if None in selected_dims:
        selected_dims = [dim for dim in selected_dims if dim != config.embed_dim]
    if not selected_dims:
        raise ValueError("no requested embedding dimension fits this model")
    special = set(special_ids if special_ids is not None else (config.pad_id, config.mask_id, config.eos_id))
    profiles = _overlap_profiles(pairs, config.max_input_len, special)
    count = len(pairs)

    # Rank every query once per setting, before selecting any overlap stratum.
    model_ranks = []
    settings = [(mode, mode, dim) for mode in modes for dim in selected_dims]
    if include_cross_mode:
        settings.append(("reasoning", "fast", None))
    for mode, doc_mode, dim in settings:
        ranks = (evaluation.retrieval_ranks(model, pairs, mode=mode, doc_mode=doc_mode,
                                          steps=steps, dim=dim, batch_size=batch_size)
                 if count else torch.empty(0, dtype=torch.long))
        metadata = {"mode": mode, "doc_mode": doc_mode, "dim": dim,
                    "embedding_dimension": config.embed_dim if dim is None else dim, "steps": steps}
        model_ranks.append((metadata, _checked_ranks(ranks, count)))
    lexical_ranks = (evaluation.lexical_retrieval_ranks(pairs, config.vocab_size, max_len=config.max_input_len)
                     if count else torch.empty(0, dtype=torch.long))
    lexical_ranks = _checked_ranks(lexical_ranks, count)

    def population(indices: list[int]) -> dict:
        return {"queries": len(indices), "candidates": count,
                "summary": _summary([profiles[index] for index in indices]),
                "model": [{**metadata, "pairs": len(indices), "queries": len(indices), "candidates": count,
                           **evaluation.ranking_metrics_from_ranks(ranks[indices], ks)}
                          for metadata, ranks in model_ranks],
                "lexical_tfidf": {"mode": "lexical-tfidf", "max_len": config.max_input_len,
                                  "pairs": len(indices), "queries": len(indices), "candidates": count,
                                  **evaluation.ranking_metrics_from_ranks(lexical_ranks[indices], ks)}}

    return {"max_input_len": config.max_input_len, "overlap_metric": "token_set_jaccard",
            "overlap_special_ids_excluded": sorted(special), "overlap_edges": list(edges),
            "candidate_pool": "all_supplied_documents", "gold_policy": "one_row_aligned_document",
            "tie_breaking": "stable_document_order", "idf_fit": "documents_only",
            "empty_lexical_overlap": "undefined_if_either_side_has_no_nonspecial_tokens",
            "overall": population(list(range(count))),
            "buckets": [{**spec, **population(indices)} for spec, indices in _strata(profiles, edges)],
            "notes": ["All query strata compete against the same full candidate document pool.",
                      "Cross-mode scores require separate alignment training before quality interpretation.",
                      "Empty populations have zero metrics and provide no quality evidence.",
                      "Overlap diagnostics alone do not establish semantic retrieval quality."]}
