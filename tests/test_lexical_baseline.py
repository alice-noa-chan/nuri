import pytest
import torch

from nuri.evaluation import (lexical_retrieval_baseline, lexical_retrieval_ranks, ranking_metrics_from_ranks,
                             ranks_from_scores)


def test_identical_pairs_are_trivial_for_the_lexical_baseline():
    pairs = [([10 + i, 50 + i, 7], [10 + i, 50 + i, 7]) for i in range(20)]
    result = lexical_retrieval_baseline(pairs, vocab_size=128)
    assert result["recall@1"] == 1.0 and result["mrr"] == 1.0 and result["pairs"] == 20


def test_pairs_without_shared_tokens_are_not_solved():
    # Every query shares only the common token 7 with every document: all scores tie.
    pairs = [([10 + i, 7], [60 + i, 7]) for i in range(20)]
    result = lexical_retrieval_baseline(pairs, vocab_size=128)
    assert result["recall@1"] == pytest.approx(1 / 20)  # stable tie-breaking gives rank 1 only to the first query


def test_common_tokens_count_less_than_rare_ones():
    # The last query shares the common token 5 with every document and the rare
    # token 99 only with its own document, which comes last. Without IDF all four
    # scores tie and the stable sort would pick document 0.
    pairs = [([2], [5, 2]), ([3], [5, 3]), ([4], [5, 4]), ([5, 99], [99, 1])]
    result = lexical_retrieval_baseline(pairs, vocab_size=128, ks=(1,))
    assert result["recall@1"] == 1.0


def test_ranking_metrics_basic():
    scores = torch.tensor([[1.0, 0.0], [1.0, 0.5]])  # query 1 ranks its document second
    ranks = ranks_from_scores(scores)
    assert ranks.tolist() == [1, 2]
    result = ranking_metrics_from_ranks(ranks, ks=(1, 2))
    assert result["recall@1"] == 0.5 and result["recall@2"] == 1.0 and result["mrr"] == 0.75


def test_max_len_matches_what_the_model_sees():
    # Identical visible prefixes, different tails: only the full text tells them apart.
    pairs = [([1, 2, 3], [1, 2, 3, 8]), ([1, 2, 3, 9], [1, 2, 3, 9])]
    assert lexical_retrieval_baseline(pairs, 16)["recall@1"] == 1.0
    truncated = lexical_retrieval_baseline(pairs, 16, max_len=3)
    assert truncated["recall@1"] == 0.5 and truncated["max_len"] == 3


def test_ranks_keep_the_full_candidate_pool_and_handle_empty_text():
    pairs = [([1, 2], [1, 2]), ([], [3]), ([4], [5])]
    ranks = lexical_retrieval_ranks(pairs, 16)
    assert ranks.tolist()[0] == 1 and len(ranks) == 3
    assert ranks.tolist()[1] == 2  # an empty query ties everywhere; stable order puts document 1 second
    assert ranking_metrics_from_ranks(ranks[:0]) == {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}
