"""Overlap strata must not change retrieval candidates or the scored token views."""

import json

import pytest
import torch

from nuri import evaluation
from nuri.config import tiny_config
from nuri.model import NuriModel
from nuri.retrieval_report import overlap_stratified_retrieval


def constant_model(**overrides):
    model = NuriModel(tiny_config(**overrides))
    model.embed = lambda ids, valid, dim=None, **kwargs: torch.ones(
        ids.shape[0], dim or model.config.embed_dim)
    return model


def bucket(report, name):
    return next(row for row in report["buckets"] if row["name"] == name)


def test_strata_keep_the_full_candidate_pool_under_tied_scores():
    model = constant_model()
    pairs = [([3], [3]), ([4], [9]), ([5], [10])]
    report = overlap_stratified_retrieval(model, pairs, ks=(1, 3))
    overall = report["overall"]
    zero = bucket(report, "zero_overlap")
    assert overall["queries"] == overall["candidates"] == 3
    assert zero["queries"] == 2 and zero["candidates"] == 3
    assert sum(row["queries"] for row in report["buckets"]) == 3
    for row in zero["model"] + [zero["lexical_tfidf"]]:
        assert row["recall@1"] == 0.0
        assert row["recall@3"] == 1.0
        assert row["mrr"] == pytest.approx((1 / 2 + 1 / 3) / 2)
        assert row["queries"] == 2 and row["candidates"] == 3
    for row in overall["model"]:
        assert row["recall@1"] == pytest.approx(1 / 3)
    # Evaluating just the zero-overlap pairs would incorrectly improve Recall@1.
    reduced = evaluation.retrieval_metrics(model, pairs[1:], ks=(1, 3))
    assert reduced["recall@1"] == 0.5
    assert zero["summary"]["jaccard"]["mean"] == 0.0


def test_truncation_matches_model_lexical_and_overlap_views():
    model = constant_model(max_input_len=3)
    pairs = [([1, 2, 3], [1, 2, 3, 8]), ([1, 2, 3, 9], [1, 2, 3, 9])]
    seen = []

    def observe(ids, valid, **kwargs):
        seen.extend(ids[row][valid[row]].tolist() for row in range(len(ids)))
        return torch.ones(len(ids), model.config.embed_dim)

    model.embed = observe
    report = overlap_stratified_retrieval(model, pairs, dims=(None,), modes=("fast",),
                                         include_cross_mode=False)
    assert seen == [[1, 2, 3]] * 4
    assert report["overall"]["model"][0]["recall@1"] == 0.5
    assert report["overall"]["lexical_tfidf"]["recall@1"] == 0.5
    assert evaluation.lexical_retrieval_baseline(pairs, 128)["recall@1"] == 1.0
    assert report["overall"]["summary"]["query_tokens"] == {
        "count": 2, "mean": 3.0, "min": 3, "max": 3}
    assert bucket(report, "jaccard_[0.9,1]")["queries"] == 2
    assert report["overall"]["summary"]["jaccard"]["mean"] == 1.0


def test_special_exclusion_is_diagnostic_only_and_empty_is_not_zero_overlap():
    model = constant_model()
    pairs = [([4], [4]), ([4, 10], [4, 11]), ([4, 12], [4, 12])]
    report = overlap_stratified_retrieval(model, pairs, special_ids=(0, 1, 2, 3, 4))
    empty = bucket(report, "empty_lexical")
    zero = bucket(report, "zero_overlap")
    assert empty["queries"] == zero["queries"] == 1
    assert empty["summary"]["jaccard"] == {"count": 0, "mean": None, "min": None, "max": None}
    assert zero["summary"]["jaccard"] == {"count": 1, "mean": 0.0, "min": 0.0, "max": 0.0}
    assert empty["summary"]["query_tokens"]["mean"] == 1.0
    assert empty["summary"]["query_lexical_tokens"]["mean"] == 0.0
    # The scores retain token 4, even though the overlap diagnostic omits it.
    expected = evaluation.lexical_retrieval_baseline(pairs, model.config.vocab_size,
                                                    max_len=model.config.max_input_len)
    assert report["overall"]["lexical_tfidf"]["mrr"] == expected["mrr"]
    assert report["overall"]["summary"]["jaccard"]["count"] == 2
    json.dumps(report, allow_nan=False)


def test_exact_overlap_boundaries_are_assigned_once():
    model = constant_model()
    pairs = [([10, 11], [10]), (list(range(10, 15)), list(range(10, 14))),
             (list(range(10, 20)), list(range(10, 19))), ([20], [20])]
    report = overlap_stratified_retrieval(model, pairs)
    assert bucket(report, "jaccard_[0.5,0.75)")["queries"] == 1
    assert bucket(report, "jaccard_[0.8,0.9)")["queries"] == 1
    assert bucket(report, "jaccard_[0.9,1]")["queries"] == 2
    assert sum(row["queries"] for row in report["buckets"]) == 4
    assert all(row["candidates"] == 4 for row in report["buckets"])


def test_document_only_idf_is_shared_by_every_stratum():
    model = constant_model()
    # Token 10 occurs in one candidate but all queries. Token 20 occurs in two
    # candidates but only one query. Fitting IDF on queries would reverse the
    # first query's document order; document-only IDF correctly ranks it first.
    pairs = [([10, 20], [10]), ([10], [20]), ([10], [20])]
    report = overlap_stratified_retrieval(model, pairs)
    assert report["idf_fit"] == "documents_only"
    assert bucket(report, "jaccard_[0.5,0.75)")["lexical_tfidf"]["recall@1"] == 1.0
    assert bucket(report, "zero_overlap")["lexical_tfidf"]["mrr"] == pytest.approx((1 / 2 + 1 / 3) / 2)


def test_real_model_prefix_and_cross_mode_rows_match_existing_metrics():
    torch.manual_seed(7)
    model = NuriModel(tiny_config())
    pairs = [([4, 5, 6], [4, 5]), ([8, 9], [8, 9]), ([11, 12], [12, 11])]
    report = overlap_stratified_retrieval(model, pairs, dims=(8, 16, None), steps=1, batch_size=2)
    rows = report["overall"]["model"]
    assert len(rows) == 7
    for row in rows:
        expected = evaluation.retrieval_metrics(model, pairs, mode=row["mode"], doc_mode=row["doc_mode"],
                                               dim=row["dim"], steps=1, batch_size=2)
        for key, value in expected.items():
            assert row[key] == value
    assert (rows[-1]["mode"], rows[-1]["doc_mode"], rows[-1]["dim"]) == ("reasoning", "fast", None)
    expected_lexical = evaluation.lexical_retrieval_baseline(pairs, 128, max_len=32)
    for key, value in expected_lexical.items():
        assert report["overall"]["lexical_tfidf"][key] == value


def test_default_release_dimension_rows_preserve_the_seven_setting_contract():
    model = constant_model(embed_dim=256)
    report = overlap_stratified_retrieval(model, [([3], [3])])
    rows = report["overall"]["model"]
    assert [(row["mode"], row["doc_mode"], row["dim"]) for row in rows] == [
        (mode, mode, dim) for mode in ("fast", "reasoning") for dim in (64, 128, None)
    ] + [("reasoning", "fast", None)]
    assert all(row["recall@1"] == 1.0 for row in rows)


def test_empty_input_has_zero_counts_without_scoring_calls(monkeypatch):
    model = constant_model()

    def forbidden(*args, **kwargs):
        pytest.fail("No scoring helper should run without queries or candidates")

    monkeypatch.setattr(evaluation, "retrieval_ranks", forbidden)
    monkeypatch.setattr(evaluation, "lexical_retrieval_ranks", forbidden)
    report = overlap_stratified_retrieval(model, [])
    assert report["overall"]["queries"] == report["overall"]["candidates"] == 0
    for population in [report["overall"], *report["buckets"]]:
        assert population["summary"]["query_tokens"]["mean"] is None
        for row in population["model"] + [population["lexical_tfidf"]]:
            assert row["recall@1"] == row["mrr"] == 0.0
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("pairs", [[([], [3])], [([3], [])], [([3], [3]), ([], [4])]])
def test_empty_token_side_fails_without_dropping_a_candidate(pairs, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Reject empty token views before scoring any population")

    monkeypatch.setattr(evaluation, "retrieval_ranks", forbidden)
    with pytest.raises(ValueError, match="non-empty query and document"):
        overlap_stratified_retrieval(constant_model(), pairs)


@pytest.mark.parametrize("options, message", [
    ({"overlap_edges": (0.8, 0.5)}, "strictly increasing"),
    ({"overlap_edges": (float("nan"),)}, "strictly increasing"),
    ({"dims": (0,)}, "positive integers"),
    ({"dims": (64,)}, "no requested embedding dimension"),
    ({"modes": ("unknown",)}, "fast and/or reasoning"),
    ({"batch_size": 0}, "batch_size"),
    ({"ks": (0,)}, "positive integers"),
])
def test_invalid_report_settings_fail_clearly(options, message):
    with pytest.raises(ValueError, match=message):
        overlap_stratified_retrieval(constant_model(), [([3], [3])], **options)
