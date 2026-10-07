import pytest
import torch

from nuri.config import tiny_config
from nuri.evaluation import (core_diagnostics, depth_curve, generation_accuracy, halting_curve, mlm_accuracy,
                             r_interventions, retrieval_metrics, rounds_curve, sample_examples,
                             shuffle_r, span_denoising, zero_r, token_f1, adjacent_repeat_rate)
from nuri.model import NuriModel

EXAMPLES = [([5, 6, 7], [8, 9]), ([10, 11], [12]), ([13, 14, 15, 16], [17, 18, 19])]


def make_model(**overrides):
    torch.manual_seed(0)
    return NuriModel(tiny_config(length_head_trained=True, **overrides)).eval()


def test_generation_accuracy_reports_counts_and_settings():
    model = make_model()
    result = generation_accuracy(model, EXAMPLES, steps=2, rounds=3, oracle_length=True)
    assert result["examples"] == 3 and (result["steps"], result["rounds"]) == (2, 3)
    assert result["length_error_rate"] == 0.0  # oracle length is always right
    assert 0.0 <= result["exact_match"] <= 1.0 and 0.0 <= result["missing_eos_rate"] <= 1.0


def test_exact_match_counts_matching_answers():
    model = make_model()
    eos = model.config.eos_id

    def eos_only_logits(hidden):
        # Every position predicts EOS: answers are empty, so only an empty gold answer would match.
        logits = torch.full((hidden.shape[0], model.config.vocab_size), -1e4)
        logits[:, eos] = 0.0
        return logits

    model.vocab_logits = eos_only_logits
    examples = [([5, 6], []), ([7], [8])]
    result = generation_accuracy(model, examples, oracle_length=True)
    assert result["exact_match"] == 0.5 and result["missing_eos_rate"] == 0.0


def test_curves_and_interventions():
    model = make_model()
    assert [r["steps"] for r in depth_curve(model, EXAMPLES, depths=(1, 2), oracle_length=True)] == [1, 2]
    assert [r["rounds"] for r in rounds_curve(model, EXAMPLES, rounds=(1, 2), oracle_length=True)] == [1, 2]
    result = r_interventions(model, EXAMPLES, oracle_length=True)
    assert set(result) == {"normal", "zero_r", "shuffled_r"}
    r = torch.arange(6.0).view(3, 2, 1)
    assert torch.equal(zero_r(r), torch.zeros_like(r))
    assert torch.equal(shuffle_r(r)[1], r[0])


def test_mlm_and_retrieval_metrics():
    model = make_model()
    docs = [list(range(3 + i, 30 + i)) for i in range(5)]
    mlm = mlm_accuracy(model, docs)
    assert mlm["targets"] > 0 and 0.0 <= mlm["accuracy"] <= 1.0
    pairs = [(list(range(3 + i, 10 + i)), list(range(3 + i, 10 + i))) for i in range(6)]
    retrieval = retrieval_metrics(model, pairs, mode="fast", ks=(1, 3))
    assert retrieval["recall@1"] == 1.0 and retrieval["mrr"] == 1.0  # identical texts retrieve themselves
    assert retrieval["recall@3"] >= retrieval["recall@1"]


def test_core_diagnostics_has_one_row_per_state():
    model = make_model()
    rows = core_diagnostics(model, [[5, 6, 7], [8, 9]], steps=3)
    assert [r["step"] for r in rows] == [0, 1, 2, 3]
    assert "relative_update" not in rows[0] and rows[1]["relative_update"] > 0
    assert all(1.0 <= r["effective_rank"] <= model.config.latent_slots for r in rows)


def test_sample_examples_is_stable():
    assert sample_examples(EXAMPLES, 2, seed=1) == sample_examples(EXAMPLES, 2, seed=1)


def test_retrieval_ties_are_not_all_rank_one():
    """Constant embeddings cannot retrieve anything; ties are broken by document order."""
    model = make_model()
    model.embed = lambda ids, valid, **kwargs: torch.ones(ids.shape[0], 4)
    pairs = [([5 + i], [9 + i]) for i in range(4)]
    result = retrieval_metrics(model, pairs, ks=(1, 4))
    assert result["recall@1"] == 0.25 and result["recall@4"] == 1.0
    assert abs(result["mrr"] - (1 + 1 / 2 + 1 / 3 + 1 / 4) / 4) < 1e-6


def test_over_limit_answers_are_excluded_not_cut():
    model = make_model()
    too_long = list(range(3, 3 + model.config.max_response_len))  # + EOS does not fit
    result = generation_accuracy(model, EXAMPLES + [([5], too_long)], oracle_length=True)
    assert result["examples"] == 3 and result["excluded"] == 1


def test_shuffle_never_leaves_a_singleton_batch():
    model = make_model()
    seen = []
    original = model.generation_memory

    def capture(x, valid, r):
        seen.append(r.shape[0])
        return original(x, valid, r)

    model.generation_memory = capture
    generation_accuracy(model, EXAMPLES, oracle_length=True, batch_size=2, transform_r=shuffle_r)
    assert seen == [3]  # the singleton tail joined the first batch
    with pytest.raises(ValueError):
        generation_accuracy(model, EXAMPLES[:1], oracle_length=True, transform_r=shuffle_r)


def test_length_only_accuracy_exposes_length_leaks():
    model = make_model()
    yes_no = [([5], [20]), ([6], [20]), ([7], [21, 22])]  # "yes" is 1 token, "no" is 2 tokens
    result = generation_accuracy(model, yes_no, oracle_length=True)
    assert result["length_only_accuracy"] == 1.0  # the canvas length alone gives every answer away
    assert "length_only_accuracy" not in generation_accuracy(model, yes_no)


def test_span_denoising_reports_each_ratio_and_is_deterministic():
    model = make_model()
    docs = [list(range(3 + i, 60 + i)) for i in range(6)]
    rows = span_denoising(model, docs, ratios=(0.5, 1.0), steps=2)
    assert [r["mask_ratio"] for r in rows] == [0.5, 1.0]
    assert all(r["spans"] == 6 and r["targets"] > 0 and r["loss"] > 0 for r in rows)
    assert rows[1]["targets"] > rows[0]["targets"]  # full masking scores every position
    assert span_denoising(model, docs, ratios=(0.5,), steps=2) == rows[:1]


def test_halting_curve_reports_mean_core_steps():
    model = make_model()
    rows = halting_curve(model, EXAMPLES, tolerances=(None, float("inf")), steps=5, oracle_length=True)
    assert [r["halt_tol"] for r in rows] == [None, float("inf")]
    assert rows[0]["mean_core_steps"] == 5.0 and rows[1]["mean_core_steps"] == 1.0


def test_cross_mode_retrieval_uses_both_modes():
    model = make_model()
    pairs = [(list(range(3 + i, 10 + i)), list(range(3 + i, 10 + i))) for i in range(5)]
    same = retrieval_metrics(model, pairs, mode="reasoning")
    cross = retrieval_metrics(model, pairs, mode="reasoning", doc_mode="fast")
    assert same["doc_mode"] == "reasoning" and cross["doc_mode"] == "fast"
    assert 0.0 <= cross["mrr"] <= 1.0


def test_token_f1_gives_partial_credit():
    assert token_f1([5, 6, 7], [5, 6, 7]) == 1.0
    assert token_f1([7, 6, 5], [5, 6, 7]) == 1.0  # bag of tokens ignores order
    assert token_f1([5, 9], [5, 6]) == pytest.approx(0.5)
    assert token_f1([5, 5, 5], [5, 6]) == pytest.approx(0.4)  # repeats count once each
    assert token_f1([9], [5]) == 0.0
    assert token_f1([], []) == 1.0 and token_f1([], [5]) == 0.0


def test_adjacent_repeat_rate():
    assert adjacent_repeat_rate([4, 4, 4]) == 1.0
    assert adjacent_repeat_rate([4, 5, 4, 4]) == pytest.approx(1 / 3)
    assert adjacent_repeat_rate([4]) == 0.0


def test_generation_accuracy_reports_partial_credit_and_repeats():
    model = make_model()
    result = generation_accuracy(model, EXAMPLES, oracle_length=True)
    for name in ("token_f1", "repeat_rate", "reference_repeat_rate"):
        assert 0.0 <= result[name] <= 1.0
    assert result["token_f1"] >= result["exact_match"]
