import torch

from nuri.config import tiny_config
from nuri.evaluation import (core_diagnostics, depth_curve, generation_accuracy, mlm_accuracy, r_interventions,
                             retrieval_metrics, rounds_curve, sample_examples, shuffle_r, zero_r)
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
