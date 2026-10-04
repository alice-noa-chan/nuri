"""Keep termination, chosen length, returned length and coverage separate."""

import pytest
import torch

import nuri.evaluation as evaluation
from nuri.config import tiny_config
from nuri.generation import GenerationResult
from nuri.model import NuriModel


@pytest.mark.parametrize("batch_size", [1, 3, 32])
@pytest.mark.parametrize("oracle_length", [False, True])
def test_generation_diagnostics_use_eligible_examples(monkeypatch, batch_size, oracle_length):
    model = NuriModel(tiny_config(max_response_len=8, length_head_trained=True))
    examples = [([5], [10, 11, 12]), ([6], [13]), ([7], [14, 15]), ([8], []), ([9], [16] * 8)]
    # First answer ends early. The second matches content but never writes EOS.
    outcomes = {5: ([10], True, 4), 6: ([13], False, 1),
                7: ([14, 15], True, 5), 8: ([], True, 1)}
    seen = []

    def controlled_generate(model, input_ids, attention_mask, **kwargs):
        rows = [outcomes[int(prompt[0])] for prompt in input_ids]
        seen.extend(int(prompt[0]) for prompt in input_ids)
        lengths = kwargs["canvas_len"]
        if lengths is None:
            lengths = torch.tensor([length for _, _, length in rows])
        tokens = [list(content) for content, _, _ in rows]
        canvas = torch.zeros(len(rows), 8, dtype=torch.long)
        for index, (content, ended, _) in enumerate(rows):
            if not ended:
                # Without EOS, generation returns every chosen canvas position.
                tokens[index].extend([16] * (int(lengths[index]) - len(content)))
            canvas[index, :len(tokens[index])] = torch.tensor(tokens[index])
            if ended:
                canvas[index, len(tokens[index])] = model.config.eos_id
        return GenerationResult(tokens, canvas, lengths,
                                [ended for _, ended, _ in rows],
                                reasoning_steps=torch.full((len(rows),), 2))

    monkeypatch.setattr(evaluation, "generate", controlled_generate)
    result = evaluation.generation_accuracy(model, examples, oracle_length=oracle_length,
                                            batch_size=batch_size, steps=2)
    assert seen == [5, 6, 7, 8]  # The over-cap answer is excluded, never truncated.
    assert result["examples"] == 4 and result["excluded"] == 1
    assert result["evaluation_coverage"] == pytest.approx(0.8)
    assert result["exact_match"] == pytest.approx(0.5 if oracle_length else 0.75)
    assert result["missing_eos_rate"] == pytest.approx(0.25)
    assert result["early_eos_rate"] == pytest.approx(0.25)
    assert result["answer_length_mae"] == pytest.approx(0.75 if oracle_length else 0.5)
    assert result["canvas_length_mae"] == pytest.approx(0.0 if oracle_length else 0.75)
    assert result["length_error_rate"] == pytest.approx(0.0 if oracle_length else 0.5)
    assert result["mean_core_steps"] == 2.0


def test_short_answer_without_eos_is_not_early_eos(monkeypatch):
    model = NuriModel(tiny_config(length_head_trained=True))

    def missing_eos(*args, **kwargs):
        return GenerationResult([[10]], torch.tensor([[10]]), torch.tensor([1]), [False])

    monkeypatch.setattr(evaluation, "generate", missing_eos)
    result = evaluation.generation_accuracy(model, [([5], [10, 11, 12])])
    assert result["early_eos_rate"] == 0.0 and result["missing_eos_rate"] == 1.0
    assert result["answer_length_mae"] == 2.0 and result["canvas_length_mae"] == 3.0


@pytest.mark.parametrize("over_cap", [False, True])
def test_no_eligible_answers_report_zero_coverage_without_generation(monkeypatch, over_cap):
    model = NuriModel(tiny_config(max_response_len=8))
    examples = [([5], [10] * 8)] if over_cap else []

    def unexpected_generation(*args, **kwargs):
        pytest.fail("No generation should run when there are no eligible answers")

    monkeypatch.setattr(evaluation, "generate", unexpected_generation)
    result = evaluation.generation_accuracy(model, examples, oracle_length=True)
    assert result["examples"] == 0 and result["excluded"] == len(examples)
    for metric in ("evaluation_coverage", "early_eos_rate", "missing_eos_rate",
                   "canvas_length_mae", "answer_length_mae", "length_error_rate", "exact_match"):
        assert result[metric] == 0.0
