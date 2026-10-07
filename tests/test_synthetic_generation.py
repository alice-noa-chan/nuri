import random

import pytest

from nuri.synthetic_generation import (
    EOS_ID,
    FIRST_CONTENT_ID,
    TASK_NAMES,
    Example,
    TaskSpec,
    branch_successors,
    canvas_target,
    make_examples,
    reference_canvas,
    repeat_rate,
    score_predictions,
    transition_validity,
)

SPEC = TaskSpec(content_size=16, input_len=(4, 8), output_len=(3, 6))


@pytest.mark.parametrize("task", TASK_NAMES)
def test_examples_stay_inside_the_content_range_and_lengths(task):
    for example in make_examples(task, 50, seed=0, spec=SPEC):
        tokens = example.input_ids + example.target
        assert all(FIRST_CONTENT_ID <= t < SPEC.vocab_size for t in tokens)
        assert SPEC.input_len[0] <= len(example.input_ids) <= SPEC.input_len[1]
        assert 1 <= len(example.target) <= SPEC.max_target_len


def test_examples_are_reproducible_and_seed_dependent():
    first = make_examples("copy", 20, seed=3, spec=SPEC)
    assert first == make_examples("copy", 20, seed=3, spec=SPEC)
    assert first != make_examples("copy", 20, seed=4, spec=SPEC)


def test_deterministic_task_targets():
    for example in make_examples("copy", 20, seed=0, spec=SPEC):
        assert example.target == example.input_ids
    for example in make_examples("reverse", 20, seed=0, spec=SPEC):
        assert example.target == example.input_ids[::-1]
    for example in make_examples("count", 20, seed=0, spec=SPEC):
        sequence = [t - FIRST_CONTENT_ID for t in example.input_ids + example.target]
        steps = {(b - a) % SPEC.content_size for a, b in zip(sequence, sequence[1:])}
        assert len(steps) == 1 and steps <= {1, 2, 3}


def test_branch_targets_follow_the_graph_and_single_takes_the_first_successor():
    successors = branch_successors(SPEC)
    assert all(len(set(pair)) == 2 and t not in pair for t, pair in successors.items())
    for task in ("branch", "branch-single"):
        for example in make_examples(task, 30, seed=1, spec=SPEC):
            walk = example.input_ids + example.target
            assert transition_validity(walk[0], walk[1:], successors) == 1.0
    for example in make_examples("branch-single", 30, seed=1, spec=SPEC):
        current = example.input_ids[-1]
        for token in example.target:
            assert token == successors[current][0]
            current = token


def test_branch_uses_both_successors_but_branch_single_uses_one():
    def first_steps(task):
        seen = {}
        for example in make_examples(task, 400, seed=2, spec=SPEC):
            seen.setdefault(example.input_ids[-1], set()).add(example.target[0])
        return seen

    assert any(len(steps) == 2 for steps in first_steps("branch").values())
    assert all(len(steps) == 1 for steps in first_steps("branch-single").values())


def test_canvas_target_eos_policy():
    example = Example([3, 4, 5], [6, 7, 8, 9])
    rng = random.Random(0)
    assert canvas_target(example, rng, eos_prob=1.0) == [6, 7, 8, 9, EOS_ID]
    for _ in range(50):
        prefix = canvas_target(example, rng, eos_prob=0.0)
        assert EOS_ID not in prefix and 1 <= len(prefix) < 4
        assert prefix == example.target[: len(prefix)]
    one_token = Example([3], [4])
    assert canvas_target(one_token, rng, eos_prob=0.0) == [4, EOS_ID]
    with pytest.raises(ValueError):
        canvas_target(example, rng, eos_prob=1.5)


def test_repeat_rate_and_validity_helpers():
    assert repeat_rate([5, 5, 5]) == 1.0
    assert repeat_rate([5, 6, 5]) == 0.0
    assert repeat_rate([5]) == 0.0
    graph = {3: (4, 5), 4: (3, 5), 5: (3, 4)}
    assert transition_validity(3, [4, 5], graph) == 1.0
    assert transition_validity(3, [3, 4], graph) == 0.5
    assert transition_validity(3, [], graph) == 1.0


def test_score_predictions_perfect_and_wrong():
    examples = make_examples("reverse", 10, seed=0, spec=SPEC)
    perfect = [reference_canvas(e) for e in examples]
    scores = score_predictions("reverse", examples, perfect, SPEC)
    assert scores["token_accuracy"] == scores["exact_match"] == 1.0
    assert scores["eos_correct"] == 1.0
    assert "transition_validity" not in scores

    repeated = [
        [FIRST_CONTENT_ID] * len(e.target) + [FIRST_CONTENT_ID] for e in examples
    ]
    scores = score_predictions("reverse", examples, repeated, SPEC)
    assert scores["eos_correct"] == 0.0 and scores["exact_match"] == 0.0
    assert scores["repeat_rate"] > 0.9


def test_branch_score_counts_legal_alternatives_as_valid():
    examples = make_examples("branch", 20, seed=5, spec=SPEC)
    successors = branch_successors(SPEC)
    predictions = []
    for example in examples:
        current, walk = example.input_ids[-1], []
        for _ in example.target:
            current = successors[current][1]
            walk.append(current)
        predictions.append(walk + [EOS_ID])
    scores = score_predictions("branch", examples, predictions, SPEC)
    assert scores["transition_validity"] == 1.0
    assert scores["eos_correct"] == 1.0


def test_score_predictions_rejects_bad_shapes():
    examples = make_examples("copy", 2, seed=0, spec=SPEC)
    with pytest.raises(ValueError):
        score_predictions("copy", examples, [[EOS_ID]], SPEC)
    with pytest.raises(ValueError):
        score_predictions("copy", examples, [[EOS_ID], [EOS_ID]], SPEC)


def test_spec_validation():
    with pytest.raises(ValueError):
        TaskSpec(content_size=2)
    with pytest.raises(ValueError):
        TaskSpec(input_len=(5, 4))
    with pytest.raises(ValueError):
        make_examples("unknown", 1, seed=0, spec=SPEC)
