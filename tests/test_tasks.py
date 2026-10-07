import random

import pytest

from nuri.tasks import (
    NAMES,
    arithmetic_example,
    reachability_example,
    reasoning_examples,
    relation_example,
)


def parse_parents(text: str) -> dict[str, str]:
    """child -> parent, from the English facts."""
    parents = {}
    for line in text.splitlines():
        if " is the parent of " in line:
            parent, child = line[2:-1].split(" is the parent of ")
            assert child not in parents, "a child must never get two parents"
            parents[child] = parent
    return parents


def test_relation_answer_follows_the_chain():
    rng = random.Random(0)
    for _ in range(300):
        messages, answer = relation_example(rng, "en")
        text = messages[0]["content"]
        parents = parse_parents(text)
        hops = int(text.split("ancestor ")[1].split(" ")[0])
        person = text.split("above ")[1].split("?")[0]
        for _ in range(hops):
            person = parents[person]
        assert person == answer


def test_splits_use_disjoint_names():
    train = {
        n
        for m, a in (
            relation_example(random.Random(i), "en", "train") for i in range(200)
        )
        for n in NAMES
        if n in m[0]["content"]
    }
    evaluation = {
        n
        for m, a in (
            relation_example(random.Random(i), "en", "eval") for i in range(200)
        )
        for n in NAMES
        if n in m[0]["content"]
    }
    assert train and evaluation and not train & evaluation


def test_arithmetic_and_reachability_answers():
    rng = random.Random(1)
    messages, answer = arithmetic_example(rng, "en")
    expression = messages[0]["content"].splitlines()[-1]
    assert int(answer) == eval(expression)
    for _ in range(50):
        _, answer = reachability_example(rng, "ko")
        assert answer in ("예", "아니오")


def test_mix_has_both_languages():
    examples = reasoning_examples(100, seed=0)
    korean = sum(any("가" <= ch <= "힣" for ch in m[0]["content"]) for m, _ in examples)
    assert 20 < korean < 80


def test_shared_split_uses_training_pools_but_new_combinations():
    def names(split, seeds):
        return {
            n
            for i in seeds
            for n in NAMES
            if n in relation_example(random.Random(i), "en", split)[0][0]["content"]
        }

    assert names("shared", range(200)) <= names("train", range(1000))
    train_prompts = {
        relation_example(random.Random(i), "en", "train")[0][0]["content"]
        for i in range(500)
    }
    shared_prompts = {
        relation_example(random.Random(10_000 + i), "en", "shared")[0][0]["content"]
        for i in range(500)
    }
    assert not train_prompts & shared_prompts


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError):
        reasoning_examples(1, split="test")


@pytest.mark.parametrize(
    "task", [relation_example, arithmetic_example, reachability_example]
)
@pytest.mark.parametrize("language", ["ja", "fr", "", None])
def test_unsupported_language_never_becomes_english(task, language):
    rng = random.Random(11)
    before = rng.getstate()
    with pytest.raises(ValueError, match="language"):
        task(rng, language)
    assert rng.getstate() == before


@pytest.mark.parametrize("languages", [(), [], "en", ("ja",), ("en", "ja")])
def test_language_declarations_are_validated_even_with_zero_examples(languages):
    with pytest.raises(ValueError, match="language"):
        reasoning_examples(0, languages=languages)


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_invalid_reasoning_count_is_rejected(count):
    with pytest.raises(ValueError, match="count"):
        reasoning_examples(count)


def test_empty_request_still_validates_split():
    with pytest.raises(ValueError, match="split"):
        reasoning_examples(0, split="test")
    assert reasoning_examples(0) == []
