"""Synthetic generation tasks with exact references, for diagnosing the readout (Q-17).

Real text continuation has many valid answers, so a failed generation does not
say whether the model, the decoder or the data is to blame. These tasks remove
that ambiguity. Each example is an input token list and a target token list
(content only; EOS is added by `canvas_target`). Token IDs follow the model's
special layout: PAD = 0, MASK = 1, EOS = 2 and content tokens start at 3.

Tasks
- copy: the target repeats the input. One correct answer, aligned positions.
- reverse: the target is the reversed input. One correct answer; the readout
  must know where the input ends.
- count: the input is an arithmetic progression (mod the content size) and
  the target continues it. One correct answer, like prefix continuation.
- branch: the input is a walk on a fixed graph where every token has two
  allowed successors, and the target continues the walk with random choices.
  Many correct answers, like real text continuation.
- branch-single: the same inputs, but the target always takes the first
  successor. One correct answer per input; this imitates training on the
  output of a teacher model (sequence-level distillation).

Metrics compare a predicted canvas with the reference canvas of the same
length. Branch tasks are also scored by transition validity, because a
different but legal continuation is not an error there.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

PAD_ID = 0
MASK_ID = 1
EOS_ID = 2
FIRST_CONTENT_ID = 3

TASK_NAMES = ("copy", "reverse", "count", "branch", "branch-single")
BRANCH_TASKS = ("branch", "branch-single")


@dataclass(frozen=True)
class TaskSpec:
    """Size limits shared by every task.

    content_size: number of ordinary tokens (IDs 3 .. 3 + content_size - 1).
    input_len: inclusive (min, max) input length.
    output_len: inclusive (min, max) target length without EOS. Copy and
        reverse ignore it, because their target length equals the input length.
    graph_seed: seed of the branch successor graph. Training and evaluation
        must use the same graph, so it is fixed separately from data seeds.
    """

    content_size: int = 32
    input_len: tuple[int, int] = (6, 12)
    output_len: tuple[int, int] = (6, 12)
    graph_seed: int = 1234

    def __post_init__(self):
        if self.content_size < 4:
            raise ValueError("content_size must be at least 4")
        for name in ("input_len", "output_len"):
            low, high = getattr(self, name)
            if not 1 <= low <= high:
                raise ValueError(f"{name} must be an ordered pair of positive lengths")

    @property
    def vocab_size(self) -> int:
        return FIRST_CONTENT_ID + self.content_size

    @property
    def max_target_len(self) -> int:
        """Longest target without EOS over all tasks."""
        return max(self.input_len[1], self.output_len[1])


@dataclass(frozen=True)
class Example:
    input_ids: list[int]
    target: list[int]  # content tokens only, without EOS


def branch_successors(spec: TaskSpec) -> dict[int, tuple[int, int]]:
    """Two distinct allowed successors for every content token, fixed by graph_seed."""
    rng = random.Random(spec.graph_seed)
    tokens = list(range(FIRST_CONTENT_ID, spec.vocab_size))
    successors = {}
    for token in tokens:
        others = [t for t in tokens if t != token]
        first, second = rng.sample(others, 2)
        successors[token] = (first, second)
    return successors


def _random_tokens(rng: random.Random, spec: TaskSpec, length: int) -> list[int]:
    return [rng.randrange(FIRST_CONTENT_ID, spec.vocab_size) for _ in range(length)]


def _progression(start: int, step: int, length: int, spec: TaskSpec) -> list[int]:
    """Content tokens start, start + step, ... wrapped inside the content range."""
    offset = start - FIRST_CONTENT_ID
    return [
        FIRST_CONTENT_ID + (offset + i * step) % spec.content_size
        for i in range(length)
    ]


def _walk(
    rng: random.Random,
    successors: dict[int, tuple[int, int]],
    start: int,
    length: int,
    always_first: bool,
) -> list[int]:
    """A walk of `length` tokens that follows `start` on the successor graph."""
    walk, current = [], start
    for _ in range(length):
        choices = successors[current]
        current = choices[0] if always_first else rng.choice(choices)
        walk.append(current)
    return walk


def make_example(
    task: str,
    rng: random.Random,
    spec: TaskSpec,
    successors: dict[int, tuple[int, int]] | None = None,
) -> Example:
    """Draw one example of `task`. Branch tasks need the successor graph."""
    input_len = rng.randint(*spec.input_len)
    output_len = rng.randint(*spec.output_len)
    if task == "copy":
        tokens = _random_tokens(rng, spec, input_len)
        return Example(tokens, list(tokens))
    if task == "reverse":
        tokens = _random_tokens(rng, spec, input_len)
        return Example(tokens, tokens[::-1])
    if task == "count":
        start = rng.randrange(FIRST_CONTENT_ID, spec.vocab_size)
        step = rng.randint(1, 3)
        sequence = _progression(start, step, input_len + output_len, spec)
        return Example(sequence[:input_len], sequence[input_len:])
    if task in BRANCH_TASKS:
        if successors is None:
            raise ValueError("branch tasks need the successor graph")
        start = rng.randrange(FIRST_CONTENT_ID, spec.vocab_size)
        prefix = [start] + _walk(rng, successors, start, input_len - 1, False)
        target = _walk(rng, successors, prefix[-1], output_len, task == "branch-single")
        return Example(prefix, target)
    raise ValueError(f"unknown task {task!r}; choose from {TASK_NAMES}")


def make_examples(task: str, count: int, seed: int, spec: TaskSpec) -> list[Example]:
    """`count` examples drawn with their own seed. Use different seeds for train and eval."""
    rng = random.Random(seed)
    successors = branch_successors(spec) if task in BRANCH_TASKS else None
    return [make_example(task, rng, spec, successors) for _ in range(count)]


def canvas_target(
    example: Example, rng: random.Random, eos_prob: float = 1.0
) -> list[int]:
    """The training canvas for one example.

    With probability eos_prob the whole target is used and EOS is appended.
    Otherwise a strict prefix (at least one token) is used without EOS. This
    imitates base span denoising, where EOS appears only when a sampled span
    happens to reach the document end.
    """
    if not 0.0 <= eos_prob <= 1.0:
        raise ValueError("eos_prob must be in [0, 1]")
    target = example.target
    if len(target) < 2 or rng.random() < eos_prob:
        return target + [EOS_ID]
    return target[: rng.randint(1, len(target) - 1)]


def reference_canvas(example: Example) -> list[int]:
    """The canvas an exact answer fills: the whole target followed by EOS."""
    return example.target + [EOS_ID]


def _content(canvas: list[int]) -> list[int]:
    """Canvas tokens before the final slot, which is reserved for EOS."""
    return canvas[:-1]


def repeat_rate(tokens: list[int]) -> float:
    """Share of adjacent token pairs that repeat the same token."""
    pairs = list(zip(tokens, tokens[1:]))
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else 0.0


def transition_validity(
    last_input: int, tokens: list[int], successors: dict[int, tuple[int, int]]
) -> float:
    """Share of steps (from the last input token through the output) that follow the graph."""
    steps = list(zip([last_input] + tokens, tokens))
    if not steps:
        return 1.0
    valid = sum(b in successors.get(a, ()) for a, b in steps)
    return valid / len(steps)


def score_predictions(
    task: str,
    examples: list[Example],
    predictions: list[list[int]],
    spec: TaskSpec,
) -> dict:
    """Average metrics over examples. Each prediction is a full canvas of reference length.

    - token_accuracy: canvas positions equal to the reference, EOS slot included.
    - exact_match: the whole canvas equals the reference.
    - eos_correct: EOS is in the final slot and nowhere earlier.
    - repeat_rate / reference_repeat_rate: adjacent repeats in the content part.
    - transition_validity (branch tasks): legal steps in the content part,
      starting from the last input token.
    """
    if len(examples) != len(predictions) or not examples:
        raise ValueError("need one prediction per example and at least one example")
    successors = branch_successors(spec) if task in BRANCH_TASKS else None
    totals = {
        "token_accuracy": 0.0,
        "exact_match": 0.0,
        "eos_correct": 0.0,
        "repeat_rate": 0.0,
        "reference_repeat_rate": 0.0,
    }
    if successors is not None:
        totals["transition_validity"] = 0.0
    for example, predicted in zip(examples, predictions):
        reference = reference_canvas(example)
        if len(predicted) != len(reference):
            raise ValueError("each prediction must have the reference canvas length")
        matches = sum(p == r for p, r in zip(predicted, reference))
        totals["token_accuracy"] += matches / len(reference)
        totals["exact_match"] += float(predicted == reference)
        totals["eos_correct"] += float(
            predicted[-1] == EOS_ID and EOS_ID not in predicted[:-1]
        )
        totals["repeat_rate"] += repeat_rate(_content(predicted))
        totals["reference_repeat_rate"] += repeat_rate(_content(reference))
        if successors is not None:
            totals["transition_validity"] += transition_validity(
                example.input_ids[-1], _content(predicted), successors
            )
    return {name: value / len(examples) for name, value in totals.items()}
