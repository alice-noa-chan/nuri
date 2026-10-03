"""Synthetic multi-step reasoning tasks in Korean and English (ARCHITECTURE.md §8-A).

Each example is (messages, answer) in the chat format used for instruction
tuning. The tasks need several steps to solve, so they show whether the
recurrent core helps:

- relation: follow a chain of parent links ("who is the 3rd ancestor of X?")
- arithmetic: evaluate a short expression with bounded numbers
- reachability: decide if one node can reach another in a small directed graph

Splits
- "train": the training pools (names, nodes, numbers 1-50).
- "shared": the same pools, new random chains, graphs and expressions.
  It tests generalization to unseen combinations and is the main eval
  split. Draw it with a seed different from the training seed.
- "eval": disjoint pools (other names and nodes, numbers 51-99). Answers
  are tokens that never were answers in training, so it is a stress test:
  CPU runs showed small models score 0% there even when they can copy
  (docs/DISCUSSION.md, §12 "Evaluation hygiene").
"""

from __future__ import annotations

import random

NAMES = ["Alex", "Bora", "Chen", "Dana", "Eun", "Farah", "Gabi", "Hana", "Ivan", "Jin", "Kai", "Lena",
         "Min", "Nora", "Omar", "Pia", "Quinn", "Rin", "Sora", "Taro", "Uma", "Vic", "Wen", "Yuna",
         "Zed", "Ari", "Beni", "Cleo", "Dov", "Ema", "Finn", "Gia", "Hugo", "Ines", "Jade", "Kenji"]
NODES = [f"N{i}" for i in range(60)]


SPLITS = ("train", "shared", "eval")


def _pool(items: list[str], split: str) -> list[str]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    half = len(items) * 2 // 3
    return items[half:] if split == "eval" else items[:half]


def relation_example(rng: random.Random, lang: str, split: str = "train"):
    hops = rng.randint(2, 4)
    people = rng.sample(_pool(NAMES, split), hops + 2)
    chain = people[:hops + 1]
    facts = [(chain[i], chain[i + 1]) for i in range(hops)]
    # Distractor: an extra child of someone in the chain. It never adds a second
    # parent, so walking up from the target always has exactly one path.
    facts.append((chain[rng.randrange(hops + 1)], people[-1]))
    rng.shuffle(facts)
    target = chain[-1]
    if lang == "ko":
        lines = "\n".join(f"- 부모: {p} → 자식: {c}" for p, c in facts)
        question = f"다음 관계를 보고 답하세요.\n{lines}\n{target}의 {hops}대 위 조상은 누구인가요? 이름만 답하세요."
    else:
        lines = "\n".join(f"- {p} is the parent of {c}." for p, c in facts)
        question = f"Read the facts and answer.\n{lines}\nWho is the ancestor {hops} generations above {target}? Answer with the name only."
    return [{"role": "user", "content": question}], chain[0]


def arithmetic_example(rng: random.Random, lang: str, split: str = "train"):
    _pool([], split)  # validates the split name
    low, high = (51, 99) if split == "eval" else (1, 50)
    a, b, c, d = (rng.randint(low, high) for _ in range(4))
    op1, op2 = rng.choice("+-"), rng.choice("+-*")
    expression = f"({a} {op1} {b}) {op2} {c} - {d}"
    value = eval(expression)  # safe: built only from integers and + - *
    question = (f"다음 식의 값을 계산하세요. 숫자만 답하세요.\n{expression}" if lang == "ko"
                else f"Compute the value of the expression. Answer with the number only.\n{expression}")
    return [{"role": "user", "content": question}], str(value)


def reachability_example(rng: random.Random, lang: str, split: str = "train"):
    nodes = rng.sample(_pool(NODES, split), 7)
    edges = {(rng.choice(nodes), rng.choice(nodes)) for _ in range(8)}
    edges = sorted((a, b) for a, b in edges if a != b)
    start, goal = rng.sample(nodes, 2)
    seen, frontier = {start}, [start]
    while frontier:
        node = frontier.pop()
        for a, b in edges:
            if a == node and b not in seen:
                seen.add(b)
                frontier.append(b)
    reachable = goal in seen
    if lang == "ko":
        lines = ", ".join(f"{a}→{b}" for a, b in edges)
        question = f"방향 그래프의 간선: {lines}\n{start}에서 {goal}(으)로 갈 수 있나요? 예 또는 아니오로 답하세요."
        answer = "예" if reachable else "아니오"
    else:
        lines = ", ".join(f"{a}->{b}" for a, b in edges)
        question = f"Edges of a directed graph: {lines}\nCan you get from {start} to {goal}? Answer yes or no."
        answer = "yes" if reachable else "no"
    return [{"role": "user", "content": question}], answer


TASKS = (relation_example, arithmetic_example, reachability_example)


def reasoning_examples(count: int, seed: int = 0, split: str = "train", languages=("ko", "en")):
    """`count` examples, mixing all tasks and languages evenly at random."""
    rng = random.Random(seed)
    return [rng.choice(TASKS)(rng, rng.choice(languages), split) for _ in range(count)]
