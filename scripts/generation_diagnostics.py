"""Small CPU experiments on synthetic tasks to separate causes of failed generation (Q-17).

Usage
  python scripts/generation_diagnostics.py --out runs/generation_diagnostics
  python scripts/generation_diagnostics.py --out runs/gd --groups tasks --seeds 0 --steps 300
  python scripts/generation_diagnostics.py --out runs/gd --shard 0 4 --threads 2   # one of 4 parallel workers
  python scripts/generation_diagnostics.py --out runs/gd --summarize-only

Every run trains a small Nuri model from scratch on one synthetic task from
nuri/synthetic_generation.py, using the real model code, the real canvas
corruption (`corrupt_canvas`), the real canvas loss (`canvas_loss`) and the
real decoder (`generate`). It then decodes held-out inputs at the oracle
canvas length with S=1 and S=4 refinement rounds.

Experiment groups (one hypothesis each; see docs/GENERATION_DIAGNOSTICS.md):
- tasks:   the unchanged recipe on all five tasks. Does generation fail even
           when there is exactly one answer, and does S=4 lose to S=1?
- draft:   train the readout to revise drafts on the branch tasks.
           "random-noise" replaces some visible canvas tokens with random
           tokens; "self-draft" fills some masked positions with the model's
           own predictions (SMART-style). Both score every canvas position.
- eos:     the unchanged recipe on count and branch-single, but EOS appears
           in only 10% of training canvases, as in base span denoising.
- readout: copy and reverse with readout depth 1 and 4 instead of 2, and
           with a start- or end-relative absolute position signal added to
           the input embeddings.

Two decoders are compared. "standard" is nuri.generation.generate
(Mask-Predict: kept tokens are never rescored). "revise" re-predicts every
position in every round and re-masks the least confident ones, so it can
correct a kept token.

Results are appended to <out>/results-<shard>.jsonl, one line per finished
run, and finished runs are skipped when the command is repeated. A summary
over seeds is printed and written to <out>/summary.json. These are small
synthetic models: they show which mechanisms can fail, not release quality.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nuri.config import NuriConfig, tiny_config  # noqa: E402
from nuri.corruption import corrupt_canvas, pad_batch  # noqa: E402
from nuri.generation import generate  # noqa: E402
from nuri.model import NuriModel  # noqa: E402
from nuri.synthetic_generation import (  # noqa: E402
    BRANCH_TASKS,
    FIRST_CONTENT_ID,
    MASK_ID,
    PAD_ID,
    TASK_NAMES,
    Example,
    TaskSpec,
    branch_successors,
    canvas_target,
    make_example,
    make_examples,
    reference_canvas,
    score_predictions,
)
from nuri.trainer import canvas_loss  # noqa: E402

GROUPS = ("tasks", "draft", "eos", "readout")
CANVAS_TRAINING = ("mask", "random-noise", "self-draft")
INPUT_POSITIONS = ("none", "from-start", "from-end")
DECODERS = ("standard", "revise")
ROUNDS = (1, 4)

# Model size shared by every run. Small enough for CPU; the unchanged recipe
# solves the count task at this size within 2,000 steps (CPU calibration).
MODEL_SIZE = dict(
    hidden_size=128,
    num_heads=4,
    ffn_size=352,
    input_layers=3,
    latent_slots=8,
    embed_dim=32,
    train_depths=(2, 4),
    default_depth=4,
)
FULL_MASK_PROB = 0.1  # the base recipe's full-canvas share
MAX_NOISE_SHARE = 0.3  # random-noise: at most this share of visible tokens is replaced
WARMUP_STEPS = 100


@dataclass(frozen=True)
class Run:
    """One training run. Everything except `seed` identifies a table row."""

    group: str
    task: str
    seed: int
    canvas_training: str = "mask"
    eos_prob: float = 1.0
    readout_layers: int = 2
    input_positions: str = "none"

    def __post_init__(self):
        if self.task not in TASK_NAMES:
            raise ValueError(f"unknown task {self.task!r}")
        if self.canvas_training not in CANVAS_TRAINING:
            raise ValueError(f"unknown canvas training {self.canvas_training!r}")
        if self.input_positions not in INPUT_POSITIONS:
            raise ValueError(f"unknown input positions {self.input_positions!r}")

    @property
    def variant(self) -> str:
        parts = [self.canvas_training]
        if self.eos_prob != 1.0:
            parts.append(f"eos{self.eos_prob:g}")
        if self.readout_layers != 2:
            parts.append(f"readout{self.readout_layers}")
        if self.input_positions != "none":
            parts.append(f"pos-{self.input_positions}")
        return "+".join(parts)

    @property
    def row(self) -> str:
        return f"{self.group}/{self.task}/{self.variant}"

    @property
    def key(self) -> str:
        return f"{self.row}/seed{self.seed}"


def experiment_runs(groups: list[str], seeds: list[int]) -> list[Run]:
    """All runs of the selected groups, in a fixed order."""
    runs = []
    for seed in seeds:
        if "tasks" in groups:
            runs += [Run("tasks", task, seed) for task in TASK_NAMES]
        if "draft" in groups:
            # Branch tasks: the unchanged recipe learns them only partly, so
            # there is room to see whether revision training changes S=4.
            for training in ("random-noise", "self-draft"):
                for task in BRANCH_TASKS:
                    runs.append(Run("draft", task, seed, canvas_training=training))
        if "eos" in groups:
            # Tasks the unchanged recipe can solve, so EOS is the only difference.
            for task in ("count", "branch-single"):
                runs.append(Run("eos", task, seed, eos_prob=0.1))
        if "readout" in groups:
            # Copy and reverse fail with the unchanged recipe on CPU calibration
            # runs; they need position-aligned reads from the input.
            for task in ("copy", "reverse"):
                runs.append(Run("readout", task, seed, readout_layers=1))
                runs.append(Run("readout", task, seed, readout_layers=4))
                for mode in ("from-start", "from-end"):
                    runs.append(Run("readout", task, seed, input_positions=mode))
    return runs


class InputPositionNuriModel(NuriModel):
    """Nuri with a learned absolute position vector added to the input embeddings.

    The unchanged model gives X only relative (RoPE) position information, and
    readout cross-attention has none, so a canvas position cannot directly
    ask for "input token i". This variant tests whether an explicit position
    signal fixes that:
    - from-start: index counted from the first input token (aligns copy).
    - from-end: index counted back from the last input token (aligns
      reverse and continuation).
    Everything else, including generate(), is inherited.
    """

    def __init__(self, config: NuriConfig, mode: str):
        if mode not in ("from-start", "from-end"):
            raise ValueError(f"unknown input position mode {mode!r}")
        super().__init__(config)
        self.position_mode = mode
        self.input_positions = nn.Embedding(config.max_input_len, config.hidden_size)
        nn.init.normal_(self.input_positions.weight, std=config.initializer_range)

    def position_ids(self, valid: torch.Tensor) -> torch.Tensor:
        """[B, N] position index of every input token; padding gets 0."""
        positions = torch.arange(valid.shape[1], device=valid.device)[None, :]
        if self.position_mode == "from-end":
            positions = valid.sum(dim=1, keepdim=True) - 1 - positions
        return positions.clamp(min=0) * valid

    def encode(self, input_ids, attention_mask=None, segments=None):
        if segments is not None:
            raise ValueError("input position variants do not support packed rows")
        valid = (
            self.input_mask(input_ids)
            if attention_mask is None
            else attention_mask.bool()
        )
        if not valid.any(dim=1).all():
            raise ValueError("every sample needs at least one valid input token")
        position_vectors = self.input_positions(self.position_ids(valid))
        embedded = self.embedding(input_ids) + position_vectors * valid[..., None]
        return self.encoder(embedded, valid), valid


def build_model(run: Run, spec: TaskSpec) -> NuriModel:
    config = tiny_config(
        **MODEL_SIZE,
        vocab_size=spec.vocab_size,
        readout_layers=run.readout_layers,
        max_input_len=spec.input_len[1],
        max_response_len=spec.max_target_len + 1,
    )
    if run.input_positions == "none":
        return NuriModel(config)
    return InputPositionNuriModel(config, run.input_positions)


def make_batch(examples: list[Example], rng: random.Random, eos_prob: float) -> dict:
    """Padded inputs and gold canvases. The canvas is not corrupted yet."""
    input_ids, input_valid = pad_batch([e.input_ids for e in examples], PAD_ID)
    targets = [canvas_target(e, rng, eos_prob) for e in examples]
    target_ids, canvas_valid = pad_batch(targets, PAD_ID)
    return {
        "input_ids": input_ids,
        "input_valid": input_valid,
        "targets": target_ids,
        "canvas_valid": canvas_valid,
    }


def _block_special_tokens(logits: torch.Tensor) -> torch.Tensor:
    """MASK and PAD are never written into an answer (same rule as generate)."""
    logits[..., [MASK_ID, PAD_ID]] = float("-inf")
    return logits


def corrupt_for_training(
    model: NuriModel, batch: dict, training: str, depth: int, generator
) -> dict:
    """Add canvas and loss_mask to a batch for one canvas-training variant.

    - mask: the base recipe. MASK only; loss on masked positions.
    - random-noise: also replace up to MAX_NOISE_SHARE of the visible tokens
      with random content tokens; loss on every canvas position.
    - self-draft: fill a random share of the masked positions with the
      model's own current predictions (no gradient); loss on every canvas
      position. Rounds 2..S of decoding see exactly this kind of canvas.
    """
    targets, valid = batch["targets"], batch["canvas_valid"]
    canvas, masked = corrupt_canvas(
        targets, valid, MASK_ID, full_mask_prob=FULL_MASK_PROB, generator=generator
    )
    rows = targets.shape[0]
    if training == "mask":
        return dict(batch, canvas=canvas, loss_mask=masked)
    if training == "random-noise":
        share = torch.rand(rows, 1, generator=generator) * MAX_NOISE_SHARE
        noisy = (
            valid & ~masked & (torch.rand(targets.shape, generator=generator) < share)
        )
        vocab_size = model.config.vocab_size
        random_ids = torch.randint(
            FIRST_CONTENT_ID, vocab_size, targets.shape, generator=generator
        )
        canvas = torch.where(noisy, random_ids, canvas)
        return dict(batch, canvas=canvas, loss_mask=valid.clone())
    if training == "self-draft":
        with torch.no_grad():
            x, input_valid = model.encode(batch["input_ids"], batch["input_valid"])
            cache = model.generation_memory(
                x, input_valid, model.reason(x, input_valid, depth)
            )
            hidden = model.readout_hidden(canvas, valid, cache)
            predicted = _block_special_tokens(model.vocab_logits(hidden)).argmax(dim=-1)
        share = torch.rand(rows, 1, generator=generator)
        drafted = masked & (torch.rand(targets.shape, generator=generator) < share)
        canvas = torch.where(drafted, predicted, canvas)
        return dict(batch, canvas=canvas, loss_mask=valid.clone())
    raise ValueError(f"unknown canvas training {training!r}")


def learning_rate_factor(step: int, total: int) -> float:
    """Linear warmup, then cosine decay to 10% of the peak."""
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def train_model(
    model: NuriModel, run: Run, spec: TaskSpec, steps: int, batch_size: int, lr: float
) -> dict:
    """Train on a fresh example stream. Returns the mean loss of the last 10% of steps."""
    rng = random.Random(run.seed)
    generator = torch.Generator().manual_seed(run.seed)
    successors = branch_successors(spec) if run.task in BRANCH_TASKS else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_factor(step, steps)
    )
    model.train()
    tail_losses = []
    for step in range(steps):
        examples = [
            make_example(run.task, rng, spec, successors) for _ in range(batch_size)
        ]
        depth = rng.choice(model.config.train_depths)
        batch = make_batch(examples, rng, run.eos_prob)
        batch = corrupt_for_training(
            model, batch, run.canvas_training, depth, generator
        )
        loss = canvas_loss(model, batch, depth)["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        if not math.isfinite(loss.item()):
            raise FloatingPointError(
                f"{run.key}: loss became {loss.item()} at step {step}"
            )
        if step >= steps - max(1, steps // 10):
            tail_losses.append(loss.item())
    return {"final_loss": statistics.fmean(tail_losses)}


@torch.no_grad()
def revise_generate(
    model: NuriModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    lengths: torch.Tensor,
    rounds: int,
) -> torch.Tensor:
    """Refinement that may change any token: re-predict all positions every round.

    Uses the same re-mask schedule as generate(): after round s, the
    ceil(L * (S - s) / S) least confident positions become MASK again. With
    S=1 it is identical to generate().
    """
    x, valid = model.encode(input_ids, attention_mask)
    r = model.reason(x, valid, model.config.default_depth)
    cache = model.generation_memory(x, valid, r)
    width = int(lengths.max())
    canvas_valid = torch.arange(width)[None, :] < lengths[:, None]
    canvas = torch.full((len(lengths), width), MASK_ID)
    canvas[~canvas_valid] = PAD_ID
    for round_index in range(1, rounds + 1):
        hidden = model.readout_hidden(canvas, canvas_valid, cache)
        probs = _block_special_tokens(model.vocab_logits(hidden).float()).softmax(
            dim=-1
        )
        confidence, tokens = probs.max(dim=-1)
        canvas = torch.where(canvas_valid, tokens, PAD_ID)
        if round_index < rounds:
            ranking = confidence.masked_fill(~canvas_valid, float("inf")).argsort(dim=1)
            for row, length in enumerate(lengths.tolist()):
                count = math.ceil(length * (rounds - round_index) / rounds)
                canvas[row, ranking[row, :count]] = MASK_ID
    return canvas


@torch.no_grad()
def decode(
    model: NuriModel, examples: list[Example], decoder: str, rounds: int
) -> list[list[int]]:
    """Predicted canvases at the oracle length (target plus EOS)."""
    model.eval()
    input_ids, input_valid = pad_batch([e.input_ids for e in examples], PAD_ID)
    lengths = torch.tensor([len(reference_canvas(e)) for e in examples])
    if decoder == "standard":
        canvas = generate(
            model, input_ids, input_valid, rounds=rounds, canvas_len=lengths
        ).canvas
    elif decoder == "revise":
        canvas = revise_generate(model, input_ids, input_valid, lengths, rounds)
    else:
        raise ValueError(f"unknown decoder {decoder!r}")
    return [
        canvas[row, :length].tolist() for row, length in enumerate(lengths.tolist())
    ]


def evaluate_model(model: NuriModel, run: Run, spec: TaskSpec, count: int) -> dict:
    """Scores for every decoder and round count on held-out examples."""
    examples = make_examples(run.task, count, seed=1_000_000 + run.seed, spec=spec)
    scores = {}
    for decoder in DECODERS:
        for rounds in ROUNDS:
            predictions = decode(model, examples, decoder, rounds)
            scores[f"{decoder}_s{rounds}"] = score_predictions(
                run.task, examples, predictions, spec
            )
    return scores


def execute(run: Run, args) -> dict:
    spec = TaskSpec()
    torch.manual_seed(run.seed)
    model = build_model(run, spec)
    started = time.perf_counter()
    training = train_model(model, run, spec, args.steps, args.batch_size, args.lr)
    scores = evaluate_model(model, run, spec, args.eval_examples)
    return {
        "key": run.key,
        "row": run.row,
        "run": asdict(run),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": round(time.perf_counter() - started, 1),
        **training,
        "scores": scores,
    }


def load_results(out: Path) -> list[dict]:
    results = []
    for path in sorted(out.glob("results-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            results += [json.loads(line) for line in handle if line.strip()]
    return results


def summarize(results: list[dict]) -> list[dict]:
    """Mean and standard deviation over seeds for every row and score."""
    rows: dict[str, list[dict]] = {}
    for result in results:
        rows.setdefault(result["row"], []).append(result)
    summary = []
    for row, members in rows.items():
        entry = {"row": row, "seeds": len(members)}
        for setting in members[0]["scores"]:
            for metric in members[0]["scores"][setting]:
                values = [m["scores"][setting][metric] for m in members]
                entry[f"{setting}.{metric}"] = statistics.fmean(values)
                entry[f"{setting}.{metric}.std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
        entry["final_loss"] = statistics.fmean(m["final_loss"] for m in members)
        summary.append(entry)
    return summary


def format_table(summary: list[dict]) -> str:
    """A compact view: token accuracy per decoder/round, then S=4 standard details."""
    header = (
        f"{'row':44} {'n':>2} {'acc s1':>7} {'acc s4':>7} {'rev s4':>7} "
        f"{'EM s4':>6} {'EOS s4':>6} {'rep s4':>6} {'valid s4':>8}"
    )
    lines = [header, "-" * len(header)]
    for entry in summary:
        validity = entry.get("standard_s4.transition_validity")
        lines.append(
            f"{entry['row']:44} {entry['seeds']:>2} "
            f"{entry['standard_s1.token_accuracy']:>7.3f} "
            f"{entry['standard_s4.token_accuracy']:>7.3f} "
            f"{entry['revise_s4.token_accuracy']:>7.3f} "
            f"{entry['standard_s4.exact_match']:>6.3f} "
            f"{entry['standard_s4.eos_correct']:>6.3f} "
            f"{entry['standard_s4.repeat_rate']:>6.3f} "
            f"{'-' if validity is None else f'{validity:.3f}':>8}"
        )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/generation_diagnostics"))
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument(
        "--shard",
        nargs=2,
        type=int,
        default=(0, 1),
        metavar=("INDEX", "COUNT"),
        help="run only runs whose position modulo COUNT equals INDEX",
    )
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    index, count = args.shard
    if count < 1 or not 0 <= index < count:
        parser.error("--shard needs 0 <= INDEX < COUNT")
    for name in ("steps", "batch_size", "eval_examples"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv=None) -> list[dict]:
    args = parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        index, count = args.shard
        runs = experiment_runs(args.groups, args.seeds)[index::count]
        done = {result["key"] for result in load_results(args.out)}
        results_path = args.out / f"results-{index}.jsonl"
        for number, run in enumerate(runs, start=1):
            if run.key in done:
                continue
            result = execute(run, args)
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")
            accuracy = result["scores"]["standard_s4"]["token_accuracy"]
            print(
                f"[{number}/{len(runs)}] {run.key}: s4 accuracy {accuracy:.3f}, "
                f"{result['seconds']}s",
                flush=True,
            )
    summary = summarize(load_results(args.out))
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(format_table(summary))
    return summary


if __name__ == "__main__":
    main()
