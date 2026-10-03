"""Train small variants under identical conditions and compare them (ARCHITECTURE.md §12).

Usage
  python scripts/ablate.py --out runs/ablate --steps 2000
  python scripts/ablate.py --out runs/ablate --variants baseline gqa mla --steps 500 --device cuda

Every variant starts from the same base config (the "tiny" preset scaled up a
little by default), the same seed and the same synthetic reasoning data
(nuri/tasks.py, "train" split), and trains for the same number of steps.
Each is then evaluated on the "shared" split with the oracle and the
predicted length, together with the length-only baseline, so a variant is
only "better" when it beats both its rivals and that baseline.

The tokenizer is a small byte-level BPE trained on the task texts, so the
script needs no data. Results go to <out>/ablation.json and a table is
printed. These are small-scale comparisons: they show whether an option
trains and how it moves the metric, not how it behaves at release scale.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nuri.config import MoEConfig, tiny_config  # noqa: E402
from nuri.data import chat_prompt_text  # noqa: E402
from nuri.evaluation import generation_accuracy  # noqa: E402
from nuri.model import NuriModel  # noqa: E402
from nuri.pipeline import instruction_stage  # noqa: E402
from nuri.tasks import reasoning_examples  # noqa: E402
from nuri.trainer import TrainSettings, train  # noqa: E402

SPECIALS = ["[PAD]", "[MASK]", "[EOS]", "[SEP]", "<|system|>", "<|user|>", "<|assistant|>"]

# Config overrides per variant, on top of the shared base config.
VARIANTS = {
    "baseline": {},
    "fixed-depth-1": {"train_depths": (1,), "default_depth": 1},
    "core-moe": {"moe": MoEConfig(num_experts=4, expert_ffn_size=96, top_k=2, input_layers=(), readout_layers=(),
                                  core=True)},
    "slot-groups": {"slot_groups": 2},
    "gqa": {"num_kv_heads": 1},
    "mla": {"kv_latent_dim": 32},
    "local-attention": {"attention_schedule": "GLG", "local_radius": 16},
}


def build_tokenizer(texts: list[str], vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer


def base_config(vocab_size: int, **overrides):
    """A small model that still has every stage: 3 input layers, 8 slots, depths {2, 4}."""
    settings = dict(vocab_size=vocab_size, hidden_size=128, num_heads=4, ffn_size=384, input_layers=3,
                    latent_slots=8, max_input_len=192, max_response_len=8, embed_dim=64,
                    train_depths=(2, 4), default_depth=4, length_head_trained=True)
    settings.update(overrides)
    return tiny_config(**settings)


def run_variant(name: str, overrides: dict, train_ex, eval_ex, vocab_size: int, args) -> dict:
    torch.manual_seed(args.seed)
    model = NuriModel(base_config(vocab_size, **overrides))
    tasks, mix = instruction_stage(train_ex, model.config, args.batch_size, seed=args.seed)
    settings = TrainSettings(out_dir=str(Path(args.out) / name), total_steps=args.steps, peak_lr=args.lr,
                             warmup_steps=max(1, args.steps // 20), log_every=max(1, args.steps // 10),
                             mix=mix, seed=args.seed, progress=False, stage=name)
    started = time.time()
    records = train(model, tasks, settings, device=torch.device(args.device))
    seconds = time.time() - started
    model.eval()
    oracle = generation_accuracy(model, eval_ex, oracle_length=True)
    predicted = generation_accuracy(model, eval_ex)
    return {"variant": name, "overrides": {k: str(v) for k, v in overrides.items()},
            "parameters": sum(p.numel() for p in model.parameters()), "train_seconds": round(seconds, 1),
            "final_loss": records[-1]["loss"], "exact_match_predicted": predicted["exact_match"],
            "exact_match_oracle": oracle["exact_match"], "length_only_baseline": oracle["length_only_accuracy"],
            "length_error_rate": predicted["length_error_rate"]}


def main(argv=None) -> list[dict]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--train-examples", type=int, default=8000)
    p.add_argument("--eval-examples", type=int, default=300)
    p.add_argument("--vocab-size", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    train_raw = reasoning_examples(args.train_examples, seed=args.seed, split="train")
    eval_raw = reasoning_examples(args.eval_examples, seed=1_000_003 + args.seed, split="shared")
    tokenizer = build_tokenizer([m[0]["content"] + " " + a for m, a in train_raw + eval_raw], args.vocab_size)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids  # noqa: E731
    train_ex = [(encode(chat_prompt_text(m)), encode(a)) for m, a in train_raw]
    eval_ex = [(encode(chat_prompt_text(m)), encode(a)) for m, a in eval_raw]
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)

    results = []
    for name in args.variants:
        results.append(run_variant(name, VARIANTS[name], train_ex, eval_ex, vocab_size, args))
        print(f"{name:<16} params {results[-1]['parameters']:>9,}  loss {results[-1]['final_loss']:.3f}  "
              f"exact(pred) {results[-1]['exact_match_predicted']:.3f}  exact(oracle) "
              f"{results[-1]['exact_match_oracle']:.3f}  length-only {results[-1]['length_only_baseline']:.3f}  "
              f"{results[-1]['train_seconds']:.0f}s", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation.json").write_text(json.dumps({"settings": vars(args), "results": results}, indent=2),
                                       encoding="utf-8")
    return results


if __name__ == "__main__":
    main()
