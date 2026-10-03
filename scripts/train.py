"""Train one release stage of Nuri.

Examples
  # base stage from scratch
  python scripts/train.py --stage base --preset release-dense --corpus $NURI_TEXT_CORPUS \
      --out runs/nuri-base --steps 20000 --batch-size 64

  # instruction tuning from a base checkpoint
  python scripts/train.py --stage it --init runs/nuri-base/final --tokenizer $NURI_TOKENIZER \
      --instructions $NURI_IT_DATA --corpus $NURI_TEXT_CORPUS --out runs/nuri-base-it --steps 3000

  # embedding training from the dense base checkpoint
  python scripts/train.py --stage embed --init runs/nuri-base/final --tokenizer $NURI_TOKENIZER \
      --pairs $NURI_PAIR_DATA --out runs/nuri-base-embed --steps 3000

Data paths can be given as flags or through the NURI_* environment variables
shown above. File formats:
  corpus        a folder written by nuri.data.write_corpus (tokens.bin + offsets.npy)
  instructions  JSONL: {"messages": [{"role": "user", "content": "..."}], "response": "..."}
  pairs         JSONL: {"query": "...", "positive": "..."}
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nuri.checkpoint import load_pretrained  # noqa: E402
from nuri.config import PRESETS  # noqa: E402
from nuri.data import TokenCorpus, chat_prompt_text  # noqa: E402
from nuri.model import NuriModel  # noqa: E402
from nuri.pipeline import base_stage, embedding_stage, instruction_stage, read_jsonl  # noqa: E402
from nuri.tasks import reasoning_examples  # noqa: E402
from nuri.trainer import TrainSettings, train  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["base", "it", "embed"], required=True)
    p.add_argument("--preset", choices=sorted(PRESETS), default="release-dense", help="model size for --stage base")
    p.add_argument("--init", help="checkpoint folder to start from (required for it/embed)")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="peak LR (default 3e-4 base, 5e-5 it/embed)")
    p.add_argument("--warmup", type=int, default=None, help="warmup steps (default 2%% of steps)")
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--corpus", default=os.environ.get("NURI_TEXT_CORPUS"))
    p.add_argument("--tokenizer", default=os.environ.get("NURI_TOKENIZER"))
    p.add_argument("--instructions", default=os.environ.get("NURI_IT_DATA"))
    p.add_argument("--pairs", default=os.environ.get("NURI_PAIR_DATA"))
    p.add_argument("--reasoning-examples", type=int, default=20000,
                   help="synthetic reasoning examples for the it stage (0 = none)")
    return p.parse_args(argv)


def require(value, flag):
    if not value:
        sys.exit(f"missing {flag} (flag or environment variable)")
    return value


def load_tokenizer(path):
    """Accepts a tokenizer.json file or a folder that contains one."""
    from tokenizers import Tokenizer
    path = Path(require(path, "--tokenizer"))
    if path.is_dir():
        path = path / "tokenizer.json"
    return Tokenizer.from_file(str(path))


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "base":
        model = NuriModel(PRESETS[args.preset]())
    else:
        model = load_pretrained(require(args.init, "--init"))
    config = model.config
    corpus = TokenCorpus(args.corpus) if args.corpus else None

    if args.stage == "base":
        tasks, mix = base_stage(require(corpus, "--corpus"), config, args.batch_size, args.seed)
    elif args.stage == "it":
        tokenizer = load_tokenizer(args.tokenizer)

        def tokenize(messages, response):
            return encode(tokenizer, chat_prompt_text(messages)), encode(tokenizer, response)

        examples = [tokenize(r["messages"], r["response"]) for r in read_jsonl(require(args.instructions, "--instructions"))]
        reasoning = [tokenize(m, a) for m, a in reasoning_examples(args.reasoning_examples, seed=args.seed)]
        tasks, mix = instruction_stage(examples, config, args.batch_size, corpus, reasoning, args.seed)
        config.length_head_trained = True  # saved into every checkpoint of this run
    else:
        tokenizer = load_tokenizer(args.tokenizer)
        pairs = [(encode(tokenizer, r["query"]), encode(tokenizer, r["positive"]))
                 for r in read_jsonl(require(args.pairs, "--pairs"))]
        tasks, mix = embedding_stage(pairs, config, args.batch_size, args.seed)

    peak_lr = args.lr or (3e-4 if args.stage == "base" else 5e-5)
    warmup = args.warmup if args.warmup is not None else max(1, args.steps // 50)
    settings = TrainSettings(out_dir=args.out, total_steps=args.steps, peak_lr=peak_lr, warmup_steps=warmup,
                             save_every=args.save_every, seed=args.seed, mix=mix)
    print(f"stage={args.stage} mix={mix} params={sum(p.numel() for p in model.parameters()):,}")
    records = train(model, tasks, settings)
    print(f"done: last loss {records[-1]['loss']:.4f}, checkpoints in {args.out}")


if __name__ == "__main__":
    main()
