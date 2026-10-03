"""Train one release stage of Nuri.

Examples
  # base stage from scratch
  python scripts/train.py --stage base --preset release-dense --corpus $NURI_TEXT_CORPUS \
      --out runs/nuri-base --steps 20000 --batch-size 64

  # instruction tuning from a base checkpoint
  python scripts/train.py --stage it --init runs/nuri-base/final --tokenizer $NURI_TOKENIZER \
      --instructions $NURI_IT_DATA --corpus $NURI_TEXT_CORPUS --out runs/nuri-base-it --steps 3000

  # continue an interrupted run (same flags plus --resume)
  python scripts/train.py --stage base ... --resume runs/nuri-base/training_state/step-2000.pt

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
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nuri.checkpoint import load_pretrained, training_state_config  # noqa: E402
from nuri.config import PRESETS  # noqa: E402
from nuri.data import TokenCorpus, chat_prompt_text  # noqa: E402
from nuri.model import NuriModel  # noqa: E402
from nuri.pipeline import base_stage, embedding_stage, instruction_stage, read_jsonl  # noqa: E402
from nuri.tasks import reasoning_examples  # noqa: E402
from nuri.tokenizer import check_tokenizer, load_tokenizer, vocab_size  # noqa: E402
from nuri.trainer import TrainSettings, train  # noqa: E402


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["base", "it", "embed"], required=True)
    p.add_argument("--preset", choices=sorted(PRESETS), default="release-dense", help="model size for --stage base")
    p.add_argument("--init", help="checkpoint folder to start from (required for it/embed unless --resume)")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="peak LR (default 3e-4 base, 5e-5 it/embed)")
    p.add_argument("--warmup", type=int, default=None, help="warmup steps (default 2%% of steps)")
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--log-every", type=positive_int, default=10,
                   help="record metrics every N optimizer steps (first and last always recorded)")
    p.add_argument("--no-progress", action="store_true", help="disable tqdm bars; keep setup messages and logs")
    p.add_argument("--tensorboard", action="store_true", help="write TensorBoard events to <out>/tensorboard")
    p.add_argument("--tensorboard-dir", help="custom event directory; also enables TensorBoard")
    p.add_argument("--resume", help="private training_state/*.pt file of an interrupted run with the same flags")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="micro-batches per optimizer step; effective batch = batch size x grad accum")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="recompute block activations in backward: less memory, about one extra forward")
    p.add_argument("--compile", action="store_true", help="torch.compile the model blocks (GPU runs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--corpus", default=os.environ.get("NURI_TEXT_CORPUS"))
    p.add_argument("--tokenizer", default=os.environ.get("NURI_TOKENIZER"),
                   help="tokenizer.json or its folder (it/embed default to the copy inside --init)")
    p.add_argument("--instructions", default=os.environ.get("NURI_IT_DATA"))
    p.add_argument("--pairs", default=os.environ.get("NURI_PAIR_DATA"))
    p.add_argument("--reasoning-examples", type=int, default=20000,
                   help="synthetic reasoning examples for the it stage (0 = none)")
    return p.parse_args(argv)


def require(value, flag):
    if not value:
        sys.exit(f"missing {flag} (flag or environment variable)")
    return value


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def main(argv=None):
    args = parse_args(argv)
    # Seed before building the model so --seed fully fixes the initial weights.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer_path = args.tokenizer
    if not tokenizer_path and args.init and (Path(args.init) / "tokenizer.json").exists():
        tokenizer_path = args.init  # every checkpoint carries a copy of its tokenizer
    print(f"[{args.stage}] Loading tokenizer and checking vocabulary options...", flush=True)
    tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path else None
    if args.stage == "base" and tokenizer is None:
        print("warning: no --tokenizer; using the preset vocabulary size and saving no tokenizer files")

    if args.resume:
        print(f"[{args.stage}] Building model from the resume configuration...", flush=True)
        # The interrupted run decides the model shape; its weights are loaded by train().
        model = NuriModel(training_state_config(args.resume))
    elif args.stage == "base":
        print(f"[{args.stage}] Initializing model weights...", flush=True)
        # The embedding needs exactly one row per token id of the real tokenizer.
        overrides = {"vocab_size": vocab_size(tokenizer)} if tokenizer else {}
        model = NuriModel(PRESETS[args.preset](**overrides))
    else:
        print(f"[{args.stage}] Loading initial checkpoint weights...", flush=True)
        model = load_pretrained(require(args.init, "--init"))
    config = model.config
    if tokenizer is not None:
        check_tokenizer(tokenizer, config)
    if args.corpus:
        print(f"[{args.stage}] Opening the token corpus...", flush=True)
    corpus = TokenCorpus(args.corpus) if args.corpus else None

    print(f"[{args.stage}] Preparing training tasks...", flush=True)
    if args.stage == "base":
        tasks, mix = base_stage(require(corpus, "--corpus"), config, args.batch_size, args.seed)
    elif args.stage == "it":
        require(tokenizer, "--tokenizer")

        def tokenize(messages, response):
            return encode(tokenizer, chat_prompt_text(messages)), encode(tokenizer, response)

        with tqdm(read_jsonl(require(args.instructions, "--instructions")),
                  desc="it | tokenizing instructions", unit="example", ascii=True,
                  dynamic_ncols=True, mininterval=0.5, disable=args.no_progress) as progress:
            examples = [tokenize(r["messages"], r["response"]) for r in progress]
        print("[it] Generating structured reasoning examples...", flush=True)
        raw_reasoning = reasoning_examples(args.reasoning_examples, seed=args.seed)
        with tqdm(raw_reasoning, desc="it | tokenizing reasoning examples", unit="example", ascii=True,
                  dynamic_ncols=True, mininterval=0.5, disable=args.no_progress) as progress:
            reasoning = [tokenize(m, a) for m, a in progress]
        tasks, mix = instruction_stage(examples, config, args.batch_size, corpus, reasoning, args.seed)
        config.length_head_trained = True  # saved into every checkpoint of this run
    else:
        require(tokenizer, "--tokenizer")
        with tqdm(read_jsonl(require(args.pairs, "--pairs")),
                  desc="embed | tokenizing pairs", unit="pair", ascii=True,
                  dynamic_ncols=True, mininterval=0.5, disable=args.no_progress) as progress:
            pairs = [(encode(tokenizer, r["query"]), encode(tokenizer, r["positive"])) for r in progress]
        tasks, mix = embedding_stage(pairs, config, args.batch_size, args.seed)

    peak_lr = args.lr or (3e-4 if args.stage == "base" else 5e-5)
    warmup = args.warmup if args.warmup is not None else max(1, args.steps // 50)
    tensorboard_dir = args.tensorboard_dir
    if args.tensorboard and not tensorboard_dir:
        tensorboard_dir = str(Path(args.out) / "tensorboard")
    settings = TrainSettings(out_dir=args.out, total_steps=args.steps, peak_lr=peak_lr, warmup_steps=warmup,
                             save_every=args.save_every, log_every=args.log_every, seed=args.seed, mix=mix,
                             stage=args.stage, progress=not args.no_progress, tensorboard_dir=tensorboard_dir,
                             tokenizer_dir=str(tokenizer_path) if tokenizer_path else None,
                             grad_accum=args.grad_accum, gradient_checkpointing=args.gradient_checkpointing,
                             compile=args.compile)
    print(f"stage={args.stage} mix={mix} params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    if tensorboard_dir:
        print(f"TensorBoard events: {tensorboard_dir}", flush=True)
    print(f"[{args.stage}] {'Restoring state and continuing' if args.resume else 'Starting'} training...", flush=True)
    records = train(model, tasks, settings, resume_from=args.resume)
    if records:
        print(f"done: last loss {records[-1]['loss']:.4f}, checkpoints in {args.out}")
    else:
        print("nothing to do: the resumed run had already finished")


if __name__ == "__main__":
    main()
