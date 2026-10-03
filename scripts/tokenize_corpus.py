"""Tokenize raw text into the corpus format used for training (see nuri/data.py).

Usage
  python scripts/tokenize_corpus.py --tokenizer $NURI_TOKENIZER --out corpus/ input1.jsonl input2.txt

Input files
  .jsonl  one JSON object per line; the text is read from --text-field (default "text")
  other   plain text; one document per line

Documents shorter than --min-tokens are skipped. Source file names are not
stored anywhere in the output (D-008).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nuri.data import write_corpus  # noqa: E402


def read_texts(paths: list[str], text_field: str) -> Iterator[str]:
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                yield json.loads(line)[text_field] if path.endswith(".jsonl") else line.rstrip("\n")


def tokenized(texts: Iterator[str], tokenizer, batch_size: int, min_tokens: int, stats: dict) -> Iterator[list[int]]:
    """Encode in batches (much faster than one by one) and yield token lists."""
    batch = []

    def flush():
        for encoding in tokenizer.encode_batch(batch, add_special_tokens=False):
            stats["documents_seen"] += 1
            if len(encoding.ids) >= min_tokens:
                stats["tokens"] += len(encoding.ids)
                yield encoding.ids
        batch.clear()

    for text in texts:
        batch.append(text)
        if len(batch) == batch_size:
            yield from flush()
    if batch:
        yield from flush()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--tokenizer", default=os.environ.get("NURI_TOKENIZER"), help="tokenizer.json or its folder")
    p.add_argument("--out", required=True)
    p.add_argument("--text-field", default="text")
    p.add_argument("--min-tokens", type=int, default=2, help="span denoising needs at least 2 tokens")
    p.add_argument("--batch-size", type=int, default=1000)
    args = p.parse_args(argv)
    if not args.tokenizer:
        sys.exit("missing --tokenizer (flag or NURI_TOKENIZER)")

    from tokenizers import Tokenizer
    path = Path(args.tokenizer)
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json" if path.is_dir() else path))

    stats = {"documents_seen": 0, "tokens": 0}
    written = write_corpus(args.out, tokenized(read_texts(args.inputs, args.text_field), tokenizer,
                                               args.batch_size, args.min_tokens, stats))
    stats["documents_written"] = written
    Path(args.out, "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
