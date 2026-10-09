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
from tools.prepare_pair_views import (PreparationError, _hash, _ignored, _inside, _local_path,
                                     _snapshot, _write_json)  # noqa: E402


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


def tokenize_splits(plan_path: str, repo: str = ".", progress: bool = True) -> dict:
    """Consume a completed grouped preparation manifest; never invent a split."""
    import re
    import numpy as np
    from tokenizers import Tokenizer
    from tqdm import tqdm
    from nuri.data import TOKEN_DTYPES, TokenCorpus

    root = Path(repo).resolve()
    private = _local_path(".private", root)
    plan_file = _inside(_local_path(plan_path, root), private)
    plan = json.loads(plan_file.read_text(encoding="utf-8-sig"))
    keys = {"schema_version", "job", "tokenizer", "tokenizer_sha256", "vocab_size", "preparation_manifest",
            "preparation_manifest_sha256", "batch_size", "min_tokens", "reuse_train"}
    if (not isinstance(plan, dict) or set(plan) - keys or plan.get("schema_version") != 1
            or not isinstance(plan.get("job"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", plan["job"])
            or type(plan.get("batch_size", 256)) is not int or plan.get("batch_size", 256) < 1
            or type(plan.get("min_tokens", 2)) is not int or plan.get("min_tokens", 2) < 1
            or type(plan.get("vocab_size")) is not int or plan["vocab_size"] < 1):
        raise PreparationError("Invalid private split-tokenization plan.")
    destination = _inside(_local_path(f"data/tokens/{plan['job']}", root), _local_path("data", root))
    report_path = private / f"tokenization-consumption-{plan['job']}.json"
    _ignored([plan_file, destination, report_path], root)
    if destination.exists() or report_path.exists():
        raise PreparationError("Split tokenization needs a fresh immutable job.")
    prepared_path = _inside(_local_path(plan["preparation_manifest"], root), private)
    if _hash(prepared_path) != plan.get("preparation_manifest_sha256"):
        raise PreparationError("The preparation manifest hash does not match the plan.")
    prepared = json.loads(prepared_path.read_text(encoding="utf-8-sig"))
    if prepared.get("status") != "complete" or set(prepared.get("splits", {})) != {"train", "valid", "test"}:
        raise PreparationError("A completed explicit three-way preparation manifest is required.")
    tokenizer_path = _local_path(plan["tokenizer"], root)
    if tokenizer_path.is_dir():
        tokenizer_path = _local_path(tokenizer_path / "tokenizer.json", root)
    tokenizer_before = _snapshot(tokenizer_path)
    if tokenizer_before["sha256"] != plan.get("tokenizer_sha256"):
        raise PreparationError("The tokenizer hash does not match the private plan.")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size(with_added_tokens=True) != plan["vocab_size"]:
        raise PreparationError("The loaded tokenizer vocabulary does not match the private plan.")
    result = {"schema_version": 1, "status": "preparing", "tokenizer": tokenizer_before,
              "vocab_size": plan["vocab_size"], "preparation_manifest": str(prepared_path),
              "preparation_manifest_sha256": _hash(prepared_path), "splits": prepared["splits"],
              "split_policy": prepared["split_policy"], "files": list(prepared["files"])}
    destination.mkdir(parents=True)
    try:
        for split, block in result["splits"].items():
            for lang, source_entry in block.get("text", {}).items():
                source = _inside(_local_path(source_entry["path"], root), _local_path("data", root))
                _ignored([source], root)
                if _hash(source) != source_entry["sha256"]:
                    raise PreparationError("A grouped text input changed before tokenization.")
                before = source.stat()
                reuse = plan.get("reuse_train", {}).get(lang) if split == "train" else None
                if reuse is not None:
                    folder = _inside(_local_path(reuse["path"], root), _local_path("data", root))
                    _ignored([folder], root)
                    completion_path = _local_path(folder.parent.parent / "COMPLETE.json", root)
                    completion = json.loads(completion_path.read_text(encoding="utf-8"))
                    if (completion.get("status") != "complete" or completion.get("vocab_size") != plan["vocab_size"]
                            or completion.get("tokenizer_sha256_at_load") != tokenizer_before["sha256"]):
                        raise PreparationError("A reused training corpus needs completed evidence for the exact loaded tokenizer.")
                    if reuse.get("prepared_sha256") != source_entry["sha256"]:
                        raise PreparationError("A reused training corpus is not bound to the identical prepared training input.")
                    for name in ("tokens.bin", "offsets.npy"):
                        if _hash(folder / name) != reuse.get("files", {}).get(name):
                            raise PreparationError("A reused training corpus failed its file hash binding.")
                    stats = dict(reuse["stats"])
                else:
                    folder = destination / lang / split
                    stats = {"documents_seen": 0, "tokens": 0}
                    texts = tqdm(read_texts([str(source)], "text"), total=source_entry["rows"],
                                 desc=f"Tokenizing {lang}/{split}", unit="document", disable=not progress)
                    written = write_corpus(folder, tokenized(texts, tokenizer, plan.get("batch_size", 256),
                                                            plan.get("min_tokens", 2), stats))
                    stats["documents_written"] = written
                    _write_json(folder / "stats.json", stats)
                corpus = TokenCorpus(folder)
                if (stats["documents_seen"] != source_entry["rows"] or len(corpus) != stats["documents_written"]
                        or int(corpus.offsets[-1]) != stats["tokens"] or corpus.tokens.dtype not in TOKEN_DTYPES):
                    raise PreparationError("A token corpus does not match its recorded document/token counts.")
                for start in range(0, len(corpus.offsets) - 1, 262144):
                    differences = np.diff(corpus.offsets[start:start + 262145])
                    if np.any(differences < plan.get("min_tokens", 2)):
                        raise PreparationError("Token corpus document boundaries failed verification.")
                for start in range(0, len(corpus.tokens), 1048576):
                    if np.any(corpus.tokens[start:start + 1048576] >= plan["vocab_size"]):
                        raise PreparationError("Token corpus contains IDs outside the loaded vocabulary.")
                if (before.st_size, before.st_mtime_ns) != (source.stat().st_size, source.stat().st_mtime_ns):
                    raise PreparationError("A grouped text input changed during tokenization.")
                files = {name: _snapshot(folder / name) for name in ("tokens.bin", "offsets.npy")}
                result["files"].extend({key: item[key] for key in ("path", "bytes", "sha256")}
                                       for item in files.values())
                source_entry["corpus"] = {"path": str(folder), "stats": stats, "files": files,
                                          "reused": reuse is not None, "minimum_tokens": plan.get("min_tokens", 2)}
        if _snapshot(tokenizer_path) != tokenizer_before:
            raise PreparationError("The tokenizer changed during split tokenization.")
        result["status"] = "complete"
        _write_json(report_path, result)
        _write_json(destination / "COMPLETE.json", {"status": "complete", "vocab_size": plan["vocab_size"],
                                                   "tokenizer_sha256_at_load": tokenizer_before["sha256"],
                                                   "consumption_manifest_sha256": _hash(report_path)})
        return result
    except (Exception, KeyboardInterrupt) as error:
        (destination / "COMPLETE.json").unlink(missing_ok=True)
        _write_json(destination / "FAILED.json", {"status": "failed"})
        if report_path.exists():
            report_path.unlink()
        if isinstance(error, PreparationError):
            raise
        raise PreparationError("Split tokenization failed; partial outputs were preserved.") from None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="*")
    p.add_argument("--plan", default=os.environ.get("NURI_TOKENIZATION_PLAN"), help="private three-way consumption plan")
    p.add_argument("--repo", default=".")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--tokenizer", default=os.environ.get("NURI_TOKENIZER"), help="tokenizer.json or its folder")
    p.add_argument("--out")
    p.add_argument("--text-field", default="text")
    p.add_argument("--min-tokens", type=int, default=2, help="span denoising needs at least 2 tokens")
    p.add_argument("--batch-size", type=int, default=1000)
    args = p.parse_args(argv)
    if args.plan:
        if args.inputs or args.out:
            p.error("--plan cannot be combined with positional inputs or --out")
        try:
            tokenize_splits(args.plan, args.repo, not args.no_progress)
        except (PreparationError, OSError, ValueError, KeyError, TypeError):
            sys.exit("Private split tokenization failed; inspect the private plan and partial job.")
        print(json.dumps({"status": "complete", "splits": ["train", "valid", "test"]}))
        return
    if not args.inputs or not args.out:
        p.error("legacy tokenization requires input files and --out")
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
