"""Audit private pairs and optionally prepare an immutable training selection.

Paths come from a plan inside .private/ or environment variables. Audit is the
default; --prepare explicitly enables the plan's training-only sampling cap.
All audit evidence stays private. Evaluation inputs are never resplit or edited.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from tokenizers import Tokenizer
from tqdm import tqdm

BLOCK_BYTES = 1024 * 1024
DEFAULT_BINS = [0.0, 0.25, 0.5, 0.75, 0.8, 0.9, 1.0]
PLAN_KEYS = {"job", "tokenizer", "train", "eval", "max_input_len", "special_token_ids",
             "overlap_bins", "selection", "expected_sha256"}


class PreparationError(Exception):
    """A generic error that is safe to print without private paths or content."""


def _local_path(value, repo: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise PreparationError("A required private path is missing.")
    path = Path(os.path.abspath(repo / value))
    for parent in (path, *path.parents):
        if parent.exists() or parent.is_symlink():
            info = parent.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise PreparationError("Symlinks and redirected paths are not allowed.")
    if path.resolve() != path:
        raise PreparationError("Redirected paths are not allowed.")
    return path


def _inside(path: Path, root: Path) -> Path:
    if not path.is_relative_to(root):
        raise PreparationError("A private plan or destination is outside its permitted directory.")
    return path


def _ignored(paths: list[Path], repo: Path) -> None:
    relative = [path.relative_to(repo).as_posix() for path in paths]
    result = subprocess.run(["git", "check-ignore", "--no-index", "-z", "--stdin"], cwd=repo,
                            input="\0".join(relative) + "\0", text=True, capture_output=True)
    if result.returncode not in (0, 1) or set(filter(None, result.stdout.split("\0"))) != set(relative):
        raise PreparationError("Private plans and destinations must be ignored by Git.")
    tracked = subprocess.run(["git", "ls-files", "-z", "--", *relative], cwd=repo,
                             text=True, capture_output=True)
    if tracked.returncode != 0 or tracked.stdout:
        raise PreparationError("Private plans and destinations must not contain tracked files.")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(BLOCK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(path: Path) -> dict:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise PreparationError("Configured inputs must be regular files.")
    return {"path": str(path), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns, "sha256": _hash(path)}


def _write_json(path: Path, value) -> None:
    # A final marker cannot be mistaken for a partially written JSON file.
    partial = path.with_name(path.name + ".partial")
    with partial.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if path.exists():
        raise PreparationError("Immutable private evidence already exists.")
    partial.rename(path)


def _validate(plan: dict, prepare: bool) -> None:
    if not isinstance(plan, dict) or set(plan) - PLAN_KEYS:
        raise PreparationError("The private pair plan has unsupported fields.")
    if not isinstance(plan.get("job"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", plan["job"]):
        raise PreparationError("The job must be a new opaque directory name.")
    if type(plan.get("max_input_len")) is not int or plan["max_input_len"] < 1:
        raise PreparationError("Set a positive max_input_len matching the evaluated model.")
    edges = plan.get("overlap_bins", DEFAULT_BINS)
    if (not isinstance(edges, list) or len(edges) < 2 or edges[0] != 0 or edges[-1] != 1
            or any(type(edge) not in (int, float) or not math.isfinite(edge) for edge in edges)
            or any(left >= right for left, right in zip(edges, edges[1:]))):
        raise PreparationError("Overlap bin edges must increase strictly from zero to one.")
    special = plan.get("special_token_ids", [])
    if not isinstance(special, list) or any(type(value) is not int or value < 0 for value in special):
        raise PreparationError("Extra special token IDs must be nonnegative integers.")
    expected = plan.get("expected_sha256", {})
    if (not isinstance(expected, dict) or set(expected) - {"train", "eval", "tokenizer"}
            or any(not isinstance(value, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", value)
                   for value in expected.values())):
        raise PreparationError("Expected input hashes must be SHA256 strings.")
    policy = plan.get("selection")
    if prepare and policy is None:
        raise PreparationError("Preparation requires an explicit selection policy.")
    if policy is not None:
        keys = {"threshold", "max_high_overlap_pairs", "min_low_overlap_pairs", "seed"}
        if not isinstance(policy, dict) or set(policy) != keys:
            raise PreparationError("Selection needs threshold, cap, minimum low-overlap support and seed.")
        threshold = policy["threshold"]
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise PreparationError("The selection threshold must be between zero and one.")
        if (type(policy["max_high_overlap_pairs"]) is not int or policy["max_high_overlap_pairs"] < 0
                or type(policy["min_low_overlap_pairs"]) is not int or policy["min_low_overlap_pairs"] < 1
                or type(policy["seed"]) is not int):
            raise PreparationError("The cap and seed must be integers; low-overlap support must be positive.")


def pair_overlap(query: list[int], positive: list[int], max_input_len: int, special_ids: set[int]) -> float | None:
    """Jaccard after retrieval truncation; a missing lexical side is undefined."""
    left = set(query[:max_input_len]) - special_ids
    right = set(positive[:max_input_len]) - special_ids
    return len(left & right) / len(left | right) if left and right else None


def _rows(path: Path, progress: bool = False, description: str = "Reading pairs"):
    ordinal = 0
    with path.open("rb") as stream, tqdm(total=path.stat().st_size, desc=description, unit="B", unit_scale=True,
                                         ascii=True, mininterval=0.5, disable=not progress) as bar:
        for line_number, raw in enumerate(stream, start=1):
            bar.update(len(raw))
            if not raw.strip():
                continue
            ordinal += 1
            try:
                row = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeError):
                raise PreparationError("A configured pair input is not valid UTF-8 JSONL.") from None
            if (not isinstance(row, dict) or set(row) != {"query", "positive"}
                    or not isinstance(row["query"], str) or not isinstance(row["positive"], str)):
                raise PreparationError("Pair inputs must contain only string query/positive fields.")
            yield ordinal, line_number, row


def _counts(edges: list[float]) -> dict:
    return {"pairs": 0, "retrieval_eligible_pairs": 0, "empty_lexical_pairs": 0, "zero_overlap_pairs": 0,
            "query_truncated_pairs": 0, "positive_truncated_pairs": 0, "low_overlap_pairs": 0,
            "high_overlap_pairs": 0,
            "overlap_bins": [{"lower": left, "upper": right, "lower_inclusive": left != 0,
                              "upper_inclusive": right == 1, "pairs": 0}
                             for left, right in zip(edges, edges[1:])]}


def _add_overlap(counts: dict, score: float | None) -> None:
    if score is None:
        counts["empty_lexical_pairs"] += 1
    elif score == 0:
        counts["zero_overlap_pairs"] += 1
    else:
        for interval in counts["overlap_bins"]:
            if score < interval["upper"] or interval["upper"] == 1:
                interval["pairs"] += 1
                break


def _audit(paths: dict, tokenizer, plan: dict, ledger: Path, prepare: bool, progress: bool):
    special = {index for index, token in tokenizer.get_added_tokens_decoder().items() if token.special}
    special.update(plan.get("special_token_ids", []))
    if any(index >= tokenizer.get_vocab_size(with_added_tokens=True) for index in special):
        raise PreparationError("A configured special token ID exceeds the tokenizer vocabulary.")
    counts = {split: _counts(plan.get("overlap_bins", DEFAULT_BINS)) for split in ("train", "eval")}
    policy, heap = plan.get("selection"), []
    with ledger.open("x", encoding="utf-8") as stream:
        for split in ("train", "eval"):
            for ordinal, line, row in _rows(paths[split], progress, f"Auditing {split} pairs"):
                query = tokenizer.encode(row["query"], add_special_tokens=False).ids
                positive = tokenizer.encode(row["positive"], add_special_tokens=False).ids
                score = pair_overlap(query, positive, plan["max_input_len"], special)
                summary = counts[split]
                summary["pairs"] += 1
                summary["retrieval_eligible_pairs"] += bool(query) and bool(positive)
                summary["query_truncated_pairs"] += len(query) > plan["max_input_len"]
                summary["positive_truncated_pairs"] += len(positive) > plan["max_input_len"]
                _add_overlap(summary, score)
                category = "empty_lexical" if score is None else "observed"
                if policy is not None and score is not None:
                    category = "high" if score >= policy["threshold"] else "low"
                    summary[f"{category}_overlap_pairs"] += 1
                stream.write(json.dumps({"split": split, "row": ordinal, "line": line,
                                         "overlap": score, "category": category}) + "\n")
                if prepare and split == "train" and category == "high" and policy["max_high_overlap_pairs"]:
                    # Keep the smallest seeded hash ranks without retaining text or all token lists.
                    rank = int.from_bytes(hashlib.sha256(f"{policy['seed']}:{ordinal}".encode()).digest(), "big")
                    candidate = (-rank, -ordinal)
                    if len(heap) < policy["max_high_overlap_pairs"]:
                        heapq.heappush(heap, candidate)
                    elif candidate > heap[0]:
                        heapq.heapreplace(heap, candidate)
    return counts, {-ordinal for _, ordinal in heap}, sorted(special)


def _selections(ledger: Path):
    with ledger.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def _selection_ledger(audit: Path, output: Path, chosen: set[int], prepare: bool) -> dict:
    retained = {"train": 0, "eval": 0}
    with output.open("x", encoding="utf-8") as stream:
        for item in _selections(audit):
            keep = item["split"] == "eval" or item["category"] != "high" or item["row"] in chosen
            item["selected"] = keep if prepare else None
            if prepare and keep:
                retained[item["split"]] += 1
            stream.write(json.dumps(item) + "\n")
    return retained


def _training_view(source: Path, destination: Path, ledger: Path, progress: bool) -> int:
    selections = (item for item in _selections(ledger) if item["split"] == "train")
    count = 0
    with destination.open("xb") as stream:
        for ordinal, line, row in _rows(source, progress, "Preparing train view"):
            item = next(selections)
            if (ordinal, line) != (item["row"], item["line"]):
                raise PreparationError("Original row membership changed during preparation.")
            if item["selected"]:
                stream.write((json.dumps({"query": row["query"], "positive": row["positive"]},
                                         ensure_ascii=False) + "\n").encode("utf-8"))
                count += 1
    if next(selections, None) is not None:
        raise PreparationError("Original row counts changed during preparation.")
    return count


def _verify_train(source: Path, destination: Path, ledger: Path) -> int:
    selections = (item for item in _selections(ledger) if item["split"] == "train")
    count = 0
    with destination.open("rb") as output:
        for ordinal, line, row in _rows(source):
            item = next(selections)
            if (ordinal, line) != (item["row"], item["line"]):
                raise PreparationError("Original row evidence changed during verification.")
            if item["selected"]:
                if json.loads(output.readline()) != row:
                    raise PreparationError("The derived training content or order failed verification.")
                count += 1
        if output.read(1) or next(selections, None) is not None:
            raise PreparationError("The derived training row count failed verification.")
    return count


def _retention_counts(counts: dict, retained: dict, chosen: set[int]) -> dict:
    return {split: {"original_pairs": summary["pairs"], "retained_pairs": retained[split],
                    "excluded_pairs": summary["pairs"] - retained[split],
                    "low_overlap_retained": summary["low_overlap_pairs"],
                    "high_overlap_retained": len(chosen) if split == "train" else summary["high_overlap_pairs"],
                    "undefined_overlap_retained": summary["empty_lexical_pairs"],
                    "retrieval_ineligible_retained": summary["pairs"] - summary["retrieval_eligible_pairs"]}
            for split, summary in counts.items()}


def _discard_completion_markers(private: Path, output: Path) -> None:
    """Invalidate only this fresh job's markers; preserve all other failure evidence."""
    for folder in (private, output):
        marker = folder / "COMPLETE.json"
        if marker.exists():
            if not marker.resolve().is_relative_to(folder) or not marker.is_file():
                raise PreparationError("A completion marker was redirected unexpectedly.")
            marker.unlink()


def prepare_pairs(plan: dict, repo: str | Path = ".", *, prepare: bool = False,
                  progress: bool = True, plan_path: Path | None = None) -> dict:
    repo = Path(repo).resolve()
    _validate(plan, prepare)
    private_root, data_root = _local_path(".private", repo), _local_path("data", repo)
    private = _inside(_local_path(private_root / "pair-views" / plan["job"], repo), private_root)
    output = _inside(_local_path(data_root / "prepared/pair-views" / plan["job"], repo), data_root)
    if plan_path is not None:
        plan_path = _inside(_local_path(plan_path, repo), private_root)
    _ignored([private, output] + ([plan_path] if plan_path is not None else []), repo)
    if private.exists() or output.exists():
        raise PreparationError("The immutable job already exists; choose a new job name.")
    paths = {key: _local_path(plan.get(key), repo) for key in ("train", "eval", "tokenizer")}
    if paths["tokenizer"].is_dir():
        paths["tokenizer"] = _local_path(paths["tokenizer"] / "tokenizer.json", repo)
    if any(not path.is_file() for path in paths.values()):
        raise PreparationError("A configured pair input or tokenizer is unavailable.")
    if paths["train"] == paths["eval"] or os.path.samefile(paths["train"], paths["eval"]):
        raise PreparationError("Training and evaluation inputs must be distinct files.")
    private.mkdir(parents=True)
    try:
        if prepare:
            output.mkdir(parents=True)
        _write_json(private / "plan.json", plan)
        snapshots = {key: _snapshot(path) for key, path in paths.items()}
        if plan_path is not None:
            snapshots["plan"] = _snapshot(plan_path)
        for key, expected in plan.get("expected_sha256", {}).items():
            if snapshots[key]["sha256"] != expected.lower():
                raise PreparationError("A configured input does not match its expected SHA256.")
        tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))
        audit = private / "rows.jsonl"
        counts, chosen, special = _audit(paths, tokenizer, plan, audit, prepare, progress)
        if prepare:
            policy = plan["selection"]
            if counts["train"]["low_overlap_pairs"] < policy["min_low_overlap_pairs"]:
                raise PreparationError("Too few low-overlap training pairs meet the configured support requirement.")
        selection = private / "selection.jsonl"
        retained = _selection_ledger(audit, selection, chosen, prepare)
        outputs = {}
        if prepare:
            ineligible = counts["train"]["pairs"] - counts["train"]["retrieval_eligible_pairs"]
            if retained["train"] - ineligible < 2:
                raise PreparationError("The derived contrastive training set needs at least two retrieval-eligible pairs.")
            actual = _training_view(paths["train"], output / "train.jsonl", selection, progress)
            if actual != retained["train"] or _verify_train(paths["train"], output / "train.jsonl", selection) != actual:
                raise PreparationError("The derived training row count failed verification.")
            with paths["eval"].open("rb") as source, (output / "eval.jsonl").open("xb") as destination:
                shutil.copyfileobj(source, destination, BLOCK_BYTES)
            outputs = {split: _snapshot(output / f"{split}.jsonl") for split in ("train", "eval")}
            if outputs["eval"]["sha256"] != snapshots["eval"]["sha256"] or retained["eval"] != counts["eval"]["pairs"]:
                raise PreparationError("Canonical evaluation preservation failed verification.")
        for evidence in snapshots.values():
            if _snapshot(Path(evidence["path"])) != evidence:
                raise PreparationError("A configured input changed during the job.")
        report = {"schema_version": 1, "status": "verified", "mode": "prepare" if prepare else "audit",
                  "inputs": snapshots, "max_input_len": plan["max_input_len"], "truncation": "first_tokens",
                  "tokenizer_vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
                  "special_token_ids": special, "counts": counts, "selection_policy": plan.get("selection"),
                  "retained_pairs": retained if prepare else None, "outputs": outputs,
                  "retention_counts": _retention_counts(counts, retained, chosen) if prepare else None,
                  "effective_plan_sha256": _hash(private / "plan.json"), "tool_sha256": _hash(Path(__file__)),
                  "audit_rows_sha256": _hash(audit), "selection_ledger_sha256": _hash(selection),
                  "verification": {"inputs_unchanged": True, "splits_preserved": True,
                                   "train_content_and_order": "passed" if prepare else "not_prepared",
                                   "eval_byte_identical": True if prepare else "not_prepared"}}
        _write_json(private / "report.json", report)
        if prepare:
            _write_json(output / "COMPLETE.json", {"schema_version": 1, "status": "complete"})
        _write_json(private / "COMPLETE.json", {"schema_version": 1, "status": "complete",
                                               "report_sha256": _hash(private / "report.json")})
        return report
    except (Exception, KeyboardInterrupt) as error:
        _discard_completion_markers(private, output)
        cancelled = isinstance(error, KeyboardInterrupt)
        _write_json(private / "FAILED.json", {"status": "cancelled" if cancelled else "failed", "error": repr(error)})
        if cancelled:
            raise PreparationError("The pair job was cancelled; details are stored privately.") from None
        if isinstance(error, PreparationError):
            raise
        raise PreparationError("The pair job failed; details are stored privately.") from None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=os.environ.get("NURI_PAIR_VIEW_PLAN"))
    parser.add_argument("--repo", default=".")
    parser.add_argument("--job", help="new opaque job name; overrides the private plan")
    parser.add_argument("--max-input-len", type=int, help="must match the evaluated checkpoint")
    parser.add_argument("--prepare", action="store_true", help="enable the explicitly configured training cap")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    try:
        repo = Path(args.repo).resolve()
        plan_path = None
        if args.plan:
            plan_path = _inside(_local_path(args.plan, repo), _local_path(".private", repo))
            _ignored([plan_path], repo)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        else:
            plan = {"job": os.environ.get("NURI_PAIR_VIEW_JOB"), "train": os.environ.get("NURI_PAIR_DATA"),
                    "eval": os.environ.get("NURI_EVAL_PAIRS"), "tokenizer": os.environ.get("NURI_TOKENIZER"),
                    "max_input_len": int(os.environ["NURI_PAIR_MAX_INPUT_LEN"])
                    if os.environ.get("NURI_PAIR_MAX_INPUT_LEN") else None}
        if args.job is not None:
            plan["job"] = args.job
        if args.max_input_len is not None:
            plan["max_input_len"] = args.max_input_len
        report = prepare_pairs(plan, repo, prepare=args.prepare, progress=not args.no_progress, plan_path=plan_path)
    except PreparationError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError):
        print("The private pair configuration is missing or invalid.", file=sys.stderr)
        return 1
    print(json.dumps({"status": "complete", "mode": report["mode"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
