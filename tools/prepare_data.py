"""Stream local private-plan inputs into ignored raw and normalized data files.

Plans and per-input reports stay in .private/. All imported and prepared data
stay in data/. Source files are read only. A COMPLETE.json file is written last;
an existing job directory is never overwritten, even after an incomplete run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from tqdm import tqdm


SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
SAFE_LANG = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}\Z")
SHA256 = re.compile(r"[a-fA-F0-9]{64}\Z")
CHUNK_BYTES = 1024 * 1024


class PreparationError(Exception):
    """A safe, generic preparation error without source identities or text."""


class InvalidRecord(Exception):
    pass


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise PreparationError("A configured path is outside its permitted directory.") from None
    return resolved


def _destination(path: Path, root: Path) -> Path:
    resolved = _inside(path, root)
    if resolved != path.absolute():
        raise PreparationError("Destination symlinks are not allowed.")
    return resolved


def _load_plan(plan_path: Path, repo: Path) -> tuple[dict, Path]:
    private = repo / ".private"
    data = repo / "data"
    if private.resolve() != private or data.resolve() != data:
        raise PreparationError("Private and data directories must be local directories.")
    plan_path = _inside(plan_path, private)
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    if not isinstance(plan, dict) or not isinstance(plan.get("job"), str) or not SAFE_NAME.fullmatch(plan["job"]):
        raise PreparationError("Invalid private preparation plan.")
    if (type(plan.get("seed", 0)) is not int or type(plan.get("max_chars", 12000)) is not int
            or type(plan.get("min_chars", 1)) is not int):
        raise PreparationError("Seed and document lengths must be integers.")
    if (plan.get("max_chars", 12000) < 1 or plan.get("min_chars", 1) < 1
            or not 0 <= plan.get("eval_fraction", 0.01) <= 1):
        raise PreparationError("Invalid document length or evaluation fraction.")
    roots = plan.get("allowed_source_roots")
    inputs = plan.get("inputs")
    if not isinstance(roots, list) or not roots or not isinstance(inputs, list) or not inputs:
        raise PreparationError("The private plan needs permitted source roots and inputs.")
    roots = [Path(value) if Path(value).is_absolute() else repo / value for value in roots]
    if any(not root.is_dir() for root in roots):
        raise PreparationError("A permitted source directory is unavailable.")
    seen = set()
    for item in inputs:
        if not isinstance(item, dict):
            raise PreparationError("Invalid private input configuration.")
        identifier, lang = item.get("id"), item.get("lang")
        if not isinstance(identifier, str) or not SAFE_NAME.fullmatch(identifier) or identifier in seen:
            raise PreparationError("Input identifiers must be unique safe names.")
        if not isinstance(lang, str) or not SAFE_LANG.fullmatch(lang):
            raise PreparationError("Languages must be safe names.")
        seen.add(identifier)
        if item.get("kind") not in {"text", "instruction", "pairs"} or item.get("split", "train") not in {"train", "eval"}:
            raise PreparationError("Invalid input task or split.")
        source = Path(item["path"])
        source = (source if source.is_absolute() else repo / source).resolve()
        if not source.is_file() or not any(source.is_relative_to(root.resolve()) for root in roots):
            raise PreparationError("An input is unavailable or outside the permitted source roots.")
        item["path"] = str(source)
        item["format"] = item.get("format", source.suffix.lstrip(".").lower())
        if item["format"] not in {"jsonl", "tsv"}:
            raise PreparationError("Only JSONL and header-based TSV inputs are supported.")
        item.setdefault("namespace", identifier)
        if not isinstance(item["namespace"], str) or not item["namespace"]:
            raise PreparationError("A grouping namespace must be a nonempty string.")
        fields = item.get("fields", {})
        if not isinstance(fields, dict) or any(not isinstance(v, str) or not v for v in fields.values()):
            raise PreparationError("Field mappings must contain nonempty field names.")
        if "group_field" in item and (not isinstance(item["group_field"], str) or not item["group_field"]):
            raise PreparationError("The grouping field must be a nonempty field name.")
        if "positive_values" in item and not isinstance(item["positive_values"], list):
            raise PreparationError("Positive pair labels must be a list.")
        if "sha256" in item and (not isinstance(item["sha256"], str) or not SHA256.fullmatch(item["sha256"])):
            raise PreparationError("Invalid expected checksum.")
        _destination(data / "raw" / f"{identifier}.{item['format']}", data)
    return plan, plan_path


def _hash_file(path: Path, progress: bool) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream, tqdm(total=path.stat().st_size, desc="Checking local input",
                                         unit="B", unit_scale=True, disable=not progress) as bar:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
            bar.update(len(chunk))
    return digest.hexdigest()


def _import(source: Path, destination: Path, expected: str | None, progress: bool) -> dict:
    before = source.stat()
    part = destination.with_name(destination.name + ".part")
    created = False
    try:
        if destination.exists():
            stat = destination.stat()
            if stat.st_nlink > 1 or (stat.st_dev, stat.st_ino) == (before.st_dev, before.st_ino):
                raise PreparationError("Imported files must be independent copies.")
            digest = _hash_file(source, progress)
            if digest != _hash_file(destination, progress):
                raise PreparationError("An immutable raw input already exists with different content.")
        else:
            digest_state = hashlib.sha256()
            with source.open("rb") as incoming, part.open("xb") as outgoing:
                created = True
                with tqdm(total=before.st_size, desc="Importing local input", unit="B", unit_scale=True,
                          disable=not progress) as bar:
                    while chunk := incoming.read(CHUNK_BYTES):
                        outgoing.write(chunk)
                        digest_state.update(chunk)
                        bar.update(len(chunk))
            digest = digest_state.hexdigest()
        after = source.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise PreparationError("A source changed during import.")
        if expected is not None and digest != expected.lower():
            raise PreparationError("A source checksum does not match the private plan.")
        if created:
            if part.stat().st_size != before.st_size:
                raise PreparationError("An imported file has an unexpected size.")
            os.utime(part, ns=(before.st_atime_ns, before.st_mtime_ns))
            os.replace(part, destination)
            created = False
        return {"sha256": digest, "size": before.st_size, "mtime_ns": before.st_mtime_ns}
    finally:
        if created:
            part.unlink(missing_ok=True)


def _field(row: dict, name: str):
    if name in row:
        return row[name]
    value = row
    for component in name.split("."):
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    return value


def _string(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidRecord
    return value


def _normalize(row: dict, item: dict) -> dict | None:
    fields = item.get("fields", {})
    get = lambda name: _field(row, fields.get(name, name))
    if item["kind"] == "text":
        return {"text": _string(get("text")), "lang": item["lang"]}
    if item["kind"] == "pairs":
        if "positive_values" in item and get("label") not in item["positive_values"]:
            return None
        return {"query": _string(get("query")), "positive": _string(get("positive"))}
    messages, response = get("messages"), get("response")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            raise InvalidRecord from None
    if messages is None:
        messages = [{"role": "user", "content": _string(get("prompt"))}]
    if not isinstance(messages, list) or not messages:
        raise InvalidRecord
    clean = []
    for message in messages:
        if (not isinstance(message, dict) or not isinstance(message.get("role"), str)
                or message["role"] not in {"system", "user", "assistant"}):
            raise InvalidRecord
        clean.append({"role": message["role"], "content": _string(message.get("content"))})
    if response is None:
        if clean[-1]["role"] != "assistant":
            raise InvalidRecord
        response = clean.pop()["content"]
    else:
        response = _string(response)
        if clean[-1]["role"] == "assistant":
            if clean[-1]["content"] != response:
                raise InvalidRecord
            clean.pop()
    if not clean or clean[-1]["role"] != "user" or not any(message["role"] == "user" for message in clean):
        raise InvalidRecord
    return {"messages": clean, "response": _string(response)}


def _rows(path: Path, format: str):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if format == "tsv":
            for row in csv.DictReader(stream, delimiter="\t"):
                yield row if None not in row and None not in row.values() else None
        else:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    row = None
                yield row if isinstance(row, dict) else None


def _group(row: dict, record: dict, item: dict, seed: int, fraction: float) -> tuple[str, bool]:
    value = _field(row, item["group_field"]) if "group_field" in item else {
        key: value for key, value in record.items() if key != "lang"}
    if value is None or value == "":
        raise InvalidRecord
    identity = _json([item["namespace"], value])
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    sample = hashlib.sha256(_json([seed, identity]).encode("utf-8")).digest()
    held_out = item.get("split", "train") == "eval" or int.from_bytes(sample[:8], "big") < int(fraction * 2**64)
    return key, held_out


def _stage(db, record: dict, item: dict, group: str, held_out: bool) -> bool:
    db.execute("INSERT INTO groups VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET held_out=max(held_out,excluded.held_out)",
               (group, held_out))
    content = {key: value for key, value in record.items() if key != "lang"}
    digest = hashlib.sha256(_json([item["kind"], content]).encode("utf-8")).hexdigest()
    inserted = db.execute("INSERT OR IGNORE INTO records VALUES (?, ?, ?, ?)",
                          (digest, item["kind"], item["lang"], _json(record))).rowcount
    db.execute("INSERT OR IGNORE INTO membership VALUES (?, ?)", (digest, group))
    return bool(inserted)


def _write_json(path: Path, value: dict) -> None:
    part = path.with_name(path.name + ".part")
    with part.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(part, path)


def _write_outputs(db, folder: Path, progress: bool) -> dict:
    # Duplicate content connects groups. Propagate evaluation membership until
    # every connected group agrees, keeping all chunks of a document together.
    with tqdm(desc="Resolving holdout groups", unit="group", disable=not progress) as bar:
        while True:
            promoted = db.execute("""UPDATE groups SET held_out=1 WHERE held_out=0 AND key IN (
                SELECT a.group_key FROM membership a JOIN membership b ON a.digest=b.digest
                JOIN groups g ON b.group_key=g.key WHERE g.held_out=1)""").rowcount
            bar.update(promoted)
            if not promoted:
                break
    counts, streams = {}, {}
    try:
        rows = db.execute("""SELECT r.kind,r.lang,r.payload,
            (SELECT max(g.held_out) FROM membership m JOIN groups g ON m.group_key=g.key
             WHERE m.digest=r.digest) FROM records r ORDER BY r.rowid""")
        for kind, lang, payload, held_out in tqdm(rows, desc="Writing normalized records", unit="record",
                                               disable=not progress):
            split = "eval" if held_out else "train"
            relative = Path("text", lang, f"{split}.jsonl") if kind == "text" else Path(
                "instructions" if kind == "instruction" else "pairs", f"{split}.jsonl")
            if relative not in streams:
                destination = folder / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                streams[relative] = destination.with_suffix(".jsonl.part").open("x", encoding="utf-8")
            streams[relative].write(payload + "\n")
            tally = counts.setdefault(kind, {}).setdefault(lang, {"train": 0, "eval": 0})
            tally[split] += 1
    finally:
        for stream in streams.values():
            stream.close()
    for relative in streams:
        destination = folder / relative
        os.replace(destination.with_suffix(".jsonl.part"), destination)
    return counts


def prepare(plan_path: str | Path, repo: str | Path = ".", progress: bool = True) -> dict:
    """Prepare one immutable job; returns only public task/language counters."""
    repo = Path(repo).resolve()
    candidate = Path(plan_path)
    candidate = candidate if candidate.is_absolute() else repo / candidate
    try:
        plan, resolved_plan = _load_plan(candidate, repo)
        data = repo / "data"
        folder = _destination(data / "prepared" / plan["job"], data)
        report_path = _destination(repo / ".private" / f"preparation-report-{plan['job']}.json", repo / ".private")
        if folder.exists() or report_path.exists():
            raise PreparationError("An existing preparation job cannot be overwritten.")
        raw = _destination(data / "raw", data)
        raw.mkdir(parents=True, exist_ok=True)
        folder.mkdir(parents=True)
        report = {"status": "preparing", "plan": str(resolved_plan), "inputs": []}
        database = folder / ".dedup.sqlite"
        with closing(sqlite3.connect(database)) as db:
            db.execute("PRAGMA cache_size=-32768")  # Bound the main page cache to about 32 MiB.
            db.executescript("""CREATE TABLE records(digest TEXT PRIMARY KEY, kind TEXT, lang TEXT, payload TEXT);
                CREATE TABLE groups(key TEXT PRIMARY KEY, held_out INTEGER);
                CREATE TABLE membership(digest TEXT, group_key TEXT, PRIMARY KEY(digest,group_key));
                CREATE INDEX member_group ON membership(group_key);""")
            try:
                ordered = sorted(plan["inputs"], key=lambda item: item.get("split", "train") != "eval")
                for item in ordered:
                    stats = {"id": item["id"], "path": item["path"], "rows": 0, "accepted": 0,
                             "dropped": 0, "malformed": 0, "duplicates": 0, "chunks": 0}
                    report["inputs"].append(stats)
                    imported = raw / f"{item['id']}.{item['format']}"
                    stats.update(_import(Path(item["path"]), imported, item.get("sha256"), progress))
                    for row in tqdm(_rows(imported, item["format"]), desc="Normalizing local input", unit="record",
                                    disable=not progress):
                        stats["rows"] += 1
                        if stats["rows"] % 1000 == 0:
                            db.commit()
                        if row is None:
                            stats["malformed"] += 1
                            stats["dropped"] += 1
                            continue
                        try:
                            record = _normalize(row, item)
                            if record is None:
                                raise InvalidRecord
                            if item["kind"] == "text" and len(record["text"]) < plan.get("min_chars", 1):
                                raise InvalidRecord
                            group, held_out = _group(row, record, item, plan.get("seed", 0),
                                                     plan.get("eval_fraction", 0.01))
                        except (InvalidRecord, TypeError):
                            stats["dropped"] += 1
                            continue
                        stats["accepted"] += 1
                        if item["kind"] == "text":
                            text = record["text"]
                            records = ({"text": text[start:start + plan.get("max_chars", 12000)], "lang": item["lang"]}
                                       for start in range(0, len(text), plan.get("max_chars", 12000)))
                        else:
                            records = (record,)
                        for normalized in records:
                            stats["chunks"] += 1
                            if not _stage(db, normalized, item, group, held_out):
                                stats["duplicates"] += 1
                    db.commit()
                counts = _write_outputs(db, folder, progress)
                db.commit()
                report.update(status="prepared", counts=counts)
                _write_json(report_path, report)
            except Exception:
                report["status"] = "failed"
                if not report_path.exists():
                    _write_json(report_path, report)
                raise
        database.unlink()
        _write_json(folder / "COMPLETE.json", {"status": "complete", "counts": counts})
        return counts
    except PreparationError:
        raise
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error, csv.Error):
        raise PreparationError("Private preparation failed; inspect the local plan and report.") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    try:
        counts = prepare(args.plan, args.repo, progress=not args.no_progress)
    except PreparationError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({"counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
