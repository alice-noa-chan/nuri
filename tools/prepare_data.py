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
import math
import os
import re
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

from tqdm import tqdm

SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
SAFE_LANG = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}\Z")
SHA256 = re.compile(r"[a-fA-F0-9]{64}\Z")
CHUNK_BYTES = 1024 * 1024
THREE_SPLITS = ("train", "valid", "test")


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
        raise PreparationError(
            "A configured path is outside its permitted directory."
        ) from None
    return resolved


def _destination(path: Path, root: Path) -> Path:
    resolved = _inside(path, root)
    if resolved != path.absolute():
        raise PreparationError("Destination symlinks are not allowed.")
    return resolved


def _ignored_private_outputs(paths: list[Path], repo: Path) -> None:
    relative = [path.relative_to(repo).as_posix() for path in paths]
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", "-z", "--stdin"],
        cwd=repo,
        input="\0".join(relative) + "\0",
        text=True,
        capture_output=True,
    )
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", *relative],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    if (
        ignored.returncode != 0
        or set(filter(None, ignored.stdout.split("\0"))) != set(relative)
        or tracked.returncode != 0
        or tracked.stdout
    ):
        raise PreparationError(
            "Three-way plans and outputs must be ignored and untracked by Git."
        )


def _load_plan(plan_path: Path, repo: Path) -> tuple[dict, Path]:
    private = repo / ".private"
    data = repo / "data"
    if private.resolve() != private or data.resolve() != data:
        raise PreparationError(
            "Private and data directories must be local directories."
        )
    plan_path = _inside(plan_path, private)
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(plan, dict)
        or not isinstance(plan.get("job"), str)
        or not SAFE_NAME.fullmatch(plan["job"])
    ):
        raise PreparationError("Invalid private preparation plan.")
    if (
        type(plan.get("seed", 0)) is not int
        or type(plan.get("max_chars", 12000)) is not int
        or type(plan.get("min_chars", 1)) is not int
    ):
        raise PreparationError("Seed and document lengths must be integers.")
    if (
        plan.get("max_chars", 12000) < 1
        or plan.get("min_chars", 1) < 1
        or not 0 <= plan.get("eval_fraction", 0.01) <= 1
    ):
        raise PreparationError("Invalid document length or evaluation fraction.")
    policy = plan.get("split_policy")
    if policy is not None:
        _validate_split_policy(policy, repo)
    roots = plan.get("allowed_source_roots")
    inputs = plan.get("inputs")
    if (
        not isinstance(roots, list)
        or not roots
        or not isinstance(inputs, list)
        or not inputs
    ):
        raise PreparationError(
            "The private plan needs permitted source roots and inputs."
        )
    roots = [
        Path(value) if Path(value).is_absolute() else repo / value for value in roots
    ]
    if any(not root.is_dir() for root in roots):
        raise PreparationError("A permitted source directory is unavailable.")
    seen = set()
    for item in inputs:
        if not isinstance(item, dict):
            raise PreparationError("Invalid private input configuration.")
        identifier, lang = item.get("id"), item.get("lang")
        if (
            not isinstance(identifier, str)
            or not SAFE_NAME.fullmatch(identifier)
            or identifier in seen
        ):
            raise PreparationError("Input identifiers must be unique safe names.")
        if not isinstance(lang, str) or not SAFE_LANG.fullmatch(lang):
            raise PreparationError("Languages must be safe names.")
        seen.add(identifier)
        allowed_splits = (
            {"train", "eval", "valid", "test"} if policy else {"train", "eval"}
        )
        if (
            item.get("kind") not in {"text", "instruction", "pairs"}
            or item.get("split", "train") not in allowed_splits
        ):
            raise PreparationError("Invalid input task or split.")
        source = Path(item["path"])
        source = (source if source.is_absolute() else repo / source).resolve()
        if not source.is_file() or not any(
            source.is_relative_to(root.resolve()) for root in roots
        ):
            raise PreparationError(
                "An input is unavailable or outside the permitted source roots."
            )
        item["path"] = str(source)
        item["format"] = item.get("format", source.suffix.lstrip(".").lower())
        if item["format"] not in {"jsonl", "tsv"}:
            raise PreparationError(
                "Only JSONL and header-based TSV inputs are supported."
            )
        item.setdefault("namespace", identifier)
        if not isinstance(item["namespace"], str) or not item["namespace"]:
            raise PreparationError("A grouping namespace must be a nonempty string.")
        fields = item.get("fields", {})
        if not isinstance(fields, dict) or any(
            not isinstance(v, str) or not v for v in fields.values()
        ):
            raise PreparationError("Field mappings must contain nonempty field names.")
        if "group_field" in item and (
            not isinstance(item["group_field"], str) or not item["group_field"]
        ):
            raise PreparationError("The grouping field must be a nonempty field name.")
        if "positive_values" in item and not isinstance(item["positive_values"], list):
            raise PreparationError("Positive pair labels must be a list.")
        if "sha256" in item and (
            not isinstance(item["sha256"], str) or not SHA256.fullmatch(item["sha256"])
        ):
            raise PreparationError("Invalid expected checksum.")
        imported = _destination(data / "raw" / f"{identifier}.{item['format']}", data)
        if "reuse_raw" in item and type(item["reuse_raw"]) is not bool:
            raise PreparationError("The immutable raw reuse flag must be boolean.")
        if item.get("reuse_raw") and (source != imported or "sha256" not in item):
            raise PreparationError(
                "Raw reuse requires the exact imported path and its expected SHA256."
            )
    return plan, plan_path


def _validate_split_policy(policy: dict, repo: Path) -> None:
    keys = {
        "mode",
        "valid_fraction",
        "test_fraction",
        "valid_share",
        "exposed_eval_rows",
        "legacy_hashes",
        "exposure_note",
    }
    if not isinstance(policy, dict) or set(policy) - keys:
        raise PreparationError("Invalid private three-way split policy.")
    mode = policy.get("mode")
    required = (
        ("valid_share",)
        if mode == "preserve_legacy_train"
        else ("valid_fraction", "test_fraction")
    )
    if mode not in {"preserve_legacy_train", "three_way"}:
        raise PreparationError("Choose an explicit three-way split mode.")
    for key in required:
        value = policy.get(key)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 < value < 1
        ):
            raise PreparationError(
                "Explicit split fractions must be finite and between zero and one."
            )
    if mode == "three_way" and policy["valid_fraction"] + policy["test_fraction"] >= 1:
        raise PreparationError(
            "Validation and test fractions must leave a training population."
        )
    if "exposed_eval_rows" in policy:
        path = _inside(repo / policy["exposed_eval_rows"], repo / ".private")
        if not path.is_file():
            raise PreparationError(
                "The private held-out exposure record is unavailable."
            )
        policy["exposed_eval_rows"] = str(path)
    hashes = policy.get("legacy_hashes", {})
    if not isinstance(hashes, dict):
        raise PreparationError("Legacy replay hashes must be a private mapping.")
    for name, value in hashes.items():
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.suffix != ".jsonl"
            or path.stem not in {"train", "eval"}
            or not isinstance(value, str)
            or not SHA256.fullmatch(value)
        ):
            raise PreparationError("Invalid legacy replay hash binding.")
    if "exposure_note" in policy and not isinstance(policy["exposure_note"], str):
        raise PreparationError("The exposure caveat must be a private text note.")


def _hash_file(path: Path, progress: bool) -> str:
    digest = hashlib.sha256()
    with (
        path.open("rb") as stream,
        tqdm(
            total=path.stat().st_size,
            desc="Checking local input",
            unit="B",
            unit_scale=True,
            disable=not progress,
        ) as bar,
    ):
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
            bar.update(len(chunk))
    return digest.hexdigest()


def _import(
    source: Path,
    destination: Path,
    expected: str | None,
    progress: bool,
    reuse_raw: bool = False,
) -> dict:
    before = source.stat()
    part = destination.with_name(destination.name + ".part")
    created = False
    try:
        if destination.exists():
            stat = destination.stat()
            same = (stat.st_dev, stat.st_ino) == (before.st_dev, before.st_ino)
            if stat.st_nlink > 1 or same and not reuse_raw:
                raise PreparationError("Imported files must be independent copies.")
            digest = _hash_file(source, progress)
            if not reuse_raw and digest != _hash_file(destination, progress):
                raise PreparationError(
                    "An immutable raw input already exists with different content."
                )
        else:
            digest_state = hashlib.sha256()
            with source.open("rb") as incoming, part.open("xb") as outgoing:
                created = True
                with tqdm(
                    total=before.st_size,
                    desc="Importing local input",
                    unit="B",
                    unit_scale=True,
                    disable=not progress,
                ) as bar:
                    while chunk := incoming.read(CHUNK_BYTES):
                        outgoing.write(chunk)
                        digest_state.update(chunk)
                        bar.update(len(chunk))
            digest = digest_state.hexdigest()
        after = source.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise PreparationError("A source changed during import.")
        if expected is not None and digest != expected.lower():
            raise PreparationError("A source checksum does not match the private plan.")
        if created:
            if part.stat().st_size != before.st_size:
                raise PreparationError("An imported file has an unexpected size.")
            os.utime(part, ns=(before.st_atime_ns, before.st_mtime_ns))
            os.replace(part, destination)
            created = False
        return {
            "sha256": digest,
            "size": before.st_size,
            "mtime_ns": before.st_mtime_ns,
        }
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

    def get(name):
        return _field(row, fields.get(name, name))

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
        if (
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or message["role"] not in {"system", "user", "assistant"}
        ):
            raise InvalidRecord
        clean.append(
            {"role": message["role"], "content": _string(message.get("content"))}
        )
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
    if (
        not clean
        or clean[-1]["role"] != "user"
        or not any(message["role"] == "user" for message in clean)
    ):
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


def _group(
    row: dict, record: dict, item: dict, seed: int, fraction: float
) -> tuple[str, bool]:
    value = (
        _field(row, item["group_field"])
        if "group_field" in item
        else {key: value for key, value in record.items() if key != "lang"}
    )
    if value is None or value == "":
        raise InvalidRecord
    identity = _json([item["namespace"], value])
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    sample = hashlib.sha256(_json([seed, identity]).encode("utf-8")).digest()
    held_out = item.get("split", "train") in {
        "eval",
        "valid",
        "test",
    } or int.from_bytes(sample[:8], "big") < int(fraction * 2**64)
    return key, held_out


def _stage(db, record: dict, item: dict, group: str, held_out: bool) -> bool:
    db.execute(
        "INSERT INTO groups VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET held_out=max(held_out,excluded.held_out)",
        (group, held_out),
    )
    content = {key: value for key, value in record.items() if key != "lang"}
    digest = hashlib.sha256(_json([item["kind"], content]).encode("utf-8")).hexdigest()
    inserted = db.execute(
        "INSERT OR IGNORE INTO records VALUES (?, ?, ?, ?)",
        (digest, item["kind"], item["lang"], _json(record)),
    ).rowcount
    db.execute("INSERT OR IGNORE INTO membership VALUES (?, ?)", (digest, group))
    return bool(inserted)


def _write_json(path: Path, value: dict) -> None:
    part = path.with_name(path.name + ".part")
    with part.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(part, path)


def _resolve_holdout(db, progress: bool) -> None:
    # Duplicate content connects groups. Propagate evaluation membership until
    # every connected group agrees, keeping all chunks of a document together.
    with tqdm(
        desc="Resolving holdout groups", unit="group", disable=not progress
    ) as bar:
        while True:
            promoted = db.execute("""UPDATE groups SET held_out=1 WHERE held_out=0 AND key IN (
                SELECT a.group_key FROM membership a JOIN membership b ON a.digest=b.digest
                JOIN groups g ON b.group_key=g.key WHERE g.held_out=1)""").rowcount
            bar.update(promoted)
            if not promoted:
                break
    # Resolve membership once from the held-out groups, rather than looking up
    # every record's groups while streaming the output. CROSS JOIN keeps the
    # small held-out index first, with indexed membership lookups after it.
    db.execute(
        "CREATE INDEX IF NOT EXISTS held_out_group_keys ON groups(key) WHERE held_out=1"
    )
    db.execute(
        "PRAGMA temp.cache_size=-8192"
    )  # Bound the TEMP page cache to about 8 MiB.
    db.execute(
        "CREATE TEMP TABLE output_held_out(digest TEXT PRIMARY KEY) WITHOUT ROWID"
    )
    db.execute("""INSERT OR IGNORE INTO temp.output_held_out
        SELECT m.digest FROM groups g INDEXED BY held_out_group_keys
        CROSS JOIN membership m INDEXED BY member_group ON m.group_key=g.key
        WHERE g.held_out=1""")


def _write_outputs(db, folder: Path, progress: bool) -> dict:
    _resolve_holdout(db, progress)
    counts, streams = {}, {}
    rows = None
    try:
        rows = db.execute("""SELECT r.kind,r.lang,r.payload,h.digest IS NOT NULL
            FROM records r LEFT JOIN temp.output_held_out h ON h.digest=r.digest
            ORDER BY r.rowid""")
        for kind, lang, payload, held_out in tqdm(
            rows, desc="Writing normalized records", unit="record", disable=not progress
        ):
            split = "eval" if held_out else "train"
            relative = (
                Path("text", lang, f"{split}.jsonl")
                if kind == "text"
                else Path(
                    "instructions" if kind == "instruction" else "pairs",
                    f"{split}.jsonl",
                )
            )
            if relative not in streams:
                destination = folder / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                streams[relative] = destination.with_suffix(".jsonl.part").open(
                    "x", encoding="utf-8"
                )
            streams[relative].write(payload + "\n")
            tally = counts.setdefault(kind, {}).setdefault(
                lang, {"train": 0, "eval": 0}
            )
            tally[split] += 1
    finally:
        if rows is not None:
            rows.close()
        for stream in streams.values():
            stream.close()
        db.execute("DROP TABLE temp.output_held_out")
    for relative in streams:
        destination = folder / relative
        os.replace(destination.with_suffix(".jsonl.part"), destination)
    return counts


def _components(db, progress: bool) -> None:
    """Find duplicate-connected held-out groups without keeping a graph in RAM."""
    db.execute(
        "CREATE TEMP TABLE components(key TEXT PRIMARY KEY, component TEXT) WITHOUT ROWID"
    )
    db.execute("INSERT INTO components SELECT key,key FROM groups WHERE held_out=1")
    with tqdm(
        desc="Connecting held-out documents", unit="group", disable=not progress
    ) as bar:
        while True:
            db.execute("""CREATE TEMP TABLE next_components AS
                SELECT a.group_key AS key,min(c.component) AS component
                FROM components own CROSS JOIN membership a INDEXED BY member_group ON a.group_key=own.key
                JOIN membership b ON b.digest=a.digest JOIN components c ON c.key=b.group_key
                GROUP BY a.group_key""")
            db.execute("CREATE UNIQUE INDEX next_component_key ON next_components(key)")
            changed = db.execute("""UPDATE components SET component=(SELECT component FROM next_components n
                WHERE n.key=components.key) WHERE component<>(SELECT component FROM next_components n
                WHERE n.key=components.key)""").rowcount
            db.execute("DROP TABLE next_components")
            bar.update(changed)
            if not changed:
                break


def _exposed_groups(db, policy: dict) -> None:
    db.execute(
        "CREATE TEMP TABLE exposed_components(component TEXT PRIMARY KEY) WITHOUT ROWID"
    )
    if "exposed_eval_rows" not in policy:
        return
    exposure = json.loads(
        Path(policy["exposed_eval_rows"]).read_text(encoding="utf-8-sig")
    )
    if not isinstance(exposure, list):
        raise PreparationError("Held-out exposure entries must be a private list.")
    wanted = {}
    for entry in exposure:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"kind", "lang", "ordinal", "digest"}
            or entry["kind"] not in {"text", "instruction", "pairs"}
            or not isinstance(entry["lang"], str)
            or not SAFE_LANG.fullmatch(entry["lang"])
            or type(entry["ordinal"]) is not int
            or entry["ordinal"] < 0
            or not isinstance(entry["digest"], str)
            or not SHA256.fullmatch(entry["digest"])
        ):
            raise PreparationError("Invalid hash-bound held-out exposure entry.")
        key = (entry["kind"], entry["lang"], entry["ordinal"])
        if key in wanted:
            raise PreparationError("Duplicate held-out exposure entry.")
        wanted[key] = entry["digest"].lower()
    positions = {}
    rows = db.execute("""SELECT r.digest,r.kind,r.lang FROM records r
        JOIN output_held_out h ON h.digest=r.digest ORDER BY r.rowid""")
    for digest, kind, lang in rows:
        stream_key = (kind, lang)
        ordinal = positions.get(stream_key, 0)
        positions[stream_key] = ordinal + 1
        expected = wanted.pop((kind, lang, ordinal), None)
        if expected is not None:
            if expected != digest:
                raise PreparationError(
                    "A held-out exposure ordinal does not match its bound content."
                )
            db.execute(
                """INSERT OR IGNORE INTO exposed_components SELECT c.component
                FROM membership m JOIN components c ON c.key=m.group_key WHERE m.digest=?""",
                (digest,),
            )
    if wanted:
        raise PreparationError(
            "A held-out exposure ordinal is outside the reconstructed population."
        )


def _split_components(db, plan: dict) -> None:
    policy = plan["split_policy"]
    _exposed_groups(db, policy)
    db.execute(
        "CREATE TEMP TABLE component_splits(component TEXT PRIMARY KEY,split TEXT) WITHOUT ROWID"
    )
    conflict = db.execute("""SELECT c.component FROM fixed_splits f JOIN components c ON c.key=f.group_key
        GROUP BY c.component HAVING count(DISTINCT f.split)>1 LIMIT 1""").fetchone()
    if conflict:
        raise PreparationError(
            "Duplicate-connected groups have conflicting explicit held-out splits."
        )
    fixed = dict(
        db.execute("""SELECT c.component,min(f.split) FROM fixed_splits f
        JOIN components c ON c.key=f.group_key GROUP BY c.component""")
    )
    share = (
        policy["valid_share"]
        if policy["mode"] == "preserve_legacy_train"
        else policy["valid_fraction"]
        / (policy["valid_fraction"] + policy["test_fraction"])
    )
    rows = db.execute("SELECT DISTINCT component FROM components ORDER BY component")
    for (component,) in rows:
        exposed = db.execute(
            "SELECT 1 FROM exposed_components WHERE component=?", (component,)
        ).fetchone()
        if exposed and fixed.get(component) == "test":
            raise PreparationError(
                "An already-consulted group cannot be fixed to test."
            )
        sample = hashlib.sha256(
            _json(["three-way-v1", plan.get("seed", 0), component]).encode()
        ).digest()
        split = (
            "valid"
            if int.from_bytes(sample[:8], "big") < int(share * 2**64)
            else "test"
        )
        db.execute(
            "INSERT INTO component_splits VALUES (?,?)",
            (component, "valid" if exposed else fixed.get(component, split)),
        )
    db.execute(
        "CREATE TEMP TABLE output_splits(digest TEXT PRIMARY KEY,component TEXT,split TEXT) WITHOUT ROWID"
    )
    db.execute("""INSERT INTO output_splits SELECT m.digest,min(c.component),min(s.split)
        FROM components c CROSS JOIN membership m INDEXED BY member_group ON m.group_key=c.key
        JOIN component_splits s ON s.component=c.component GROUP BY m.digest""")


def _relative_output(kind: str, lang: str, split: str) -> Path:
    return (
        Path("text", lang, f"{split}.jsonl")
        if kind == "text"
        else Path(
            "instructions" if kind == "instruction" else "pairs", f"{split}.jsonl"
        )
    )


def _write_three_outputs(
    db, folder: Path, private: Path, plan: dict, progress: bool
) -> tuple[dict, dict]:
    _resolve_holdout(db, progress)
    _components(db, progress)
    _split_components(db, plan)
    counts, streams, hashes, file_rows, legacy_hashes = {}, {}, {}, {}, {}
    populations = list(db.execute("SELECT DISTINCT kind,lang FROM records"))
    for kind, lang in populations:
        counts.setdefault(kind, {})[lang] = dict.fromkeys(THREE_SPLITS, 0)
        for split in THREE_SPLITS:
            relative = _relative_output(kind, lang, split)
            if relative not in streams:
                path = folder / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                streams[relative] = path.with_suffix(".jsonl.part").open("xb")
                hashes[relative], file_rows[relative] = hashlib.sha256(), 0
    ledger = private / f"preparation-rows-{plan['job']}.jsonl"
    groups = private / f"preparation-groups-{plan['job']}.jsonl"
    try:
        with ledger.open("x", encoding="utf-8") as evidence:
            rows = db.execute("""SELECT r.digest,r.kind,r.lang,r.payload,s.component,coalesce(s.split,'train')
                FROM records r LEFT JOIN output_splits s ON s.digest=r.digest ORDER BY r.rowid""")
            for digest, kind, lang, payload, component, split in tqdm(
                rows,
                desc="Writing three partitions",
                unit="record",
                disable=not progress,
            ):
                relative = _relative_output(kind, lang, split)
                # Replay the legacy writer's platform line ending when preserving
                # its byte-bound training corpus; fresh three-way jobs use LF.
                ending = (
                    os.linesep
                    if plan["split_policy"]["mode"] == "preserve_legacy_train"
                    else "\n"
                )
                raw = (payload + ending).encode("utf-8")
                streams[relative].write(raw)
                hashes[relative].update(raw)
                counts[kind][lang][split] += 1
                evidence.write(
                    _json(
                        {
                            "digest": digest,
                            "kind": kind,
                            "lang": lang,
                            "split": split,
                            "ordinal": file_rows[relative],
                            "component": component,
                        }
                    )
                    + "\n"
                )
                file_rows[relative] += 1
                old_relative = _relative_output(
                    kind, lang, "eval" if component else "train"
                )
                legacy_hashes.setdefault(
                    old_relative.as_posix(), hashlib.sha256()
                ).update(raw)
        with groups.open("x", encoding="utf-8") as evidence:
            for (
                key,
                component,
                split,
            ) in db.execute("""SELECT g.key,c.component,coalesce(s.split,'train')
                FROM groups g LEFT JOIN components c ON c.key=g.key
                LEFT JOIN component_splits s ON s.component=c.component ORDER BY g.key"""):
                evidence.write(
                    _json({"group": key, "component": component, "split": split}) + "\n"
                )
    finally:
        for stream in streams.values():
            stream.close()
    replay = {name: digest.hexdigest() for name, digest in legacy_hashes.items()}
    for name, expected in plan["split_policy"].get("legacy_hashes", {}).items():
        if replay.get(name) != expected.lower():
            raise PreparationError(
                "Reconstructed legacy partition bytes do not match the private binding."
            )
    manifest = {
        "format_version": 1,
        "status": "complete",
        "split_policy": plan["split_policy"],
        "splits": {split: {} for split in THREE_SPLITS},
        "legacy_replay_sha256": replay,
        "row_assignments": str(ledger),
        "group_assignments": str(groups),
        "files": [],
    }
    for relative in streams:
        path = folder / relative
        os.replace(path.with_suffix(".jsonl.part"), path)
        entry = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashes[relative].hexdigest(),
            "rows": file_rows[relative],
        }
        manifest["files"].append(
            {key: entry[key] for key in ("path", "bytes", "sha256")}
        )
        split = relative.stem
        if relative.parts[0] == "text":
            manifest["splits"][split].setdefault("text", {})[relative.parts[1]] = entry
        else:
            manifest["splits"][split][relative.parts[0]] = entry
    return counts, manifest


def prepare(
    plan_path: str | Path, repo: str | Path = ".", progress: bool = True
) -> dict:
    """Prepare one immutable job; returns only public task/language counters."""
    repo = Path(repo).resolve()
    candidate = Path(plan_path)
    candidate = candidate if candidate.is_absolute() else repo / candidate
    try:
        plan, resolved_plan = _load_plan(candidate, repo)
        data = repo / "data"
        folder = _destination(data / "prepared" / plan["job"], data)
        report_path = _destination(
            repo / ".private" / f"preparation-report-{plan['job']}.json",
            repo / ".private",
        )
        consumption_path = (
            repo / ".private" / f"preparation-consumption-{plan['job']}.json"
        )
        evidence_paths = [
            consumption_path,
            *(
                repo / ".private" / f"preparation-{name}-{plan['job']}.jsonl"
                for name in ("rows", "groups")
            ),
        ]
        if "split_policy" in plan:
            exposure = plan["split_policy"].get("exposed_eval_rows")
            _ignored_private_outputs(
                [resolved_plan, folder, report_path, *evidence_paths]
                + ([Path(exposure)] if exposure else []),
                repo,
            )
        if (
            folder.exists()
            or report_path.exists()
            or any(path.exists() for path in evidence_paths)
        ):
            raise PreparationError("An existing preparation job cannot be overwritten.")
        raw = _destination(data / "raw", data)
        raw.mkdir(parents=True, exist_ok=True)
        folder.mkdir(parents=True)
        report = {"status": "preparing", "plan": str(resolved_plan), "inputs": []}
        database = folder / ".dedup.sqlite"
        with closing(sqlite3.connect(database)) as db:
            db.execute(
                "PRAGMA cache_size=-32768"
            )  # Bound the main page cache to about 32 MiB.
            db.executescript("""CREATE TABLE records(digest TEXT PRIMARY KEY, kind TEXT, lang TEXT, payload TEXT);
                CREATE TABLE groups(key TEXT PRIMARY KEY, held_out INTEGER);
                CREATE TABLE membership(digest TEXT, group_key TEXT, PRIMARY KEY(digest,group_key));
                CREATE INDEX member_group ON membership(group_key);""")
            db.execute(
                "CREATE TABLE fixed_splits(group_key TEXT,split TEXT,PRIMARY KEY(group_key,split))"
            )
            try:
                ordered = sorted(
                    plan["inputs"],
                    key=lambda item: item.get("split", "train") == "train",
                )
                for item in ordered:
                    stats = {
                        "id": item["id"],
                        "path": item["path"],
                        "rows": 0,
                        "accepted": 0,
                        "dropped": 0,
                        "malformed": 0,
                        "duplicates": 0,
                        "chunks": 0,
                    }
                    report["inputs"].append(stats)
                    imported = raw / f"{item['id']}.{item['format']}"
                    stats.update(
                        _import(
                            Path(item["path"]),
                            imported,
                            item.get("sha256"),
                            progress,
                            item.get("reuse_raw", False),
                        )
                    )
                    for row in tqdm(
                        _rows(imported, item["format"]),
                        desc="Normalizing local input",
                        unit="record",
                        disable=not progress,
                    ):
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
                            if item["kind"] == "text" and len(
                                record["text"]
                            ) < plan.get("min_chars", 1):
                                raise InvalidRecord
                            policy = plan.get("split_policy", {})
                            fraction = (
                                policy["valid_fraction"] + policy["test_fraction"]
                                if policy.get("mode") == "three_way"
                                else plan.get("eval_fraction", 0.01)
                            )
                            group, held_out = _group(
                                row, record, item, plan.get("seed", 0), fraction
                            )
                        except (InvalidRecord, TypeError):
                            stats["dropped"] += 1
                            continue
                        stats["accepted"] += 1
                        if item.get("split") in {"valid", "test"}:
                            db.execute(
                                "INSERT OR IGNORE INTO fixed_splits VALUES (?,?)",
                                (group, item["split"]),
                            )
                        if item["kind"] == "text":
                            text = record["text"]
                            records = (
                                {
                                    "text": text[
                                        start : start + plan.get("max_chars", 12000)
                                    ],
                                    "lang": item["lang"],
                                }
                                for start in range(
                                    0, len(text), plan.get("max_chars", 12000)
                                )
                            )
                        else:
                            records = (record,)
                        for normalized in records:
                            stats["chunks"] += 1
                            if not _stage(db, normalized, item, group, held_out):
                                stats["duplicates"] += 1
                    db.commit()
                if "split_policy" in plan:
                    counts, consumption = _write_three_outputs(
                        db, folder, repo / ".private", plan, progress
                    )
                    _write_json(consumption_path, consumption)
                    report["consumption_manifest"] = str(consumption_path)
                else:
                    counts = _write_outputs(db, folder, progress)
                db.commit()
                report.update(status="prepared", counts=counts)
                _write_json(report_path, report)
                _write_json(
                    folder / "COMPLETE.json", {"status": "complete", "counts": counts}
                )
            except (Exception, KeyboardInterrupt):
                (folder / "COMPLETE.json").unlink(missing_ok=True)
                consumption_path.unlink(missing_ok=True)
                report["status"] = "failed"
                _write_json(report_path, report)
                raise
        database.unlink()
        return counts
    except PreparationError:
        raise
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error, csv.Error):
        raise PreparationError(
            "Private preparation failed; inspect the local plan and report."
        ) from None
    except KeyboardInterrupt:
        raise PreparationError(
            "Private preparation was cancelled; partial evidence was preserved."
        ) from None


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
