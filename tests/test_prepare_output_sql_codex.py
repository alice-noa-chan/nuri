"""Indexed output streaming preserves the original holdout query's exact files."""

import os
import sqlite3
from pathlib import Path

import pytest

from tools.prepare_data import _stage, _write_outputs


OLD_OUTPUT_QUERY = """SELECT r.kind,r.lang,r.payload,
    (SELECT max(g.held_out) FROM membership m JOIN groups g ON m.group_key=g.key
     WHERE m.digest=r.digest) FROM records r ORDER BY r.rowid"""


class QueryPlanConnection(sqlite3.Connection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_plans = []

    def execute(self, sql, parameters=()):
        if sql.startswith("INSERT OR IGNORE INTO temp.output_held_out") or sql.startswith("SELECT r.kind"):
            plan = super().execute("EXPLAIN QUERY PLAN " + sql, parameters).fetchall()
            self.output_plans.append((sql, [row[-1] for row in plan]))
        return super().execute(sql, parameters)


def open_database(path):
    database = sqlite3.connect(path, factory=QueryPlanConnection)
    database.execute("PRAGMA cache_size=-32768")
    database.executescript("""CREATE TABLE records(digest TEXT PRIMARY KEY, kind TEXT, lang TEXT, payload TEXT);
        CREATE TABLE groups(key TEXT PRIMARY KEY, held_out INTEGER);
        CREATE TABLE membership(digest TEXT, group_key TEXT, PRIMARY KEY(digest,group_key));
        CREATE INDEX member_group ON membership(group_key);""")
    return database


def stage_varied_groups(database, initial_holdout, late_eval):
    def add(kind, lang, content, groups):
        record = dict(content)
        if kind == "text":
            record["lang"] = lang
        for group in groups:
            _stage(database, record, {"kind": kind, "lang": lang}, group, group in initial_holdout)

    # Deliberately interleave task kinds and file destinations in record order.
    add("text", "en", {"text": "First chunk\nwith whitespace  "}, ["g_a"])
    add("pairs", "ja", {"query": "Query α", "positive": "Answer β"}, ["g_d"])
    add("text", "ko", {"text": "Bridge A/B Ω"}, ["g_a", "g_b"])
    add("instruction", "en", {"messages": [{"role": "user", "content": "Prompt one"}],
                                "response": "Response one"}, ["g_e"])
    add("text", "en", {"text": "Bridge B/C"}, ["g_b", "g_c"])
    add("text", "ko", {"text": "Last group C chunk"}, ["g_c"])
    for index in range(60):
        add("text", "en" if index % 3 else "ko", {"text": f"Ordered standalone row {index:02}"},
            ["g_d" if index % 2 else "g_e"])
    if late_eval:
        # Duplicate content excludes language from its identity; its first
        # payload/language stays intact while the later eval membership wins.
        inserted = _stage(database, {"text": "First chunk\nwith whitespace  ", "lang": "ko"},
                          {"kind": "text", "lang": "ko"}, "explicit_eval", True)
        assert not inserted
    database.execute("INSERT INTO groups VALUES ('empty_eval_group',1)")
    database.commit()


def original_query_outputs(database):
    while database.execute("""UPDATE groups SET held_out=1 WHERE held_out=0 AND key IN (
        SELECT a.group_key FROM membership a JOIN membership b ON a.digest=b.digest
        JOIN groups g ON b.group_key=g.key WHERE g.held_out=1)""").rowcount:
        pass
    files, counts = {}, {}
    for kind, lang, payload, held_out in database.execute(OLD_OUTPUT_QUERY):
        split = "eval" if held_out else "train"
        relative = Path("text", lang, f"{split}.jsonl") if kind == "text" else Path(
            "instructions" if kind == "instruction" else "pairs", f"{split}.jsonl")
        files.setdefault(relative, bytearray()).extend((payload + os.linesep).encode("utf-8"))
        tally = counts.setdefault(kind, {}).setdefault(lang, {"train": 0, "eval": 0})
        tally[split] += 1
    return {path: bytes(content) for path, content in files.items()}, counts


@pytest.mark.parametrize("initial_holdout", [set(), {"g_a"}, {"g_c"}, {"g_d"},
                                            {"g_a", "g_b", "g_c", "g_d", "g_e"}])
@pytest.mark.parametrize("late_eval", [False, True])
def test_indexed_output_matches_old_query_bytes_counts_order_and_holdout(
    tmp_path, initial_holdout, late_eval
):
    original = open_database(tmp_path / "original.sqlite")
    optimized = open_database(tmp_path / "optimized.sqlite")
    try:
        for database in (original, optimized):
            stage_varied_groups(database, initial_holdout, late_eval)
        expected_files, expected_counts = original_query_outputs(original)
        destination = tmp_path / "output"
        actual_counts = _write_outputs(optimized, destination, False)
        actual_files = {path.relative_to(destination): path.read_bytes()
                        for path in destination.rglob("*.jsonl")}
        assert actual_files == expected_files and actual_counts == expected_counts
        expected_groups = original.execute("SELECT * FROM groups ORDER BY key").fetchall()
        assert optimized.execute("SELECT * FROM groups ORDER BY key").fetchall() == expected_groups
        connected_holdout = late_eval or bool(initial_holdout & {"g_a", "g_b", "g_c"})
        assert all(dict(expected_groups)[key] == int(connected_holdout) for key in ("g_a", "g_b", "g_c"))
        assert not list(destination.rglob("*.part"))
    finally:
        original.close()
        optimized.close()


def test_output_plan_uses_sparse_holdout_membership_and_ordered_record_join(tmp_path):
    database = open_database(tmp_path / "planned.sqlite")
    try:
        stage_varied_groups(database, {"g_d"}, False)
        _write_outputs(database, tmp_path / "output", False)
        materialization, output = [plan for _, plan in database.output_plans]
        assert "held_out_group_keys" in materialization[0]
        assert any("SEARCH m" in step and "member_group" in step for step in materialization[1:])
        assert any("SCAN r" in step for step in output)
        assert any("SEARCH h" in step and "PRIMARY KEY" in step for step in output)
        assert not any("CORRELATED" in step or "TEMP B-TREE" in step for step in output)
        assert database.execute("PRAGMA cache_size").fetchone()[0] == -32768
        assert database.execute("PRAGMA temp.cache_size").fetchone()[0] == -8192
        assert not database.execute("SELECT name FROM sqlite_temp_master WHERE type='table'").fetchall()
    finally:
        database.close()


def test_empty_outputs_and_failed_file_creation_release_the_temporary_set(tmp_path):
    database = open_database(tmp_path / "empty.sqlite")
    try:
        assert _write_outputs(database, tmp_path / "empty", False) == {}
        assert not database.execute("SELECT name FROM sqlite_temp_master WHERE type='table'").fetchall()
        stage_varied_groups(database, set(), False)
        destination = tmp_path / "failed"
        sentinel = destination / "text/en/train.jsonl.part"
        sentinel.parent.mkdir(parents=True)
        sentinel.write_bytes(b"immutable incomplete output")
        with pytest.raises(FileExistsError):
            _write_outputs(database, destination, False)
        assert sentinel.read_bytes() == b"immutable incomplete output"
        assert not database.execute("SELECT name FROM sqlite_temp_master WHERE type='table'").fetchall()
        assert _write_outputs(database, tmp_path / "fresh", False)["text"]["en"]["train"] > 0
    finally:
        database.close()
