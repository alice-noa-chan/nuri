import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402

from nuri.data import write_corpus  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WORDS = ["[PAD]", "[MASK]", "[EOS]", "[SEP]", "<|system|>", "<|user|>", "<|assistant|>",
         "[UNK]"] + [f"w{i}" for i in range(120)]  # fmt: skip


def load_easy_run():
    spec = importlib.util.spec_from_file_location("easy_run", ROOT / "easy_run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def easy(tmp_path, monkeypatch):
    module = load_easy_run()
    work = tmp_path / "work"
    monkeypatch.setattr(module, "WORK", work)
    monkeypatch.setattr(module, "STATE_PATH", work / "state.json")
    monkeypatch.setattr(module, "LOG_PATH", work / "easy_run.log")
    monkeypatch.setattr(module, "OUTPUT", tmp_path / "easy_run_output")
    monkeypatch.setattr(module, "ARCHIVE", tmp_path / "easy_run_output.tar")
    monkeypatch.setattr(module, "CONFIG_PATH", tmp_path / "easy_run.json")
    return module


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def make_data(folder: Path) -> None:
    """A tiny but complete data folder: tokenizer, corpora and JSONL inputs per split."""
    tok = Tokenizer(
        models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]")
    )
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    (folder / "tok").mkdir(parents=True)
    tok.save(str(folder / "tok" / "tokenizer.json"))
    for split, offset in (("train", 0), ("valid", 3), ("test", 7)):
        write_corpus(folder / split / "corpus",
                     [[8 + (i + j + offset) % 100 for j in range(30)] for i in range(40)])  # fmt: skip
        write_jsonl(folder / split / "it.jsonl", [
            {"messages": [{"role": "user", "content": f"w{i + offset} w{i + 1}"}],
             "response": f"w{i + 2}"} for i in range(20)])  # fmt: skip
        write_jsonl(folder / split / "pairs.jsonl", [
            {"query": f"w{i + offset} w{i + 1}", "positive": f"w{i + 1} w{i + offset}"}
            for i in range(12)])  # fmt: skip


def make_archive(tmp_path: Path) -> bytes:
    source = tmp_path / "source"
    make_data(source)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as bundle:
        for item in sorted(source.rglob("*")):
            bundle.add(item, arcname=str(item.relative_to(source)), recursive=False)
    return buffer.getvalue()


def make_config(payload: bytes, **budget) -> dict:
    return {
        "schema_version": 1,
        "data": {"url": "https://example.invalid/data.tar",
                 "sha256": hashlib.sha256(payload).hexdigest()},
        "inputs": {"tokenizer": "tok", "corpus": ["train/corpus=1"],
                   "instructions": "train/it.jsonl", "pairs": "train/pairs.jsonl"},
        "holdouts": {
            split: {"corpus": [f"{split}/corpus"], "instructions": f"{split}/it.jsonl",
                    "pairs": f"{split}/pairs.jsonl"}
            for split in ("valid", "test")
        },
        "evaluation": {"eval_corpus": "valid/corpus", "eval_pairs": "valid/pairs.jsonl"},
        "budget": {"total_hours": 0.05, "safety": 1.0, "reserve_hours": 0, "max_epochs": 2,
                   "shares": {"base-dense": 4, "base-moe": 2, "it-dense": 1, "it-moe": 1,
                              "embed": 1}, **budget},
        "train": {"batch_size": 4, "embed_batch_size": 4, "reasoning_examples": 0,
                  "validations_per_stage": 2},
        "recipes": {"base": {"span_end_prob": 0.5}},
    }  # fmt: skip


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, status: int):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def serve(monkeypatch, easy, payload: bytes, fail_first_after: int | None = None):
    """Answer urlopen with the payload, honoring Range; optionally break the first transfer."""
    calls = []

    def urlopen(request, timeout=None):
        header = request.headers.get("Range")
        start = int(header.split("=")[1].rstrip("-")) if header else 0
        calls.append(start)
        if fail_first_after is not None and len(calls) == 1:
            part = easy.WORK / "data.tar.part"
            part.write_bytes(payload[:fail_first_after])
            raise OSError("connection reset")
        return FakeResponse(payload[start:], 206 if start else 200)

    monkeypatch.setattr(easy.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(easy.time, "sleep", lambda seconds: None)
    return calls


def test_config_validation_names_the_problem(easy, tmp_path):
    with pytest.raises(SystemExit, match="easy_run.json is missing"):
        easy.load_config()
    config = make_config(b"x")
    config["budget"]["shares"].pop("embed")
    easy.CONFIG_PATH.write_text(json.dumps(config))
    with pytest.raises(SystemExit, match="budget.shares"):
        easy.load_config()
    config = make_config(b"x")
    config["data"]["url"] = "ftp://x"
    easy.CONFIG_PATH.write_text(json.dumps(config))
    with pytest.raises(SystemExit, match="data.url"):
        easy.load_config()


def test_paths_resolve_inside_the_data_folder_and_keep_weights(easy):
    config = make_config(b"x")
    base = easy.WORK / "data"
    assert easy.resolve(config, "train/corpus=0.6") == f"{base / 'train/corpus'}=0.6"
    assert easy.resolve(config, ["a", "b=2"]) == [str(base / "a"), f"{base / 'b'}=2"]
    assert easy.strip_weight("folder=0.25") == "folder"
    assert easy.strip_weight("odd=name") == "odd=name"


def test_download_resumes_and_extraction_refuses_unsafe_paths(
    easy, monkeypatch, tmp_path
):
    payload = make_archive(tmp_path)
    calls = serve(monkeypatch, easy, payload, fail_first_after=1000)
    easy.WORK.mkdir(parents=True)
    easy.download("https://example.invalid/data.tar", easy.WORK / "data.tar")
    assert calls == [0, 1000]
    assert (easy.WORK / "data.tar").read_bytes() == payload

    bad = io.BytesIO()
    with tarfile.open(fileobj=bad, mode="w") as bundle:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        bundle.addfile(info, io.BytesIO(b"x"))
    (tmp_path / "bad.tar").write_bytes(bad.getvalue())
    with pytest.raises(SystemExit, match="unsafe path"):
        easy.safe_extract(tmp_path / "bad.tar", tmp_path / "out")


def test_a_wrong_checksum_stops_before_unpacking(easy, monkeypatch, tmp_path):
    payload = make_archive(tmp_path)
    serve(monkeypatch, easy, payload)
    config = make_config(payload)
    config["data"]["sha256"] = "0" * 64
    easy.WORK.mkdir(parents=True)
    with pytest.raises(SystemExit, match="does not match"):
        easy.fetch_data(config, {})
    assert not (easy.WORK / "data").exists()


def test_step_time_uses_the_mean_after_warmup(easy, tmp_path):
    log = tmp_path / "train_log.jsonl"
    rows = [
        {"step": i, "seconds": t}
        for i, t in enumerate([0, 5, 5.5, 6, 7, 7.5, 8.5, 9, 10])
    ]
    write_jsonl(log, rows)
    # rows 3..8: (10 - 6) / (8 - 3)
    assert easy.seconds_per_step(log) == pytest.approx(0.8)


def test_allocation_caps_small_stages_and_gives_the_rest_to_base_dense(easy, tmp_path):
    make_data(easy.WORK / "data")
    config = make_config(b"x", total_hours=10)
    state = {"started_at": easy.now(),
             "seconds_per_step": dict.fromkeys(easy.STAGES, 1.0)}  # fmt: skip
    steps = easy.allocate_steps(config, state)
    # IT: 2 epochs of 20 examples at batch 4; embed: 2 epochs of 12 pairs.
    assert steps["it-dense"] == steps["it-moe"] == 10 and steps["embed"] == 6
    assert steps["base-moe"] == int(10 * 3600 * 2 / 9)
    assert sum(steps.values()) == pytest.approx(10 * 3600, abs=3)
    config["budget"]["total_hours"] = 0.01
    with pytest.raises(SystemExit, match="too few steps"):
        easy.allocate_steps(config, state)


def test_end_to_end_on_cpu_with_tiny_presets(easy, monkeypatch, tmp_path):
    """Download, calibrate, plan, train every stage, evaluate, export and package."""
    stages = {name: dict(spec) for name, spec in easy.STAGES.items()}
    stages["base-dense"]["preset"] = stages["base-moe"]["preset"] = "tiny"
    monkeypatch.setattr(easy, "STAGES", stages)
    monkeypatch.setattr(
        easy, "EVALUATE", {"base": ["mlm"], "it": [], "embed": ["retrieval"]}
    )
    monkeypatch.setattr(easy, "CALIBRATION_STEPS", 6)
    monkeypatch.setattr(easy, "MIN_BASE_STEPS", 2)
    monkeypatch.setattr(
        easy, "STEPS", [s for s in easy.STEPS if s[0] not in ("preflight", "install")]
    )
    payload = make_archive(tmp_path)
    serve(monkeypatch, easy, payload)
    monkeypatch.setattr(
        easy, "now", lambda: 0.0
    )  # the test's own runtime is not budget
    config = make_config(payload, total_hours=0.01)
    easy.CONFIG_PATH.write_text(json.dumps(config))

    easy.main()

    state = json.loads(easy.STATE_PATH.read_text())
    assert state["done"] == ["data", "calibrate", "plan", "train", "package"]
    assert set(state["seconds_per_step"]) == set(stages)
    plan = json.loads((easy.WORK / "plan.json").read_text())
    assert plan["stages"]["base-dense"]["recipe_config"].endswith("recipe-base.json")
    assert plan["stages"]["it-dense"]["init"] == "base-dense"
    output = easy.OUTPUT
    for name in (
        "nuri-base",
        "nuri-base-moe",
        "nuri-base-it",
        "nuri-base-moe-it",
        "nuri-base-embed",
    ):
        assert (output / "models" / name / "model.safetensors").is_file()
    assert (output / "eval" / "base-dense.json").is_file()
    assert (output / "logs" / "base-dense" / "train_log.jsonl").is_file()
    assert (output / "logs" / "base-dense" / "holdout_log.jsonl").is_file()
    with tarfile.open(easy.ARCHIVE) as bundle:
        assert "easy_run_output/models/nuri-base/model.safetensors" in bundle.getnames()
    assert not (easy.WORK / "data.tar").exists()  # deleted after unpacking

    # A second run finds every step finished and does nothing.
    easy.main()
    assert "finished steps: ['data'" in easy.LOG_PATH.read_text().splitlines()[-2]


def test_example_config_passes_validation_once_filled_in(easy):
    config = json.loads((ROOT / "configs" / "easy_run.example.json").read_text())
    config["data"]["url"] = "https://example.invalid/data.tar"
    config["data"]["sha256"] = "a" * 64
    easy.CONFIG_PATH.write_text(json.dumps(config))
    loaded = easy.load_config()
    assert set(loaded["budget"]["shares"]) == set(easy.STAGES)
    assert loaded["budget"]["total_hours"] == 168
