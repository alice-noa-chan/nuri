import importlib.util
import json
import sys
from pathlib import Path

import torch

from nuri import data
from nuri.checkpoint import save_pretrained
from nuri.config import tiny_config
from nuri.model import NuriModel


def load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "readout_probe.py"
    spec = importlib.util.spec_from_file_location("readout_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = load_script()


def write_inputs(tmp_path):
    torch.manual_seed(0)
    model = NuriModel(tiny_config())
    save_pretrained(model, tmp_path / "init", dtype=torch.float32)
    for name, offset in (("train", 0), ("valid", 7)):
        documents = [
            [3 + (row * 5 + i + offset) % 120 for i in range(30 + row % 10)]
            for row in range(40)
        ]
        data.write_corpus(tmp_path / name, documents)
    return model


def run(tmp_path, *extra):
    argv = [
        "--init", str(tmp_path / "init"), "--train", str(tmp_path / "train"),
        "--valid", str(tmp_path / "valid"), "--out", str(tmp_path / "out"),
        "--steps", "3", "--batch-size", "2", "--pool", "6", "--eval-examples", "4",
        "--input-len", "16", "--span-len", "8", "--threads", "1", *extra,
    ]  # fmt: skip
    return script.main(argv)


def test_probe_trains_every_arm_and_reports_both_eval_sets(tmp_path):
    write_inputs(tmp_path)
    results = run(tmp_path)
    assert list(results["arms"]) == ["initial", "baseline", "memory", "memory-eos"]
    for arm, entry in results["arms"].items():
        evaluation = entry["evaluation"]
        assert set(evaluation) == {"mid", "end"}
        end = evaluation["end"]["s4"]
        assert 0.0 <= end["eos_correct"] <= 1.0
        assert end["eos_examples"] == 4  # every "end" target carries EOS
        mid = evaluation["mid"]["s1"]
        assert mid.get("eos_examples", 0) <= 4
        assert evaluation["mid"]["full_mask_ce"] > 0
        if arm != "initial":
            assert entry["training"]["final_loss"] > 0
    saved = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert "init" not in saved["settings"] and "train" not in saved["settings"]
    assert "seconds" in saved


def test_only_the_readout_is_trained(tmp_path):
    initial = write_inputs(tmp_path)
    model = script.load_pretrained(tmp_path / "init")
    script.add_memory_positions(model)
    args = script.parse_args(
        ["--init", "x", "--train", "x", "--valid", "x", "--steps", "2", "--batch-size", "2",
         "--pool", "4", "--input-len", "16", "--span-len", "8"]
    )  # fmt: skip
    corpus = data.open_corpus(str(tmp_path / "train"))
    pairs = script.make_pairs(corpus, 4, script.random.Random(0), args, 0.0, 2)
    cached = script.encode_pairs(model, pairs)
    script.train_readout(model, pairs, cached, args, seed=0)
    before = initial.state_dict()
    for name, tensor in model.state_dict().items():
        if name.startswith("readout."):
            continue
        assert torch.equal(tensor, before[name]), name
    assert model.readout.memory_positions.from_start.weight.any()


def test_end_pairs_always_carry_eos(tmp_path):
    write_inputs(tmp_path)
    args = script.parse_args(
        [
            "--init",
            "x",
            "--train",
            "x",
            "--valid",
            "x",
            "--input-len",
            "16",
            "--span-len",
            "8",
        ]
    )
    corpus = data.open_corpus(str(tmp_path / "valid"))
    pairs = script.make_pairs(corpus, 10, script.random.Random(0), args, 1.0, 2)
    for prefix, target in pairs:
        assert target[-1] == 2 and 2 not in target[:-1]
        assert 1 <= len(prefix) <= 16 and len(target) <= 9
