import importlib.util
import json
from pathlib import Path


def load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "ablate.py"
    spec = importlib.util.spec_from_file_location("ablate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_variant_trains_and_reports(tmp_path):
    script = load_script()
    results = script.main(["--out", str(tmp_path), "--steps", "3", "--batch-size", "4", "--train-examples", "40",
                           "--eval-examples", "6", "--vocab-size", "400", "--device", "cpu"])
    assert [r["variant"] for r in results] == list(script.VARIANTS)
    for r in results:
        assert 0.0 <= r["exact_match_predicted"] <= 1.0 and 0.0 <= r["length_only_baseline"] <= 1.0
    by_name = {r["variant"]: r for r in results}
    assert by_name["gqa"]["parameters"] < by_name["baseline"]["parameters"]
    assert by_name["core-moe"]["parameters"] > by_name["baseline"]["parameters"]
    saved = json.loads((tmp_path / "ablation.json").read_text(encoding="utf-8"))
    assert len(saved["results"]) == len(script.VARIANTS)
