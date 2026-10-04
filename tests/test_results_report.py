import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import results_report as rr  # noqa: E402


def generation(exact, examples=10, excluded=0, length_only=None):
    r = {"exact_match": exact, "examples": examples, "excluded": excluded,
         "evaluation_coverage": examples / (examples + excluded), "early_eos_rate": 0.1,
         "missing_eos_rate": 0.0, "answer_length_mae": 1.5}
    if length_only is not None:
        r["length_only_accuracy"] = length_only
    return r


def test_every_section_renders_with_its_context():
    report = {
        "mlm": {"mask_rate": 0.15, "loss": 3.2, "accuracy": 0.41, "targets": 1000},
        "denoise": [{"mask_ratio": 0.5, "steps": 4, "loss": 4.0, "accuracy": 0.3, "targets": 500}],
        "instructions": {"input_examples": 12, "predicted_length": generation(0.2, 8, 4),
                         "oracle_length": generation(0.3, 8, 4, length_only=0.05)},
        "reasoning": {"shared": {"oracle_length": generation(0.5, length_only=0.2)}},
        "depth_curve": [{"steps": 1, "exact_match": 0.1}, {"steps": 8, "exact_match": 0.4}],
        "rounds_curve": [{"rounds": 1, "exact_match": 0.2}],
        "interventions": {"normal": {"exact_match": 0.4}, "zero_r": {"exact_match": 0.1}},
        "retrieval": [{"mode": "fast", "doc_mode": "fast", "dim": None, "recall@1": 0.7, "recall@5": 0.9,
                       "recall@10": 0.95, "mrr": 0.8, "pairs": 100}],
        "retrieval_lexical_baseline": {"recall@1": 0.6, "recall@5": 0.9, "recall@10": 0.95, "mrr": 0.7,
                                       "pairs": 100},
    }
    text = rr.stage_markdown("it-dense", report)
    for heading in ("Masked tokens", "Span reconstruction", "Held-out instructions", "Synthetic reasoning",
                    "depth curve", "rounds curve", "interventions", "Retrieval"):
        assert heading in text, heading
    assert "| predicted length | 0.200 | 8 | 4 | 0.667 |" in text  # coverage stays next to the score
    assert "Answer-length-only baseline (oracle length): 0.050." in text
    assert "| TF-IDF (no model) | 0.600 |" in text
    assert text.isascii()  # prints on any console


def test_overlap_strata_from_the_real_report_module():
    from nuri.config import tiny_config
    from nuri.model import NuriModel
    from nuri.retrieval_report import overlap_stratified_retrieval

    torch.manual_seed(0)
    model = NuriModel(tiny_config()).eval()
    pairs = [([5 + i, 6 + i, 7 + i], [5 + i, 6 + i, 30 + i]) for i in range(12)]
    overlap = overlap_stratified_retrieval(model, pairs)
    text = "\n".join(rr.overlap_lines(overlap))
    assert "overlap stratum" in text and "TF-IDF" in text
    assert sum(1 for line in text.splitlines() if line.startswith("| jaccard")) >= 1


def test_model_card_section_is_inserted_once_and_replaced(tmp_path):
    card = tmp_path / "README.md"
    card.write_text("# nuri-base-it\n\nintro\n\n## License\n\nMIT\n", encoding="utf-8")
    rr.update_model_card(card, "### A\n\nfirst\n")
    rr.update_model_card(card, "### A\n\nsecond\n")
    text = card.read_text(encoding="utf-8")
    assert text.count("## Evaluation") == 1 and "second" in text and "first" not in text
    assert text.index("## Evaluation") < text.index("## License")


def test_cli_writes_markdown_and_updates_the_card(tmp_path):
    report = tmp_path / "embed.json"
    report.write_text(json.dumps({"retrieval": [{"mode": "fast", "dim": 64, "recall@1": 0.5, "pairs": 10}]}),
                      encoding="utf-8")
    release = tmp_path / "nuri-base-embed"
    release.mkdir()
    (release / "README.md").write_text("# card\n\n## License\n\nMIT\n", encoding="utf-8")
    rr.main([str(report), "--output", str(tmp_path / "out.md"), "--model-card", str(release)])
    assert "## embed" in (tmp_path / "out.md").read_text(encoding="utf-8")
    assert "fast, dim 64" in (release / "README.md").read_text(encoding="utf-8")
    assert "No supported evaluation results" in rr.stage_markdown("x", {"checkpoint": "c"})
