"""Turn scripts/evaluate.py reports into Markdown result tables (ARCHITECTURE.md §12).

Every table keeps the context a number needs: how many examples were
evaluated and how many were excluded, and the baseline the score should be
read against (answer-length-only accuracy for generation, a shared-token
TF-IDF ranking for retrieval). Nothing here compares Nuri with other models.

Usage
  # one Markdown file for all stages (the stage name is the JSON file name)
  python tools/results_report.py runs/release/eval/*.json --output .private/results.md

  # also put the stage's tables into an exported model card (README.md),
  # replacing an earlier evaluation section if there is one
  python tools/results_report.py runs/release/eval/it-dense.json \
      --model-card runs/release/release/nuri-base-it
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BEGIN = "<!-- nuri-evaluation:begin -->"
END = "<!-- nuri-evaluation:end -->"


def _num(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def _table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_num(cell) if not isinstance(cell, str) else cell for cell in row) + " |"
              for row in rows]
    return lines + [""]


def _generation_row(label: str, result: dict) -> list:
    return [label, result.get("exact_match"), result.get("examples"), result.get("excluded"),
            result.get("evaluation_coverage"), result.get("early_eos_rate"), result.get("missing_eos_rate"),
            result.get("answer_length_mae")]


GENERATION_HEADER = ["setting", "exact match", "examples", "excluded", "coverage",
                     "early EOS", "missing EOS", "answer length MAE"]


def mlm_section(report: dict) -> list[str]:
    m = report["mlm"]
    return ["### Masked tokens (held-out text)", ""] + _table(
        ["mask rate", "loss", "accuracy", "targets"],
        [[m.get("mask_rate"), m.get("loss"), m.get("accuracy"), m.get("targets")]])


def denoise_section(report: dict) -> list[str]:
    rows = [[r.get("mask_ratio"), r.get("steps"), r.get("loss"), r.get("accuracy"), r.get("targets")]
            for r in report["denoise"]]
    return ["### Span reconstruction (held-out text)", ""] + _table(
        ["mask ratio", "T", "loss", "accuracy", "targets"], rows)


def instructions_section(report: dict) -> list[str]:
    s = report["instructions"]
    rows, notes = [], []
    for key, label in (("predicted_length", "predicted length"), ("oracle_length", "oracle length")):
        if key in s:
            rows.append(_generation_row(label, s[key]))
            if s[key].get("length_only_accuracy") is not None:
                notes.append(f"Answer-length-only baseline (oracle length): "
                             f"{_num(s[key]['length_only_accuracy'])}.")
    lines = ["### Held-out instructions", "",
             f"Supplied examples: {_num(s.get('input_examples'))}. Answers longer than the canvas are "
             "excluded, never cut; coverage is the evaluated share.", ""]
    return lines + _table(GENERATION_HEADER, rows) + notes + ([""] if notes else [])


def reasoning_section(report: dict) -> list[str]:
    lines = ["### Synthetic reasoning tasks", ""]
    rows = []
    for split, result in report["reasoning"].items():
        for key, label in (("predicted_length", "predicted"), ("oracle_length", "oracle")):
            if key in result:
                row = _generation_row(f"{split}, {label} length", result[key])
                rows.append(row + [result[key].get("length_only_accuracy")])
    return lines + _table(GENERATION_HEADER + ["length-only baseline"], rows)


def curve_section(report: dict, key: str, label: str, title: str) -> list[str]:
    rows = [[r.get(label), r.get("exact_match"), r.get("mean_core_steps"), r.get("answer_length_mae")]
            for r in report[key]]
    return [f"### {title}", ""] + _table([label, "exact match", "mean core steps", "answer length MAE"], rows)


def interventions_section(report: dict) -> list[str]:
    rows = [[name, r.get("exact_match"), r.get("answer_length_mae")] for name, r in report["interventions"].items()]
    return ["### Latent-state interventions", "",
            "If zeroing or shuffling the reasoning state does not lower the score, the core is not being used.",
            ""] + _table(["state", "exact match", "answer length MAE"], rows)


def _retrieval_label(r: dict) -> str:
    doc = r.get("doc_mode") or r.get("mode")
    mode = r.get("mode") if doc == r.get("mode") else f"{r.get('mode')} -> {doc}"
    return f"{mode}, dim {r.get('dim') or 'full'}"


def retrieval_section(report: dict) -> list[str]:
    rows = [[_retrieval_label(r), r.get("recall@1"), r.get("recall@5"), r.get("recall@10"), r.get("mrr"),
             r.get("pairs")] for r in report["retrieval"]]
    baseline = report.get("retrieval_lexical_baseline")
    if baseline:
        rows.append(["TF-IDF (no model)", baseline.get("recall@1"), baseline.get("recall@5"),
                     baseline.get("recall@10"), baseline.get("mrr"), baseline.get("pairs")])
    lines = ["### Retrieval (held-out pairs)", "",
             "Each query is ranked against every held-out document. The TF-IDF row scores shared "
             "tokens only; an embedding result means something only clearly above it.", ""]
    lines += _table(["setting", "R@1", "R@5", "R@10", "MRR", "pairs"], rows)
    overlap = report.get("retrieval_overlap")
    if overlap:
        lines += overlap_lines(overlap)
    return lines


def overlap_lines(overlap: dict) -> list[str]:
    """Recall@1 per lexical-overlap stratum; every stratum keeps the full candidate pool."""
    def full_dim(models: list[dict], mode: str):
        for m in models:
            if m.get("mode") == mode and m.get("doc_mode", mode) == mode and m.get("dim") is None:
                return m.get("recall@1")
        return None

    rows = []
    for bucket in overlap.get("buckets", []):
        if not bucket.get("queries"):
            continue
        rows.append([bucket["name"], bucket["queries"], full_dim(bucket.get("model", []), "fast"),
                     full_dim(bucket.get("model", []), "reasoning"),
                     (bucket.get("lexical_tfidf") or {}).get("recall@1")])
    if not rows:
        return []
    return ["Recall@1 by query-document token overlap (Jaccard). Semantic gains show in the low-overlap "
            "rows; high-overlap rows mostly test word matching.", ""] + _table(
        ["overlap stratum", "queries", "fast", "reasoning", "TF-IDF"], rows)


SECTIONS = [
    ("mlm", mlm_section),
    ("denoise", denoise_section),
    ("instructions", instructions_section),
    ("reasoning", reasoning_section),
    ("depth_curve", lambda r: curve_section(r, "depth_curve", "steps", "Reasoning depth curve (T)")),
    ("rounds_curve", lambda r: curve_section(r, "rounds_curve", "rounds", "Refinement rounds curve (S)")),
    ("interventions", interventions_section),
    ("retrieval", retrieval_section),
]


def stage_body(report: dict) -> str:
    """The ### sections for one report, without a stage heading."""
    lines = []
    for key, render in SECTIONS:
        if report.get(key):
            lines += render(report)
    if not lines:
        lines = ["No supported evaluation results in this report.", ""]
    return "\n".join(lines).rstrip() + "\n"


def stage_markdown(name: str, report: dict) -> str:
    return f"## {name}\n\n" + stage_body(report)


def update_model_card(card: Path, body: str) -> None:
    """Insert or replace the evaluation section of an exported README.md (before "## License")."""
    text = card.read_text(encoding="utf-8")
    block = f"{BEGIN}\n## Evaluation\n\n{body}{END}\n"
    if BEGIN in text and END in text:
        start, end = text.index(BEGIN), text.index(END) + len(END)
        text = text[:start] + block.rstrip("\n") + text[end:]
    elif "\n## License" in text:
        text = text.replace("\n## License", "\n" + block + "\n## License", 1)
    else:
        text = text.rstrip("\n") + "\n\n" + block
    card.write_text(text, encoding="utf-8")


def main(argv=None) -> str:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reports", nargs="+", help="JSON files written by scripts/evaluate.py --output")
    p.add_argument("--output", help="write the combined Markdown here (default: print it)")
    p.add_argument("--model-card", help="exported release folder whose README.md gets the evaluation "
                                        "section (use with exactly one report)")
    args = p.parse_args(argv)
    if args.model_card and len(args.reports) != 1:
        p.error("--model-card takes exactly one report")

    reports = [(Path(path).stem, json.loads(Path(path).read_text(encoding="utf-8"))) for path in args.reports]
    markdown = "\n".join(stage_markdown(name, report) for name, report in reports)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    if args.model_card:
        update_model_card(Path(args.model_card) / "README.md", stage_body(reports[0][1]))
    return markdown


if __name__ == "__main__":
    main()
