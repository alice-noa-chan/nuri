"""Measure bounded TRAIN batch policies on CPU, without loading a model.

Keep the configuration and output in .private/. Corpus paths are supplied only
by that configuration. Counts describe sampled batch-builder behavior, never
historical training, gradient contributions, model quality or GPU throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from nuri.config import NuriConfig  # noqa: E402
from nuri.data import (TokenCorpus, build_mlm_batch, build_packed_mlm_batch,
                       build_span_batch, has_mlm_target)  # noqa: E402


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def validate(spec):
    """Bound work and require deliberately declared TRAIN-only resources."""
    allowed = {"schema_version", "model_config", "corpora", "seed", "batches",
               "batch_size", "pack_mlm", "bucket_span", "mlm_rate", "min_span"}
    if (not isinstance(spec, dict) or set(spec) - allowed
            or type(spec.get("schema_version")) is not int or spec["schema_version"] != 1):
        raise ValueError("unsupported audit schema or fields")
    for name, default, minimum, maximum in (("batches", 32, 1, 512), ("batch_size", 2, 1, 64),
                                          ("seed", 0, 0, 2**31 - 1), ("min_span", 16, 1, 4096)):
        value = spec.get(name, default)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"invalid {name}")
    for name in ("pack_mlm", "bucket_span"):
        if name in spec and type(spec[name]) is not bool:
            raise ValueError(f"invalid {name}")
    rate = spec.get("mlm_rate", 0.3)
    if type(rate) not in (float, int) or not math.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError("invalid mlm_rate")
    if not isinstance(spec.get("model_config"), str) or not spec["model_config"].strip():
        raise ValueError("model_config must be a path")
    sources = spec.get("corpora")
    if not isinstance(sources, list) or not sources or len(sources) > 3:
        raise ValueError("corpora must declare one to three TRAIN resources")
    languages = []
    for source in sources:
        if (not isinstance(source, dict) or set(source) != {"language", "path", "weight", "split"}
                or source["split"] != "train"):
            raise ValueError("every resource must explicitly declare split=train")
        if source["language"] not in ("ko", "en", "ja"):
            raise ValueError("unknown language")
        languages.append(source["language"])
        if not isinstance(source["path"], str) or not source["path"].strip():
            raise ValueError("invalid corpus path")
        weight = source["weight"]
        if type(weight) not in (float, int) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("invalid corpus weight")
    if len(set(languages)) != len(languages):
        raise ValueError("language resources must be unique")


def counts():
    return {"sampled_documents": 0, "input_positions": 0, "canvas_positions": 0,
            "supervised_targets": 0, "rows": 0}


def mixed_mlm(corpora, languages, weights, config, spec):
    """Count actual segment positions; a packed row can contain many languages."""
    rng = random.Random(spec.get("seed", 0))
    generator = torch.Generator().manual_seed(spec.get("seed", 0))
    report = {language: {**counts(), "draws": 0, "rejected_documents": 0} for language in languages}
    batch_size = spec.get("batch_size", 2)
    total_rows = 0
    for _ in tqdm(range(spec.get("batches", 32)), desc="CPU mixed MLM batches"):
        accepted = []

        def draw():
            for _ in range(100):
                index = rng.choices(range(len(corpora)), weights)[0]
                language = languages[index]
                report[language]["draws"] += 1
                doc = corpora[index].random_document(rng)
                if has_mlm_target(doc, config):
                    accepted.append(language)
                    report[language]["sampled_documents"] += 1
                    return doc
                report[language]["rejected_documents"] += 1
            raise ValueError("too few usable TRAIN documents")

        if spec.get("pack_mlm", True):
            batch = build_packed_mlm_batch(draw, batch_size, config, rng, generator,
                                          rate=spec.get("mlm_rate", 0.3))
            cursor = 0
            for row in range(len(batch["input_ids"])):
                segments = batch["segments"][row]
                for segment in range(1, int(segments.max()) + 1):
                    language = accepted[cursor]
                    cursor += 1
                    positions = (segments == segment) & batch["input_valid"][row]
                    report[language]["input_positions"] += int(positions.sum())
                    report[language]["supervised_targets"] += int((positions & batch["target_mask"][row]).sum())
            if cursor != len(accepted):
                raise ValueError("packed document/segment attribution mismatch")
        else:
            batch = build_mlm_batch([draw() for _ in range(batch_size)], config, rng, generator,
                                    rate=spec.get("mlm_rate", 0.3))
            if len(batch["input_ids"]) != len(accepted):
                raise ValueError("unexpected usable MLM row drop")
            for row, language in enumerate(accepted):
                report[language]["rows"] += 1
                report[language]["input_positions"] += int(batch["input_valid"][row].sum())
                report[language]["supervised_targets"] += int(batch["target_mask"][row].sum())
        total_rows += len(batch["input_ids"])
    for block in report.values():
        if spec.get("pack_mlm", True):
            block["rows"] = None  # A multilingual packed row has no single language.
    return {"total_rows": total_rows, "by_language": report}


def isolated_span(corpus, config, spec, seed):
    """Observe one-language bucketed span counts without inferring mixed shares."""
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    report = counts()
    size = spec.get("batch_size", 2)
    bucket = (size * config.max_input_len, size * config.max_response_len, size * 4)
    offered = size * 8 if spec.get("bucket_span", True) else size
    for _ in tqdm(range(spec.get("batches", 32)), desc="CPU isolated span batches"):
        docs = corpus.sample(offered, rng, usable=lambda doc: len(doc) >= 2)
        batch = build_span_batch(docs, config, rng, generator, min_span=spec.get("min_span", 16),
                                 bucket=bucket if spec.get("bucket_span", True) else None)
        report["sampled_documents"] += len(docs)
        report["rows"] += len(batch["input_ids"])
        report["input_positions"] += int(batch["input_valid"].sum())
        report["canvas_positions"] += int(batch["canvas_valid"].sum())
        report["supervised_targets"] += int(batch["loss_mask"].sum())
    report["offered_eligible_documents"] = report.pop("sampled_documents")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    path, output = Path(args.config).resolve(), Path(args.output).resolve()
    if ".private" not in path.parts or ".private" not in output.parts:
        raise ValueError("configuration and receipt must stay inside .private")
    spec = json.loads(path.read_text(encoding="utf-8"))
    validate(spec)
    if output.exists():
        raise ValueError("use a fresh output path")

    def resolve(value):
        item = Path(value)
        return item.resolve() if item.is_absolute() else (path.parent / item).resolve()

    model_config = resolve(spec["model_config"])
    config = NuriConfig.load(model_config)
    if config.max_input_len > 4096 or config.max_response_len > 4096:
        raise ValueError("batch geometry exceeds the bounded CPU audit")
    if spec.get("min_span", 16) > config.max_response_len:
        raise ValueError("min_span exceeds the configured response length")
    folders = [resolve(source["path"]) for source in spec["corpora"]]
    if len(set(folders)) != len(folders):
        raise ValueError("corpus resources must not alias")
    resources = [path, model_config, *[folder / name for folder in folders
                                      for name in ("tokens.bin", "offsets.npy")]]
    print("Binding TRAIN inputs before CPU batch simulation", flush=True)
    bindings = {str(item): digest(item) for item in resources}
    before = {str(item): (item.stat().st_size, item.stat().st_mtime_ns) for item in resources}
    sources = [Path(__file__).resolve(), ROOT / "nuri" / "data.py", ROOT / "nuri" / "corruption.py",
               ROOT / "nuri" / "config.py"]
    source_binding = {str(item.relative_to(ROOT)): digest(item) for item in sources}
    corpora = [TokenCorpus(folder) for folder in folders]
    if any(len(corpus) == 0 for corpus in corpora):
        raise ValueError("TRAIN corpus is empty")
    torch.set_num_threads(2)
    languages = [source["language"] for source in spec["corpora"]]
    report = {"schema_version": 1, "status": "cpu_batch_policy_simulation",
              "input_sha256": bindings, "runtime_source_sha256": source_binding,
              "configuration": spec, "mlm_mixed": mixed_mlm(corpora, languages,
                  [source["weight"] for source in spec["corpora"]], config, spec),
              "span_isolated_by_language": {language: isolated_span(corpus, config, spec,
                  spec.get("seed", 0) + 1009 * (index + 1))
                  for index, (language, corpus) in enumerate(zip(languages, corpora))},
              "model_loaded": False, "optimizer_updates": 0, "gpu_used": False, "test_used": False,
              "limits": ["TRAIN membership is caller-declared, not established by this utility.",
                         "Counts measure these simulated batch policies, not historical training exposure.",
                         "Isolated span batches cannot establish mixed-language bucket selection shares.",
                         "Source weights are document weights, never target, gradient or compute shares."]}
    after = {str(item): (item.stat().st_size, item.stat().st_mtime_ns) for item in resources}
    if (before != after or bindings != {str(item): digest(item) for item in resources}
            or source_binding != {str(item.relative_to(ROOT)): digest(item) for item in sources}):
        raise ValueError("inputs or runtime sources changed during the simulation")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("CPU batch-policy receipt complete; no model updates", flush=True)


if __name__ == "__main__":
    main()
