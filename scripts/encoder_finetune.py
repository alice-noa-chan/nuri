"""Run one frozen-protocol TRAIN/VALID encoder-adaptation candidate on a fresh copy.

This is a separate diagnostic worker. It never reads TEST or changes an input
checkpoint. A private outer controller must equalize LR/seed search budgets.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nuri.moe import MoEFFN  # noqa: E402
from scripts import encoder_features as features  # noqa: E402
from scripts.encoder_probe import classification_metrics, select_candidate  # noqa: E402


MAX_ROWS = 20000
MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_ENCODER_PARAMETERS = 250000000


def number(value, name, *, positive=False):
    if type(value) not in (float, int):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number")
    return value


def bounded_integer(value, name, maximum):
    features.positive_integer(value, name)
    if value > maximum:
        raise ValueError(f"{name} exceeds the bounded worker limit {maximum}")
    return value


def validate_protocol(data, learning_rate, seed):
    required = {"schema_version", "kind", "learning_rates", "seeds", "epochs", "batch_size",
                "max_length", "label_count", "task_type"}
    if (not isinstance(data, dict) or required - data.keys() or data.keys() - required - {"optimizer"}
            or type(data.get("schema_version")) is not int or data["schema_version"] != 1
            or data.get("kind") != "encoder_finetune"):
        raise ValueError("Adaptation protocol needs schema_version 1, encoder_finetune and supported fields")
    rates, seeds = data["learning_rates"], data["seeds"]
    if not isinstance(rates, list) or not 1 <= len(rates) <= 8:
        raise ValueError("Declare 1..8 learning rates")
    rates = [number(rate, "learning rate", positive=True) for rate in rates]
    if rates != sorted(set(rates)):
        raise ValueError("Learning rates must be strictly increasing")
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 8
            or any(type(value) is not int or not 0 <= value < 2 ** 32 for value in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Declare 1..8 unique integer seeds in 0..2**32-1")
    learning_rate = number(learning_rate, "candidate learning rate", positive=True)
    if learning_rate not in rates or type(seed) is not int or seed not in seeds:
        raise ValueError("The candidate LR and seed must be explicitly declared in the protocol")
    result = copy.deepcopy(data)
    result["learning_rates"] = rates
    for name, maximum in (("epochs", 20), ("batch_size", 64), ("max_length", 128), ("label_count", 1024)):
        bounded_integer(result[name], name, maximum)
    if result["label_count"] < 2 or result["task_type"] not in ("single", "pair"):
        raise ValueError("Declare at least two classes and single or pair task geometry")
    optimizer = result.get("optimizer", {})
    if not isinstance(optimizer, dict) or optimizer.keys() - {"weight_decay", "grad_clip"}:
        raise ValueError("Unsupported optimizer setting")
    result["optimizer"] = {"weight_decay": number(optimizer.get("weight_decay", 0.01), "weight_decay"),
                           "grad_clip": number(optimizer.get("grad_clip", 1.0), "grad_clip", positive=True)}
    return result


def load_inputs(manifest_path, protocol):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed = {"schema_version", "model", "inputs", "label_count", "protocol", "assessment_protocol",
               "model_artifacts_sha256", "tokenizer_artifacts_sha256"}
    if (not isinstance(manifest, dict) or set(manifest) - allowed
            or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1):
        raise ValueError("Unsupported adaptation manifest schema")
    if not isinstance(manifest.get("inputs"), dict) or set(manifest["inputs"]) != {"train", "valid"}:
        raise ValueError("Declare exactly TRAIN and VALID inputs; TEST is reserved")
    if type(manifest.get("label_count")) is not int or manifest["label_count"] != protocol["label_count"]:
        raise ValueError("Manifest and protocol class universes differ")
    geometry = manifest.get("protocol", {})
    allowed_geometry = {"batch_size", "max_length", "pooling", "pair_features", "dtype", "device"}
    if not isinstance(geometry, dict) or geometry.keys() - allowed_geometry:
        raise ValueError("Unknown shared feature geometry")
    fixed = {"pooling": "masked_mean_content", "pair_features": "concat_u_v_absdiff_product", "dtype": "float32",
             "device": "cpu"}
    if any(name in geometry and geometry[name] != value for name, value in fixed.items()):
        raise ValueError("Preserve the shared content-mean feature geometry")
    if geometry.get("max_length", 128) != protocol["max_length"]:
        raise ValueError("Adaptation must use the same declared visible-token limit")
    if "max_length" in geometry:
        features.positive_integer(geometry["max_length"], "shared max_length")
    if "batch_size" in geometry:
        features.positive_integer(geometry["batch_size"], "shared batch_size")
    records, rows, seen_ids, seen_paths = [], {}, set(), set()
    for split in ("train", "valid"):
        declared = manifest["inputs"][split]
        if not isinstance(declared, dict) or set(declared) != {"path", "sha256"}:
            raise ValueError("Input bindings require path and SHA256")
        path = features.resolve_path(manifest_path.parent, declared["path"])
        if path in seen_paths or path.stat().st_size > MAX_INPUT_BYTES:
            raise ValueError("Input paths must be distinct and each at most 32 MiB")
        seen_paths.add(path)
        digest = features.file_digest(path)
        if digest != declared["sha256"]:
            raise ValueError("Input hash mismatch")
        rows[split] = features.read_rows(path, split, protocol["label_count"])
        if len(rows[split]) > MAX_ROWS:
            raise ValueError("Input row count exceeds the bounded worker limit")
        ids = {row["id"] for row in rows[split]}
        if ids & seen_ids:
            raise ValueError("TRAIN and VALID row IDs must be disjoint")
        seen_ids.update(ids)
        if ("text_pair" in rows[split][0]) != (protocol["task_type"] == "pair"):
            raise ValueError("Input rows and protocol task geometry differ")
        records.append({"name": split, "path": str(path), "bytes": path.stat().st_size, "sha256": digest})
    if manifest.get("assessment_protocol") is not None:
        declared = manifest["assessment_protocol"]
        if not isinstance(declared, dict) or set(declared) != {"path", "sha256"}:
            raise ValueError("Assessment protocol needs path and SHA256")
        path = features.private_path(features.resolve_path(manifest_path.parent, declared["path"]))
        digest = features.file_digest(path)
        if digest != declared["sha256"]:
            raise ValueError("Assessment protocol hash mismatch")
        records.append({"name": "assessment_protocol", "path": str(path), "bytes": path.stat().st_size, "sha256": digest})
    spec = manifest.get("model")
    model_keys = {"backend", "path_or_id", "revision", "tokenizer_path_or_id", "tokenizer_revision",
                  "random_initialization_seed"}
    if (not isinstance(spec, dict) or set(spec) - model_keys or spec.get("backend") not in ("nuri", "transformers")
            or not isinstance(spec.get("revision"), str) or not spec["revision"]):
        raise ValueError("Declare a supported local encoder and explicit pinned revision")
    for name in ("path_or_id", "tokenizer_path_or_id"):
        value = spec.get(name, spec.get("path_or_id"))
        if not isinstance(value, str) or not value:
            raise ValueError("Model and tokenizer need local paths")
        path = features.resolve_path(manifest_path.parent, value)
        if not (path.is_dir() or (name == "tokenizer_path_or_id" and spec["backend"] == "nuri" and path.is_file())):
            raise ValueError("Prepare model and tokenizer locally; this worker never downloads them")
    return manifest, rows, records


def verify_records(records):
    for record in records:
        path = Path(record["path"])
        if path.stat().st_size != record["bytes"] or features.file_digest(path) != record["sha256"]:
            raise ValueError("Bound source or input bytes changed during adaptation")


def runtime_source_records():
    root = Path(__file__).resolve().parent.parent
    paths = [Path(__file__).resolve(), Path(features.__file__).resolve(),
             Path(sys.modules[classification_metrics.__module__].__file__).resolve()]
    paths += sorted((root / "nuri").glob("*.py"))
    return [{"name": str(path.relative_to(root)).replace("\\", "/"), "path": str(path),
             "bytes": path.stat().st_size, "sha256": features.file_digest(path)} for path in paths]


def pooled_batch(model, tokenizer, special, backend, rows, max_length, device):
    pooled, counts = [], {"fields": 0, "valid_input_positions": 0, "content_positions": 0, "truncated_fields": 0}
    fields = ("text", "text_pair") if "text_pair" in rows[0] else ("text",)
    for field in fields:
        inputs, content, stats = features.tokenize_texts(backend, tokenizer, special, model,
                                                        [row[field] for row in rows], max_length)
        if any(stats["truncated"]):
            raise ValueError("Common-visible input needs tokenizer truncation; prepare a matched prefix first")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        hidden = model(**inputs) if backend == "nuri" else model(**inputs, return_dict=True).last_hidden_state
        pooled.append(features.content_mean(hidden, content.to(device)))
        counts["fields"] += len(rows)
        counts["valid_input_positions"] += sum(stats["retained_tokens"])
        counts["content_positions"] += sum(stats["retained_content_tokens"])
        counts["truncated_fields"] += sum(stats["truncated"])
    if len(pooled) == 1:
        return pooled[0], counts
    u, v = pooled
    return torch.cat((u, v, (u - v).abs(), u * v), dim=1), counts


def audit_tokenization(model, tokenizer, special, backend, rows, protocol, *, progress=False):
    """Check all supplied fields in bounded batches before any learning."""
    observed = {}
    for split, values in rows.items():
        counts = {"fields": 0, "valid_input_positions": 0, "content_positions": 0, "truncated_fields": 0}
        fields = ("text", "text_pair") if protocol["task_type"] == "pair" else ("text",)
        for start in tqdm(range(0, len(values), protocol["batch_size"]), desc=f"token audit | {split}",
                          unit="batch", disable=not progress):
            batch = values[start:start + protocol["batch_size"]]
            for field in fields:
                _, content, stats = features.tokenize_texts(backend, tokenizer, special, model,
                                                            [row[field] for row in batch], protocol["max_length"])
                if any(stats["truncated"]):
                    raise ValueError("Common-visible input needs tokenizer truncation; prepare a matched prefix first")
                if not bool(content.any(dim=1).all()):
                    raise ValueError("Each supplied text field needs at least one content token")
                add_counts(counts, {"fields": len(batch), "valid_input_positions": sum(stats["retained_tokens"]),
                                    "content_positions": sum(stats["retained_content_tokens"]), "truncated_fields": 0})
        observed[split] = counts
    return observed


def add_counts(total, values):
    for name, value in values.items():
        total[name] = total.get(name, 0) + value


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def verify_finite_state(model, head, optimizer):
    """Bound temporary checks to one million values, without cloning parameters."""
    tensors = list(model.parameters()) + list(head.parameters())
    tensors += [value for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value)]
    for tensor in tensors:
        values = tensor.detach().reshape(-1)
        for start in range(0, values.numel(), 1024 * 1024):
            if not bool(torch.isfinite(values[start:start + 1024 * 1024]).all()):
                raise ValueError("Nonfinite encoder, head or optimizer state")


@torch.no_grad()
def assess(model, head, tokenizer, special, backend, rows, protocol, device, *, progress=False, split="valid"):
    model.eval()
    head.eval()
    predictions, loss_sum, counts = [], 0.0, {}
    for start in tqdm(range(0, len(rows), protocol["batch_size"]), desc=f"assess | {split}",
                      unit="batch", disable=not progress):
        batch = rows[start:start + protocol["batch_size"]]
        pooled, observed = pooled_batch(model, tokenizer, special, backend, batch, protocol["max_length"], device)
        logits = head(pooled)
        labels = torch.tensor([row["label"] for row in batch], device=device)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Nonfinite assessment logits")
        loss = nn.functional.cross_entropy(logits, labels)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite assessment loss")
        loss_sum += float(loss) * len(batch)
        predictions.extend(logits.argmax(1).cpu().tolist())
        add_counts(counts, observed)
    metrics = classification_metrics(np.array([row["label"] for row in rows]), np.array(predictions), protocol["label_count"])
    metrics.update(loss=loss_sum / len(rows), counts=counts, predictions=predictions)
    return metrics


def run_candidate(manifest_path, protocol_path, output_dir, *, learning_rate, seed, device="cpu",
                  threads=2, tensorboard_dir=None, progress=True):
    manifest_path, protocol_path, output_dir = map(features.private_path, (manifest_path, protocol_path, output_dir))
    source_records = runtime_source_records()
    if output_dir.exists():
        raise FileExistsError("Use a fresh private output directory")
    bounded_integer(threads, "threads", 2)
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Select cpu or an available cuda device explicitly")
    device = torch.device(device)
    protocol = validate_protocol(json.loads(protocol_path.read_text(encoding="utf-8")), learning_rate, seed)
    learning_rate = float(learning_rate)
    manifest, rows, records = load_inputs(manifest_path, protocol)
    records += [{"name": name, "path": str(path), "bytes": path.stat().st_size, "sha256": features.file_digest(path)}
                for name, path in (("manifest", manifest_path), ("adaptation_protocol", protocol_path))]
    event_path = features.private_path(tensorboard_dir) if tensorboard_dir else None
    spec = manifest["model"]
    for name in ("path_or_id", "tokenizer_path_or_id"):
        resource = features.resolve_path(manifest_path.parent, spec.get(name, spec["path_or_id"]))
        folder = resource if resource.is_dir() else resource.parent
        if folder in output_dir.parents or (event_path is not None and (folder == event_path or folder in event_path.parents)):
            raise ValueError("Keep result/event outputs outside source checkpoint/tokenizer folders")
    if event_path is not None and event_path.exists():
        raise FileExistsError("Use a fresh private TensorBoard directory")
    writer = None
    if event_path is not None:
        from torch.utils.tensorboard import SummaryWriter
    torch.set_num_threads(threads)
    random.seed(seed)
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    features.write_json(output_dir / "IN_PROGRESS.json", {"status": "running", "protocol": protocol,
                                                          "learning_rate": learning_rate, "seed": seed})
    began = time.perf_counter()
    try:
        loader = features.load_nuri if spec["backend"] == "nuri" else features.load_transformers
        original, tokenizer, special, model_records, tokenizer_records, details = loader(spec, manifest_path.parent)
        if any(isinstance(module, MoEFFN) for module in original.modules()):
            raise ValueError("This matched worker currently accepts dense encoders only")
        parameters = sum(parameter.numel() for parameter in original.parameters())
        if parameters > MAX_ENCODER_PARAMETERS or protocol["max_length"] > details["maximum_length"]:
            raise ValueError("Encoder parameter or position limits exceed this worker")
        for name, value in (("model_artifacts_sha256", model_records), ("tokenizer_artifacts_sha256", tokenizer_records)):
            if name in manifest and manifest[name] != features.digest_json(features.records_identity(value)):
                raise ValueError("Prepared model/tokenizer artifact binding differs")
        records += model_records + tokenizer_records
        verify_records(records)
        verify_records(source_records)
        tokenization = audit_tokenization(original, tokenizer, special, spec["backend"], rows, protocol, progress=progress)
        initial_tensor_sha = features.tensor_digest(original)
        # Clone all parameters, including memory-mapped loader tensors, before updates.
        model = copy.deepcopy(original).to(device).requires_grad_(True)
        del original
        torch.manual_seed(seed)  # head initialization is independent of loader RNG use
        dimension = model.config.hidden_size * (4 if protocol["task_type"] == "pair" else 1)
        head = nn.Linear(dimension, protocol["label_count"]).to(device)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=learning_rate,
                                      weight_decay=protocol["optimizer"]["weight_decay"])
        shuffle = torch.Generator().manual_seed(seed)
        if event_path is not None:
            writer = SummaryWriter(log_dir=str(event_path))
        epochs, presentations, optimizer_updates, train_counts = [], 0, 0, {}
        training_seconds = assessment_seconds = 0.0
        runtime = {"torch": torch.__version__, "numpy": np.__version__, "device": str(device), "threads": threads,
                   "tokenizers": importlib.metadata.version("tokenizers"),
                   "safetensors": importlib.metadata.version("safetensors"), "cuda": torch.version.cuda,
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "float32_matmul_precision": torch.get_float32_matmul_precision(),
                   "worker_sha256": source_records[0]["sha256"],
                   "extractor_sha256": source_records[1]["sha256"],
                   "metric_source_sha256": source_records[2]["sha256"],
                   "nuri_source_sha256": features.digest_json({record["name"]: record["sha256"] for record in source_records[3:]})}
        if spec["backend"] == "transformers":
            import transformers
            runtime["transformers"] = transformers.__version__
        for epoch in range(1, protocol["epochs"] + 1):
            model.train()
            head.train()
            order = torch.randperm(len(rows["train"]), generator=shuffle).tolist()
            loss_sum, assessed_rows = 0.0, 0
            sync(device)
            started = time.perf_counter()
            batches = tqdm(range(0, len(order), protocol["batch_size"]), desc=f"epoch {epoch} | train",
                           unit="batch", disable=not progress)
            for start in batches:
                batch = [rows["train"][index] for index in order[start:start + protocol["batch_size"]]]
                optimizer.zero_grad(set_to_none=True)
                pooled, counts = pooled_batch(model, tokenizer, special, spec["backend"], batch, protocol["max_length"], device)
                logits = head(pooled)
                labels = torch.tensor([row["label"] for row in batch], device=device)
                loss = nn.functional.cross_entropy(logits, labels)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite training loss before optimizer update")
                loss.backward()
                norm = nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()),
                                               protocol["optimizer"]["grad_clip"])
                if not bool(torch.isfinite(norm)):
                    raise ValueError("Nonfinite gradient before optimizer update")
                optimizer.step()
                value = float(loss.detach())
                loss_sum += value * len(batch)
                assessed_rows += len(batch)
                presentations += len(batch)
                optimizer_updates += 1
                add_counts(train_counts, counts)
                batches.set_postfix(loss=f"{value:.4f}", update=optimizer_updates)
                if writer is not None:
                    writer.add_scalar("train/observed_loss", value, optimizer_updates)
                    writer.add_scalar("train/grad_norm_preclip", float(norm), optimizer_updates)
            sync(device)
            training_seconds += time.perf_counter() - started
            started = time.perf_counter()
            train_metrics = assess(model, head, tokenizer, special, spec["backend"], rows["train"], protocol, device,
                                   progress=progress, split="train")
            valid_metrics = assess(model, head, tokenizer, special, spec["backend"], rows["valid"], protocol, device,
                                   progress=progress, split="valid")
            sync(device)
            assessment_seconds += time.perf_counter() - started
            epochs.append({"epoch": epoch, "train": train_metrics, "valid": valid_metrics,
                           "train_observed_loss": loss_sum / assessed_rows})
            if progress:
                tqdm.write(f"epoch={epoch} train_f1={train_metrics['macro_f1']:.4f} valid_f1={valid_metrics['macro_f1']:.4f}")
            if writer is not None:
                for split, metrics in (("train", train_metrics), ("valid", valid_metrics)):
                    for name in ("loss", "accuracy", "macro_f1"):
                        writer.add_scalar(f"{split}/{name}", metrics[name], epoch)
                writer.flush()
        selected = select_candidate(epochs)
        verify_finite_state(model, head, optimizer)
        final_tensor_sha = features.tensor_digest(model.cpu())
        verify_records(records)
        verify_records(source_records)
        if writer is not None:
            writer.close()
            writer = None
        report = {"schema_version": 1, "status": "complete", "protocol": protocol,
                  "protocol_sha256": next(record["sha256"] for record in records if record["name"] == "adaptation_protocol"),
                  "manifest_sha256": next(record["sha256"] for record in records if record["name"] == "manifest"),
                  "candidate": {"learning_rate": learning_rate, "seed": seed}, "model": spec, "runtime": runtime,
                  "model_sha256": features.digest_json({"artifacts": features.records_identity(model_records),
                                                        "initial_tensor_sha256": initial_tensor_sha, "details": details}),
                  "tokenizer_sha256": features.digest_json(features.records_identity(tokenizer_records)),
                  "source_bindings": source_records,
                  "assessment_protocol_sha256": next((r["sha256"] for r in records if r["name"] == "assessment_protocol"), None),
                  "model_artifacts": model_records, "tokenizer_artifacts": tokenizer_records, "input_bindings": records[:2],
                  "encoder_details": details, "encoder_parameters": parameters, "head_parameters": sum(p.numel() for p in head.parameters()),
                  "initial_encoder_tensor_sha256": initial_tensor_sha, "final_encoder_tensor_sha256": final_tensor_sha,
                  "final_tensor_epoch": protocol["epochs"], "finite_model_head_optimizer_state": True,
                  "pre_update_tokenization": tokenization, "tokenizer_truncation_allowed": False,
                  "source_and_inputs_unchanged": True, "epochs": epochs, "selected_epoch": selected + 1,
                  "selected": epochs[selected], "selection": "VALID macro_f1, then accuracy, then earliest epoch",
                  "rows": {split: {"count": len(values), "row_ids": [row["id"] for row in values],
                                   "row_ids_sha256": features.row_ids_digest([row["id"] for row in values]),
                                   "labels": [row["label"] for row in values]} for split, values in rows.items()},
                  "budget": {"optimizer_updates": optimizer_updates, "training_presentations": presentations,
                             "actual_training_counts": train_counts, "training_seconds": training_seconds,
                             "assessment_seconds": assessment_seconds, "total_seconds": time.perf_counter() - began},
                  "test_used": False, "adapted_weights_retained": False,
                  "limits": "single candidate; outer LR/seed search and independent completion assessment remain separate"}
        json.dumps(report, allow_nan=False)  # refuse nonfinite receipts before publication
        features.write_json(output_dir / "REPORT.json", report)
        features.write_json(output_dir / "COMPLETE.json", {"status": "complete", "report_sha256": features.file_digest(output_dir / "REPORT.json")})
        return report
    except Exception as error:
        features.write_json(output_dir / "FAILED.json", {"status": "failed", "error_type": type(error).__name__})
        raise
    finally:
        if writer is not None:
            writer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("manifest", "protocol", "output"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--tensorboard-dir")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    report = run_candidate(args.manifest, args.protocol, args.output, learning_rate=args.learning_rate,
                           seed=args.seed, device=args.device, threads=args.threads,
                           tensorboard_dir=args.tensorboard_dir, progress=not args.no_progress)
    print(f"selected_epoch={report['selected_epoch']} valid_accuracy={report['selected']['valid']['accuracy']:.4f} "
          f"valid_macro_f1={report['selected']['valid']['macro_f1']:.4f}; TEST unused")


if __name__ == "__main__":
    main()
