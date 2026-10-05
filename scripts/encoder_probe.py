"""Fit identical frozen-feature ridge probes using TRAIN and VALID only.

This supervised diagnostic does not fine-tune an encoder. VALID selects one
regularization value, so its selected score is not an independent test score.
Input features and results must remain private; no model checkpoint is loaded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import zipfile

import numpy as np
from tqdm.auto import tqdm


ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
CHUNK_ROWS = 512
FEATURE_KEYS = {"features", "labels", "row_ids", "metadata_json"}
SHARED_METADATA = ("backend", "model_id", "model_revision", "model_sha256",
                   "tokenizer_sha256", "protocol_sha256", "pooling", "pair_features", "label_count")
HASH_FIELDS = ("model_sha256", "tokenizer_sha256", "protocol_sha256", "raw_input_sha256",
               "row_ids_sha256")


@dataclass
class FeatureSet:
    features: np.ndarray
    labels: np.ndarray
    row_ids: list[str]
    metadata: dict
    sha256: str
    uncompressed_bytes: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def private_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if ".private" not in resolved.parts:
        raise ValueError("probe output and TensorBoard paths must be inside a .private directory")
    return resolved


def inspect_npz(path: Path, max_bytes: int) -> int:
    """Check declared NPY sizes before NumPy can allocate their arrays."""
    total = 0
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != len(FEATURE_KEYS) or {m.filename for m in members} != {
                f"{key}.npy" for key in FEATURE_KEYS}:
            raise ValueError("feature NPZ must contain exactly the four declared arrays")
        for member in members:
            total += member.file_size
            if total > max_bytes:
                raise ValueError("feature arrays exceed the CPU memory budget")
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("unsupported NPY header version")
                if dtype.hasobject:
                    raise ValueError("object arrays are forbidden")
                payload_bytes = math.prod(shape) * dtype.itemsize
                if payload_bytes + stream.tell() != member.file_size:
                    raise ValueError("feature NPY header does not match its payload size")
    return total


def load_features(path: Path, split: str, max_bytes: int) -> FeatureSet:
    size = inspect_npz(path, max_bytes)
    digest = file_sha256(path)
    with np.load(path, allow_pickle=False) as arrays:
        features, labels = arrays["features"], arrays["labels"]
        ids, raw_metadata = arrays["row_ids"], arrays["metadata_json"]
    if features.dtype != np.float32 or features.ndim != 2 or not all(features.shape):
        raise ValueError("features must be a nonempty float32 [rows, dimensions] array")
    if labels.dtype != np.int64 or labels.shape != (len(features),):
        raise ValueError("labels must be int64 with one label per feature row")
    if ids.dtype.kind != "U" or ids.shape != labels.shape:
        raise ValueError("row_ids must be Unicode strings with one ID per row")
    row_ids = ids.tolist()
    if any(not item for item in row_ids) or len(set(row_ids)) != len(row_ids):
        raise ValueError("row IDs must be nonempty and unique within each split")
    if raw_metadata.dtype.kind != "U" or raw_metadata.shape != ():
        raise ValueError("metadata_json must be a Unicode scalar")
    metadata = json.loads(raw_metadata.item())
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("feature metadata needs schema_version 1")
    if metadata.get("split") != split:
        raise ValueError(f"expected {split.upper()} features; TEST is never used by this probe")
    if any(key not in metadata for key in SHARED_METADATA + HASH_FIELDS):
        raise ValueError("feature metadata is missing a required binding")
    for key in HASH_FIELDS + (("assessment_protocol_sha256",) if "assessment_protocol_sha256" in metadata else ()):
        value = metadata[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{key} must be a lowercase SHA256")
    classes = metadata["label_count"]
    if type(classes) is not int or not 2 <= classes <= 1024:
        raise ValueError("label_count must declare between 2 and 1024 classes")
    expected_pooling = "hashed_character_ngrams" if metadata["backend"] == "lexical" else "masked_mean_content"
    if metadata["pooling"] != expected_pooling or metadata["pair_features"] not in (
            "none", "concat_u_v_absdiff_product"):
        raise ValueError("unsupported frozen-feature pooling or pair representation")
    for key in ("backend", "model_id", "model_revision"):
        if not isinstance(metadata[key], str) or not metadata[key]:
            raise ValueError(f"{key} must be a nonempty string")
    if metadata["row_ids_sha256"] != row_ids_sha256(row_ids):
        raise ValueError("row ID digest does not match feature metadata")
    if np.any(labels < 0) or np.any(labels >= classes):
        raise ValueError("labels are outside the declared class universe")
    for start in range(0, len(features), CHUNK_ROWS):
        if not np.isfinite(features[start:start + CHUNK_ROWS]).all():
            raise ValueError("feature values must be finite")
    if file_sha256(path) != digest:
        raise ValueError("feature file changed during loading")
    return FeatureSet(features, labels, row_ids, metadata, digest, size)


def validate_splits(train: FeatureSet, valid: FeatureSet) -> None:
    if len(train.features) < 2:
        raise ValueError("TRAIN needs at least two rows")
    if train.features.shape[1] != valid.features.shape[1]:
        raise ValueError("TRAIN and VALID feature dimensions differ")
    if any(train.metadata[key] != valid.metadata[key] for key in SHARED_METADATA):
        raise ValueError("TRAIN and VALID model/protocol bindings differ")
    if train.metadata.get("assessment_protocol_sha256") != valid.metadata.get("assessment_protocol_sha256"):
        raise ValueError("TRAIN and VALID assessment protocol bindings differ")
    if set(train.row_ids).intersection(valid.row_ids):
        raise ValueError("TRAIN and VALID row IDs overlap")


def workspace_bytes(train: FeatureSet, valid: FeatureSet) -> int:
    """Conservative array workspace, excluding the interpreter and BLAS runtime."""
    rows, dims = train.features.shape
    classes = train.metadata["label_count"]
    system_dim = min(rows, dims)
    return (train.uncompressed_bytes + valid.uncompressed_bytes + 3 * rows * dims * 8
            + 6 * system_dim * system_dim * 8 + 3 * rows * classes * 8
            + 3 * dims * classes * 8 + 2 * classes * classes * 8 + CHUNK_ROWS * (dims + classes) * 8
            + (rows + len(valid.features)) * 32 + 4 * dims * 8)


def fit_standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit location and scale on TRAIN only; constant columns keep scale one."""
    mean = features.mean(axis=0, dtype=np.float64)
    scale = features.std(axis=0, dtype=np.float64)
    scale[scale == 0] = 1.0
    return mean, scale


def ridge_system(features: np.ndarray, labels: np.ndarray, classes: int):
    targets = np.eye(classes, dtype=np.float64)[labels]
    intercept = targets.mean(axis=0)
    targets -= intercept
    dual = features.shape[1] > features.shape[0]
    gram = features @ features.T if dual else features.T @ features
    rhs = targets if dual else features.T @ targets
    return gram, rhs, intercept, dual


def ridge_weights(features: np.ndarray, gram: np.ndarray, rhs: np.ndarray,
                  alpha: float, dual: bool) -> np.ndarray:
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("ridge alpha must be finite and positive")
    regularized = gram.copy()
    regularized.flat[::len(regularized) + 1] += alpha
    solution = np.linalg.solve(regularized, rhs)
    return features.T @ solution if dual else solution


def predict(features: np.ndarray, weights: np.ndarray, intercept: np.ndarray,
            mean: np.ndarray | None = None, scale: np.ndarray | None = None) -> np.ndarray:
    predictions = np.empty(len(features), dtype=np.int64)
    for start in range(0, len(features), CHUNK_ROWS):
        chunk = features[start:start + CHUNK_ROWS]
        if mean is not None:
            chunk = (chunk.astype(np.float64) - mean) / scale
        scores = chunk @ weights + intercept
        if not np.isfinite(scores).all():
            raise ValueError("ridge predictions are nonfinite")
        predictions[start:start + len(chunk)] = scores.argmax(axis=1)
    return predictions


def classification_metrics(labels: np.ndarray, predictions: np.ndarray, classes: int) -> dict:
    if labels.shape != predictions.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("metrics need nonempty aligned label and prediction rows")
    if np.any(labels < 0) or np.any(labels >= classes) or np.any(predictions < 0) or np.any(predictions >= classes):
        raise ValueError("metric labels are outside the class universe")
    confusion = np.bincount(labels * classes + predictions, minlength=classes * classes).reshape(classes, classes)
    correct = np.diag(confusion)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2.0 * correct, denominator, out=np.zeros(classes), where=denominator != 0)
    return {"accuracy": float(correct.sum() / len(labels)), "macro_f1": float(f1.mean()),
            "rows": len(labels), "support": confusion.sum(axis=1).tolist(), "per_class_f1": f1.tolist()}


def select_candidate(candidates: list[dict]) -> int:
    return max(range(len(candidates)), key=lambda i: (candidates[i]["valid"]["macro_f1"],
                                                     candidates[i]["valid"]["accuracy"], -i))


def paired_bootstrap(labels: np.ndarray, predictions: np.ndarray, classes: int, *,
                     samples: int = 1000, seed: int = 0,
                     reference: np.ndarray | None = None) -> dict:
    """Resample the same validation rows for both models, one replicate at a time."""
    if type(samples) is not int or not 1 <= samples <= 100000:
        raise ValueError("bootstrap samples must be between 1 and 100000")
    observed = classification_metrics(labels, predictions, classes)
    if reference is not None:
        reference_metrics = classification_metrics(labels, reference, classes)
    draws = {name: np.empty(samples) for name in ("accuracy", "macro_f1")}
    rng = np.random.default_rng(seed)
    for index in range(samples):
        rows = rng.integers(0, len(labels), size=len(labels))
        measured = classification_metrics(labels[rows], predictions[rows], classes)
        if reference is not None:
            compared = classification_metrics(labels[rows], reference[rows], classes)
        for name in draws:
            draws[name][index] = measured[name] - compared[name] if reference is not None else measured[name]
    result = {}
    for name, values in draws.items():
        estimate = observed[name] - reference_metrics[name] if reference is not None else observed[name]
        low, high = np.percentile(values, [2.5, 97.5])
        result[name] = {"estimate": estimate, "low": float(low), "high": float(high)}
    return {"samples": samples, "seed": seed, "confidence": 0.95, "row_paired": reference is not None,
            "metrics": result,
            "limitations": "Conditional on VALID-selected alpha; row independence assumed; no selection or training uncertainty."}


def reference_predictions(path: Path, valid: FeatureSet) -> tuple[np.ndarray, dict]:
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("reference report exceeds the bounded JSON size")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("schema_version") != 1 or report.get("status") != "completed"
            or report.get("kind") != "frozen_multiclass_ridge" or report.get("test_used") is not False):
        raise ValueError("reference must be a completed validation-only ridge probe")
    if report.get("frozen_protocol_sha256") != valid.metadata["assessment_protocol_sha256"]:
        raise ValueError("reference assessment protocol binding differs")
    metadata = report["inputs"]["valid"]["metadata"]
    for key in ("assessment_protocol_sha256", "raw_input_sha256", "row_ids_sha256", "label_count"):
        if metadata[key] != valid.metadata[key]:
            raise ValueError("reference VALID protocol or input bindings differ")
    rows = report["validation_predictions"]
    if rows["row_ids"] != valid.row_ids or rows["labels"] != valid.labels.tolist():
        raise ValueError("reference VALID rows or labels differ")
    predictions = np.asarray(rows["predictions"])
    if predictions.dtype.kind not in "iu" or predictions.shape != valid.labels.shape:
        raise ValueError("reference predictions must be aligned integer class IDs")
    classification_metrics(valid.labels, predictions, valid.metadata["label_count"])
    return predictions.astype(np.int64), {"sha256": file_sha256(path),
                                          "model_sha256": metadata["model_sha256"],
                                          "model_id": metadata["model_id"]}


def write_immutable_json(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".probe-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link publishes complete bytes and refuses an existing result.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_probe(train_path: str | Path, valid_path: str | Path, output: str | Path, *,
              protocol_path: str | Path,
              bootstrap_samples: int = 1000, seed: int = 0, max_memory_mib: int = 512,
              reference_reports: tuple[str | Path, ...] = (), progress: bool = True,
              tensorboard_dir: str | Path | None = None) -> dict:
    output = private_path(output)
    if output.exists():
        raise FileExistsError("probe output already exists; preserve it and choose a new result path")
    if type(max_memory_mib) is not int or max_memory_mib <= 0:
        raise ValueError("max_memory_mib must be positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(bootstrap_samples) is not int or not 1 <= bootstrap_samples <= 100000:
        raise ValueError("bootstrap samples must be between 1 and 100000")
    protocol_path = private_path(protocol_path)
    if protocol_path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("protocol JSON exceeds the bounded input size")
    json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_digest = file_sha256(protocol_path)
    train_path, valid_path = Path(train_path).resolve(), Path(valid_path).resolve()
    if train_path == valid_path or output in (train_path, valid_path):
        raise ValueError("feature splits and output need distinct paths")
    max_bytes = max_memory_mib * 1024 * 1024
    if inspect_npz(train_path, max_bytes) + inspect_npz(valid_path, max_bytes) > max_bytes:
        raise ValueError("combined feature arrays exceed the CPU memory budget")
    train = load_features(train_path, "train", max_bytes)
    valid = load_features(valid_path, "valid", max_bytes)
    validate_splits(train, valid)
    if train.metadata.get("assessment_protocol_sha256") != protocol_digest:
        raise ValueError("feature assessment digest differs from the supplied frozen protocol")
    estimated_bytes = workspace_bytes(train, valid)
    if estimated_bytes > max_bytes:
        raise ValueError("ridge workspace exceeds the CPU memory budget; use fewer rows or a larger explicit budget")
    references = [(reference_predictions(Path(path), valid)) for path in reference_reports]
    writer = None
    if tensorboard_dir is not None:
        directory = private_path(tensorboard_dir)
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise ValueError("TensorBoard requires the optional tensorboard dependency") from error
        writer = SummaryWriter(log_dir=str(directory))
    try:
        mean, scale = fit_standardizer(train.features)
        standardized = (train.features.astype(np.float64) - mean) / scale
        classes = train.metadata["label_count"]
        gram, rhs, intercept, dual = ridge_system(standardized, train.labels, classes)
        candidates, selected_predictions = [], []
        bar = tqdm(ALPHAS, desc="Frozen ridge: TRAIN fit / VALID selection", disable=not progress)
        for step, alpha in enumerate(bar, start=1):
            weights = ridge_weights(standardized, gram, rhs, alpha, dual)
            train_prediction = predict(standardized, weights, intercept)
            valid_prediction = predict(valid.features, weights, intercept, mean, scale)
            candidate = {"alpha": alpha, "train": classification_metrics(train.labels, train_prediction, classes),
                         "valid": classification_metrics(valid.labels, valid_prediction, classes)}
            candidates.append(candidate)
            selected_predictions.append(valid_prediction)
            bar.set_postfix(alpha=alpha, valid_accuracy=f"{candidate['valid']['accuracy']:.4f}",
                            valid_macro_f1=f"{candidate['valid']['macro_f1']:.4f}")
            if writer is not None:
                for split in ("train", "valid"):
                    for name in ("accuracy", "macro_f1"):
                        writer.add_scalar(f"{split}/{name}", candidate[split][name], step)
                writer.add_scalar("probe/alpha", alpha, step)
        selected = select_candidate(candidates)
        predictions = selected_predictions[selected]
        counts = np.bincount(train.labels, minlength=classes)
        majority = int(counts.argmax())
        control = np.full(len(valid.labels), majority, dtype=np.int64)
        report = {
            "schema_version": 1, "status": "completed", "kind": "frozen_multiclass_ridge",
            "test_used": False, "encoder_updated": False,
            "frozen_protocol_sha256": protocol_digest,
            "inputs": {split: {"sha256": data.sha256, "metadata": data.metadata}
                       for split, data in (("train", train), ("valid", valid))},
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "script_sha256": file_sha256(Path(__file__))},
            "protocol": {"alpha_grid": list(ALPHAS), "selection": "VALID macro_f1, then accuracy, then grid order",
                         "standardization": "TRAIN mean/population standard deviation; zero variance scale one",
                         "intercept": "unpenalized TRAIN class-prior intercept",
                         "macro_f1": "all declared classes, including absent classes, zero denominator gives zero",
                         "seed": seed, "label_count": classes},
            "geometry": {"train_rows": len(train.labels), "valid_rows": len(valid.labels),
                         "dimensions": train.features.shape[1], "formulation": "dual" if dual else "primal",
                         "estimated_array_workspace_bytes": estimated_bytes,
                         "max_memory_bytes": max_bytes, "prediction_chunk_rows": CHUNK_ROWS},
            "standardizer_sha256": hashlib.sha256(mean.tobytes() + scale.tobytes()).hexdigest(),
            "candidates": candidates, "selected_alpha": candidates[selected]["alpha"],
            "selected_metrics": {split: candidates[selected][split] for split in ("train", "valid")},
            "controls": {"train_class_counts": counts.tolist(), "train_class_priors": (counts / counts.sum()).tolist(),
                         "majority_label": majority, "majority_valid": classification_metrics(valid.labels, control, classes),
                         "majority_train": classification_metrics(train.labels, np.full_like(train.labels, majority), classes),
                         "train_prior_argmax_valid": classification_metrics(valid.labels, control, classes),
                         "train_prior_rule": "Argmax TRAIN class prior equals majority; smallest class ID breaks ties."},
            "validation_predictions": {"row_ids": valid.row_ids, "labels": valid.labels.tolist(),
                                       "predictions": predictions.tolist()},
            "bootstrap": paired_bootstrap(valid.labels, predictions, classes, samples=bootstrap_samples, seed=seed),
            "paired_references": [{"reference": identity, "delta": paired_bootstrap(
                valid.labels, predictions, classes, samples=bootstrap_samples, seed=seed, reference=reference)}
                for reference, identity in references],
            "limitations": ["Supervised frozen-feature diagnostic, not task-specific encoder fine-tuning.",
                            "VALID is used for alpha selection; reserve TEST for independent completion.",
                            "Row bootstrap assumes independent rows and omits alpha-selection and training variance.",
                            "No lexical control is fitted here; assess it separately on the same rows."],
        }
        if (file_sha256(train_path) != train.sha256 or file_sha256(valid_path) != valid.sha256
                or file_sha256(protocol_path) != protocol_digest):
            raise ValueError("feature inputs changed during the probe")
        write_immutable_json(output, report)
        return report
    finally:
        if writer is not None:
            writer.close()


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--valid-features", required=True)
    parser.add_argument("--output", required=True, help="new JSON path inside .private")
    parser.add_argument("--protocol", required=True, help="frozen protocol JSON inside .private")
    parser.add_argument("--reference-report", action="append", default=[])
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-memory-mib", type=int, default=512)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--tensorboard-dir")
    args = parser.parse_args(argv)
    try:
        report = run_probe(args.train_features, args.valid_features, args.output,
                           protocol_path=args.protocol,
                           bootstrap_samples=args.bootstrap_samples, seed=args.seed,
                           max_memory_mib=args.max_memory_mib,
                           reference_reports=tuple(args.reference_report), progress=not args.no_progress,
                           tensorboard_dir=args.tensorboard_dir)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile, np.linalg.LinAlgError) as error:
        parser.exit(2, f"encoder probe failed: {error}\n")
    metrics = report["selected_metrics"]["valid"]
    print(f"Frozen probe complete: alpha={report['selected_alpha']:g}, VALID rows={metrics['rows']}, "
          f"accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}; TEST unused")
    return report


if __name__ == "__main__":
    main()
