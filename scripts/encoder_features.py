"""Extract frozen encoder features using an ignored private JSON manifest.

Manifest schema 1 has model, inputs, label_count and optional protocol fields.
model supplies backend (nuri or transformers), path_or_id, a pinned HF revision,
and optionally tokenizer_path_or_id / tokenizer_revision. Nuri alone supports
random_initialization_seed. inputs maps split names to {path, sha256}; paths are
relative to the manifest. JSONL rows contain id, text, optional text_pair, label
and split. protocol defaults to batch_size=8 and max_length=128 per text field.

No core or projection head participates. HF fields retain their normal special
tokens for contextualization, then exclude inserted wrapper tokens and PAD from
pooling. Unknown tokens are content when the tokenizer marks them as content.
Both backends encode each pair field separately. Shared raw prefixes should be
prepared before extraction; tokenizer-dependent truncation is reported.

Run: python scripts/encoder_features.py --manifest <private-json> --output
<new-private-folder> --splits train valid --threads 2. Test is reserved unless
--allow-test-completion is explicitly provided. Existing outputs are never reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer
from torch import nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nuri.config import NuriConfig
from nuri.encoder import InputEncoder
from nuri.moe import MoEFFN


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                     sort_keys=True).encode("utf-8")).hexdigest()


def row_ids_digest(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def private_path(path) -> Path:
    resolved = Path(path).resolve()
    if ".private" not in resolved.parts:
        raise ValueError("Manifest and output must be inside an ignored .private directory")
    return resolved


def positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def resolve_path(base: Path, value: str) -> Path:
    return (base / value).resolve()


def read_rows(path: Path, split: str, label_count: int) -> list[dict]:
    rows, ids, paired = [], set(), None
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            required = {"id", "text", "label", "split"}
            if not isinstance(row, dict) or not required <= row.keys() or row.keys() - required - {"text_pair"}:
                raise ValueError(f"Invalid row schema at line {number}")
            if not isinstance(row["id"], str) or not row["id"] or row["id"] in ids:
                raise ValueError(f"Invalid or duplicate row ID at line {number}")
            if row["split"] != split:
                raise ValueError(f"Row split mismatch at line {number}")
            label = row["label"]
            if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < label_count:
                raise ValueError(f"Invalid class label at line {number}")
            is_pair = "text_pair" in row
            if paired is not None and paired != is_pair:
                raise ValueError("A split cannot mix single-text and pair rows")
            paired = is_pair
            for field in ("text", "text_pair") if is_pair else ("text",):
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ValueError(f"Empty or invalid text at line {number}")
            ids.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError("Every selected split must contain rows")
    return rows


def content_mean(hidden: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
    """Mean only content positions; do not silently replace empty input with zero."""
    if hidden.ndim != 3 or content.shape != hidden.shape[:2]:
        raise ValueError("Hidden states and content mask must have matching batch/length")
    counts = content.sum(dim=1)
    if not bool((counts > 0).all()):
        raise ValueError("Every text needs at least one retained content token")
    return hidden.float().masked_fill(~content[..., None], 0).sum(dim=1) / counts[:, None]


def combine_pair(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    if u.shape != v.shape or u.ndim != 2:
        raise ValueError("Pair representations must have matching [rows, dimension] shapes")
    return np.concatenate((u, v, np.abs(u - v), u * v), axis=1).astype(np.float32)


class NuriEncoderOnly(nn.Module):
    """Only the native input embedding and input encoder exist in this object."""

    def __init__(self, config: NuriConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder = InputEncoder(config)

    def forward(self, input_ids, attention_mask):
        return self.encoder(self.embedding(input_ids), attention_mask.bool())


def tensor_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(digest_json([name, str(value.dtype), list(value.shape)]).encode("ascii"))
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def artifact_records(folder: Path, names: list[str]) -> list[dict]:
    return [{"name": name, "path": str(folder / name), "bytes": (folder / name).stat().st_size,
             "sha256": file_digest(folder / name)} for name in sorted(set(names))]


def records_identity(records: list[dict]) -> list[dict]:
    return [{key: row[key] for key in ("name", "bytes", "sha256")} for row in records]


def loading_info_json(value):
    """Normalize version-dependent loader sets into deterministic JSON values."""
    if isinstance(value, dict):
        return {key: loading_info_json(item) for key, item in value.items()}
    if isinstance(value, set):
        return [loading_info_json(item) for item in sorted(value, key=repr)]
    if isinstance(value, (tuple, list)):
        return [loading_info_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("Unsupported encoder loading diagnostic; cannot bind it safely")


def load_nuri(spec: dict, base: Path):
    folder = resolve_path(base, spec["path_or_id"])
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    config = NuriConfig.from_dict(config.get("nuri", config))
    random_seed = spec.get("random_initialization_seed")
    if random_seed is not None:
        if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
            raise ValueError("random_initialization_seed must be an integer >= 0")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(random_seed)
            model = NuriEncoderOnly(config)

            def initialize(module):
                if isinstance(module, (nn.Embedding, nn.Linear)):
                    nn.init.normal_(module.weight, std=config.initializer_range)
                elif isinstance(module, MoEFFN):
                    for value in (module.w_gate, module.w_up, module.w_down):
                        nn.init.normal_(value, std=config.initializer_range)

            model.apply(initialize)
        names = ["config.json"]
    else:
        with torch.device("meta"):
            model = NuriEncoderOnly(config)
        expected = set(model.state_dict())
        with safe_open(str(folder / "model.safetensors"), framework="pt", device="cpu") as stream:
            keys = set(stream.keys())
            prefix = "" if "embedding.weight" in keys else "model."
            selected = {name: stream.get_tensor(prefix + name).float() for name in sorted(expected)}
        model.load_state_dict(selected, strict=True, assign=True)
        names = ["config.json", "model.safetensors"]
    model = model.float().cpu().eval().requires_grad_(False)
    tokenizer_path = resolve_path(base, spec.get("tokenizer_path_or_id", spec["path_or_id"]))
    if tokenizer_path.is_dir():
        tokenizer_path /= "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("Nuri tokenizer vocabulary must match the encoder")
    for name, token_id in (("[PAD]", config.pad_id), ("[MASK]", config.mask_id), ("[EOS]", config.eos_id)):
        if tokenizer.token_to_id(name) != token_id:
            raise ValueError("Nuri tokenizer special IDs must match the encoder")
    special = {config.pad_id, config.mask_id, config.eos_id}
    model_records = artifact_records(folder, names)
    tokenizer_records = artifact_records(tokenizer_path.parent, [tokenizer_path.name])
    return model, tokenizer, special, model_records, tokenizer_records, {
        "random_initialization_seed": random_seed, "selected_tensor_sha256": tensor_digest(model),
        "loaded_tensor_names": sorted(model.state_dict()), "maximum_length": config.max_input_len,
        "parameter_scope": "input embedding and encoder only; total, not per-token MoE active count",
    }


def hf_folder(identifier: str, revision: str, base: Path) -> Path:
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("HF revisions must be immutable 40-character commit hashes")
    local = resolve_path(base, identifier)
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(identifier, revision=revision,
                                  allow_patterns=["*.json", "*.txt", "*.model", "*.spm",
                                                  "*.safetensors", "*.tiktoken"]))


def load_transformers(spec: dict, base: Path):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    if "random_initialization_seed" in spec:
        raise ValueError("random_initialization_seed is supported only for nuri")
    revision = spec.get("revision")
    folder = hf_folder(spec["path_or_id"], revision, base)
    tokenizer_folder = hf_folder(spec.get("tokenizer_path_or_id", spec["path_or_id"]),
                                 spec.get("tokenizer_revision", revision), base)
    config = AutoConfig.from_pretrained(str(folder), revision=revision, trust_remote_code=False,
                                       local_files_only=True)
    if hasattr(config, "reference_compile"):
        config.reference_compile = False
    model, loading_info = AutoModel.from_pretrained(str(folder), config=config, revision=revision, trust_remote_code=False,
                                                   local_files_only=True, use_safetensors=True, dtype=torch.float32,
                                                   attn_implementation="eager", output_loading_info=True)
    loading_info = loading_info_json(loading_info)
    if getattr(model.config, "is_encoder_decoder", False) or getattr(model.config, "is_decoder", False):
        raise ValueError("The transformers backend requires a bidirectional encoder")
    # Base poolers are unrelated to mean hidden-state features; do not execute them.
    if hasattr(model, "pooler"):
        model.pooler = None
    missing = loading_info.get("missing_keys", [])
    mismatched = loading_info.get("mismatched_keys", [])
    mismatch_names = [value[0] if isinstance(value, (list, tuple)) else value for value in mismatched]
    if (any(not name.startswith("pooler.") for name in missing)
            or mismatch_names or loading_info.get("error_msgs")):
        raise ValueError("Required encoder weights did not load completely; no initialized fallback is allowed")
    model = model.float().cpu().eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_folder), trust_remote_code=False,
                                             local_files_only=True,
                                             revision=spec.get("tokenizer_revision", revision))
    if tokenizer.pad_token_id is None:
        raise ValueError("The encoder tokenizer must define PAD")
    tokenizer.padding_side = tokenizer.truncation_side = "right"
    weights = [file.name for file in folder.glob("*.safetensors")]
    if not weights:
        raise ValueError("The encoder checkpoint must have safetensors weights")
    model_records = artifact_records(folder, ["config.json", *weights,
                                              *[file.name for file in folder.glob("*.safetensors.index.json")]])
    tokenizer_names = [file.name for file in tokenizer_folder.iterdir() if file.is_file()
                       and file.suffix in (".json", ".txt", ".model", ".spm", ".tiktoken")
                       and not file.name.endswith(".safetensors.index.json")]
    tokenizer_records = artifact_records(tokenizer_folder, tokenizer_names)
    return model, tokenizer, set(), model_records, tokenizer_records, {
        "random_initialization_seed": None,
        "maximum_length": getattr(model.config, "max_position_embeddings", 128),
        "parameter_scope": "AutoModel input encoder; optional unused pooler removed",
        "attention_implementation": "eager",
        "reference_compile": getattr(model.config, "reference_compile", "not_exposed"),
        "loading_info": loading_info,
        "content_policy": "exclude tokenizer-marked inserted wrappers and every PAD; retain content UNK",
        "local_revision_note": "local files are hash-bound; revision lineage is declared by the private manifest",
    }


def tokenize_texts(backend, tokenizer, special, model, texts, max_length):
    """Return model inputs, content mask and observed per-field length evidence."""
    if backend == "nuri":
        full = [tokenizer.encode(text, add_special_tokens=False).ids for text in texts]
        kept = [row[:max_length] for row in full]
        width = max(map(len, kept))
        ids = torch.full((len(kept), width), model.config.pad_id, dtype=torch.long)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        for index, row in enumerate(kept):
            ids[index, :len(row)] = torch.tensor(row, dtype=torch.long)
            valid[index, :len(row)] = True
        valid &= ids != model.config.pad_id
        inputs = {"input_ids": ids, "attention_mask": valid}
    else:
        untruncated = tokenizer(texts, add_special_tokens=True, truncation=False,
                                return_special_tokens_mask=True)
        full = untruncated["input_ids"]
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                           return_tensors="pt", return_special_tokens_mask=True)
        wrappers = inputs.pop("special_tokens_mask").bool()
        inputs["attention_mask"] &= inputs["input_ids"] != tokenizer.pad_token_id
        ids, valid = inputs["input_ids"], inputs["attention_mask"].bool()
    if backend == "nuri":
        content = valid & ~torch.isin(ids, torch.tensor(sorted(special), dtype=torch.long))
        full_content = [sum(token not in special for token in row) for row in full]
    else:
        content = valid & ~wrappers
        full_content = [sum(not marked and token != tokenizer.pad_token_id for token, marked in zip(row, mask))
                        for row, mask in zip(full, untruncated["special_tokens_mask"])]
    stats = {"full_tokens": list(map(len, full)), "retained_tokens": valid.sum(dim=1).tolist(),
             "full_content_tokens": full_content, "retained_content_tokens": content.sum(dim=1).tolist(),
             "raw_characters": list(map(len, texts)), "truncated": [len(row) > max_length for row in full]}
    return inputs, content, stats


@torch.inference_mode()
def encode_field(backend, model, tokenizer, special, texts, batch_size, max_length, *, progress=True, description="text"):
    arrays, stats = [], {}
    tokenize_seconds = forward_seconds = 0.0
    for start in tqdm(range(0, len(texts), batch_size), desc=description, unit="batch", disable=not progress):
        began = time.perf_counter()
        inputs, content, observed = tokenize_texts(backend, tokenizer, special, model,
                                                  texts[start:start + batch_size], max_length)
        if not bool(content.any(dim=1).all()):
            raise ValueError("Every text needs at least one retained content token")
        tokenize_seconds += time.perf_counter() - began
        began = time.perf_counter()
        result = model(**inputs) if backend == "nuri" else model(**inputs, return_dict=True).last_hidden_state
        arrays.append(content_mean(result, content).cpu().numpy())
        forward_seconds += time.perf_counter() - began
        for key, values in observed.items():
            stats.setdefault(key, []).extend(values)
    features = np.concatenate(arrays).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Encoder features must be finite")
    return features, {**stats, "truncated_fields": sum(stats["truncated"]),
                      "tokenize_seconds": tokenize_seconds, "forward_pool_seconds": forward_seconds}


def write_json(path: Path, data) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def save_features(path: Path, features: np.ndarray, labels: np.ndarray,
                  row_ids: list[str], metadata: dict) -> str:
    """Write the shared NPZ contract without pickle or existing-file reuse."""
    path = private_path(path)
    if features.dtype != np.float32 or features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("features must be a finite float32 [rows, dimension] array")
    if labels.dtype != np.int64 or labels.shape != (features.shape[0],) or len(row_ids) != features.shape[0]:
        raise ValueError("labels must be int64 [rows] with matching row IDs")
    if not all(isinstance(value, str) and value for value in row_ids):
        raise ValueError("row_ids must be nonempty strings")
    with path.open("xb") as stream:
        np.savez_compressed(stream, features=features, labels=labels, row_ids=np.asarray(row_ids),
                            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)))
        stream.flush()
        os.fsync(stream.fileno())
    return file_digest(path)


def extract_features(manifest_path, output_dir, splits=("train", "valid"), *,
                     allow_test_completion=False, threads=2, progress=True) -> dict:
    """Create one fresh private cache; never infer completion from existing files."""
    manifest_path, output_dir = private_path(manifest_path), private_path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Use a new output directory; feature caches are immutable")
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits) or set(splits) - {"train", "valid", "test"}:
        raise ValueError("Select distinct train, valid or test splits")
    if "test" in splits and not allow_test_completion:
        raise ValueError("TEST is reserved; explicit completion permission is required")
    positive_integer(threads, "threads")
    if threads > 2:
        raise ValueError("CPU extraction uses at most two threads")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported feature manifest schema")
    label_count = positive_integer(manifest["label_count"], "label_count")
    if not 2 <= label_count <= 1024:
        raise ValueError("label_count must declare between 2 and 1024 classes")
    protocol = {"batch_size": 8, "max_length": 128, "pooling": "masked_mean_content",
                "pair_features": "concat_u_v_absdiff_product", "dtype": "float32", "device": "cpu"}
    overrides = manifest.get("protocol", {})
    if overrides.keys() - {"batch_size", "max_length", "pooling", "pair_features", "dtype", "device"}:
        raise ValueError("Unknown feature protocol option")
    for key in ("pooling", "pair_features", "dtype", "device"):
        if key in overrides and overrides[key] != protocol[key]:
            raise ValueError("The frozen content-mean protocol cannot be changed")
    protocol.update(overrides)
    positive_integer(protocol["batch_size"], "batch_size")
    positive_integer(protocol["max_length"], "max_length")
    if protocol["max_length"] > 128:
        raise ValueError("The shared encoded field budget is at most 128 tokens")
    base, inputs, rows, seen_ids, pair_shapes = manifest_path.parent, [], {}, set(), set()
    assessment = manifest.get("assessment_protocol")
    assessment_record = None
    if assessment is not None:
        path = resolve_path(base, assessment["path"])
        sha = file_digest(path)
        if sha != assessment["sha256"]:
            raise ValueError("Assessment protocol bytes differ from the private manifest binding")
        assessment_record = {"name": "assessment_protocol", "path": str(path), "bytes": path.stat().st_size, "sha256": sha}
    for split in splits:
        declared = manifest["inputs"][split]
        path = resolve_path(base, declared["path"])
        sha = file_digest(path)
        if sha != declared["sha256"]:
            raise ValueError("Input bytes differ from the private manifest binding")
        rows[split] = read_rows(path, split, label_count)
        ids = {row["id"] for row in rows[split]}
        if seen_ids & ids:
            raise ValueError("Row IDs must be disjoint across selected splits")
        seen_ids.update(ids)
        pair_shapes.add("text_pair" in rows[split][0])
        inputs.append({"name": split, "path": str(path), "bytes": path.stat().st_size, "sha256": sha})
    if len(pair_shapes) != 1:
        raise ValueError("Selected splits must have the same single/pair geometry")
    paired = pair_shapes.pop()
    protocol.update(pair_features=protocol["pair_features"] if paired else "none", threads=threads)
    protocol_sha = digest_json(protocol)
    manifest_sha = file_digest(manifest_path)
    torch.set_num_threads(threads)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "IN_PROGRESS.json", {"schema_version": 1, "manifest_sha256": manifest_sha,
                                              "protocol_sha256": protocol_sha, "selected_splits": splits})
    began = time.perf_counter()
    if progress:
        tqdm.write("Loading frozen encoder and verifying artifacts", file=sys.stderr)
    spec = manifest["model"]
    backend = spec["backend"]
    if backend not in ("nuri", "transformers"):
        raise ValueError("backend must be nuri or transformers")
    loader = load_nuri if backend == "nuri" else load_transformers
    model, tokenizer, special, model_records, tokenizer_records, model_details = loader(spec, base)
    load_seconds = time.perf_counter() - began
    if protocol["max_length"] > model_details["maximum_length"]:
        raise ValueError("The shared field budget exceeds this encoder's position limit")
    runtime = {"torch": torch.__version__, "numpy": np.__version__,
               "extractor_sha256": file_digest(Path(__file__))}
    if backend == "nuri":
        runtime["nuri_source_sha256"] = digest_json({path.name: file_digest(path)
                                                   for path in sorted((Path(__file__).parent.parent / "nuri").glob("*.py"))})
    else:
        import transformers
        runtime["transformers"] = transformers.__version__
    model_sha = digest_json({"backend": backend, "artifacts": records_identity(model_records),
                             "details": model_details, "runtime": runtime})
    tokenizer_sha = digest_json(records_identity(tokenizer_records))
    identity = {"schema_version": 1, "backend": backend, "model_id": spec["path_or_id"],
                "model_revision": spec.get("revision") or model_details.get("selected_tensor_sha256"),
                "pooling": protocol["pooling"],
                "pair_features": protocol["pair_features"], "label_count": label_count,
                "assessment_protocol_sha256": assessment_record["sha256"] if assessment_record else None,
                "protocol_sha256": protocol_sha, "model_sha256": model_sha, "tokenizer_sha256": tokenizer_sha}
    report = {**identity, "status": "complete", "manifest_sha256": manifest_sha, "protocol": protocol,
              "runtime": runtime, "model_details": model_details, "model_artifacts": model_records,
              "tokenizer_artifacts": tokenizer_records, "assessment_protocol": assessment_record,
              "load_seconds": load_seconds,
              "used_parameter_count": sum(value.numel() for value in model.parameters()), "splits": {}}
    for split in splits:
        split_rows = rows[split]
        features, first = encode_field(backend, model, tokenizer, special, [row["text"] for row in split_rows],
                                       protocol["batch_size"], protocol["max_length"], progress=progress,
                                       description=f"{split}/text")
        field_stats = {"text": first}
        if paired:
            second, stats = encode_field(backend, model, tokenizer, special, [row["text_pair"] for row in split_rows],
                                         protocol["batch_size"], protocol["max_length"], progress=progress,
                                         description=f"{split}/text_pair")
            features = combine_pair(features, second)
            field_stats["text_pair"] = stats
        ids = [row["id"] for row in split_rows]
        input_sha = next(row["sha256"] for row in inputs if row["name"] == split)
        metadata = {**identity, "split": split, "raw_input_sha256": input_sha,
                    "row_ids_sha256": row_ids_digest(ids)}
        destination = output_dir / f"{split}.npz"
        feature_sha = save_features(destination, features, np.asarray([row["label"] for row in split_rows], dtype=np.int64),
                                    ids, metadata)
        report["splits"][split] = {"rows": len(split_rows), "feature_dimension": features.shape[1],
                                   "raw_input_sha256": input_sha, "row_ids_sha256": metadata["row_ids_sha256"],
                                   "feature_file": destination.name, "feature_sha256": feature_sha,
                                   "observed_lengths": field_stats}
    for record in [*inputs, *model_records, *tokenizer_records, *([assessment_record] if assessment_record else [])]:
        if file_digest(Path(record["path"])) != record["sha256"]:
            raise ValueError("A bound input or model/tokenizer artifact changed during extraction")
    if file_digest(manifest_path) != manifest_sha:
        raise ValueError("The private manifest changed during extraction")
    write_json(output_dir / "report.json", report)
    write_json(output_dir / "COMPLETE.json", {"schema_version": 1, "status": "complete",
                                            "manifest_sha256": manifest_sha, "protocol_sha256": protocol_sha,
                                            "model_sha256": model_sha, "tokenizer_sha256": tokenizer_sha,
                                            "assessment_protocol_sha256": identity["assessment_protocol_sha256"],
                                            "report_sha256": file_digest(output_dir / "report.json"),
                                            "features": {split: report["splits"][split]["feature_sha256"] for split in splits}})
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "valid", "test"), default=("train", "valid"))
    parser.add_argument("--allow-test-completion", action="store_true")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    report = extract_features(args.manifest, args.output, args.splits,
                              allow_test_completion=args.allow_test_completion, threads=args.threads,
                              progress=not args.no_progress)
    print(json.dumps({"status": report["status"], "splits": {split: row["rows"] for split, row in report["splits"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
