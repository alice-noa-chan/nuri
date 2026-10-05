"""Prepare a verified position-aware warm start without changing a legacy checkpoint.

The output contains weights, configuration and any tokenizer files. It contains
no optimizer state. Keep the hash/seed lineage receipt in an ignored .private
directory, and use scripts/train.py --init for a fresh repair run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nuri.checkpoint import add_canvas_positions, atomic_write, load_pretrained, save_pretrained


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initialize(source: Path, output: Path, receipt: Path, seed: int = 0) -> dict:
    source, output, receipt = source.resolve(), output.resolve(), receipt.resolve()
    if ".private" not in receipt.parts:
        raise ValueError("Keep the initialization receipt under an ignored .private directory.")
    staging = output.with_name(output.name + ".building")
    if output.exists() or staging.exists() or receipt.exists():
        raise ValueError("Output, staging directory and receipt must be fresh paths.")
    weight_path = source / "model.safetensors"
    hashes = {name: file_hash(source / name) for name in ("config.json", "model.safetensors")}
    with safe_open(str(weight_path), framework="pt") as weights:
        dtypes = {weights.get_slice(name).get_dtype() for name in weights.keys()}
    supported = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
    if len(dtypes) != 1 or next(iter(dtypes)) not in supported:
        raise ValueError("The initializer requires one supported floating weight dtype.")
    dtype = supported[next(iter(dtypes))]
    model = load_pretrained(source, dtype=dtype)
    old_config = model.config.to_dict()
    original = load_file(str(weight_path))
    if original and all(name.startswith("model.") for name in original):
        original = {name[6:]: value for name, value in original.items()}
    add_canvas_positions(model, seed)
    save_pretrained(model, staging, dtype=dtype,
                    tokenizer_dir=source if (source / "tokenizer.json").exists() else None)
    saved = load_file(str(staging / "model.safetensors"))
    new_key = "readout.position_embedding.weight"
    if set(saved) != set(original) | {new_key}:
        raise RuntimeError("Unexpected tensor keys in the migrated initializer.")
    for name, tensor in original.items():
        if tensor.dtype != saved[name].dtype or not torch.equal(tensor, saved[name]):
            raise RuntimeError(f"An existing tensor changed: {name}")
    restored = load_pretrained(staging, dtype=dtype)
    expected_config = dict(old_config, canvas_position_encoding="learned")
    if restored.config.to_dict() != expected_config:
        raise RuntimeError("Unexpected configuration change in the initializer.")
    if any(file_hash(source / name) != value for name, value in hashes.items()):
        raise RuntimeError("The source checkpoint changed during initialization.")
    result = {
        "status": "verified_warm_start",
        "source": str(source), "output": str(output), "seed": seed,
        "source_hashes": hashes,
        "output_hashes": {name: file_hash(staging / name) for name in hashes},
        "new_tensor": new_key, "new_parameters": saved[new_key].numel(),
        "preserved_tensor_count": len(original), "preserved_weight_dtype": str(dtype),
        "optimizer_state_copied": False, "exact_resume": False,
    }
    staging.rename(output)
    atomic_write(receipt, lambda temporary: temporary.write_text(json.dumps(result, indent=2), encoding="utf-8"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = initialize(args.source, args.out, args.receipt, args.seed)
    print(f"Verified warm start: {result['preserved_tensor_count']} unchanged tensors; "
          f"{result['new_parameters']:,} new position weights. Use --init, with a fresh optimizer.")


if __name__ == "__main__":
    main()
