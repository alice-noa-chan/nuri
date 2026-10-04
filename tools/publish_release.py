"""Publish the release folders of a finished run to the Hugging Face Hub (ARCHITECTURE.md §15).

Each stage with a "release_name" in the run plan becomes one repository:
its final export (written by scripts/run_release.py) goes to the main
branch, and with --intermediate every saved step-* and pre-decay
checkpoint of that stage is exported and published as a branch of the same
repository.

Nothing leaves the machine by default: the tool prints what it would upload.
Uploading needs --push, a --namespace and an explicit --visibility. Before
any upload every folder must pass three checks:

1. export verification: "<folder>.verify.json" exists and says "ok",
2. an allow-list of file names (weights, config, tokenizer files, model code,
   LICENSE, README); training states, logs and anything else are refused,
3. no file contains a line of .private/provenance-markers.txt (matched
   case-insensitively; the match itself is never printed).

With --tokenizer-repo NAME the shared tokenizer is also published as its own
repository (section 15 lists "nuri-tokenizer"). Its files are taken from the
release folders, which must all carry the identical tokenizer; the folder is
verified by loading the tokenizer, matching its vocabulary size and special
token ids to the model config, and round-tripping a short trilingual text.

Usage
  python tools/publish_release.py plan.json --intermediate              # dry run
  python tools/publish_release.py plan.json --intermediate --push \
      --namespace <user-or-org> --visibility private
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKERS = ROOT / ".private" / "provenance-markers.txt"
CODE_FILES = {"config.py", "layers.py", "moe.py", "encoder.py", "core.py", "readout.py", "model.py",
              "generation.py", "configuration_nuri.py", "modeling_nuri.py"}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
ALLOWED_FILES = {"model.safetensors", "config.json", "LICENSE", "README.md", *TOKENIZER_FILES} | CODE_FILES
REQUIRED = {"model": {"model.safetensors", "config.json", "LICENSE", "README.md"},
            "tokenizer": {"tokenizer.json", "LICENSE", "README.md"}}
ROUND_TRIP_TEXT = "Nuri reads 한국어, English and 日本語."
TEXT_SUFFIXES = {".json", ".md", ".py", ""}  # LICENSE has no suffix


class PublishError(Exception):
    """A folder failed a check; nothing was uploaded for it."""


def load_markers(path: Path | None = None) -> list[str]:
    path = MARKERS if path is None else path
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return [line.strip().casefold() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def check_folder(folder: Path, markers: list[str], kind: str = "model") -> None:
    """Raise PublishError unless the folder is verified, allow-listed and marker-free."""
    report = folder.parent / f"{folder.name}.verify.json"
    if not report.exists() or not json.loads(report.read_text(encoding="utf-8")).get("ok"):
        raise PublishError(f"{folder.name}: no passing export verification ({report.name})")
    files = [p for p in folder.rglob("*") if p.is_file()]
    unexpected = sorted(p.relative_to(folder).as_posix() for p in files if p.name not in ALLOWED_FILES
                        or p.parent != folder)
    if unexpected:
        raise PublishError(f"{folder.name}: files outside the allow-list: {', '.join(unexpected)}")
    missing = REQUIRED[kind] - {p.name for p in files}
    if missing:
        raise PublishError(f"{folder.name}: missing {', '.join(sorted(missing))}")
    for path in files:
        if path.suffix in TEXT_SUFFIXES and markers:
            text = path.read_text(encoding="utf-8", errors="replace").casefold()
            if any(marker in text for marker in markers):
                raise PublishError(f"{folder.name}: {path.name} contains a private provenance marker")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_tokenizer_folder(release_folders: list[Path], out: Path, name: str) -> None:
    """Write out/<name> with the shared tokenizer, LICENSE, a short card and a verification report."""
    sources = [f for f in release_folders if (f / "tokenizer.json").exists()]
    if not sources:
        raise PublishError(f"{name}: no release folder contains tokenizer.json")
    first = sources[0]
    for other in sources[1:]:
        for file in TOKENIZER_FILES:
            a, b = first / file, other / file
            if a.exists() != b.exists() or (a.exists() and _digest(a) != _digest(b)):
                raise PublishError(f"{name}: {other.name} carries a different {file}")
    out.mkdir(parents=True, exist_ok=True)
    for file in TOKENIZER_FILES:
        if (first / file).exists():
            shutil.copy2(first / file, out / file)
    shutil.copy2(ROOT / "LICENSE", out / "LICENSE")
    report = verify_tokenizer(out, first)
    (out / "README.md").write_text(tokenizer_card(name, report), encoding="utf-8")
    (out.parent / f"{out.name}.verify.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def verify_tokenizer(folder: Path, release: Path) -> dict:
    """Load the tokenizer, check it against the model config, and round-trip a short text."""
    sys.path.insert(0, str(ROOT))
    from nuri.config import NuriConfig
    from nuri.tokenizer import check_tokenizer, load_tokenizer, vocab_size

    tokenizer = load_tokenizer(folder)
    config = NuriConfig.from_dict(json.loads((release / "config.json").read_text(encoding="utf-8"))["nuri"])
    report = {"vocab_size": vocab_size(tokenizer), "pad_id": config.pad_id, "mask_id": config.mask_id,
              "eos_id": config.eos_id}
    try:
        check_tokenizer(tokenizer, config)
        ids = tokenizer.encode(ROUND_TRIP_TEXT, add_special_tokens=False).ids
        report["round_trip"] = tokenizer.decode(ids).replace(" ", "") == ROUND_TRIP_TEXT.replace(" ", "")
        report["ok"] = report["round_trip"]
    except ValueError as error:
        report.update(ok=False, error=str(error))
    return report


def tokenizer_card(name: str, report: dict) -> str:
    return f"""---
license: mit
language: [ko, en, ja]
library_name: tokenizers
tags: [nuri, tokenizer]
---

# {name}

The tokenizer shared by every Nuri model: {report['vocab_size']:,} entries for Korean and English
(primary) and some Japanese. Special token ids: `[PAD]` {report['pad_id']}, `[MASK]` {report['mask_id']},
`[EOS]` {report['eos_id']}; chat roles use `<|system|>`, `<|user|>` and `<|assistant|>`.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{name}")
```

## License

MIT
"""


def intermediate_checkpoints(stage_dir: Path) -> list[Path]:
    """Saved public checkpoints other than final: step-* in step order, then pre-decay."""
    steps = sorted((p for p in stage_dir.glob("step-*") if (p / "model.safetensors").exists()),
                   key=lambda p: int(p.name.split("-", 1)[1]))
    pre_decay = stage_dir / "pre-decay"
    return steps + ([pre_decay] if (pre_decay / "model.safetensors").exists() else [])


def export_intermediate(checkpoint: Path, out: Path, name: str, stage: str) -> None:
    """Export one intermediate checkpoint with the same script as the final release."""
    if (out.parent / f"{out.name}.verify.json").exists() and (out / "model.safetensors").exists():
        return  # already exported and verified earlier
    subprocess.run([sys.executable, str(ROOT / "scripts" / "export_release.py"), "--checkpoint", str(checkpoint),
                    "--out", str(out), "--name", name, "--stage", stage], check=True, cwd=ROOT)


def plan_uploads(plan: dict, intermediate: bool, export: bool = True) -> list[dict]:
    """(repository name, revision, folder) for every upload, in order: main first."""
    out = Path(plan["out"])
    uploads = []
    for stage_name, spec in plan["stages"].items():
        name = spec.get("release_name")
        if not name:
            continue
        uploads.append({"repo": name, "revision": "main", "folder": out / "release" / name, "kind": "model"})
        if intermediate:
            for checkpoint in intermediate_checkpoints(out / stage_name):
                folder = out / "release-revisions" / name / checkpoint.name
                if export:
                    export_intermediate(checkpoint, folder, name, spec["stage"])
                uploads.append({"repo": name, "revision": checkpoint.name, "folder": folder, "kind": "model"})
    return uploads


def push(uploads: list[dict], namespace: str, private: bool, api=None) -> None:
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    for item in uploads:
        repo_id = f"{namespace}/{item['repo']}"
        api.create_repo(repo_id, private=private, exist_ok=True)
        if item["revision"] != "main":
            api.create_branch(repo_id, branch=item["revision"], exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=str(item["folder"]), revision=item["revision"],
                          commit_message=f"Upload {item['repo']} ({item['revision']})")
        print(f"uploaded {repo_id}@{item['revision']}", flush=True)


def main(argv=None, api=None) -> list[dict]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plan", help="the run plan given to scripts/run_release.py")
    p.add_argument("--intermediate", action="store_true", help="also export and publish step-* / pre-decay")
    p.add_argument("--push", action="store_true", help="really upload (default: dry run)")
    p.add_argument("--namespace", help="Hub user or organization (required with --push)")
    p.add_argument("--visibility", choices=("public", "private"), help="required with --push")
    p.add_argument("--tokenizer-repo", help='also publish the shared tokenizer, e.g. "nuri-tokenizer"')
    args = p.parse_args(argv)
    if args.push and not (args.namespace and args.visibility):
        p.error("--push needs --namespace and --visibility")

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    uploads = plan_uploads(plan, args.intermediate, export=args.push or args.intermediate)
    failures = []
    if args.tokenizer_repo:
        folder = Path(plan["out"]) / "release" / args.tokenizer_repo
        try:
            build_tokenizer_folder([u["folder"] for u in uploads if u["revision"] == "main"], folder,
                                   args.tokenizer_repo)
        except PublishError as error:
            failures.append(str(error))
        uploads.insert(0, {"repo": args.tokenizer_repo, "revision": "main", "folder": folder, "kind": "tokenizer"})
    markers = load_markers()
    for item in uploads:
        try:
            check_folder(item["folder"], markers, item["kind"])
            item["checked"] = True
        except PublishError as error:
            item["checked"] = False
            failures.append(str(error))
        print(f"{'ok  ' if item['checked'] else 'FAIL'} {item['repo']}@{item['revision']}", flush=True)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        sys.exit("checks failed: nothing was uploaded")
    if not args.push:
        print(f"dry run: {len(uploads)} uploads would be made; add --push to upload")
        return uploads
    push(uploads, args.namespace, args.visibility == "private", api)
    return uploads


if __name__ == "__main__":
    main()
