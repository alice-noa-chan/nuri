"""Build the two archives that easy_run.py needs on the GPU host.

Usage
  python tools/pack_easy_run.py data --spec <private data spec JSON> --out <data archive.tar>
  python tools/pack_easy_run.py code --config <private easy_run.json> --out nuri-easy-run.tar.gz

data  Packs local input folders and files under the archive names that
      easy_run.json refers to, then prints the archive SHA256 for
      easy_run.json "data.sha256". Upload the archive and put its download URL
      in "data.url". The spec is private JSON: {"entries": {"<name in archive>":
      "<local path>", ...}}. Folders are packed recursively; links are refused.

code  Packs the committed project (git archive HEAD, so uncommitted edits are
      never shipped) plus the private easy_run.json at the archive root. On the
      host: unpack it and run `python easy_run.py`.

Keep both archives and the data spec in .private/ or another ignored folder,
because they name private inputs (D-008).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pack_data(spec_path: Path, out: Path) -> str:
    """Write a plain tar of the spec entries; returns its SHA256."""
    entries = json.loads(spec_path.read_text(encoding="utf-8"))["entries"]
    if not entries:
        raise SystemExit("the data spec has no entries")
    for name, local in entries.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise SystemExit(f"archive names must be relative and stay inside: {name}")
        if not Path(local).exists():
            raise SystemExit(f"missing input for {name}: {local}")
    temporary = out.with_suffix(out.suffix + ".building")
    with tarfile.open(temporary, "w") as bundle:
        for name, local in sorted(entries.items()):
            add_tree(bundle, Path(local), name)
    temporary.replace(out)
    return file_sha256(out)


def add_tree(bundle: tarfile.TarFile, local: Path, name: str) -> None:
    """Add a file or a folder (recursively), refusing symbolic links."""
    if local.is_symlink():
        raise SystemExit(f"links are not packed: {local}")
    bundle.add(local, arcname=name, recursive=False)
    if local.is_dir():
        for child in sorted(local.iterdir()):
            add_tree(bundle, child, f"{name}/{child.name}")


def pack_code(config_path: Path, out: Path) -> None:
    """git archive HEAD plus easy_run.json at the root, as one .tar.gz."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "<" in config.get("data", {}).get("url", "<"):
        raise SystemExit("easy_run.json still has the placeholder data URL")
    committed = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    with tarfile.open(out, "w:gz") as bundle:
        with tarfile.open(fileobj=io.BytesIO(committed)) as source:
            for member in source.getmembers():
                content = source.extractfile(member) if member.isfile() else None
                bundle.addfile(member, content)
        bundle.add(config_path, arcname="easy_run.json")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="kind", required=True)
    data = sub.add_parser("data", help="pack the data archive")
    data.add_argument("--spec", type=Path, required=True)
    data.add_argument("--out", type=Path, required=True)
    code = sub.add_parser("code", help="pack the committed code with easy_run.json")
    code.add_argument("--config", type=Path, required=True)
    code.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.kind == "data":
        digest = pack_data(args.spec, args.out)
        print(f"{args.out}: {args.out.stat().st_size / 1e9:.2f} GB, sha256 {digest}")
    else:
        pack_code(args.config, args.out)
        print(f"{args.out}: {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
