"""Build the single offline bundle that easy_run.py needs on the GPU host.

Usage
  python tools/pack_easy_run.py bundle --config <private easy_run.json> \
      --data-spec <private data spec JSON> --offline <private offline folder> \
      --out <private folder>/nuri-easy-run.tar.zst

The GPU host has no internet access, so the bundle carries everything:

- the committed project (git archive HEAD, so uncommitted edits are never
  shipped) and the private easy_run.json at its root;
- offline/: one python-build-standalone CPython 3.11 archive
  (offline/python/cpython-*.tar.gz), the pinned Linux wheels
  (offline/wheels/*.whl) and offline/requirements.txt;
- data/: the inputs named in the data spec, which is private JSON
  {"entries": {"<path inside data/>": "<local file or folder>", ...}},
  plus data/SHA256SUMS (sha256sum format) for every data file. A corpus
  tokens.bin stored as uint32 (older corpora) is packed as uint16, which
  halves the biggest part of the bundle; the local file is not changed.

Everything sits under one top folder, nuri/, compressed with zstd
(the zstandard Python package). The SHA256 of the bundle is printed and
written next to it as <bundle>.sha256. On the host:

  sha256sum -c nuri-easy-run.tar.zst.sha256
  tar --zstd -xf nuri-easy-run.tar.zst      (or: zstd -dc ... | tar -x)
  cd nuri && python3 easy_run.py

Keep the bundle, the config and the data spec in .private/ or another
ignored folder, because they contain or name private inputs (D-008).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TOP = "nuri"  # the folder the bundle unpacks into
BLOCK = 16 * 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


class HashingReader:
    """Wrap a binary file so that reading it also computes its SHA256."""

    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        return block


class NarrowingReader:
    """Read a uint32 tokens.bin as the same ids in uint16 (little-endian).

    Every block is checked before it is narrowed, so an id above 65535
    stops the packing instead of being wrapped around silently.
    """

    def __init__(self, stream):
        self.stream = stream
        self.pending = memoryview(b"")

    def next_block(self) -> memoryview:
        block = self.stream.read(BLOCK)  # BLOCK is a multiple of 4 bytes
        if len(block) % 4:
            raise SystemExit("a uint32 tokens.bin ended inside a token id")
        ids = np.frombuffer(block, dtype="<u4")
        if ids.size and int(ids.max()) > np.iinfo(np.uint16).max:
            raise SystemExit(
                "a tokens.bin holds an id above 65535; it cannot be uint16"
            )
        return memoryview(ids.astype("<u2").tobytes())

    def read(self, size=-1):
        pieces, wanted = [], size if size >= 0 else float("inf")
        while wanted > 0:
            if not self.pending:
                self.pending = self.next_block()
                if not self.pending:
                    break
            piece = self.pending[: min(wanted, len(self.pending))]
            self.pending = self.pending[len(piece) :]
            pieces.append(piece)
            wanted -= len(piece)
        return b"".join(pieces)


def stored_as_uint32(path: Path) -> bool:
    """True for a corpus tokens.bin whose size is 4 bytes per token of its offsets."""
    offsets_path = path.with_name("offsets.npy")
    if path.name != "tokens.bin" or not offsets_path.is_file():
        return False
    token_count = int(np.load(offsets_path, mmap_mode="r", allow_pickle=False)[-1])
    return token_count > 0 and path.stat().st_size == 4 * token_count


def data_files(spec_path: Path) -> list[tuple[str, Path]]:
    """(path inside data/, local file) for every file of the spec, sorted."""
    entries = json.loads(spec_path.read_text(encoding="utf-8"))["entries"]
    if not entries:
        raise SystemExit("the data spec has no entries")
    files = []
    for name, local in entries.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise SystemExit(f"data names must be relative and stay inside: {name}")
        local = Path(local)
        if not local.exists():
            raise SystemExit(f"missing input for {name}: {local}")
        files += list(walk(local, name))
    return sorted(files)


def walk(local: Path, name: str):
    """Yield (name, path) for a file or every file below a folder; links are refused."""
    if local.is_symlink():
        raise SystemExit(f"links are not packed: {local}")
    if local.is_dir():
        for child in sorted(local.iterdir()):
            yield from walk(child, f"{name}/{child.name}")
    else:
        yield name, local


def offline_files(offline: Path) -> list[tuple[str, Path]]:
    """(path inside offline/, local file) after checking the expected layout."""
    archives = sorted((offline / "python").glob("cpython-*.tar.gz"))
    wheels = sorted((offline / "wheels").glob("*.whl"))
    requirements = offline / "requirements.txt"
    if len(archives) != 1 or not wheels or not requirements.is_file():
        raise SystemExit(
            f"{offline} must hold python/cpython-*.tar.gz (exactly one), "
            "wheels/*.whl and requirements.txt"
        )
    files = [
        ("requirements.txt", requirements),
        (f"python/{archives[0].name}", archives[0]),
    ]
    return files + [(f"wheels/{wheel.name}", wheel) for wheel in wheels]


def check_config(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "<" in json.dumps(config.get("inputs", {})):
        raise SystemExit("easy_run.json still has placeholder input paths")
    if "url" in config.get("data", {}):
        raise SystemExit("easy_run.json names a data URL; the bundle carries the data")


def add_file(
    bundle: tarfile.TarFile, path: Path, name: str, narrow: bool = False
) -> str:
    """Add one file under TOP/name; returns the SHA256 of the packed bytes.

    narrow=True packs a uint32 tokens.bin as uint16 (half the size).
    """
    info = bundle.gettarinfo(str(path), arcname=f"{TOP}/{name}")
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    if narrow:
        info.size //= 2
    with path.open("rb") as stream:
        reader = HashingReader(NarrowingReader(stream) if narrow else stream)
        bundle.addfile(info, reader)
    return reader.digest.hexdigest()


def add_bytes(bundle: tarfile.TarFile, payload: bytes, name: str, mode=0o644) -> None:
    info = tarfile.TarInfo(f"{TOP}/{name}")
    info.size, info.mode = len(payload), mode
    bundle.addfile(info, io.BytesIO(payload))


def pack_bundle(
    config_path: Path, spec_path: Path, offline: Path, out: Path, level: int = 6
) -> str:
    """Write the zstd-compressed bundle and its .sha256 file; returns the SHA256."""
    import zstandard  # only the packing PC needs it

    check_config(config_path)
    data = data_files(spec_path)
    extra = offline_files(offline)
    committed = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    temporary = out.with_name(out.name + ".building")
    compressor = zstandard.ZstdCompressor(level=level, threads=-1)
    with temporary.open("wb") as raw, compressor.stream_writer(raw) as compressed:
        with tarfile.open(
            fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT
        ) as bundle:
            with tarfile.open(fileobj=io.BytesIO(committed)) as source:
                for member in source.getmembers():
                    content = source.extractfile(member) if member.isfile() else None
                    member.name = f"{TOP}/{member.name}"
                    bundle.addfile(member, content)
            add_bytes(bundle, config_path.read_bytes(), "easy_run.json")
            for name, path in extra:
                add_file(bundle, path, f"offline/{name}")
            sums = []
            for name, path in data:
                narrow = stored_as_uint32(path)
                print(
                    f"adding data/{name} ({path.stat().st_size / 1e9:.2f} GB"
                    + (", uint32 -> uint16)" if narrow else ")"),
                    flush=True,
                )
                digest = add_file(bundle, path, f"data/{name}", narrow)
                sums.append(f"{digest}  {name}\n")
            add_bytes(bundle, "".join(sums).encode("utf-8"), "data/SHA256SUMS")
    temporary.replace(out)
    digest = file_sha256(out)
    # Bytes, not text: on Windows write_text would end the line with CRLF,
    # and `sha256sum -c` on the Linux host would then look for "name\r".
    out.with_name(out.name + ".sha256").write_bytes(
        f"{digest}  {out.name}\n".encode("ascii")
    )
    return digest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="kind", required=True)
    bundle = sub.add_parser(
        "bundle", help="pack code, config, offline packages and data"
    )
    bundle.add_argument("--config", type=Path, required=True)
    bundle.add_argument("--data-spec", type=Path, required=True)
    bundle.add_argument("--offline", type=Path, required=True)
    bundle.add_argument("--out", type=Path, required=True)
    bundle.add_argument("--level", type=int, default=6, help="zstd level (default 6)")
    args = parser.parse_args(argv)
    digest = pack_bundle(
        args.config, args.data_spec, args.offline, args.out, args.level
    )
    print(f"{args.out}: {args.out.stat().st_size / 1e9:.2f} GB, sha256 {digest}")


if __name__ == "__main__":
    main()
