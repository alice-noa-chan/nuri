import importlib.util
import json
import subprocess
import tarfile

import numpy as np
import pytest

from nuri.data import TokenCorpus, write_corpus
from test_easy_run import ROOT, load_easy_run, make_config, make_data

zstandard = pytest.importorskip("zstandard")


def load_packer():
    spec = importlib.util.spec_from_file_location(
        "pack_easy_run", ROOT / "tools" / "pack_easy_run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_offline(folder):
    (folder / "python").mkdir(parents=True)
    (
        folder
        / "python"
        / "cpython-3.11.17-x86_64-unknown-linux-gnu-install_only.tar.gz"
    ).write_bytes(b"py")
    (folder / "wheels").mkdir()
    (folder / "wheels" / "torch-2.10.0-cp311-cp311-linux_x86_64.whl").write_bytes(
        b"wheel"
    )
    (folder / "requirements.txt").write_text("torch==2.10.0\n")  # fmt: skip


def unpack(bundle_path, destination):
    with bundle_path.open("rb") as raw:
        with zstandard.ZstdDecompressor().stream_reader(raw) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as bundle:
                bundle.extractall(destination)


def is_git_checkout() -> bool:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True)
    return result.returncode == 0


@pytest.mark.skipif(not is_git_checkout(), reason="packing needs the git history")
def test_bundle_holds_code_config_offline_packages_and_checked_data(tmp_path):
    packer = load_packer()
    make_data(tmp_path / "local")
    # An older uint32 corpus is packed as uint16 with the same documents.
    train = tmp_path / "local" / "train" / "corpus"
    documents = [TokenCorpus(train).document(i) for i in range(40)]
    write_corpus(train, documents, dtype=np.uint32)
    make_offline(tmp_path / "offline")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"entries": {
        "tok": str(tmp_path / "local" / "tok"),
        "train": str(tmp_path / "local" / "train"),
        "valid": str(tmp_path / "local" / "valid"),
        "test": str(tmp_path / "local" / "test"),
    }}))  # fmt: skip
    config = tmp_path / "easy_run.json"
    config.write_text(json.dumps(make_config("data")))
    out = tmp_path / "bundle.tar.zst"
    digest = packer.pack_bundle(config, spec, tmp_path / "offline", out)
    # Exact bytes: sha256sum -c on Linux fails on a CRLF line ending.
    sidecar = (tmp_path / "bundle.tar.zst.sha256").read_bytes()
    assert sidecar == f"{digest}  bundle.tar.zst\n".encode("ascii")

    unpack(out, tmp_path / "host")
    top = tmp_path / "host" / "nuri"
    for name in ("easy_run.py", "scripts/run_release.py", "nuri/model.py", "easy_run.json",
                 "offline/requirements.txt", "data/SHA256SUMS", "data/tok/tokenizer.json",
                 "data/train/corpus/tokens.bin"):  # fmt: skip
        assert (top / name).is_file(), name
    assert not (top / ".private").exists()
    packed = TokenCorpus(top / "data" / "train" / "corpus")
    assert packed.tokens.dtype == np.uint16
    assert [packed.document(i) for i in range(40)] == documents
    assert TokenCorpus(train).tokens.dtype == np.uint32  # the local file is unchanged

    # The unpacked bundle passes easy_run's own checks.
    easy = load_easy_run()
    easy.ROOT, easy.OFFLINE = top, top / "offline"
    easy.WORK = tmp_path / "work"
    easy.LOG_PATH = easy.WORK / "easy_run.log"
    easy.DATA_MARKER = easy.WORK / "data_verified"
    easy.offline_files()
    easy.verify_data(json.loads((top / "easy_run.json").read_text()), {})
    assert easy.DATA_MARKER.is_file()


def test_bundle_inputs_are_checked_before_packing(tmp_path):
    packer = load_packer()
    make_offline(tmp_path / "offline")
    config = tmp_path / "easy_run.json"
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"entries": {"../x": str(tmp_path)}}))
    with pytest.raises(SystemExit, match="relative"):
        packer.data_files(spec)
    spec.write_text(json.dumps({"entries": {"x": str(tmp_path / "missing")}}))
    with pytest.raises(SystemExit, match="missing input"):
        packer.data_files(spec)

    with_url = make_config("data")
    with_url["data"]["url"] = "https://example.invalid/data.tar"
    config.write_text(json.dumps(with_url))
    with pytest.raises(SystemExit, match="data URL"):
        packer.check_config(config)
    placeholder = make_config("data")
    placeholder["inputs"]["tokenizer"] = "<tokenizer folder>"
    config.write_text(json.dumps(placeholder))
    with pytest.raises(SystemExit, match="placeholder"):
        packer.check_config(config)

    (tmp_path / "offline" / "requirements.txt").unlink()
    with pytest.raises(SystemExit, match="requirements.txt"):
        packer.offline_files(tmp_path / "offline")


def test_narrowing_keeps_ids_and_refuses_ids_above_uint16(tmp_path):
    packer = load_packer()
    ids = np.arange(0, 65536, 7, dtype=np.uint32)
    write_corpus(tmp_path / "ok", [ids.tolist()], dtype=np.uint32)
    assert packer.stored_as_uint32(tmp_path / "ok" / "tokens.bin")
    write_corpus(tmp_path / "small", [ids.tolist()])
    assert not packer.stored_as_uint32(tmp_path / "small" / "tokens.bin")
    assert not packer.stored_as_uint32(tmp_path / "ok" / "offsets.npy")

    packer.BLOCK = 12  # several blocks, and reads that cross block borders
    with (tmp_path / "ok" / "tokens.bin").open("rb") as stream:
        reader = packer.NarrowingReader(stream)
        narrowed = b"".join(iter(lambda: reader.read(5), b""))
    assert np.array_equal(np.frombuffer(narrowed, dtype="<u2"), ids)

    np.array([5, 65536], dtype=np.uint32).tofile(tmp_path / "big.bin")
    with (tmp_path / "big.bin").open("rb") as stream:
        with pytest.raises(SystemExit, match="above 65535"):
            packer.NarrowingReader(stream).read(4)
