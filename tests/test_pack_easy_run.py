import hashlib
import importlib.util
import json
import tarfile

import pytest

from test_easy_run import ROOT, load_easy_run, make_data


def load_packer():
    spec = importlib.util.spec_from_file_location(
        "pack_easy_run", ROOT / "tools" / "pack_easy_run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_data_archive_unpacks_with_easy_run(tmp_path):
    packer, easy = load_packer(), load_easy_run()
    make_data(tmp_path / "local")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"entries": {
        "tok": str(tmp_path / "local" / "tok"),
        "train/corpus": str(tmp_path / "local" / "train" / "corpus"),
    }}))  # fmt: skip
    digest = packer.pack_data(spec, tmp_path / "data.tar")
    assert digest == hashlib.sha256((tmp_path / "data.tar").read_bytes()).hexdigest()
    easy.safe_extract(tmp_path / "data.tar", tmp_path / "unpacked")
    assert (tmp_path / "unpacked" / "tok" / "tokenizer.json").is_file()
    assert (tmp_path / "unpacked" / "train" / "corpus" / "tokens.bin").is_file()
    spec.write_text(json.dumps({"entries": {"../x": str(tmp_path)}}))
    with pytest.raises(SystemExit, match="relative"):
        packer.pack_data(spec, tmp_path / "bad.tar")


def test_code_archive_holds_committed_files_and_the_config(tmp_path):
    packer = load_packer()
    config = tmp_path / "easy_run.json"
    config.write_text(json.dumps({"data": {"url": "<placeholder>"}}))
    with pytest.raises(SystemExit, match="placeholder"):
        packer.pack_code(config, tmp_path / "code.tar.gz")
    config.write_text(json.dumps({"data": {"url": "https://example.invalid/d.tar"}}))
    packer.pack_code(config, tmp_path / "code.tar.gz")
    with tarfile.open(tmp_path / "code.tar.gz") as bundle:
        names = set(bundle.getnames())
    assert {"easy_run.json", "scripts/run_release.py", "nuri/model.py"} <= names
    assert not any(name.startswith((".private", "data/", "runs/")) for name in names)
