import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from scripts import encoder_probe as probe


def write_features(path, split, features, labels, *, protocol, metadata_changes=None, ids=None):
    labels = np.asarray(labels, dtype=np.int64)
    ids = ids or [f"{split}-{i}" for i in range(len(labels))]
    metadata = {
        "schema_version": 1, "split": split, "backend": "nuri", "model_id": "encoder-control",
        "model_revision": "revision-1", "pooling": "masked_mean_content", "pair_features": "none",
        "label_count": 2, "protocol_sha256": "a" * 64,
        "assessment_protocol_sha256": probe.file_sha256(protocol),
        "raw_input_sha256": ("b" if split == "train" else "c") * 64,
        "model_sha256": "d" * 64, "tokenizer_sha256": "e" * 64,
        "row_ids_sha256": probe.row_ids_sha256(ids),
    }
    metadata.update(metadata_changes or {})
    np.savez(path, features=np.asarray(features, dtype=np.float32), labels=labels,
             row_ids=np.asarray(ids, dtype=str), metadata_json=np.asarray(json.dumps(metadata)))
    return path


@pytest.fixture
def inputs(tmp_path):
    directory = tmp_path / ".private"
    directory.mkdir()
    protocol = directory / "protocol.json"
    protocol.write_text('{"schema_version":1,"selection":"valid"}\n', encoding="utf-8")
    train = write_features(directory / "train.npz", "train",
                           [[-3, 1], [-2, 1], [-1, 1], [1, 1], [2, 1], [3, 1]],
                           [0, 0, 0, 1, 1, 1], protocol=protocol)
    valid = write_features(directory / "valid.npz", "valid", [[-2, 1], [-1, 1], [1, 1], [2, 1]],
                           [0, 0, 1, 1], protocol=protocol)
    return directory, protocol, train, valid


def run(inputs, **kwargs):
    directory, protocol, train, valid = inputs
    return probe.run_probe(train, valid, directory / "report.json", protocol_path=protocol,
                           bootstrap_samples=40, progress=False, **kwargs)


def test_standardizer_uses_train_only_and_handles_constant_columns():
    train = np.array([[1, 9], [3, 9]], dtype=np.float32)
    mean, scale = probe.fit_standardizer(train)
    np.testing.assert_array_equal(mean, [2, 9])
    np.testing.assert_array_equal(scale, [1, 1])
    valid = np.array([[1000, -1000]], dtype=np.float32)
    transformed = (valid - mean) / scale
    np.testing.assert_array_equal(transformed, [[998, -1009]])
    np.testing.assert_array_equal(probe.fit_standardizer(train)[0], mean)


def test_primal_and_dual_solve_same_centered_ridge_with_unpenalized_intercept():
    features = np.array([[-2, 1, 0], [-1, 0, 2], [0, -1, 0], [1, 0, -1], [2, 1, 2]], dtype=np.float64)
    features -= features.mean(axis=0)
    labels = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    targets = np.eye(2)[labels]
    intercept = targets.mean(axis=0)
    targets -= intercept
    primal = probe.ridge_weights(features, features.T @ features, features.T @ targets, 0.1, False)
    dual = probe.ridge_weights(features, features @ features.T, targets, 0.1, True)
    np.testing.assert_allclose(primal, dual, atol=1e-12)
    np.testing.assert_allclose(intercept, [0.6, 0.4])
    scores = features @ primal + intercept
    np.testing.assert_allclose(scores.mean(axis=0), intercept, atol=1e-12)


def test_dual_selected_when_dimensions_exceed_train_rows():
    features = np.zeros((3, 20), dtype=np.float64)
    gram, rhs, intercept, dual = probe.ridge_system(features, np.array([0, 0, 1]), 2)
    assert dual and gram.shape == (3, 3) and rhs.shape == (3, 2)
    weights = probe.ridge_weights(features, gram, rhs, 100, dual)
    assert weights.shape == (20, 2)
    np.testing.assert_array_equal(probe.predict(features, weights, intercept), [0, 0, 0])


def test_macro_f1_keeps_declared_missing_classes():
    labels = np.array([0, 0, 0], dtype=np.int64)
    measured = probe.classification_metrics(labels, labels, 3)
    assert measured["accuracy"] == 1
    assert measured["macro_f1"] == pytest.approx(1 / 3)
    assert measured["support"] == [3, 0, 0]
    assert measured["per_class_f1"] == [1, 0, 0]


def test_candidate_selection_uses_valid_then_accuracy_then_grid_order():
    candidates = [
        {"train": {"macro_f1": 1}, "valid": {"macro_f1": 0.4, "accuracy": 1}},
        {"train": {"macro_f1": 0}, "valid": {"macro_f1": 0.5, "accuracy": 0.7}},
        {"train": {"macro_f1": 1}, "valid": {"macro_f1": 0.5, "accuracy": 0.8}},
        {"train": {"macro_f1": 0}, "valid": {"macro_f1": 0.5, "accuracy": 0.8}},
    ]
    assert probe.select_candidate(candidates) == 2


def test_bootstrap_is_deterministic_and_identical_paired_models_have_zero_delta():
    labels = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    current = np.array([0, 1, 1, 0, 1], dtype=np.int64)
    reference = np.array([0, 0, 1, 1, 0], dtype=np.int64)
    first = probe.paired_bootstrap(labels, current, 2, samples=80, seed=3, reference=reference)
    assert first == probe.paired_bootstrap(labels, current, 2, samples=80, seed=3, reference=reference)
    assert first["row_paired"]
    identity = probe.paired_bootstrap(labels, current, 2, samples=80, seed=3, reference=current)
    assert all(values == {"estimate": 0, "low": 0, "high": 0} for values in identity["metrics"].values())


def test_complete_probe_binds_inputs_and_fixed_grid_without_encoder_updates(inputs):
    directory, protocol, train, valid = inputs
    before = [probe.file_sha256(path) for path in (protocol, train, valid)]
    report = run(inputs)
    assert report["test_used"] is False and report["encoder_updated"] is False
    assert report["protocol"]["alpha_grid"] == [0.01, 0.1, 1, 10, 100]
    assert report["selected_alpha"] == 0.01
    assert report["selected_metrics"]["valid"]["accuracy"] == 1
    assert report["selected_metrics"]["valid"]["macro_f1"] == 1
    assert report["controls"]["train_class_priors"] == [0.5, 0.5]
    assert report["controls"]["majority_label"] == 0
    assert report["controls"]["majority_valid"]["accuracy"] == 0.5
    assert report["geometry"]["train_rows"] == 6 and report["geometry"]["valid_rows"] == 4
    assert report["frozen_protocol_sha256"] == before[0]
    assert report["inputs"]["train"]["sha256"] == before[1]
    assert json.loads((directory / "report.json").read_text(encoding="utf-8")) == report
    assert [probe.file_sha256(path) for path in (protocol, train, valid)] == before
    with pytest.raises(FileExistsError, match="already exists"):
        run(inputs)


def test_cross_model_pairing_matches_assessment_and_rows_not_tokenizer_or_feature_recipe(inputs):
    directory, protocol, train, valid = inputs
    first = run(inputs)
    changes = {"backend": "lexical", "model_id": "lexical-control", "model_revision": "revision-2",
               "model_sha256": "f" * 64, "tokenizer_sha256": "0" * 64,
               "protocol_sha256": "1" * 64, "pooling": "hashed_character_ngrams"}
    train2 = write_features(directory / "lexical-train.npz", "train", [[-1], [-1], [-1], [1], [1], [1]],
                            [0, 0, 0, 1, 1, 1], protocol=protocol, metadata_changes=changes)
    valid2 = write_features(directory / "lexical-valid.npz", "valid", [[-1], [-1], [1], [1]],
                            [0, 0, 1, 1], protocol=protocol, metadata_changes=changes)
    second = probe.run_probe(train2, valid2, directory / "lexical.json", protocol_path=protocol,
                             bootstrap_samples=40, progress=False, reference_reports=(directory / "report.json",))
    assert second["paired_references"][0]["reference"]["model_sha256"] == first["inputs"]["valid"]["metadata"]["model_sha256"]
    assert second["paired_references"][0]["delta"]["metrics"]["accuracy"]["high"] == 0


def test_reference_refuses_changed_row_order_even_with_updated_digest(inputs):
    directory, _, _, valid = inputs
    report = run(inputs)
    report["validation_predictions"]["row_ids"].reverse()
    report["inputs"]["valid"]["metadata"]["row_ids_sha256"] = probe.row_ids_sha256(report["validation_predictions"]["row_ids"])
    reference = directory / "wrong-reference.json"
    reference.write_text(json.dumps(report), encoding="utf-8")
    loaded = probe.load_features(valid, "valid", 1024 * 1024)
    with pytest.raises(ValueError, match="bindings differ"):
        probe.reference_predictions(reference, loaded)


@pytest.mark.parametrize("changes, match", [
    ({"split": "test"}, "TEST"),
    ({"label_count": 1}, "label_count"),
    ({"row_ids_sha256": "0" * 64}, "row ID digest"),
    ({"model_revision": ""}, "model_revision"),
    ({"pooling": "hashed_character_ngrams"}, "unsupported"),
])
def test_metadata_validation_refuses_test_and_invalid_bindings(inputs, changes, match):
    directory, protocol, _, _ = inputs
    path = write_features(directory / "invalid.npz", "valid", [[0], [1]], [0, 1],
                          protocol=protocol, metadata_changes=changes)
    with pytest.raises(ValueError, match=match):
        probe.load_features(path, "valid", 1024 * 1024)


def test_split_overlap_and_different_model_are_rejected(inputs):
    directory, protocol, train, valid = inputs
    loaded_train = probe.load_features(train, "train", 1024 * 1024)
    overlapping = write_features(directory / "overlap.npz", "valid", [[0], [1]], [0, 1], protocol=protocol,
                                 ids=["train-0", "valid-new"])
    # Same feature dimension so the row overlap, rather than geometry, is tested.
    loaded_train.features = loaded_train.features[:, :1]
    with pytest.raises(ValueError, match="row IDs overlap"):
        probe.validate_splits(loaded_train, probe.load_features(overlapping, "valid", 1024 * 1024))
    loaded_valid = probe.load_features(valid, "valid", 1024 * 1024)
    loaded_train.features = np.zeros((6, 2), dtype=np.float32)
    loaded_valid.metadata["model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bindings differ"):
        probe.validate_splits(loaded_train, loaded_valid)


def test_nonfinite_features_and_duplicate_ids_refused(inputs):
    directory, protocol, _, _ = inputs
    bad = write_features(directory / "nan.npz", "valid", [[np.nan], [1]], [0, 1], protocol=protocol)
    with pytest.raises(ValueError, match="finite"):
        probe.load_features(bad, "valid", 1024 * 1024)
    duplicate = write_features(directory / "duplicate.npz", "valid", [[0], [1]], [0, 1],
                               protocol=protocol, ids=["same", "same"])
    with pytest.raises(ValueError, match="unique"):
        probe.load_features(duplicate, "valid", 1024 * 1024)


def test_npz_header_size_refused_before_array_allocation(inputs, monkeypatch):
    directory, _, _, valid = inputs
    malformed = directory / "malformed.npz"
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": "<f4", "fortran_order": False,
                                               "shape": (1000000000, 1000000000)})
    with zipfile.ZipFile(valid) as original, zipfile.ZipFile(malformed, "w") as target:
        for member in original.infolist():
            target.writestr(member.filename, header.getvalue() if member.filename == "features.npy"
                            else original.read(member))
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("invalid header reached array allocation"))
    with pytest.raises(ValueError, match="payload size"):
        probe.load_features(malformed, "valid", 1024 * 1024)


def test_workspace_refused_before_solve(inputs, monkeypatch):
    directory, protocol, _, _ = inputs
    train = write_features(directory / "large-train.npz", "train", np.zeros((200, 100)),
                           np.arange(200) % 2, protocol=protocol)
    valid = write_features(directory / "large-valid.npz", "valid", np.zeros((20, 100)),
                           np.arange(20) % 2, protocol=protocol)
    monkeypatch.setattr(np.linalg, "solve", lambda *a, **kw: pytest.fail("oversized workspace reached solver"))
    with pytest.raises(ValueError, match="workspace exceeds"):
        probe.run_probe(train, valid, directory / "large.json", protocol_path=protocol,
                        max_memory_mib=1, progress=False)


def test_frozen_assessment_protocol_must_match(inputs):
    directory, protocol, _, _ = inputs
    protocol.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="assessment digest"):
        run(inputs)
    assert not (directory / "report.json").exists()


def test_input_change_during_fit_cannot_publish_a_receipt(inputs, monkeypatch):
    directory, _, train, _ = inputs
    original = probe.ridge_weights
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        with train.open("ab") as stream:
            stream.write(b"mutation")
        return result
    monkeypatch.setattr(probe, "ridge_weights", changed)
    with pytest.raises(ValueError, match="changed during"):
        run(inputs)
    assert not (directory / "report.json").exists()


def test_cli_metrics_and_optional_tensorboard_are_train_valid_only(inputs, monkeypatch, capsys):
    directory, protocol, train, valid = inputs
    scalars = []
    closed = []
    writer = SimpleNamespace(add_scalar=lambda tag, value, step: scalars.append((tag, value, step)),
                             close=lambda: closed.append(True))
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", SimpleNamespace(SummaryWriter=lambda **kw: writer))
    result = probe.main(["--train-features", str(train), "--valid-features", str(valid),
                         "--protocol", str(protocol), "--output", str(directory / "cli.json"),
                         "--bootstrap-samples", "20", "--no-progress", "--tensorboard-dir", str(directory / "events")])
    assert result["test_used"] is False
    assert {tag for tag, _, _ in scalars} == {"train/accuracy", "train/macro_f1", "valid/accuracy", "valid/macro_f1", "probe/alpha"}
    assert len(scalars) == 25 and closed == [True]
    output = capsys.readouterr().out
    assert "VALID rows=4" in output and "TEST unused" in output
    assert str(directory) not in output


def test_outputs_must_remain_private(inputs, tmp_path):
    _, protocol, train, valid = inputs
    with pytest.raises(ValueError, match=".private"):
        probe.run_probe(train, valid, tmp_path / "public.json", protocol_path=protocol, progress=False)


def test_cli_does_not_accept_test_feature_inputs(inputs):
    directory, protocol, train, valid = inputs
    with pytest.raises(SystemExit) as error:
        probe.main(["--train-features", str(train), "--valid-features", str(valid), "--test-features", str(valid),
                    "--protocol", str(protocol), "--output", str(directory / "refused.json")])
    assert error.value.code == 2
