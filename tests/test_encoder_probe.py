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


def wider_protocol(inputs, grid=None, search_changes=None):
    directory, original, _, _ = inputs
    search = {"schema_version": 2, "alpha_grid": grid if grid is not None else [0.01, 1, 100, 10000, 1000000],
              "feature_assessment_protocol": {"path": original.name, "sha256": probe.file_sha256(original)}}
    search.update(search_changes or {})
    path = directory / "wider-protocol.json"
    path.write_text(json.dumps({"schema_version": 2, "ridge_probe": search}), encoding="utf-8")
    return path


def test_wider_search_preserves_cached_features_and_default_numerics(inputs):
    directory, original, train, valid = inputs
    before = {path: path.read_bytes() for path in (original, train, valid)}
    legacy = run(inputs)
    protocol = wider_protocol(inputs, list(probe.ALPHAS))
    widened = probe.run_probe(train, valid, directory / "wider.json", protocol_path=protocol,
                             bootstrap_samples=40, progress=False)
    assert widened["frozen_protocol_sha256"] == probe.file_sha256(protocol)
    assert widened["feature_assessment_protocol"]["sha256"] == probe.file_sha256(original)
    assert widened["inputs"]["valid"]["metadata"]["assessment_protocol_sha256"] == probe.file_sha256(original)
    assert widened["selection_boundary"] == {"lower": True, "upper": False, "requires_grid_review": True}
    for field in ("candidates", "selected_alpha", "selected_metrics", "controls", "validation_predictions",
                  "bootstrap", "standardizer_sha256", "geometry"):
        assert widened[field] == legacy[field]
    assert "selection_boundary" not in legacy and "feature_assessment_protocol" not in legacy
    assert "ridge_probe_schema_version" not in legacy["protocol"]
    assert all(path.read_bytes() == contents for path, contents in before.items())
    saved = (directory / "wider.json").read_bytes()
    with pytest.raises(FileExistsError):
        probe.run_probe(train, valid, directory / "wider.json", protocol_path=protocol, progress=False)
    assert (directory / "wider.json").read_bytes() == saved


@pytest.mark.parametrize("grid", [None, [], [1], [float(i + 1) for i in range(65)],
                                 [0, 1], [-1, 1], [1, float("inf")], [1, float("nan")],
                                 [1, True], [1, "2"], [2, 1], [1, 1], [1, 1.0], [1, 10 ** 400]])
def test_invalid_wider_grid_fails_before_loading_features(inputs, monkeypatch, grid):
    directory, _, train, valid = inputs
    protocol = wider_protocol(inputs, search_changes={"alpha_grid": grid})
    monkeypatch.setattr(probe, "inspect_npz", lambda *a: pytest.fail("invalid grid reached feature arrays"))
    with pytest.raises(ValueError, match="alpha_grid"):
        probe.run_probe(train, valid, directory / "invalid-grid.json", protocol_path=protocol, progress=False)
    assert not (directory / "invalid-grid.json").exists()


@pytest.mark.parametrize("changes", [{"schema_version": True}, {"schema_version": 1}, {"unknown": 1},
                                      {"feature_assessment_protocol": None},
                                      {"feature_assessment_protocol": {"path": "protocol.json", "sha256": "0" * 64}},
                                      {"feature_assessment_protocol": {"path": "protocol.json", "sha256": "A" * 64}},
                                      {"feature_assessment_protocol": {"path": "protocol.json", "sha256": "a" * 64,
                                                                       "extra": True}}])
def test_invalid_search_schema_and_original_binding_are_rejected(inputs, changes):
    directory, _, train, valid = inputs
    protocol = wider_protocol(inputs, search_changes=changes)
    with pytest.raises(ValueError):
        probe.run_probe(train, valid, directory / "invalid-search.json", protocol_path=protocol, progress=False)
    assert not (directory / "invalid-search.json").exists()


def test_correctly_hashed_original_protocol_must_match_cached_assessment(inputs):
    directory, _, train, valid = inputs
    other = directory / "other-assessment.json"
    other.write_text('{"different":true}', encoding="utf-8")
    protocol = wider_protocol(inputs, search_changes={"feature_assessment_protocol": {
        "path": other.name, "sha256": probe.file_sha256(other)}})
    with pytest.raises(ValueError, match="feature assessment digest"):
        probe.run_probe(train, valid, directory / "mismatch.json", protocol_path=protocol, progress=False)


@pytest.mark.parametrize("grid, expected", [
    ([100, 10000, 1000000], {"lower": True, "upper": False, "requires_grid_review": True}),
    ([0.01, 1, 100], {"lower": False, "upper": True, "requires_grid_review": True}),
    ([0.01, 1, 100, 10000, 1000000], {"lower": False, "upper": False, "requires_grid_review": False}),
])
def test_real_candidate_selection_reports_both_boundaries_and_interior(inputs, grid, expected):
    directory, original, _, _ = inputs
    rng = np.random.default_rng(1)
    train_values = rng.normal(size=(12, 8))
    train_labels = rng.integers(0, 2, size=12)
    valid_values = rng.normal(size=(8, 8))
    valid_labels = rng.integers(0, 2, size=8)
    train = write_features(directory / "noisy-train.npz", "train", train_values, train_labels, protocol=original)
    valid = write_features(directory / "noisy-valid.npz", "valid", valid_values, valid_labels, protocol=original)
    protocol = wider_protocol(inputs, grid)
    report = probe.run_probe(train, valid, directory / "boundaries.json", protocol_path=protocol,
                             bootstrap_samples=20, progress=False)
    assert report["selected_alpha"] == 100
    assert report["selection_boundary"] == expected
    assert [candidate["alpha"] for candidate in report["candidates"]] == grid


def test_wider_references_require_same_active_search_not_just_same_cache(inputs):
    directory, original, train, valid = inputs
    run(inputs)
    protocol = wider_protocol(inputs)
    with pytest.raises(ValueError, match="assessment protocol binding differs"):
        probe.run_probe(train, valid, directory / "mixed.json", protocol_path=protocol,
                        reference_reports=(directory / "report.json",), progress=False)
    assert not (directory / "mixed.json").exists()
    first = probe.run_probe(train, valid, directory / "first-wider.json", protocol_path=protocol,
                            bootstrap_samples=20, progress=False)
    second = probe.run_probe(train, valid, directory / "second-wider.json", protocol_path=protocol,
                             bootstrap_samples=20, progress=False, reference_reports=(directory / "first-wider.json",))
    assert second["paired_references"][0]["delta"]["metrics"]["accuracy"]["estimate"] == 0
    assert first["inputs"]["valid"]["metadata"]["assessment_protocol_sha256"] == probe.file_sha256(original)
    with pytest.raises(ValueError, match="assessment protocol binding differs"):
        probe.run_probe(train, valid, directory / "legacy-mixed.json", protocol_path=original,
                        reference_reports=(directory / "first-wider.json",), progress=False)
    protocol = wider_protocol(inputs, [0.01, 1, 100, 10000])
    with pytest.raises(ValueError, match="assessment protocol binding differs"):
        probe.run_probe(train, valid, directory / "different-grid.json", protocol_path=protocol,
                        reference_reports=(directory / "first-wider.json",), progress=False)


@pytest.mark.parametrize("artifact", ["original", "active", "reference"])
def test_mutated_wider_search_dependencies_cannot_publish(inputs, monkeypatch, artifact):
    directory, original, train, valid = inputs
    protocol = wider_protocol(inputs)
    reference = directory / "reference.json"
    probe.run_probe(train, valid, reference, protocol_path=protocol, bootstrap_samples=20, progress=False)
    target = {"original": original, "active": protocol, "reference": reference}[artifact]
    old_solve = probe.ridge_weights
    def mutate(*args, **kwargs):
        result = old_solve(*args, **kwargs)
        with target.open("ab") as stream:
            stream.write(b" ")
        return result
    monkeypatch.setattr(probe, "ridge_weights", mutate)
    with pytest.raises(ValueError, match="changed during"):
        probe.run_probe(train, valid, directory / "mutated.json", protocol_path=protocol, progress=False,
                        reference_reports=(reference,))
    assert not (directory / "mutated.json").exists()


def test_wider_prediction_retention_is_counted_in_memory_guard(inputs, monkeypatch):
    directory, _, train, valid = inputs
    protocol = wider_protocol(inputs, list(range(1, 65)))
    monkeypatch.setattr(probe, "workspace_bytes", lambda *a: 1024 * 1024 - 1)
    monkeypatch.setattr(probe, "ridge_weights", lambda *a: pytest.fail("oversized grid reached solver"))
    with pytest.raises(ValueError, match="workspace exceeds"):
        probe.run_probe(train, valid, directory / "memory.json", protocol_path=protocol,
                        max_memory_mib=1, progress=False)


def test_wider_tensorboard_directory_must_be_fresh(inputs):
    directory, _, train, valid = inputs
    protocol = wider_protocol(inputs)
    events = directory / "events"
    events.mkdir()
    with pytest.raises(FileExistsError):
        probe.run_probe(train, valid, directory / "existing-events.json", protocol_path=protocol,
                        tensorboard_dir=events, progress=False)
    assert not (directory / "existing-events.json").exists()


def test_actual_wider_tensorboard_and_progress_remain_train_valid_only(inputs, capsys):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    directory, _, train, valid = inputs
    grid = [0.01, 1, 100, 10000, 1000000]
    protocol = wider_protocol(inputs, grid)
    result = probe.main(["--train-features", str(train), "--valid-features", str(valid),
                         "--protocol", str(protocol), "--output", str(directory / "events-report.json"),
                         "--bootstrap-samples", "20", "--tensorboard-dir", str(directory / "fresh-events")])
    events = EventAccumulator(str(directory / "fresh-events"), size_guidance={"scalars": 0}).Reload()
    assert set(events.Tags()["scalars"]) == {
        "train/accuracy", "train/macro_f1", "valid/accuracy", "valid/macro_f1", "probe/alpha",
        "probe/selected_lower_boundary", "probe/selected_upper_boundary", "probe/requires_grid_review"}
    for split in ("train", "valid"):
        for metric in ("accuracy", "macro_f1"):
            points = events.Scalars(f"{split}/{metric}")
            assert [point.step for point in points] == list(range(1, len(grid) + 1))
            np.testing.assert_allclose([point.value for point in points],
                                       [candidate[split][metric] for candidate in result["candidates"]], atol=1e-7)
    np.testing.assert_allclose([point.value for point in events.Scalars("probe/alpha")], grid, rtol=1e-7)
    for flag, tag in (("lower", "selected_lower_boundary"), ("upper", "selected_upper_boundary"),
                      ("requires_grid_review", "requires_grid_review")):
        points = events.Scalars(f"probe/{tag}")
        assert [(point.step, point.value) for point in points] == [(len(grid), int(result["selection_boundary"][flag]))]
    captured = capsys.readouterr()
    assert "Frozen ridge: TRAIN fit / VALID selection" in captured.err
    assert "selected lower candidate" in captured.out and "TEST unused" in captured.out
    assert not events.Tags()["tensors"]
