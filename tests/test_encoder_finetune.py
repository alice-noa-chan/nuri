"""Synthetic CPU mechanics for the matched TRAIN/VALID adaptation worker."""

import copy
import json

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from nuri.checkpoint import save_pretrained
from nuri.config import tiny_config
from nuri.model import NuriModel
from scripts import encoder_finetune as worker


@pytest.fixture
def private(tmp_path):
    folder = tmp_path / ".private"
    folder.mkdir()
    return folder


def checkpoint(private, backend):
    folder = private / backend
    torch.manual_seed(3)
    if backend == "nuri":
        config = tiny_config(vocab_size=8, hidden_size=8, num_heads=2, ffn_size=16,
                             input_layers=1, readout_layers=1, latent_slots=2,
                             max_input_len=16, max_response_len=8, embed_dim=4)
        model = NuriModel(config)
        # The selected encoder must never use these unrelated task modules.
        with torch.no_grad():
            for parameter in model.core.parameters():
                parameter.fill_(float("nan"))
            model.embed_proj.weight.fill_(float("nan"))
        save_pretrained(model, folder, dtype=torch.float32)
        tokenizer = Tokenizer(models.WordLevel({"[PAD]": 0, "[MASK]": 1, "[EOS]": 2,
                                                "[UNK]": 3, "hello": 4, "world": 5,
                                                "alpha": 6, "beta": 7}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(str(folder / "tokenizer.json"))
    else:
        transformers = pytest.importorskip("transformers")
        config = transformers.BertConfig(vocab_size=9, hidden_size=8, num_hidden_layers=1,
                                          num_attention_heads=2, intermediate_size=16,
                                          max_position_embeddings=16)
        transformers.BertModel(config).save_pretrained(folder)
        tokenizer = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "[CLS]": 2,
                                                "[SEP]": 3, "[MASK]": 4, "hello": 5,
                                                "world": 6, "alpha": 7, "beta": 8}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]",
                                                                  special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
        transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]",
                                             unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]",
                                             mask_token="[MASK]").save_pretrained(folder)
    return folder


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def prepare(private, backend="nuri", paired=False, epochs=2):
    folder = checkpoint(private, backend)
    inputs = {}
    for split in ("train", "valid"):
        rows = [{"id": f"{split}-{index}", "text": text, "label": index % 2, "split": split}
                for index, text in enumerate(("hello", "world alpha", "unseen", "beta world"))]
        if paired:
            for row, text in zip(rows, ("alpha", "hello beta", "world", "unseen")):
                row["text_pair"] = text
        path = private / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        inputs[split] = {"path": path.name, "sha256": worker.features.file_digest(path)}
    manifest = {"schema_version": 1, "model": {"backend": backend, "path_or_id": folder.name,
                                               "revision": "a" * 40},
                "inputs": inputs, "label_count": 2, "protocol": {"max_length": 8}}
    protocol = {"schema_version": 1, "kind": "encoder_finetune", "learning_rates": [0.001, 0.01],
                "seeds": [0, 1], "epochs": epochs, "batch_size": 2, "max_length": 8,
                "label_count": 2, "task_type": "pair" if paired else "single"}
    manifest_path, protocol_path = private / "manifest.json", private / "protocol.json"
    write_json(manifest_path, manifest)
    write_json(protocol_path, protocol)
    return manifest_path, protocol_path, folder, manifest, protocol


def run(private, prepared, output="result", **kwargs):
    manifest_path, protocol_path = prepared[:2]
    return worker.run_candidate(manifest_path, protocol_path, private / output,
                                learning_rate=0.01, seed=0, progress=False, **kwargs)


@pytest.mark.parametrize("backend", ["nuri", "transformers"])
@pytest.mark.parametrize("paired", [False, True])
def test_actual_encoder_gradients_updates_and_source_isolation(private, monkeypatch, backend, paired):
    prepared = prepare(private, backend, paired)
    folder = prepared[2]
    hashes = {path.name: worker.features.file_digest(path) for path in folder.iterdir() if path.is_file()}
    load_name = "load_nuri" if backend == "nuri" else "load_transformers"
    original_loader = getattr(worker.features, load_name)
    loaded = []

    def load(*args):
        result = original_loader(*args)
        loaded.append((result[0], worker.features.tensor_digest(result[0])))
        return result

    monkeypatch.setattr(worker.features, load_name, load)
    adam = torch.optim.AdamW
    gradient_checks = []

    class CheckedAdam(adam):
        def step(self, *args, **kwargs):
            source = loaded[0][0]
            parameters = self.param_groups[0]["params"]
            encoder_count = len(list(source.parameters()))
            source_names = [name for name, _ in source.named_parameters()]
            contextual_indices = [index for index, name in enumerate(source_names)
                                  if name.startswith("encoder.")]
            assert contextual_indices
            assert all(parameter.requires_grad for parameter in parameters)
            assert all(copy_parameter.data_ptr() != original.data_ptr()
                       for copy_parameter, original in zip(parameters, source.parameters()))
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                       for parameter in parameters[:encoder_count])
            assert any(parameters[index].grad.abs().sum() > 0 for index in contextual_indices)
            before = [parameters[index].detach().clone() for index in contextual_indices]
            result = super().step(*args, **kwargs)
            assert any(not torch.equal(old, parameters[index]) for old, index in zip(before, contextual_indices))
            worker.verify_finite_state(source, torch.nn.Identity(), self)
            gradient_checks.append(True)
            return result

    monkeypatch.setattr(torch.optim, "AdamW", CheckedAdam)
    report = run(private, prepared)
    assert len(gradient_checks) == report["budget"]["optimizer_updates"] == 4
    assert report["budget"]["training_presentations"] == 8
    assert report["initial_encoder_tensor_sha256"] != report["final_encoder_tensor_sha256"]
    assert worker.features.tensor_digest(loaded[0][0]) == loaded[0][1]
    assert {path.name: worker.features.file_digest(path) for path in folder.iterdir() if path.is_file()} == hashes
    assert report["head_parameters"] == (8 * (4 if paired else 1) + 1) * 2
    assert report["pre_update_tokenization"]["train"]["fields"] == 4 * (2 if paired else 1)
    assert report["budget"]["actual_training_counts"]["fields"] == 8 * (2 if paired else 1)
    assert report["budget"]["actual_training_counts"]["truncated_fields"] == 0
    assert report["test_used"] is False and report["adapted_weights_retained"] is False
    assert report["finite_model_head_optimizer_state"] is True
    assert report["final_tensor_epoch"] == 2
    complete = json.loads((private / "result" / "COMPLETE.json").read_text())
    assert complete["report_sha256"] == worker.features.file_digest(private / "result" / "REPORT.json")
    assert not list((private / "result").glob("*.safetensors"))
    for record in report["source_bindings"]:
        assert worker.features.file_digest(worker.Path(record["path"])) == record["sha256"]


def test_seeded_replay_and_validation_epoch_selection(private):
    prepared = prepare(private, epochs=3)
    first, second = run(private, prepared, "first"), run(private, prepared, "second")
    assert first["epochs"] == second["epochs"]
    assert first["final_encoder_tensor_sha256"] == second["final_encoder_tensor_sha256"]
    assert first["selected_epoch"] == worker.select_candidate(first["epochs"]) + 1
    candidates = [{"train": {"macro_f1": 1}, "valid": {"macro_f1": .5, "accuracy": .8}},
                  {"train": {"macro_f1": 0}, "valid": {"macro_f1": .6, "accuracy": .7}},
                  {"train": {"macro_f1": 1}, "valid": {"macro_f1": .6, "accuracy": .7}}]
    assert worker.select_candidate(candidates) == 1
    candidates[2]["valid"]["accuracy"] = .9
    assert worker.select_candidate(candidates) == 2


@pytest.mark.parametrize("update", [
    {"schema_version": True}, {"test": {}}, {"epochs": 0}, {"batch_size": 65},
    {"max_length": 129}, {"label_count": 1}, {"task_type": "unsupported"},
    {"learning_rates": [0.01, 0.01]}, {"learning_rates": [float("nan")]},
    {"seeds": [True]}, {"seeds": [{}]}, {"optimizer": {"schedule": "other"}},
])
def test_invalid_protocol_fails_before_loading(private, monkeypatch, update):
    prepared = prepare(private)
    protocol = prepared[4] | update
    write_json(prepared[1], protocol)
    monkeypatch.setattr(worker.features, "load_nuri", lambda *args: pytest.fail("Model allocation must not occur"))
    with pytest.raises(ValueError):
        run(private, prepared)
    assert not (private / "result").exists()


@pytest.mark.parametrize("kwargs", [{"learning_rate": 0.1, "seed": 0}, {"learning_rate": .01, "seed": 2}])
def test_undeclared_candidate_rejected(private, kwargs):
    prepared = prepare(private)
    with pytest.raises(ValueError, match="explicitly declared"):
        worker.run_candidate(*prepared[:2], private / "result", progress=False, **kwargs)


@pytest.mark.parametrize("bad", ["test_field", "test_row", "shared_ids", "pair_geometry", "input_hash"])
def test_bad_input_refusal_precedes_model_loading(private, monkeypatch, bad):
    prepared = prepare(private)
    manifest = prepared[3]
    if bad == "test_field":
        manifest["inputs"]["test"] = {"path": "never-opened.jsonl", "sha256": "0" * 64}
    else:
        path = private / "valid.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if bad == "test_row":
            rows[0]["split"] = "test"
        elif bad == "shared_ids":
            rows[0]["id"] = "train-0"
        elif bad == "pair_geometry":
            for row in rows:
                row["text_pair"] = "hello"
        else:
            rows[0]["text"] += " alpha"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        if bad != "input_hash":
            manifest["inputs"]["valid"]["sha256"] = worker.features.file_digest(path)
    write_json(prepared[0], manifest)
    monkeypatch.setattr(worker.features, "load_nuri", lambda *args: pytest.fail("Model allocation must not occur"))
    with pytest.raises(ValueError):
        run(private, prepared)
    assert not (private / "result").exists()


@pytest.mark.parametrize("backend", ["nuri", "transformers"])
def test_all_validation_text_audited_before_first_update(private, monkeypatch, backend):
    prepared = prepare(private, backend, paired=True)
    path = private / "valid.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["text_pair"] = "hello " * 20
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    prepared[3]["inputs"]["valid"]["sha256"] = worker.features.file_digest(path)
    write_json(prepared[0], prepared[3])
    monkeypatch.setattr(torch.optim, "AdamW", lambda *args, **kwargs: pytest.fail("Truncation precedes optimizer allocation"))
    with pytest.raises(ValueError, match="truncation"):
        run(private, prepared)
    assert (private / "result" / "FAILED.json").exists()
    assert not (private / "result" / "COMPLETE.json").exists()


def test_assessment_eval_no_grad_and_pair_field_gradients(private, monkeypatch):
    prepared = prepare(private, paired=True)
    model, tokenizer, special, *_ = worker.features.load_nuri(prepared[3]["model"], private)
    model.requires_grad_(True)
    rows = worker.features.read_rows(private / "train.jsonl", "train", 2)
    pooled, counts = worker.pooled_batch(model, tokenizer, special, "nuri", rows, 8, torch.device("cpu"))
    pooled.sum().backward()
    assert model.embedding.weight.grad[6].abs().sum() > 0  # pair field alpha
    assert model.embedding.weight.grad[7].abs().sum() > 0  # pair field beta
    assert counts["fields"] == 8 and counts["content_positions"] > 8
    model.zero_grad(set_to_none=True)
    head = torch.nn.Linear(32, 2)
    original = worker.pooled_batch
    checks = []

    def checked(*args, **kwargs):
        assert not torch.is_grad_enabled() and not args[0].training
        checks.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(worker, "pooled_batch", checked)
    assessed = worker.assess(model, head, tokenizer, special, "nuri", rows, prepared[4], torch.device("cpu"))
    assert len(checks) == 2 and assessed["rows"] == 4
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("resource", ["input", "source"])
def test_post_job_mutation_refuses_completion(private, monkeypatch, resource):
    prepared = prepare(private)
    fake_source = private / "synthetic_runtime.py"
    fake_source.write_text("original")
    initial = worker.runtime_source_records()
    initial.append({"name": "synthetic_runtime.py", "path": str(fake_source),
                    "bytes": fake_source.stat().st_size, "sha256": worker.features.file_digest(fake_source)})
    monkeypatch.setattr(worker, "runtime_source_records", lambda: copy.deepcopy(initial))
    original = worker.assess

    def mutated(*args, **kwargs):
        result = original(*args, **kwargs)
        target = private / "train.jsonl" if resource == "input" else fake_source
        target.write_text(target.read_text() + " ")
        return result

    monkeypatch.setattr(worker, "assess", mutated)
    with pytest.raises(ValueError, match="changed during adaptation"):
        run(private, prepared)
    assert (private / "result" / "FAILED.json").exists()
    assert not (private / "result" / "COMPLETE.json").exists()
    assert not (private / "result" / "REPORT.json").exists()


def test_nonfinite_gradients_and_optimizer_state_cannot_promote(private, monkeypatch):
    prepared = prepare(private)
    original = worker.pooled_batch

    def poisoned(*args, **kwargs):
        pooled, counts = original(*args, **kwargs)
        if torch.is_grad_enabled():
            pooled.register_hook(lambda grad: grad * float("nan"))
        return pooled, counts

    monkeypatch.setattr(worker, "pooled_batch", poisoned)
    with pytest.raises(ValueError, match="Nonfinite gradient"):
        run(private, prepared)
    assert not (private / "result" / "COMPLETE.json").exists()
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter])
    optimizer.state[parameter] = {"exp_avg": torch.tensor([0., float("inf")])}
    with pytest.raises(ValueError, match="optimizer state"):
        worker.verify_finite_state(torch.nn.Linear(1, 1), torch.nn.Identity(), optimizer)


def test_finite_loss_with_damaged_optimizer_state_cannot_complete(private, monkeypatch):
    prepared = prepare(private, epochs=1)
    adam = torch.optim.AdamW

    class DamagedAdam(adam):
        updates = 0

        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            self.updates += 1
            if self.updates == 2:
                first = self.param_groups[0]["params"][0]
                self.state[first]["exp_avg"].view(-1)[0] = float("nan")
            return result

    monkeypatch.setattr(torch.optim, "AdamW", DamagedAdam)
    with pytest.raises(ValueError, match="optimizer state"):
        run(private, prepared)
    assert (private / "result" / "FAILED.json").exists()
    assert not (private / "result" / "COMPLETE.json").exists()


def test_source_snapshot_precedes_model_loader(private, monkeypatch):
    prepared = prepare(private)
    fake = private / "runtime-fixture.py"
    fake.write_text("before loader")
    snapshots = worker.runtime_source_records()
    snapshots.append({"name": fake.name, "path": str(fake), "bytes": fake.stat().st_size,
                      "sha256": worker.features.file_digest(fake)})
    monkeypatch.setattr(worker, "runtime_source_records", lambda: copy.deepcopy(snapshots))
    load = worker.features.load_nuri

    def changed(*args):
        fake.write_text("changed by loader")
        return load(*args)

    monkeypatch.setattr(worker.features, "load_nuri", changed)
    monkeypatch.setattr(torch.optim, "AdamW", lambda *args, **kwargs: pytest.fail("Source refusal must precede updates"))
    with pytest.raises(ValueError, match="changed during adaptation"):
        run(private, prepared)
    assert not (private / "result" / "COMPLETE.json").exists()


def test_numeric_tensorboard_and_cli_receipt(private, capsys):
    event_accumulator = pytest.importorskip("tensorboard.backend.event_processing.event_accumulator")
    prepared = prepare(private, epochs=1)
    events = private / "events"
    worker.main(["--manifest", str(prepared[0]), "--protocol", str(prepared[1]),
                 "--output", str(private / "result"), "--learning-rate", ".01", "--seed", "0",
                 "--tensorboard-dir", str(events), "--no-progress"])
    assert "TEST unused" in capsys.readouterr().out
    accumulator = event_accumulator.EventAccumulator(str(events)).Reload()
    tags = set(accumulator.Tags()["scalars"])
    assert tags == {"train/observed_loss", "train/grad_norm_preclip", "train/loss", "train/accuracy",
                    "train/macro_f1", "valid/loss", "valid/accuracy", "valid/macro_f1"}
    assert [event.step for event in accumulator.Scalars("train/observed_loss")] == [1, 2]
    assert all(np.isfinite(event.value) for tag in tags for event in accumulator.Scalars(tag))
    assert not accumulator.Tags()["tensors"]


def test_outputs_are_fresh_and_outside_checkpoint(private):
    prepared = prepare(private, epochs=1)
    run(private, prepared)
    with pytest.raises(FileExistsError, match="fresh"):
        run(private, prepared)
    with pytest.raises(ValueError, match="outside source"):
        worker.run_candidate(*prepared[:2], prepared[2] / ".private" / "result",
                             learning_rate=.01, seed=0, progress=False)
    assert not (prepared[2] / ".private").exists()


def test_assessment_and_prepared_artifact_bindings(private, monkeypatch):
    prepared = prepare(private)
    assessment = private / "assessment.json"
    write_json(assessment, {"schema_version": 1})
    prepared[3]["assessment_protocol"] = {"path": assessment.name, "sha256": worker.features.file_digest(assessment)}
    model, _, _, model_records, tokenizer_records, _ = worker.features.load_nuri(prepared[3]["model"], private)
    del model
    prepared[3]["model_artifacts_sha256"] = worker.features.digest_json(worker.features.records_identity(model_records))
    prepared[3]["tokenizer_artifacts_sha256"] = worker.features.digest_json(worker.features.records_identity(tokenizer_records))
    write_json(prepared[0], prepared[3])
    report = run(private, prepared)
    assert report["assessment_protocol_sha256"] == worker.features.file_digest(assessment)
    assert report["manifest_sha256"] == worker.features.file_digest(prepared[0])
    assert report["protocol_sha256"] == worker.features.file_digest(prepared[1])
    write_json(assessment, {"schema_version": 2})
    monkeypatch.setattr(worker.features, "load_nuri", lambda *args: pytest.fail("Assessment refusal precedes loading"))
    with pytest.raises(ValueError, match="Assessment protocol hash"):
        run(private, prepared, "changed")
