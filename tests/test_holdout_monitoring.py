"""Fixed held-out losses must not alter training, resume or completion markers."""

import json
import random
from dataclasses import replace

import pytest
import torch

from nuri import trainer
from nuri.config import MoEConfig, tiny_config
from nuri.data import TokenCorpus, build_instruction_batch, write_corpus
from nuri.model import NuriModel
from nuri.pipeline import base_stage, embedding_stage
from nuri.trainer import HoldoutSet, Task, TrainSettings, evaluate_holdout, train


def small_config(moe=False):
    sparse = MoEConfig(num_experts=4, expert_ffn_size=8, top_k=2,
                      input_layers=(1,), readout_layers=(1,)) if moe else None
    return tiny_config(hidden_size=16, num_heads=2, ffn_size=32, input_layers=1,
                       readout_layers=1, embed_dim=8, moe=sparse)


def fresh(tmp_path, name, moe=False):
    corpus_dir = tmp_path / "corpus"
    if not corpus_dir.exists():
        write_corpus(corpus_dir, [[3 + (row + position) % 60 for position in range(14)] for row in range(12)])
    config = small_config(moe)
    torch.manual_seed(17)
    model = NuriModel(config)
    tasks, mix = base_stage(TokenCorpus(corpus_dir), config, batch_size=3, seed=9)
    valid_tasks, _ = base_stage(TokenCorpus(corpus_dir), config, batch_size=2, seed=71)
    test_tasks, _ = base_stage(TokenCorpus(corpus_dir), config, batch_size=2, seed=93)
    settings = TrainSettings(str(tmp_path / name), 4, warmup_steps=1, log_every=1, save_every=2,
                             mix=mix, progress=False, embed_dims=(4, 8), grad_accum=2)
    return model, tasks, settings, HoldoutSet(valid_tasks, 2), HoldoutSet(test_tasks, 1)


def same_state(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same_state(a, b)
    else:
        assert left == right


def without_seconds(report):
    return {key: value for key, value in report.items() if key != "seconds"}


def test_actual_assessment_repeats_masks_and_restores_rng_modes_and_gradients(tmp_path):
    model, tasks, settings, valid, _ = fresh(tmp_path, "run", moe=True)
    model.train()
    model.encoder.blocks[0].eval()  # restore even a deliberately mixed mode tree
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    rng_state = random.getstate()
    torch_state = torch.get_rng_state()
    task_states = trainer._task_random_state(tasks)
    holdout_states = trainer._task_random_state(valid.tasks)
    modes = [module.training for module in model.modules()]
    first = evaluate_holdout(model, valid, settings, torch.device("cpu"), "valid", 0, tasks)
    second = evaluate_holdout(model, valid, settings, torch.device("cpu"), "valid", 0, tasks)
    assert without_seconds(first) == without_seconds(second)
    assert random.getstate() == rng_state
    assert torch.equal(torch.get_rng_state(), torch_state)
    same_state(trainer._task_random_state(tasks), task_states)
    same_state(trainer._task_random_state(valid.tasks), holdout_states)
    assert [module.training for module in model.modules()] == modes
    assert torch.equal(parameter.grad, torch.ones_like(parameter))
    assert all(not layer.aux_records and layer.last_aux is None for layer in model.moe_layers())
    assert first["coverage"] == "bounded_sample"
    assert first["assessed_batches"] == 4 and first["assessed_rows"] == 8
    assert all(len(task["target_sha256"]) == 64 for task in first["tasks"].values())


@pytest.mark.parametrize("moe", [False, True])
def test_validation_and_test_leave_optimizer_weights_and_sampling_identical(tmp_path, moe):
    plain, tasks, settings, _, _ = fresh(tmp_path, "plain", moe)
    plain_records = train(plain, tasks, settings, torch.device("cpu"))
    observed, tasks, settings, valid, test = fresh(tmp_path, "observed", moe)
    settings.valid_every = 2
    records = train(observed, tasks, settings, torch.device("cpu"), validation=valid, test=test)
    assert [without_seconds(record) for record in records] == [without_seconds(record) for record in plain_records]
    same_state(plain.state_dict(), observed.state_dict())
    before = torch.load(tmp_path / "plain/training_state/final.pt", weights_only=False)
    after = torch.load(tmp_path / "observed/training_state/final.pt", weights_only=False)
    same_state(before["optimizer"], after["optimizer"])
    same_state(before["extra"]["random"], after["extra"]["random"])
    reports = [json.loads(line) for line in (tmp_path / "observed/holdout_log.jsonl").read_text().splitlines()]
    assert [(report["split"], report["step"]) for report in reports] == [("valid", 2), ("valid", 4), ("test", 4)]
    for name in valid.tasks:
        assert reports[0]["tasks"][name]["batch_sha256"] == reports[1]["tasks"][name]["batch_sha256"]


def test_exception_restores_rng_sources_modes_and_moe_telemetry(tmp_path, monkeypatch):
    model, tasks, settings, valid, _ = fresh(tmp_path, "exception", moe=True)
    layer = model.moe_layers()[0]
    marker = {"balance": torch.tensor(1.0), "z": torch.tensor(2.0), "counts": torch.tensor([1, 2, 3, 4])}
    layer.last_aux, layer.aux_records = marker, [marker]
    layer.last_backend, layer.used_backends = "grouped", {"grouped"}
    original_rng, original_torch = random.getstate(), torch.get_rng_state()
    sources = trainer._task_random_state(valid.tasks)
    modes = [module.training for module in model.modules()]

    def failure(*args):
        random.random()
        torch.rand(3)
        layer.last_aux = None
        layer.aux_records = []
        layer.used_backends.add("loop")
        raise RuntimeError("assessment failure")

    monkeypatch.setattr(trainer, "_task_loss", failure)
    with pytest.raises(RuntimeError, match="assessment failure"):
        evaluate_holdout(model, valid, settings, torch.device("cpu"), "valid", 1, tasks)
    assert original_rng == random.getstate()
    assert torch.equal(original_torch, torch.get_rng_state())
    same_state(trainer._task_random_state(valid.tasks), sources)
    assert [module.training for module in model.modules()] == modes
    assert layer.last_aux is marker and layer.aux_records[0] is marker
    assert layer.last_backend == "grouped" and layer.used_backends == {"grouped"}


def test_losses_use_actual_token_and_eligible_row_denominators(tmp_path, monkeypatch):
    config = small_config()
    model = NuriModel(config)
    rng = random.Random(2)
    calls = []

    def mlm_batch():
        count = 1 if rng.random() < 0.5 else 3
        calls.append(count)
        mask = torch.arange(3)[None, :] < count
        return {"input_ids": torch.tensor([[1, 4, 5]]), "input_valid": torch.ones(1, 3, dtype=torch.bool),
                "labels": torch.tensor([[3, 4, 5]]), "target_mask": mask}

    def fake_loss(model, kind, batch, depth, settings):
        return {"loss": torch.tensor(float(batch["target_mask"].sum()))}

    task = Task("mlm", mlm_batch, rng)
    holdout = HoldoutSet({"mlm": task}, 4)
    monkeypatch.setattr(trainer, "_task_loss", fake_loss)
    report = evaluate_holdout(model, holdout, TrainSettings(str(tmp_path), 1), torch.device("cpu"), "valid", 1)
    result = report["tasks"]["mlm"]
    assert len(set(calls)) == 2
    assert result["metrics"]["loss"] == pytest.approx(sum(count ** 2 for count in calls) / sum(calls))
    assert result["loss_denominators"]["loss"] == sum(calls)
    assert result["counts"]["targets"] == sum(calls)

    canvas = build_instruction_batch([([3, 4], [5, 6]), ([7, 8], [9])], config,
                                     torch.Generator().manual_seed(3))
    canvas["loss_mask"][1] = False  # this row still contributes a natural length target
    canvas_task = Task("instruction", lambda: canvas)
    monkeypatch.setattr(trainer, "_task_loss", lambda *args: {"loss": torch.tensor(2.7),
                        "denoise": torch.tensor(2.0), "length": torch.tensor(7.0)})
    result = evaluate_holdout(model, HoldoutSet({"instruction": canvas_task}, 1),
                             TrainSettings(str(tmp_path), 1), torch.device("cpu"), "test", 1)["tasks"]["instruction"]
    assert result["loss_denominators"] == {"denoise": 1, "length": 2}
    assert result["metrics"]["loss"] == pytest.approx(2.7)


def test_stop_during_final_test_resumes_assessment_before_public_marker(tmp_path):
    model, tasks, settings, _, test = fresh(tmp_path, "stopped")
    settings.total_steps = 1
    calls = []
    original = test.tasks["mlm"].next_batch

    def stop_in_test():
        calls.append(1)
        assert not (tmp_path / "stopped/final/model.safetensors").exists()
        trainer.request_stop()
        return original()

    test.tasks["mlm"].next_batch = stop_in_test
    with pytest.raises(trainer.TrainingInterrupted) as error:
        train(model, tasks, settings, torch.device("cpu"), test=test)
    assert error.value.step == 1 and len(calls) == 1
    assert not (tmp_path / "stopped/final/model.safetensors").exists()
    saved = torch.load(error.value.path, weights_only=False)
    assert saved["step"] == 1
    resumed, tasks, _, _, test = fresh(tmp_path, "resumed")
    records = train(resumed, tasks, replace(settings, out_dir=str(tmp_path / "resumed")), torch.device("cpu"),
                    resume_from=error.value.path, test=test)
    assert records == []
    assert (tmp_path / "resumed/final/model.safetensors").exists()
    report = json.loads((tmp_path / "resumed/holdout_log.jsonl").read_text())
    assert report["split"] == "test" and report["step"] == 1
    same_state(model.state_dict(), resumed.state_dict())


def test_no_test_runs_after_early_training_stop(tmp_path):
    model, tasks, settings, _, test = fresh(tmp_path, "early")
    original = next(iter(tasks.values())).next_batch
    settings.mix = {"mlm": 1.0}

    def stop_after_batch():
        trainer.request_stop()
        return original()

    tasks["mlm"].next_batch = stop_after_batch
    for task in test.tasks.values():
        task.next_batch = lambda: pytest.fail("test was consulted before completion")
    with pytest.raises(trainer.TrainingInterrupted):
        train(model, tasks, settings, torch.device("cpu"), test=test)
    assert not (tmp_path / "early/holdout_log.jsonl").exists()


def test_nonfinite_test_does_not_promote_final_weights(tmp_path, monkeypatch):
    model, tasks, settings, _, test = fresh(tmp_path, "bad")
    settings.total_steps = 1
    original = trainer._task_loss

    def nonfinite(model, *args):
        return original(model, *args) if model.training else {"loss": torch.tensor(float("nan"))}

    monkeypatch.setattr(trainer, "_task_loss", nonfinite)
    with pytest.raises(ValueError, match="non-finite"):
        train(model, tasks, settings, torch.device("cpu"), test=test)
    assert model.training
    assert not (tmp_path / "bad/final/model.safetensors").exists()
    assert torch.load(tmp_path / "bad/training_state/latest.pt", weights_only=False)["step"] == 1


def test_completed_resume_does_not_consult_already_recorded_test_again(tmp_path):
    model, tasks, settings, valid, test = fresh(tmp_path, "complete")
    settings.total_steps = 1
    train(model, tasks, settings, torch.device("cpu"), validation=valid, test=test)
    old_log = (tmp_path / "complete/holdout_log.jsonl").read_bytes()
    resumed, tasks, _, valid, test = fresh(tmp_path, "complete")
    for holdout in (valid, test):
        for task in holdout.tasks.values():
            task.next_batch = lambda: pytest.fail("completed holdout was consulted twice")
    assert train(resumed, tasks, settings, torch.device("cpu"),
                 resume_from=tmp_path / "complete/training_state/final.pt", validation=valid, test=test) == []
    assert (tmp_path / "complete/holdout_log.jsonl").read_bytes() == old_log


def test_recorded_assessment_requires_exact_weights_config_and_objective_settings(tmp_path):
    model, tasks, settings, _, test = fresh(tmp_path, "binding")
    report = evaluate_holdout(model, test, settings, torch.device("cpu"), "test", 1, tasks)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(report))
    assert trainer._assessment_recorded(path, "test", 1, test, model, settings)
    for field, changed in (("length_weight", 0.7), ("embed_dims", (4,)),
                           ("temperature", 0.2), ("cross_mode_weight", 0.5)):
        assert not trainer._assessment_recorded(path, "test", 1, test, model, replace(settings, **{field: changed}))
    old_theta = model.config.rope_theta
    model.config.rope_theta = old_theta + 1
    assert not trainer._assessment_recorded(path, "test", 1, test, model, settings)
    model.config.rope_theta = old_theta
    with torch.no_grad():
        model.embedding.weight[3, 0] += 0.01
    assert not trainer._assessment_recorded(path, "test", 1, test, model, settings)


@pytest.mark.parametrize("damage", ["truncated", "missing_task", "nan_metric", "wrong_denominator", "not_complete"])
def test_malformed_receipt_cannot_skip_final_assessment(tmp_path, damage):
    model, tasks, settings, _, test = fresh(tmp_path, "invalid-receipt")
    report = evaluate_holdout(model, test, settings, torch.device("cpu"), "test", 1, tasks)
    path = tmp_path / "receipt.json"
    if damage == "truncated":
        path.write_text('{"status":"complete",')
    else:
        if damage == "missing_task":
            del report["tasks"]["span"]
        elif damage == "nan_metric":
            report["tasks"]["mlm"]["metrics"]["loss"] = float("nan")
        elif damage == "wrong_denominator":
            report["tasks"]["mlm"]["loss_denominators"]["loss"] += 1
        else:
            report["status"] = "in_progress"
        path.write_text(json.dumps(report))
    assert not trainer._assessment_recorded(path, "test", 1, test, model, settings)


def test_streamed_model_binding_uses_bounded_chunks_and_noncontiguous_values(tmp_path):
    model = NuriModel(small_config())
    original = trainer._model_assessment_sha256(model, chunk_bytes=32)
    assert original == trainer._model_assessment_sha256(model, chunk_bytes=1024)
    with torch.no_grad():
        parameter = model.encoder.blocks[0].attn.q_proj.weight
        parameter.set_(parameter.T.contiguous().T)  # same values, non-contiguous storage
    assert not parameter.is_contiguous()
    assert original == trainer._model_assessment_sha256(model, chunk_bytes=32)


def test_mixed_run_completed_output_is_rejected_before_test_and_preserves_files(tmp_path):
    model, tasks, settings, _, test = fresh(tmp_path, "original")
    settings.total_steps = 1
    train(model, tasks, settings, torch.device("cpu"), test=test)
    paths = [tmp_path / "original/final/model.safetensors", tmp_path / "original/training_state/final.pt",
             tmp_path / "original/holdout/test-step1.json"]
    originals = [path.read_bytes() for path in paths]
    changed = torch.load(paths[1], weights_only=False)
    changed["model"]["embedding.weight"][3, 0] += 1.0
    alternate = tmp_path / "other-run.pt"
    torch.save(changed, alternate)
    resumed, tasks, _, _, test = fresh(tmp_path, "original")
    for task in test.tasks.values():
        task.next_batch = lambda: pytest.fail("test was consulted for another run's output")
    with pytest.raises(ValueError, match="fresh out directory"):
        train(resumed, tasks, settings, torch.device("cpu"), resume_from=alternate, test=test)
    assert [path.read_bytes() for path in paths] == originals


def test_same_model_with_damaged_receipt_reassesses_and_repairs_final_state(tmp_path):
    model, tasks, settings, _, test = fresh(tmp_path, "damaged")
    settings.total_steps = 1
    train(model, tasks, settings, torch.device("cpu"), test=test)
    assert trainer._public_checkpoint_matches(model, tmp_path / "damaged/final", chunk_bytes=32)
    receipt = tmp_path / "damaged/holdout/test-step1.json"
    receipt.write_text('{"status":"incomplete"}')
    resumed, tasks, _, _, test = fresh(tmp_path, "damaged")
    calls = []
    for task in test.tasks.values():
        original = task.next_batch

        def observed_batch(original=original):
            calls.append(1)
            return original()

        task.next_batch = observed_batch
    assert train(resumed, tasks, settings, torch.device("cpu"),
                 resume_from=tmp_path / "damaged/training_state/final.pt", test=test) == []
    assert len(calls) == 2
    assert json.loads(receipt.read_text())["status"] == "complete"
    assert trainer._public_checkpoint_matches(resumed, tmp_path / "damaged/final", chunk_bytes=32)


def test_pair_holdout_keeps_fixed_in_batch_candidate_pool(tmp_path):
    config = small_config()
    tasks, _ = embedding_stage([([3 + i], [10 + i, 20 + i]) for i in range(5)], config, 3, seed=8)
    settings = TrainSettings(str(tmp_path), 1, embed_dims=(4, 8), cross_mode_weight=0.3)
    result = evaluate_holdout(NuriModel(config), HoldoutSet(tasks, 2), settings,
                             torch.device("cpu"), "test", 1)["tasks"]["pairs"]
    assert result["candidate_rows_per_batch"] == [3, 3]
    assert result["counts"]["rows"] == 6 and result["counts"]["valid_input_tokens"] == 18
    assert result["loss_denominators"] == {"fast": 6, "reasoning": 6, "cross_mode": 6}
    metrics = result["metrics"]
    assert metrics["loss"] == pytest.approx((metrics["fast"] + metrics["reasoning"]) / 2 + 0.3 * metrics["cross_mode"])


def test_tensorboard_separates_numeric_train_valid_test_objectives(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    model, tasks, settings, valid, test = fresh(tmp_path, "events", moe=True)
    settings.mix = {"mlm": 1.0}
    settings.valid_every = 2
    settings.tensorboard_dir = str(tmp_path / "tb")
    records = train(model, tasks, settings, torch.device("cpu"), validation=valid, test=test)
    assert all(record["objective_loss"] < record["loss"] for record in records)
    events = EventAccumulator(str(tmp_path / "tb"), size_guidance={"scalars": 0}).Reload()
    assert [point.step for point in events.Scalars("train/mlm/objective_loss")] == [1, 2, 3, 4]
    assert [point.step for point in events.Scalars("valid/mlm/objective_loss")] == [2, 4]
    assert [point.step for point in events.Scalars("test/mlm/objective_loss")] == [4]
    assert [point.value for point in events.Scalars("test/assessed_batches")] == [2]
    allowed_text = {f"{split}/{name}/text_summary" for split in ("valid", "test")
                    for name in ("coverage", "policy_sha256", "mlm/batch_sha256", "mlm/target_sha256",
                                 "span/batch_sha256", "span/target_sha256")}
    assert set(events.Tags()["tensors"]) == allowed_text
    reports = [json.loads(line) for line in (tmp_path / "events/holdout_log.jsonl").read_text().splitlines()]
    digest = reports[-1]["tasks"]["mlm"]["target_sha256"]
    assert events.Tensors("test/mlm/target_sha256/text_summary")[-1].tensor_proto.string_val == [digest.encode()]


def test_validation_sources_are_explicit_and_not_training_aliases(tmp_path):
    model, tasks, settings, _, _ = fresh(tmp_path, "invalid")
    with pytest.raises(ValueError, match="separate"):
        train(model, tasks, settings, torch.device("cpu"), validation=HoldoutSet(tasks, 1))
    with pytest.raises(ValueError, match="explicit"):
        train(model, tasks, replace(settings, valid_every=2), torch.device("cpu"))
