import importlib.util
import json
from pathlib import Path

import pytest
import torch

from nuri.config import MoEConfig, tiny_config
from nuri.model import NuriModel
from tools import benchmark_nuri as benchmark


def small_config(**overrides):
    values = dict(hidden_size=16, num_heads=2, ffn_size=32, input_layers=1,
                  readout_layers=1, embed_dim=256)
    values.update(overrides)
    return tiny_config(**values)


def budget_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "plan_budget.py"
    spec = importlib.util.spec_from_file_location("pairs_plan_budget", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_pairs_measure_the_real_shared_memory_loss_and_embedding_optimizer(monkeypatch, dtype):
    captured = {}
    memories = []
    encode_calls = []
    loss_calls = []
    updates = []
    real_pair_loss = benchmark.pair_loss
    real_optimizer = benchmark.make_optimizer

    def capture_model(config):
        model = NuriModel(config)
        captured["model"] = model
        encode = model.encode
        embed_memory = model.embed_memory

        def counted_encode(*args, **kwargs):
            encode_calls.append(args[0].shape)
            return encode(*args, **kwargs)

        def counted_embed_memory(x, valid, *args, **kwargs):
            memories.append((kwargs["mode"], x))
            return embed_memory(x, valid, *args, **kwargs)

        model.encode = counted_encode
        model.embed_memory = counted_embed_memory
        return model

    def recorded_loss(model, batch, depth, dims, temperature, cross_mode_weight):
        loss_calls.append((depth, dims, temperature, cross_mode_weight))
        assert batch["false_negatives"].shape == (2, 2)
        assert not batch["false_negatives"].diagonal().any()
        return real_pair_loss(model, batch, depth, dims, temperature, cross_mode_weight)

    def recorded_optimizer(model, settings):
        optimizer = real_optimizer(model, settings)
        captured["optimizer"] = optimizer
        step = optimizer.step

        def recorded_step(*args, **kwargs):
            names = ("embedding.weight", "core.slots", "embed_proj.weight", "reason_proj.weight")
            parameters = dict(model.named_parameters())
            before = {name: parameters[name].detach().clone() for name in names}
            for name in names:
                gradient = parameters[name].grad
                assert gradient is not None and torch.isfinite(gradient).all()
                assert gradient.abs().sum() > 0
            result = step(*args, **kwargs)
            updates.append(all(not torch.equal(before[name], parameters[name]) for name in names))
            return result

        optimizer.step = recorded_step
        return optimizer

    monkeypatch.setattr(benchmark, "NuriModel", capture_model)
    monkeypatch.setattr(benchmark, "pair_loss", recorded_loss)
    monkeypatch.setattr(benchmark, "make_optimizer", recorded_optimizer)
    report = benchmark.run_benchmark(small_config(), task="pairs", training=True, dtype=dtype,
                                     batch_size=2, input_length=4, steps=2, warmup=1, iterations=2)

    assert report["task"] == "pairs" and report["training"]
    assert report["input_sequences_per_sample"] == 2
    assert loss_calls == [(2, (64, 128, 256), 0.05, 0.0)] * 3
    assert encode_calls == [torch.Size([2, 4])] * 6
    assert updates == [True] * 3
    for offset in range(0, len(memories), 4):
        fast_q, fast_d, reasoning_q, reasoning_d = memories[offset:offset + 4]
        assert [item[0] for item in (fast_q, fast_d, reasoning_q, reasoning_d)] == [
            "fast", "fast", "reasoning", "reasoning"]
        assert fast_q[1] is reasoning_q[1]
        assert fast_d[1] is reasoning_d[1]
        assert fast_q[1] is not fast_d[1]
    optimizer = captured["optimizer"]
    assert all(group["lr"] == 5e-5 and group["betas"] == (0.9, 0.95)
               for group in optimizer.param_groups)
    assert {group["weight_decay"] for group in optimizer.param_groups} == {0.0, 0.1}
    assert captured["model"].length_head.weight.grad is None
    assert all(parameter.grad is None for parameter in captured["model"].readout.parameters())


def test_pairs_include_and_clear_auxiliary_graphs_from_both_encoder_and_core_calls(monkeypatch):
    captured = {}
    records = []
    real_pair_loss = benchmark.pair_loss

    def capture_model(config):
        model = NuriModel(config)
        captured["model"] = model
        captured["routers_before"] = [layer.router.weight.detach().clone() for layer in model.moe_layers()]
        return model

    def isolate_auxiliary_loss(model, *args, **kwargs):
        result = real_pair_loss(model, *args, **kwargs)
        records.append([len(layer.aux_records) for layer in model.moe_layers()])
        # Keep the real pair computation, but isolate the auxiliary gradient contribution.
        result["loss"] = result["loss"] * 0
        return result

    config = small_config(moe=MoEConfig(num_experts=2, expert_ffn_size=8, top_k=2,
                                       input_layers=(1,), readout_layers=(), core=True))
    monkeypatch.setattr(benchmark, "NuriModel", capture_model)
    monkeypatch.setattr(benchmark, "pair_loss", isolate_auxiliary_loss)
    report = benchmark.run_benchmark(config, task="pairs", training=True,
                                     batch_size=2, input_length=4, steps=2, warmup=1, iterations=1)
    model = captured["model"]
    assert records == [[2, 4], [2, 4]]
    for before, layer in zip(captured["routers_before"], model.moe_layers()):
        assert layer.aux_records == []
        assert layer.router.weight.grad is not None
        assert torch.isfinite(layer.router.weight.grad).all()
        assert layer.router.weight.grad.abs().sum() > 0
        assert not torch.equal(before, layer.router.weight)
    assert report["moe_backend"] == "portable_expert_loop"


@pytest.mark.parametrize("task,training,batch_size,message", [
    ("pairs", False, 2, "require --training"),
    ("pairs", True, 1, "batch_size >= 2"),
    ("embed-fast", True, 2, "Training benchmarks"),
    ("embed-reasoning", True, 2, "Training benchmarks"),
])
def test_pair_scope_errors_happen_before_model_allocation(monkeypatch, task, training, batch_size, message):
    def forbidden_allocation(*args, **kwargs):
        pytest.fail("Invalid benchmark scope allocated a model")

    monkeypatch.setattr(benchmark, "NuriModel", forbidden_allocation)
    with pytest.raises(ValueError, match=message):
        benchmark.run_benchmark(small_config(), task=task, training=training, batch_size=batch_size)


def test_pair_token_accounting_transfers_to_budget_and_keeps_legacy_reports(monkeypatch, tmp_path):
    clock = iter((10.0, 12.0))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    report = benchmark.run_benchmark(small_config(max_response_len=2), task="pairs", training=True,
                                     batch_size=2, input_length=4, steps=2, warmup=0, iterations=1)
    assert report["input_tokens_per_second"] == 8
    assert report["latency_seconds"]["mean"] == 2
    assert report["reasoning_steps"] == 2
    assert report["output_length"] is None
    assert report["canvas_sequences_tokens_per_second"] is None
    assert report["generation_canvas_positions_processed"] == 0
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    module = budget_module()
    budget = module.budget_stage("embed", {"stage": "embed", "batch_size": 2},
                                 {"hours": 1, "benchmarks": {"pairs": str(path)}}, 1.0)
    assert budget["seconds_per_step"] == 2
    assert budget["steps"] == 1800
    assert budget["tokens_per_step"] == 16
    assert budget["processed_tokens"] == 28_800
    assert module.tokens_per_iteration({"batch_size": 2, "input_length": 4, "output_length": 3}) == 14


def test_pairs_cli_runs_a_real_cpu_training_measurement_and_writes_json(tmp_path, capsys):
    path = tmp_path / "pair-calibration.json"
    assert benchmark.main(["--preset", "tiny", "--task", "pairs", "--training", "--device", "cpu",
                           "--batch-size", "2", "--input-length", "4", "--steps", "2",
                           "--warmup", "0", "--iterations", "1", "--output", str(path)]) == 0
    report = json.loads(path.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == report
    assert report["task"] == "pairs" and report["training"]
    assert report["input_sequences_per_sample"] == 2
    assert report["autocast_dtype"] == "float32"
    assert report["input_kind"] == "synthetic_random_tokens_not_a_quality_evaluation"
    assert report["latency_seconds"]["mean"] > 0
