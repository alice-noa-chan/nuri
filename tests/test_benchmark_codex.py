import pytest

from nuri.config import tiny_config
from tools.benchmark_nuri import run_benchmark


@pytest.mark.parametrize("task", ["mlm", "denoise", "embed-fast", "embed-reasoning", "generate"])
def test_real_cpu_benchmark_reports_task_and_measurement_scope(task):
    config = tiny_config(hidden_size=16, num_heads=2, ffn_size=32, input_layers=1, readout_layers=1, embed_dim=8)
    result = run_benchmark(config, task=task, input_length=4, output_length=3, rounds=2, warmup=0, iterations=1)
    assert result["latency_seconds"]["mean"] > 0
    assert result["input_tokens_per_second"] > 0
    assert result["total_parameters"] > 0
    assert result["peak_cuda_allocated_bytes"] is None
    assert result["input_kind"].startswith("synthetic")
    assert result["refinement_rounds"] == (2 if task == "generate" else None)


def test_training_benchmark_runs_real_backward_and_optimizer_step():
    config = tiny_config(hidden_size=16, num_heads=2, ffn_size=32, input_layers=1, readout_layers=1, embed_dim=8)
    result = run_benchmark(config, training=True, input_length=4, output_length=3, warmup=0, iterations=1)
    assert result["training"]
    assert result["moe_backend"] == "dense"


def test_benchmark_rejects_invalid_scope_before_allocating_model():
    with pytest.raises(ValueError, match="Training benchmarks"):
        run_benchmark(tiny_config(), task="generate", training=True)
    with pytest.raises(ValueError, match="exceed"):
        run_benchmark(tiny_config(), input_length=33)
