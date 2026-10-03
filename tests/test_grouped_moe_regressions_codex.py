"""Grouped expert execution keeps precision, fallback and telemetry honest."""

import copy

import pytest
import torch

from nuri.config import MoEConfig, tiny_config
from nuri.moe import MoEFFN, _compute_dtype
from tools.benchmark_nuri import run_benchmark


def make_moe(backend="grouped", dtype=torch.float32):
    torch.manual_seed(41)
    module = MoEFFN(16, 4, 8, 2, 1, backend=backend).to(dtype)
    for parameter in (module.w_gate, module.w_up, module.w_down):
        torch.nn.init.normal_(parameter, std=0.1)
    return module


def loop_copy(module):
    reference = copy.deepcopy(module)
    reference.backend = "loop"
    return reference


def test_unsupported_double_keeps_complete_outputs_and_gradients():
    grouped = make_moe(dtype=torch.float64)
    loop = loop_copy(grouped)
    inputs = torch.randn(2, 4, 16, dtype=torch.float64, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    valid = torch.tensor([[True, True, False, False], [True, True, True, True]])
    actual = grouped(inputs, valid)
    expected = loop(reference_inputs, valid)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
    for (name, parameter), (reference_name, reference) in zip(
        grouped.named_parameters(), loop.named_parameters()
    ):
        assert name == reference_name and parameter.grad is not None
        torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)
    assert grouped.last_backend == "loop" and grouped.used_backends == {"loop"}
    assert int(grouped.last_aux["counts"].sum()) == 12


def test_compute_dtype_respects_autocast_ineligible_double():
    inputs = torch.randn(4, 16, dtype=torch.float64)
    weight = torch.randn(16, 8, dtype=torch.float64)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert _compute_dtype(inputs) == (inputs @ weight).dtype == torch.float64


def test_cpu_autocast_fallback_uses_the_matmul_precision(monkeypatch):
    grouped = make_moe()
    loop = loop_copy(grouped)
    inputs = torch.randn(5, 16)
    counts = torch.tensor([0, 2, 0, 3])

    def unexpected_kernel(*args, **kwargs):
        raise AssertionError("CPU bf16 must use the supported portable fallback")

    monkeypatch.setattr(torch, "_grouped_mm", unexpected_kernel, raising=False)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = grouped._run_experts(inputs, counts)
        expected = loop._run_experts(inputs, counts)
    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert grouped.last_backend == "loop" and grouped.used_backends == {"loop"}


@pytest.mark.parametrize("error", [
    torch.OutOfMemoryError("synthetic grouped kernel OOM"),
    RuntimeError("synthetic unexpected grouped kernel corruption"),
])
def test_runtime_memory_and_unknown_errors_are_not_hidden(monkeypatch, error):
    module = make_moe()
    assert module.last_backend == "not_run" and module.used_backends == set()

    def failed_kernel(*args, **kwargs):
        raise error

    monkeypatch.setattr(torch, "_grouped_mm", failed_kernel, raising=False)
    with pytest.raises(type(error), match=str(error)):
        module(torch.randn(2, 4, 16))
    assert module.last_backend == "not_run" and module.used_backends == set()


@pytest.mark.parametrize("error", [
    NotImplementedError("grouped GEMM is not implemented by this build"),
    RuntimeError("torch._grouped_mm is only supported on CUDA devices with compute capability = 9.0"),
])
def test_known_unsupported_kernel_reports_the_actual_loop(monkeypatch, error):
    module = make_moe()
    reference = loop_copy(module)

    def unsupported_kernel(*args, **kwargs):
        raise error

    monkeypatch.setattr(torch, "_grouped_mm", unsupported_kernel, raising=False)
    inputs = torch.randn(2, 4, 16)
    torch.testing.assert_close(module(inputs), reference(inputs), rtol=0, atol=0)
    assert module.last_backend == "loop" and module.used_backends == {"loop"}


@pytest.mark.skipif(not hasattr(torch, "_grouped_mm"), reason="native grouped GEMM is unavailable")
def test_grouped_execution_normalizes_every_weight_and_reports_success(monkeypatch):
    module = make_moe()
    # Keep the same expert dimensions with a physically strided up projection.
    module.w_up = torch.nn.Parameter(torch.randn(4, 16, 16)[..., ::2])
    reference = loop_copy(module)
    native = torch._grouped_mm
    calls = []

    def inspect_layout(inputs, weight, *, offs):
        assert inputs.is_contiguous() and weight.is_contiguous()
        assert offs.dtype == torch.int32 and offs.device == inputs.device
        calls.append((inputs.dtype, weight.dtype))
        return native(inputs, weight, offs=offs)

    monkeypatch.setattr(torch, "_grouped_mm", inspect_layout)
    inputs = torch.randn(2, 4, 16, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    actual = module(inputs)
    expected = reference(reference_inputs)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(module.w_up.grad, reference.w_up.grad, rtol=2e-5, atol=2e-6)
    assert calls == [(torch.float32, torch.float32)] * 3
    assert module.last_backend == "grouped" and module.used_backends == {"grouped"}
    module.backend = "loop"
    module(inputs.detach())
    assert module.last_backend == "loop" and module.used_backends == {"grouped", "loop"}


def benchmark_config(input_layers=(1,)):
    moe = MoEConfig(num_experts=4, expert_ffn_size=32, top_k=2,
                    input_layers=input_layers, readout_layers=(), backend="grouped")
    return tiny_config(moe=moe)


def observe_expert_calls(monkeypatch):
    original = MoEFFN._run_experts
    calls = []

    def observed(module, inputs, counts):
        result = original(module, inputs, counts)
        calls.append((module, module.last_backend))
        return result

    monkeypatch.setattr(MoEFFN, "_run_experts", observed)
    return calls


def count_native_calls(monkeypatch):
    native = getattr(torch, "_grouped_mm", None)
    if native is None:
        pytest.skip("this PyTorch build has no native grouped GEMM")
    calls = []

    def observed(inputs, weight, *, offs):
        calls.append((inputs.shape, weight.shape))
        return native(inputs, weight, offs=offs)

    monkeypatch.setattr(torch, "_grouped_mm", observed)
    return calls


def test_benchmark_reports_loop_when_grouped_eligibility_fails(monkeypatch):
    import nuri.moe as moe_module

    monkeypatch.setattr(moe_module, "_grouped_mm_usable", lambda *args: False)

    def unexpected_kernel(*args, **kwargs):
        raise AssertionError("a failed eligibility check must not invoke grouped GEMM")

    monkeypatch.setattr(torch, "_grouped_mm", unexpected_kernel, raising=False)
    calls = observe_expert_calls(monkeypatch)
    report = run_benchmark(benchmark_config(), task="mlm", training=True,
                           input_length=8, output_length=4, warmup=1, iterations=2)
    assert [path for _, path in calls] == ["loop", "loop", "loop"]
    assert report["moe_backend_requested"] == "grouped"
    assert report["moe_backend"] == "portable_expert_loop"
    assert report["moe_expert_paths_used"] == ["loop"]


def test_benchmark_discards_grouped_calls_that_only_happened_in_warmup(monkeypatch):
    import nuri.moe as moe_module

    native_calls = count_native_calls(monkeypatch)
    usable = moe_module._grouped_mm_usable
    eligibility_calls = []

    def warmup_only(*args):
        eligibility_calls.append(1)
        return len(eligibility_calls) == 1 and usable(*args)

    monkeypatch.setattr(moe_module, "_grouped_mm_usable", warmup_only)
    calls = observe_expert_calls(monkeypatch)
    report = run_benchmark(benchmark_config(), task="mlm", training=True,
                           input_length=8, output_length=4, warmup=1, iterations=2)
    assert [path for _, path in calls] == ["grouped", "loop", "loop"]
    assert len(native_calls) == 3  # the three real expert projections in warmup
    assert calls[0][0].used_backends == {"loop"}
    assert report["moe_backend_requested"] == "grouped"
    assert report["moe_backend"] == "portable_expert_loop"
    assert report["moe_expert_paths_used"] == ["loop"]


def test_benchmark_reports_mixed_paths_from_real_measured_layers(monkeypatch):
    import nuri.moe as moe_module

    native_calls = count_native_calls(monkeypatch)
    usable = moe_module._grouped_mm_usable
    eligibility_calls = []

    def first_layer_only(*args):
        eligibility_calls.append(1)
        return len(eligibility_calls) == 1 and usable(*args)

    monkeypatch.setattr(moe_module, "_grouped_mm_usable", first_layer_only)
    calls = observe_expert_calls(monkeypatch)
    report = run_benchmark(benchmark_config((1, 2)), task="mlm", training=True,
                           input_length=8, output_length=4, warmup=0, iterations=1)
    assert [path for _, path in calls] == ["grouped", "loop"]
    assert calls[0][0] is not calls[1][0] and len(native_calls) == 3
    assert calls[0][0].used_backends == {"grouped"} and calls[1][0].used_backends == {"loop"}
    assert report["moe_backend_requested"] == "grouped"
    assert report["moe_backend"] == "mixed_grouped_and_loop"
    assert report["moe_expert_paths_used"] == ["grouped", "loop"]
