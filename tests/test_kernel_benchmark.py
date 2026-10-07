"""Synthetic calibration records actual paths, including arbitrary-mask fallback."""

import json

import pytest
import torch
import torch.nn.functional as F

from nuri.config import tiny_config
from nuri.model import NuriModel
from tools import benchmark_nuri as benchmark


def configuration(**overrides):
    return tiny_config(
        hidden_size=16,
        num_heads=2,
        num_kv_heads=1,
        ffn_size=32,
        input_layers=1,
        readout_layers=1,
        embed_dim=8,
        **overrides,
    )


@pytest.mark.parametrize("task", ["mlm", "denoise"])
def test_chunked_benchmark_matches_reference_gradients_and_optimizer_update(
    monkeypatch, task
):
    models = []

    def capture(config):
        model = NuriModel(config)
        models.append(model)
        return model

    monkeypatch.setattr(benchmark, "NuriModel", capture)
    arguments = {
        "task": task,
        "training": True,
        "batch_size": 2,
        "input_length": 5,
        "output_length": 4,
        "steps": 2,
        "seed": 31,
        "warmup": 0,
        "iterations": 1,
    }
    reference = benchmark.run_benchmark(configuration(), **arguments)
    chunked = benchmark.run_benchmark(
        configuration(), **arguments, vocab_loss_backend="chunked", vocab_chunk_size=2
    )
    assert reference["vocab_loss_execution"] == "torch_full_logits"
    assert chunked["vocab_loss_execution"] == "torch_checkpointed_chunks"
    assert chunked["attention_paths_used"] == ["sdpa"]
    assert chunked["peak_cuda_allocated_bytes"] is None
    for (name, left), (_, right) in zip(
        models[0].named_parameters(), models[1].named_parameters()
    ):
        torch.testing.assert_close(right, left, atol=1e-6, rtol=2e-5, msg=name)
        if left.grad is not None:
            torch.testing.assert_close(
                right.grad, left.grad, atol=2e-6, rtol=2e-5, msg=name
            )


@pytest.fixture
def fake_flash(monkeypatch):
    monkeypatch.setattr(benchmark, "validate_attention_backend", lambda *a: None)
    monkeypatch.setattr("nuri.layers.validate_attention_backend", lambda *a: None)

    def attention(q, k, v, mask, backend):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None if mask is None else mask[:, None, None, :],
            enable_gqa=q.shape[1] != k.shape[1],
        )

    monkeypatch.setattr("nuri.layers.flash_attention", attention)


@pytest.mark.parametrize("packed", [False, True])
def test_requested_flash_is_reported_as_sdpa_for_packed_mlm(fake_flash, packed):
    report = benchmark.run_benchmark(
        configuration(),
        task="mlm",
        training=True,
        dtype="bfloat16",
        attention_backend="flash2",
        input_length=8,
        output_length=4,
        pack_mlm=packed,
        packed_document_length=3,
        warmup=0,
        iterations=1,
    )
    expected = "sdpa" if packed else "flash2"
    assert report["attention_backend_requested"] == "flash2"
    assert report["attention_backend"] == expected and report[
        "attention_paths_used"
    ] == [expected]
    assert report["attention_execution"] == (
        "repeat_kv_gqa" if packed else "compact_flash_gqa"
    )


def test_local_encoder_and_global_core_report_mixed_paths(fake_flash):
    report = benchmark.run_benchmark(
        configuration(attention_schedule="L", local_radius=2),
        task="denoise",
        training=True,
        dtype="bfloat16",
        attention_backend="flash2",
        input_length=6,
        output_length=4,
        steps=1,
        warmup=0,
        iterations=1,
    )
    assert report["attention_backend"] == "mixed"
    assert report["attention_paths_used"] == ["flash2", "sdpa"]
    assert report["attention_execution"] == "mixed_gqa"
    assert report["gqa_paths_used"] == ["compact_flash_gqa", "repeat_kv_gqa"]


@pytest.mark.parametrize(
    "options,match",
    [
        ({"attention_backend": "flash2"}, "bfloat16"),
        ({"attention_backend": "flash3", "dtype": "bfloat16"}, "NVIDIA CUDA"),
        ({"liger_kernels": True}, "NVIDIA CUDA"),
        ({"vocab_loss_backend": "liger"}, "NVIDIA CUDA"),
        ({"vocab_loss_backend": "unknown"}, "backend"),
        ({"vocab_chunk_size": 0}, "positive integer"),
        (
            {"task": "embed-fast", "training": False, "vocab_loss_backend": "chunked"},
            "mlm or denoise",
        ),
    ],
)
def test_invalid_kernel_options_fail_before_model_allocation(
    monkeypatch, options, match
):
    monkeypatch.setattr(
        benchmark, "NuriModel", lambda *a: pytest.fail("model allocated")
    )
    arguments = {
        "task": "mlm",
        "training": True,
        "device": "cpu",
        "input_length": 4,
        "output_length": 3,
    }
    arguments.update(options)
    with pytest.raises(ValueError, match=match):
        benchmark.run_benchmark(configuration(), **arguments)


def test_cli_records_chunk_size_and_actual_paths(tmp_path):
    path = tmp_path / "calibration.json"
    assert (
        benchmark.main(
            [
                "--preset",
                "tiny",
                "--training",
                "--task",
                "mlm",
                "--device",
                "cpu",
                "--vocab-loss-backend",
                "chunked",
                "--vocab-chunk-size",
                "2",
                "--input-length",
                "4",
                "--output-length",
                "3",
                "--warmup",
                "0",
                "--iterations",
                "1",
                "--output",
                str(path),
            ]
        )
        == 0
    )
    result = json.loads(path.read_text())
    assert (
        result["vocab_chunk_size"] == 2
        and result["vocab_loss_execution"] == "torch_checkpointed_chunks"
    )
    assert (
        result["attention_backend_requested"] == result["attention_backend"] == "sdpa"
    )
