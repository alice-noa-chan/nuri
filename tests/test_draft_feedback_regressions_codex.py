"""Independent leakage, cache, and compute regressions for draft feedback."""

import pytest
import torch

from nuri.config import tiny_config
from nuri.evaluation import span_denoising
from nuri.generation import generate
from nuri.model import NuriModel
from nuri.trainer import canvas_loss
from tools import benchmark_nuri


def feedback_model(**overrides):
    torch.manual_seed(7)
    config = tiny_config(hidden_size=16, num_heads=2, ffn_size=32,
                         input_layers=1, readout_layers=1, embed_dim=8,
                         draft_feedback=True, **overrides)
    return NuriModel(config).eval()


def observed_generation(monkeypatch, tolerance):
    model = feedback_model()
    observed = []
    original = model.reason

    def capture(*args, **kwargs):
        out = original(*args, **kwargs)
        observed.append({"draft": kwargs.get("draft") is not None,
                         "tolerance": kwargs.get("halt_tol"),
                         "used": model.core.last_steps_used.clone()})
        return out

    monkeypatch.setattr(model, "reason", capture)
    result = generate(model, torch.tensor([[3, 4, 5], [6, 7, 0]]),
                      canvas_len=torch.tensor([4, 2]), steps=4, rounds=3, halt_tol=tolerance)
    return result, observed


def test_feedback_propagates_halting_to_every_actual_core_call(monkeypatch):
    result, observed = observed_generation(monkeypatch, float("inf"))
    assert observed and any(call["draft"] for call in observed)
    assert all(call["tolerance"] == float("inf") for call in observed)
    assert all(call["used"].tolist() == [1, 1] for call in observed)
    expected = torch.stack([call["used"] for call in observed]).sum(dim=0)
    assert torch.equal(result.reasoning_steps, expected)


@pytest.mark.parametrize("tolerance", [None, 0.0])
def test_feedback_reports_all_core_steps_including_any_initial_call(monkeypatch, tolerance):
    result, observed = observed_generation(monkeypatch, tolerance)
    expected = torch.stack([call["used"] for call in observed]).sum(dim=0)
    assert expected.tolist() == [4 * len(observed), 4 * len(observed)]
    assert torch.equal(result.reasoning_steps, expected)


def training_batch():
    return {
        "input_ids": torch.tensor([[3, 4, 5], [6, 7, 0]]),
        "input_valid": torch.tensor([[True, True, True], [True, True, False]]),
        "canvas": torch.tensor([[1, 44, 1], [1, 0, 0]]),
        "canvas_valid": torch.tensor([[True, True, True], [True, False, False]]),
        "loss_mask": torch.tensor([[True, False, True], [True, False, False]]),
        "targets": torch.tensor([[8, 44, 2], [9, 0, 0]]),
        "lengths": torch.tensor([3, 1]),
    }


def test_hidden_training_labels_never_change_reasoning_or_readout_states(monkeypatch):
    model = feedback_model().train()
    batch = training_batch()
    changed = {key: value.clone() for key, value in batch.items()}
    changed["targets"][changed["loss_mask"]] += 11
    states, hidden_states = [], []
    reason, readout = model.reason, model.readout_hidden

    def capture_reason(*args, **kwargs):
        canvas, _ = kwargs["draft"]
        assert torch.all(canvas[batch["loss_mask"]] == model.config.mask_id)
        out = reason(*args, **kwargs)
        states.append(out.detach().clone())
        return out

    def capture_hidden(*args, **kwargs):
        out = readout(*args, **kwargs)
        hidden_states.append(out.detach().clone())
        return out

    monkeypatch.setattr(model, "reason", capture_reason)
    monkeypatch.setattr(model, "readout_hidden", capture_hidden)
    first = canvas_loss(model, batch, depth=2)
    second = canvas_loss(model, changed, depth=2)
    assert torch.equal(states[0], states[1])
    assert torch.equal(hidden_states[0], hidden_states[1])
    assert not torch.equal(first["denoise"], second["denoise"])


def test_padded_draft_values_do_not_enter_reasoning_memory():
    model = feedback_model()
    batch = training_batch()
    changed = batch["canvas"].clone()
    changed[~batch["canvas_valid"]] = torch.tensor([77, 88])
    with torch.no_grad():
        x, valid = model.encode(batch["input_ids"], batch["input_valid"])
        original = model.reason(x, valid, steps=2, draft=(batch["canvas"], batch["canvas_valid"]))
        padded = model.reason(x, valid, steps=2, draft=(changed, batch["canvas_valid"]))
    torch.testing.assert_close(original, padded, rtol=0, atol=0)


def test_length_training_inputs_are_independent_of_gold_canvas_shape_and_tokens(monkeypatch):
    model = feedback_model(length_head_trained=True).train()
    batch = training_batch()
    changed = {key: value.clone() for key, value in batch.items()}
    changed.update(canvas=torch.tensor([[1, 55, 1, 56], [1, 57, 1, 0]]),
                   canvas_valid=torch.tensor([[True, True, True, True], [True, True, True, False]]),
                   loss_mask=torch.tensor([[True, False, True, False], [True, False, True, False]]),
                   targets=torch.tensor([[10, 55, 2, 56], [11, 57, 2, 0]]),
                   lengths=torch.tensor([4, 3]))
    length_inputs = []
    original = model.length_logits

    def capture(u, r):
        length_inputs.append((u.detach().clone(), r.detach().clone()))
        return original(u, r)

    monkeypatch.setattr(model, "length_logits", capture)
    canvas_loss(model, batch, depth=2, length_weight=0.1)
    canvas_loss(model, changed, depth=2, length_weight=0.1)
    assert len(length_inputs) == 2
    assert torch.equal(length_inputs[0][0], length_inputs[1][0])
    assert torch.equal(length_inputs[0][1], length_inputs[1][1])


def test_feedback_reuses_source_and_refreshes_readout_cache_without_repredicting_length(monkeypatch):
    model = feedback_model(length_head_trained=True)
    encoded, reasons, memories, readouts, length_calls = [], [], [], [], []
    encode, reason = model.encode, model.reason
    memory, readout, length = model.generation_memory, model.readout_hidden, model.length_logits

    def capture_encode(*args, **kwargs):
        out = encode(*args, **kwargs)
        encoded.append(out)
        return out

    def capture_reason(x, valid, *args, **kwargs):
        assert x is encoded[0][0] and valid is encoded[0][1]
        out = reason(x, valid, *args, **kwargs)
        reasons.append((out, kwargs.get("draft")))
        return out

    def capture_memory(x, valid, r):
        assert x is encoded[0][0] and valid is encoded[0][1]
        assert r is reasons[-1][0]
        cache = memory(x, valid, r)
        memories.append(cache)
        return cache

    def capture_readout(canvas, valid, cache):
        assert cache is memories[-1]
        assert reasons[-1][1] is not None
        draft, draft_valid = reasons[-1][1]
        assert torch.equal(canvas, draft) and torch.equal(valid, draft_valid)
        readouts.append(cache)
        return readout(canvas, valid, cache)

    def capture_length(u, r):
        assert reasons[-1][1] is None
        length_calls.append(1)
        logits = length(u, r)
        logits[:] = -100
        logits[:, 3] = 100  # choose length 4 once, before any drafted reasoning
        return logits

    monkeypatch.setattr(model, "encode", capture_encode)
    monkeypatch.setattr(model, "reason", capture_reason)
    monkeypatch.setattr(model, "generation_memory", capture_memory)
    monkeypatch.setattr(model, "readout_hidden", capture_readout)
    monkeypatch.setattr(model, "length_logits", capture_length)
    result = generate(model, torch.tensor([[3, 4, 5]]), steps=2, rounds=3)
    assert len(encoded) == 1 and length_calls == [1]
    assert len(readouts) == 3 and len({id(cache) for cache in readouts}) == 3
    assert result.lengths.tolist() == [4]
    assert not (result.canvas == model.config.mask_id).any()


def test_span_denoising_evaluation_uses_the_same_corrupted_draft_as_training(monkeypatch):
    model = feedback_model()
    drafts, canvases = [], []
    reason, readout = model.reason, model.readout_hidden

    def capture_reason(*args, **kwargs):
        assert kwargs.get("draft") is not None
        canvas, valid = kwargs["draft"]
        drafts.append((canvas.clone(), valid.clone()))
        return reason(*args, **kwargs)

    def capture_readout(canvas, valid, cache):
        assert torch.equal(canvas, drafts[-1][0])
        assert torch.equal(valid, drafts[-1][1])
        canvases.append(1)
        return readout(canvas, valid, cache)

    monkeypatch.setattr(model, "reason", capture_reason)
    monkeypatch.setattr(model, "readout_hidden", capture_readout)
    documents = [[3 + (row + position) % 90 for position in range(24 + row)] for row in range(2)]
    reports = span_denoising(model, documents, ratios=(0.5,), steps=2, batch_size=2, seed=7)
    assert len(drafts) == len(canvases) == 1
    assert reports[0]["spans"] == 2


@pytest.mark.parametrize("training", [False, True])
def test_denoising_benchmark_executes_the_configured_feedback_path(monkeypatch, training):
    model = feedback_model()
    calls, models = [], []
    original = NuriModel.reason

    def capture_reason(self, *args, **kwargs):
        assert kwargs.get("draft") is not None
        canvas, valid = kwargs["draft"]
        assert canvas.shape == valid.shape
        assert ((canvas == self.config.mask_id) & valid).any()
        calls.append(1)
        models.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NuriModel, "reason", capture_reason)
    report = benchmark_nuri.run_benchmark(
        model.config, task="denoise", training=training, input_length=4,
        output_length=3, steps=2, warmup=0, iterations=1,
    )
    assert calls == [1]
    assert report["latency_seconds"]["mean"] > 0
    if training:
        assert models[0].draft_embedding.grad is not None
        assert models[0].draft_embedding.grad.abs().sum() > 0
