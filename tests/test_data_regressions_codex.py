"""Independent regressions for excluded samples and multi-forward training.

These tests use small synthetic token lists. They do not require a GPU or
external data and do not prescribe how the implementation batches forwards.
"""

import pytest
import torch

from nuri.config import MoEConfig, tiny_config
from nuri.data import TokenCorpus, build_pair_batch, write_corpus
from nuri.model import NuriModel
from nuri.pipeline import base_stage, embedding_stage, instruction_stage
from nuri.trainer import mlm_loss, pair_loss


def _pair_model():
    torch.manual_seed(0)
    return NuriModel(tiny_config(moe=MoEConfig(
        num_experts=4,
        expert_ffn_size=16,
        top_k=2,
        input_layers=(2,),
        readout_layers=(2,),
    )))


def _pair_batch(config):
    return build_pair_batch([
        ([3, 4, 5], [6, 7]),
        ([8, 9], [10, 11, 12, 13]),
    ], config)


def test_pair_loss_reports_all_processed_encoder_tokens():
    model = _pair_model()
    processed = []

    def count_tokens(module, inputs, output):
        processed.append(int(inputs[1].sum()))

    handle = model.encoder.register_forward_hook(count_tokens)
    try:
        result = pair_loss(model, _pair_batch(model.config), 2, (16, 32), 0.05)
    finally:
        handle.remove()

    assert result["tokens"] == sum(processed)


def test_pair_aux_includes_every_routing_forward():
    model = _pair_model()
    routed = []

    def count_assignments(module, inputs, output):
        if module.last_aux is not None:
            routed.append(int(module.last_aux["counts"].sum()))

    handles = [layer.register_forward_hook(count_assignments) for layer in model.moe_layers()]
    try:
        result = pair_loss(model, _pair_batch(model.config), 2, (16, 32), 0.05)
        aux = model.pop_moe_aux()
    finally:
        for handle in handles:
            handle.remove()

    assert torch.isfinite(result["loss"])
    assert aux["loss"] is not None
    assert sum(int(count.sum()) for count in aux["counts"]) == sum(routed)


def test_span_task_recovers_when_sampled_documents_are_too_short(tmp_path):
    config = tiny_config()
    folder = tmp_path / "tokens"
    write_corpus(folder, [[7], [8], [10, 11, 12, 13]])
    tasks, _ = base_stage(TokenCorpus(folder), config, batch_size=2, seed=0)

    batch = tasks["span"].next_batch()

    assert batch["input_valid"].any(dim=1).all()
    assert batch["canvas_valid"].any(dim=1).all()
    assert batch["loss_mask"].any(dim=1).all()


def test_mlm_task_excludes_windows_without_maskable_targets(tmp_path):
    config = tiny_config()
    folder = tmp_path / "tokens"
    write_corpus(folder, [[6, 7], [config.eos_id], [config.eos_id]])
    tasks, _ = base_stage(TokenCorpus(folder), config, batch_size=2, seed=0)

    batch = tasks["mlm"].next_batch()
    result = mlm_loss(NuriModel(config), batch)

    assert batch["target_mask"].any(dim=1).all()
    assert result["targets"] > 0
    assert torch.isfinite(result["loss"])


def test_instruction_stage_rejects_an_entirely_excluded_pool():
    config = tiny_config()
    examples = [
        ([], [3]),
        ([4], [5] * config.max_response_len),
    ]

    # Validation must happen before a next_batch loop can retry forever.
    with pytest.raises(ValueError):
        instruction_stage(examples, config, batch_size=2)


def test_embedding_stage_requires_at_least_one_negative():
    config = tiny_config()

    # With one row, symmetric in-batch InfoNCE is identically zero.
    with pytest.raises(ValueError):
        embedding_stage([([3, 4], [5, 6])], config, batch_size=1)


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_inference_does_not_queue_or_replace_training_auxiliaries(grad_enabled):
    model = _pair_model().eval()
    ids = torch.tensor([[3, 4, 5]])

    with torch.set_grad_enabled(grad_enabled):
        for _ in range(3):
            assert torch.isfinite(model.embed(ids)).all()

    assert all(not layer.aux_records for layer in model.moe_layers())

    # Diagnostic inference must also preserve an already pending training loss.
    model.train()
    model.encode(ids)
    model.eval()
    with torch.no_grad():
        model.embed(torch.tensor([[6, 7]]))
    aux = model.pop_moe_aux()

    assert aux["loss"] is not None and aux["loss"].requires_grad
    assert sum(int(count.sum()) for count in aux["counts"]) == ids.numel() * model.config.moe.top_k


def test_mlm_task_never_returns_an_empty_special_only_window_batch(tmp_path):
    config = tiny_config()
    folder = tmp_path / "tokens"
    # The document is usable, but seed 0 initially samples only its EOS tail.
    write_corpus(folder, [[3] + [config.eos_id] * (3 * config.max_input_len)])
    tasks, _ = base_stage(TokenCorpus(folder), config, batch_size=2, seed=0)

    try:
        batch = tasks["mlm"].next_batch()
    except ValueError as error:
        # Bounded rejection with a useful explanation is also an acceptable policy.
        assert "target" in str(error).lower() or "usable" in str(error).lower()
        return

    assert batch["input_ids"].shape[0] > 0
    assert batch["target_mask"].any(dim=1).all()
    assert torch.isfinite(mlm_loss(NuriModel(config), batch)["loss"])
