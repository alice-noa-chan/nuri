"""Block boundaries, conditioning cache and single-block sampler equivalence."""

import json
import math

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from nuri import generation
from nuri.checkpoint import save_pretrained
from nuri.config import tiny_config
from nuri.model import NuriModel
from tools.infer_nuri import infer, main


def small_config(**overrides):
    values = dict(vocab_size=12, hidden_size=8, num_heads=2, ffn_size=8,
                  input_layers=1, readout_layers=2, latent_slots=2,
                  max_input_len=8, max_response_len=8, embed_dim=4,
                  default_depth=2, refine_rounds=3)
    values.update(overrides)
    return tiny_config(**values)


class ScriptedModel(NuriModel):
    """Change predictions on every call so accidental rewriting is observable."""

    def __init__(self, **overrides):
        super().__init__(small_config(**overrides))
        self.encode_calls = self.reason_calls = self.memory_calls = 0
        self.readout_inputs = []
        self.projected_positions = []
        self.cache = object()

    def encode(self, input_ids, attention_mask=None):
        self.encode_calls += 1
        x = torch.zeros(*input_ids.shape, self.config.hidden_size)
        return x, input_ids != self.config.pad_id

    def reason(self, x, valid, steps=None):
        self.reason_calls += 1
        self.core.last_steps_used = torch.full((x.shape[0],), steps, dtype=torch.long)
        return x.new_zeros(x.shape[0], self.config.latent_slots, self.config.hidden_size)

    def generation_memory(self, x, valid, r):
        self.memory_calls += 1
        return self.cache

    def readout_hidden(self, canvas, valid, cache):
        assert cache is self.cache
        self.readout_inputs.append((canvas.clone(), valid.clone()))
        rows, positions = torch.meshgrid(torch.arange(canvas.shape[0]),
                                        torch.arange(canvas.shape[1]), indexing="ij")
        return torch.stack((rows, positions), dim=-1).float()

    def vocab_logits(self, hidden):
        coordinates = hidden.long().tolist()
        self.projected_positions.append(coordinates)
        call = len(self.readout_inputs) - 1
        logits = hidden.new_full((len(coordinates), self.config.vocab_size), -torch.inf)
        for index, (row, position) in enumerate(coordinates):
            token = 3 + (row + position + call) % (self.config.vocab_size - 3)
            # Round 2 raises new predictions above the old visible position 0.
            probability = (0.90, 0.30, 0.80)[position % 3] if call % 3 == 0 else 0.98
            logits[index, 2:] = math.log((1 - probability) / (self.config.vocab_size - 3))
            logits[index, token] = math.log(probability)
        return logits


def test_blocks_freeze_prefix_leave_future_masked_and_preserve_padding():
    model = ScriptedModel().eval()
    lengths = torch.tensor([1, 5, 7])
    result = generation.generate(model, torch.tensor([[3, 0], [4, 5], [6, 0]]),
                                 canvas_len=lengths, rounds=3, block_size=3, keep_history=True)
    assert result.readout_calls == len(result.history) == 9
    assert (model.encode_calls, model.reason_calls, model.memory_calls) == (1, 1, 1)
    assert result.reasoning_steps.tolist() == [2, 2, 2]
    for call, ((before, valid), after, projected) in enumerate(
            zip(model.readout_inputs, result.history, model.projected_positions)):
        start = (call // 3) * 3
        end = min(start + 3, 7)
        round_index = call % 3 + 1
        for row, length in enumerate(lengths.tolist()):
            assert valid[row].tolist() == [position < length for position in range(7)]
            assert (before[row, length:] == model.config.pad_id).all()
            assert (after[row, length:] == model.config.pad_id).all()
            assert (before[row, end:length] == model.config.mask_id).all()
            assert (after[row, end:length] == model.config.mask_id).all()
            if start:
                completed = result.history[(call // 3) * 3 - 1][row, :start]
                assert torch.equal(before[row, :start], completed)
                assert torch.equal(after[row, :start], completed)
            active_length = max(0, min(end, length) - start)
            if round_index == 1:
                assert (before[row, start:min(end, length)] == model.config.mask_id).all()
            expected_masks = math.ceil(active_length * (3 - round_index) / 3)
            assert int((after[row, start:min(end, length)] == model.config.mask_id).sum()) == expected_masks
        assert all(start <= position < min(end, int(lengths[row])) for row, position in projected)
    assert not (result.canvas == model.config.mask_id).any()
    # Original confidence at visible position 0 stays lower than round-2 fills.
    assert model.projected_positions[1] == [[0, 0], [1, 1], [1, 2], [2, 1], [2, 2]]
    assert model.projected_positions[2] == [[0, 0], [1, 0], [2, 0]]


@pytest.mark.parametrize("block_size", [5, 8, 100])
@pytest.mark.parametrize("temperature,top_k", [(0.0, None), (1.4, None), (1.4, 3)])
def test_one_block_matches_default_tokens_confidence_history_and_rng(monkeypatch, block_size, temperature, top_k):
    torch.manual_seed(7)
    model = NuriModel(small_config()).eval()
    ids = torch.tensor([[3, 4, 0], [5, 6, 7]])
    confidences = []
    original = generation._remask_least_confident

    def capture(canvas, confidence, *args):
        confidences.append(confidence.clone())
        return original(canvas, confidence, *args)

    monkeypatch.setattr(generation, "_remask_least_confident", capture)
    seed = torch.Generator().manual_seed(11)
    settings = dict(canvas_len=torch.tensor([3, 5]), rounds=3, keep_history=True,
                    temperature=temperature, top_k=top_k)
    default = generation.generate(model, ids, generator=seed, **settings)
    default_confidences = confidences.copy()
    default_rng = seed.get_state()
    confidences.clear()
    seed = torch.Generator().manual_seed(11)
    single = generation.generate(model, ids, block_size=block_size, generator=seed, **settings)
    assert torch.equal(default.canvas, single.canvas)
    assert default.tokens == single.tokens and default.ended_with_eos == single.ended_with_eos
    assert torch.equal(default.lengths, single.lengths)
    assert torch.equal(default.reasoning_steps, single.reasoning_steps)
    assert default.readout_calls == single.readout_calls == 3
    assert all(torch.equal(a, b) for a, b in zip(default.history, single.history))
    assert len(default_confidences) == len(confidences) == 2
    assert all(torch.equal(a, b) for a, b in zip(default_confidences, confidences))
    assert torch.equal(default_rng, seed.get_state())


def test_real_model_reuses_memory_and_samples_repeatably(monkeypatch):
    torch.manual_seed(4)
    model = NuriModel(small_config()).eval()
    memories = []
    original = model.generation_memory

    def capture_memory(*args):
        cache = original(*args)
        memories.append(cache)
        return cache

    monkeypatch.setattr(model, "generation_memory", capture_memory)
    readout = model.readout_hidden
    seen_caches = []

    def capture_readout(canvas, valid, cache):
        seen_caches.append(cache)
        return readout(canvas, valid, cache)

    monkeypatch.setattr(model, "readout_hidden", capture_readout)
    ids = torch.tensor([[3, 4], [5, 6]])
    settings = dict(canvas_len=torch.tensor([2, 7]), rounds=2, block_size=3,
                    temperature=1.2, top_k=4, halt_tol=float("inf"))
    first = generation.generate(model, ids, generator=torch.Generator().manual_seed(9), **settings)
    second = generation.generate(model, ids, generator=torch.Generator().manual_seed(9), **settings)
    assert torch.equal(first.canvas, second.canvas)
    assert first.readout_calls == second.readout_calls == 6
    assert first.reasoning_steps.tolist() == [1, 1]
    assert len(memories) == 2
    assert all(cache is memories[0] for cache in seen_caches[:6])
    assert all(cache is memories[1] for cache in seen_caches[6:])
    assert not (first.canvas == model.config.mask_id).any()
    assert (first.canvas[0, 2:] == model.config.pad_id).all()


def test_eos_does_not_stop_later_blocks_and_final_answer_is_truncated():
    model = ScriptedModel().eval()

    def prefer_eos(hidden):
        logits = torch.zeros(hidden.shape[0], model.config.vocab_size)
        logits[:, model.config.eos_id] = 100
        return logits

    model.vocab_logits = prefer_eos
    result = generation.generate(model, torch.tensor([[3], [4]]), rounds=2,
                                 canvas_len=torch.tensor([3, 5]), block_size=2)
    assert result.tokens == [[], []] and result.ended_with_eos == [True, True]
    assert result.readout_calls == len(model.readout_inputs) == 6
    assert not (result.canvas == model.config.mask_id).any()
    assert (result.canvas[0, 3:] == model.config.pad_id).all()


@pytest.mark.parametrize("block_size", [0, -2, 1.5, True, "3"])
def test_invalid_block_size_rejected_before_encoding(block_size):
    model = ScriptedModel().eval()
    with pytest.raises(ValueError, match="block_size"):
        generation.generate(model, torch.tensor([[3]]), canvas_len=5, block_size=block_size)
    assert model.encode_calls == 0


def test_draft_feedback_combination_rejected_before_encoding():
    model = ScriptedModel(draft_feedback=True).eval()
    with pytest.raises(ValueError, match="draft_feedback"):
        generation.generate(model, torch.tensor([[3]]), canvas_len=5, block_size=8)
    assert model.encode_calls == model.reason_calls == model.memory_calls == 0


def test_text_inference_and_cli_expose_block_cost(tmp_path, capsys):
    vocabulary = {"[PAD]": 0, "[MASK]": 1, "[EOS]": 2, "hello": 3, "world": 4}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="hello"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    model = NuriModel(small_config(vocab_size=5)).eval()
    save_pretrained(model, tmp_path, dtype=torch.float32)
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    result = infer(model, tokenizer, ["hello", "hello world"], canvas_len=5, rounds=2, block_size=2)
    assert result["block_size"] == 2 and result["readout_calls"] == 6
    assert result["refinement_rounds"] == 2 and result["canvas_lengths"] == [5, 5]
    assert main(["--checkpoint", str(tmp_path), "--text", "hello", "--canvas-len", "5",
                 "--rounds", "2", "--block-size", "2", "--device", "cpu"]) == 0
    cli = json.loads(capsys.readouterr().out)
    assert cli["block_size"] == 2 and cli["readout_calls"] == 6
    with pytest.raises(ValueError, match="only to generation"):
        infer(model, tokenizer, ["hello"], task="embed", block_size=2)
