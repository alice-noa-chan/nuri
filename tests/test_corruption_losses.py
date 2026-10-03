import random

import torch

from nuri.corruption import corrupt_canvas, mask_for_mlm, pad_batch, split_span
from nuri.losses import contrastive_loss, length_loss, sample_mean_ce

PAD, MASK, EOS = 0, 1, 2


def test_pad_batch():
    ids, valid = pad_batch([[5, 6, 7], [8]], PAD)
    assert ids.tolist() == [[5, 6, 7], [8, 0, 0]]
    assert valid.tolist() == [[True] * 3, [True, False, False]]


def test_mlm_masks_only_real_tokens():
    g = torch.Generator().manual_seed(0)
    ids, valid = pad_batch([list(range(3, 23)), [3, 4, EOS]], PAD)
    corrupted, targets = mask_for_mlm(ids, valid, MASK, 100, (PAD, MASK, EOS), rate=0.3, generator=g)
    assert not targets[~valid].any() and not targets[ids == EOS].any()
    assert targets.sum(1).tolist() == [6, 1]
    assert torch.equal(corrupted[~targets], ids[~targets])


def test_canvas_targets_are_replaced_by_mask():
    g = torch.Generator().manual_seed(0)
    targets, valid = pad_batch([[5, 6, 7, 8, EOS], [9, EOS]], PAD)
    for _ in range(20):
        canvas, loss_mask = corrupt_canvas(targets, valid, MASK, generator=g)
        assert (loss_mask.sum(1) >= 1).all()
        assert not loss_mask[~valid].any()
        assert (canvas[loss_mask] == MASK).all()  # no answer leaks into the input
        assert torch.equal(canvas[~loss_mask], targets[~loss_mask])


def test_full_mask_probability():
    g = torch.Generator().manual_seed(0)
    targets, valid = pad_batch([list(range(3, 13))], PAD)
    _, loss_mask = corrupt_canvas(targets, valid, MASK, full_mask_prob=1.0, generator=g)
    assert loss_mask.all()


def test_split_span_eos_only_at_document_end():
    rng = random.Random(0)
    doc = list(range(10, 110))
    for _ in range(500):
        source, target = split_span(doc, max_input_len=20, max_response_len=16, eos_id=EOS, min_span=4, rng=rng)
        assert 1 <= len(source) <= 20 and 1 <= len(target) <= 16
        raw = target[:-1] if target[-1] == EOS else target
        start = doc.index(raw[0])
        assert doc[start - len(source):start] == source  # prefix sits right before the span
        assert doc[start:start + len(raw)] == raw
        assert (target[-1] == EOS) == (start + len(raw) == len(doc))


def test_split_span_moves_cut_when_eos_does_not_fit():
    class FixedRng:
        def __init__(self, values):
            self.values = list(values)

        def randint(self, a, b):
            return self.values.pop(0)

    doc = list(range(10, 30))  # 20 tokens
    source, target = split_span(doc, 50, 8, EOS, min_span=4, rng=FixedRng([12, 8]))
    assert target == doc[13:] + [EOS] and len(target) == 8 and source == doc[:13]


def test_sample_mean_ce_weights_samples_equally():
    targets = torch.tensor([[3, 3, 3, 3], [3, 0, 0, 0]])
    loss_mask = torch.tensor([[True] * 4, [True, False, False, False]])
    logits = torch.zeros(5, 4)
    logits[:4, 3] = 10.0  # first sample: confident and right
    loss = sample_mean_ce(logits, targets, loss_mask)
    wrong = torch.nn.functional.cross_entropy(torch.zeros(1, 4), torch.tensor([3]))
    assert torch.allclose(loss, wrong / 2, atol=1e-3)


def test_length_and_contrastive_losses():
    assert length_loss(torch.tensor([[0.0, 9.0, 0.0]]), torch.tensor([2])) < 0.01
    z = torch.randn(4, 8)
    good = contrastive_loss(z, z, dims=(4, 8))
    bad = contrastive_loss(z, z.roll(1, dims=0), dims=(4, 8))
    assert good < bad
