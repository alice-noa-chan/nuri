import random

import torch

from nuri.config import tiny_config
from nuri.data import TokenCorpus, build_packed_mlm_batch, write_corpus
from nuri.model import NuriModel
from nuri.pipeline import base_stage
from nuri.trainer import TrainSettings, mlm_loss, train

CONFIG = tiny_config()  # N = 32


def short_documents(rng):
    return lambda: [rng.randrange(3, 120) for _ in range(rng.randint(5, 12))]


def test_rows_are_filled_with_numbered_documents():
    rng = random.Random(0)
    batch = build_packed_mlm_batch(short_documents(rng), rows=4, config=CONFIG, rng=rng,
                                   generator=torch.Generator().manual_seed(0), min_piece=4)
    segments, valid = batch["segments"], batch["input_valid"]
    assert batch["input_ids"].shape == (4, CONFIG.max_input_len)
    assert (valid.sum(dim=1) > CONFIG.max_input_len - 4).all()  # at most min_piece-1 padding
    assert (segments.max(dim=1).values >= 3).all()  # several documents per row
    for row in range(4):
        real = segments[row][valid[row]].tolist()
        assert real == sorted(real) and real[0] == 1  # contiguous, numbered from 1
    assert batch["target_mask"].any(dim=1).all() and not batch["target_mask"][~valid].any()


def test_documents_in_a_row_do_not_see_each_other():
    torch.manual_seed(0)
    for schedule in ("", "GL"):
        model = NuriModel(tiny_config(attention_schedule=schedule, local_radius=8)).eval()
        ids = torch.randint(3, 120, (1, 20))
        segments = torch.tensor([[1] * 8 + [2] * 12])
        x, _ = model.encode(ids, segments=segments)
        changed = ids.clone()
        changed[0, 8:] = torch.randint(3, 120, (12,))
        x_changed, _ = model.encode(changed, segments=segments)
        assert torch.allclose(x[0, :8], x_changed[0, :8], atol=1e-6)
        # The first document alone gives the same X as inside the packed row.
        alone, _ = model.encode(ids[:, :8])
        assert torch.allclose(x[0, :8], alone[0], atol=1e-5)


def test_packed_base_stage_trains(tmp_path):
    rng = random.Random(0)
    write_corpus(tmp_path / "corpus", [short_documents(rng)() for _ in range(50)])
    torch.manual_seed(0)
    model = NuriModel(CONFIG)
    tasks, mix = base_stage(TokenCorpus(tmp_path / "corpus"), CONFIG, batch_size=4, pack_mlm=True)
    batch = tasks["mlm"].next_batch()
    assert "segments" in batch and torch.isfinite(mlm_loss(model, batch)["loss"])
    assert tasks["mlm"].options["pack_mlm"] is True
    settings = TrainSettings(out_dir=str(tmp_path / "run"), total_steps=3, peak_lr=1e-3, warmup_steps=1,
                             mix={"mlm": 1.0}, progress=False)
    assert train(model, tasks, settings, device=torch.device("cpu"))
