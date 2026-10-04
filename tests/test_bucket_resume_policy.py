"""Resume must reproduce the same bucket algorithm, not merely the same flag."""

from dataclasses import replace

import pytest
import torch

from nuri import pipeline
from nuri.config import tiny_config
from nuri.data import TokenCorpus, write_corpus
from nuri.model import NuriModel
from nuri.trainer import TrainSettings, train


def run_inputs(tmp_path, bucket=True, stage="base"):
    config = tiny_config()
    corpus_path = tmp_path / "corpus"
    if not corpus_path.exists():
        write_corpus(corpus_path, [[3, 4, 5, 6] * length for length in range(1, 20)])
    corpus = TokenCorpus(corpus_path)
    if stage == "base":
        tasks, mix = pipeline.base_stage(corpus, config, batch_size=3, seed=5, bucket=bucket)
    else:
        examples = [([3] * (length + 1), [4] * (length % 8 + 1)) for length in range(1, 20)]
        tasks, mix = pipeline.instruction_stage(examples, config, batch_size=3,
                                              corpus=corpus, seed=5, bucket=bucket)
    torch.manual_seed(5)
    model = NuriModel(config)
    settings = TrainSettings(str(tmp_path / "original"), total_steps=3, warmup_steps=1,
                             save_every=1, log_every=1, seed=5, mix=mix, progress=False)
    return model, tasks, settings


@pytest.mark.parametrize("stage", ["base", "it"])
def test_current_policy_resumes_exactly(tmp_path, stage):
    model, tasks, settings = run_inputs(tmp_path, stage=stage)
    expected = train(model, tasks, settings, device=torch.device("cpu"))
    resumed, tasks, _ = run_inputs(tmp_path, stage=stage)
    observed = train(resumed, tasks, replace(settings, out_dir=str(tmp_path / "resumed")),
                     device=torch.device("cpu"),
                     resume_from=tmp_path / "original/training_state/step-1.pt")
    assert all(torch.equal(value, resumed.state_dict()[key]) for key, value in model.state_dict().items())
    assert [row["batch_shapes"] for row in observed] == [row["batch_shapes"] for row in expected[1:]]


@pytest.mark.parametrize("change", ["legacy", "candidate_pool", "row_limit", "version"])
def test_old_or_changed_bucket_policy_is_rejected(tmp_path, monkeypatch, change):
    model, tasks, settings = run_inputs(tmp_path)
    train(model, tasks, settings, device=torch.device("cpu"))
    path = tmp_path / "original/training_state/step-1.pt"
    if change in ("legacy", "version"):
        state = torch.load(path, weights_only=False)
        for options in state["extra"]["task_options"].values():
            if change == "legacy":
                del options["bucket_policy"]
            else:
                options["bucket_policy"]["version"] = 1
        path = tmp_path / "legacy.pt"
        torch.save(state, path)
    elif change == "candidate_pool":
        monkeypatch.setattr(pipeline, "BUCKET_POOL", 7)
    else:
        monkeypatch.setattr(pipeline, "BUCKET_MAX_ROWS", 3)
    resumed, tasks, _ = run_inputs(tmp_path)
    with pytest.raises(ValueError, match="same batch construction"):
        train(resumed, tasks, replace(settings, out_dir=str(tmp_path / "resumed")),
              device=torch.device("cpu"), resume_from=path)
    assert not (tmp_path / "resumed/train_log.jsonl").exists()


def test_unbucketed_options_stay_compatible_with_prior_states(tmp_path):
    model, tasks, settings = run_inputs(tmp_path, bucket=False)
    assert all("bucket_policy" not in task.options for task in tasks.values())
    train(model, tasks, settings, device=torch.device("cpu"))
    resumed, tasks, _ = run_inputs(tmp_path, bucket=False)
    train(resumed, tasks, replace(settings, out_dir=str(tmp_path / "resumed")),
          device=torch.device("cpu"), resume_from=tmp_path / "original/training_state/step-1.pt")
    assert all(torch.equal(value, resumed.state_dict()[key]) for key, value in model.state_dict().items())
