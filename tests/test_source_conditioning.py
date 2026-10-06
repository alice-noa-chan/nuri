"""Input sensitivity needs correct answers, equal-length controls and preserved state."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

import nuri.evaluation as evaluation
from nuri.config import tiny_config
from nuri.generation import GenerationResult
from nuri.model import NuriModel
from test_instruction_evaluation_cli import make_checkpoint, write_rows


def load_script(name):
    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"conditioning_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_model(trained=True, **kwargs):
    return NuriModel(tiny_config(length_head_trained=trained, max_response_len=8, **kwargs))


def generator_for(outcomes, seen=None, consume_rng=False):
    def generate(model, ids, valid, **kwargs):
        if consume_rng:
            torch.rand(3)
        if seen is not None:
            seen.extend(row[mask].tolist() for row, mask in zip(ids, valid))
        tokens, ended = zip(*(outcomes[int(row[0])] for row in ids))
        lengths = kwargs["canvas_len"]
        if lengths is None:
            lengths = torch.tensor([len(t) + int(e) for t, e in zip(tokens, ended)])
        canvas = torch.full((len(ids), int(lengths.max())), model.config.pad_id, dtype=torch.long)
        for i, (answer, eos) in enumerate(zip(tokens, ended)):
            canvas[i, :len(answer)] = torch.tensor(answer)
            if eos and len(answer) < int(lengths[i]):
                canvas[i, len(answer)] = model.config.eos_id
        assert kwargs["temperature"] == 0.0
        return GenerationResult([list(t) for t in tokens], canvas, lengths, list(ended), readout_calls=4)
    return generate


@pytest.mark.parametrize("oracle", [True, False])
def test_responsive_correct_outputs_use_actual_replaced_prompts(monkeypatch, oracle):
    model = fixture_model()
    examples = [([5], [10]), ([6], [11]), ([7], [12, 13]), ([8], [14, 15])]
    seen = []
    monkeypatch.setattr(evaluation, "generate", generator_for({5: ([10], True), 6: ([11], True),
                                                             7: ([12, 13], True), 8: ([14, 15], True)}, seen))
    report = evaluation.source_conditioning(model, examples, oracle_length=oracle, batch_size=2, sample_count=4)
    assert seen == [[5], [6], [6], [5], [7], [8], [8], [7]]
    assert report["original_token_sequence_exact_match"] == 1
    assert report["swapped_own_target_token_sequence_exact_match"] == 1
    assert report["swapped_original_target_token_sequence_exact_match"] == 0
    assert report["both_targets_correct_rate"] == report["response_change_rate"] == 1
    assert report["conditioning_gap"] == 1 and report["generated_rows"] == 8
    assert report["readout_calls"] == 16
    assert all(len(s["original_reference"]) == len(s["swapped_reference"]) for s in report["samples"])
    assert all(s["original_canvas"][-1] == model.config.eos_id for s in report["samples"])
    assert report["length_only_token_sequence_exact_match"] == (0.5 if oracle else None)


@pytest.mark.parametrize("changing", [False, True])
def test_constant_or_merely_changing_wrong_outputs_do_not_pass(monkeypatch, changing):
    model = fixture_model()
    examples = [([5], [10]), ([6], [11])]
    outputs = {5: ([20], True), 6: ([21 if changing else 20], True)}
    monkeypatch.setattr(evaluation, "generate", generator_for(outputs))
    report = evaluation.source_conditioning(model, examples)
    assert report["response_change_rate"] == int(changing)
    assert report["both_targets_correct_rate"] == report["conditioning_gap"] == 0
    assert report["original_token_sequence_exact_match"] == 0


def test_content_match_without_eos_is_reported_as_termination_failure(monkeypatch):
    model = fixture_model()
    monkeypatch.setattr(evaluation, "generate", generator_for({5: ([10], False), 6: ([11], False)}))
    report = evaluation.source_conditioning(model, [([5], [10]), ([6], [11])])
    assert report["original_content_exact_match"] == report["response_change_rate"] == 1
    assert report["original_token_sequence_exact_match"] == report["both_targets_correct_rate"] == 0
    assert report["original_missing_eos_rate"] == report["swapped_missing_eos_rate"] == 1
    assert report["original_early_eos_rate"] == 0


@pytest.mark.parametrize("examples", [[], [([5], [])], [([5], [10]), ([6], [10])],
                                      [([5], [10]), ([6], [11, 12])],
                                      [([8, 5], [10]), ([9, 5], [11])]])
def test_missing_real_length_matched_partners_are_unassessed(monkeypatch, examples):
    model = fixture_model(max_input_len=1)
    monkeypatch.setattr(evaluation, "generate", lambda *a, **k: pytest.fail("No valid source pair"))
    report = evaluation.source_conditioning(model, examples)
    assert report["status"] == "unassessed_no_length_matched_pairs"
    assert report["paired_examples"] == 0
    assert report["original_token_sequence_exact_match"] is None
    assert report["conditioning_gap"] is None and report["generated_rows"] == 0


def test_population_bounds_selection_and_partner_reuse_are_explicit(monkeypatch):
    model = fixture_model(max_input_len=1)
    examples = [([], [10]), ([5], [11] * 8), ([9, 5], [10]), ([6], [11]), ([7], [12]), ([8], [13])]
    monkeypatch.setattr(evaluation, "generate", generator_for({5: ([10], True), 6: ([11], True),
                                                             7: ([12], True), 8: ([13], True)}))
    first = evaluation.source_conditioning(model, examples, seed=9, max_examples=3)
    second = evaluation.source_conditioning(model, examples, seed=9, max_examples=3)
    assert first["selection_sha256"] == second["selection_sha256"]
    assert first["eligible_examples"] == 4 and first["selected_examples"] == 3
    assert first["excluded_empty_prompt"] == first["excluded_over_cap_answer"] == 1
    assert first["eligible_truncated_prompts"] == 1
    assert first["evaluation_coverage"] == 0.5 and first["paired_examples"] == 3
    assert first["unique_partner_examples"] == 2 and first["reused_partner_presentations"] == 1


def test_eval_restores_mixed_module_modes_rng_and_source_parameters(monkeypatch):
    model = fixture_model().train()
    model.encoder.blocks[0].eval()
    modes = [m.training for m in model.modules()]
    parameters = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    monkeypatch.setattr(evaluation, "generate", generator_for({5: ([10], True), 6: ([11], True)}, consume_rng=True))
    evaluation.source_conditioning(model, [([5], [10]), ([6], [11])])
    assert [m.training for m in model.modules()] == modes
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value, parameters[name]) for name, value in model.state_dict().items())


def test_eval_restores_modes_and_rng_after_generation_failure(monkeypatch):
    model = fixture_model().train()
    rng = torch.get_rng_state().clone()

    def failed(*a, **k):
        torch.rand(3)
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(evaluation, "generate", failed)
    with pytest.raises(RuntimeError, match="fixture failure"):
        evaluation.source_conditioning(model, [([5], [10]), ([6], [11])])
    assert model.training and all(m.training for m in model.modules())
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("kwargs", [{"max_examples": 0}, {"max_examples": 1025}, {"max_examples": True},
                                    {"batch_size": 0}, {"sample_count": 33}, {"seed": True},
                                    {"steps": -1}, {"rounds": 0}, {"oracle_length": 1}])
def test_invalid_bounds_are_refused_before_generation(kwargs):
    with pytest.raises(ValueError):
        evaluation.source_conditioning(fixture_model(), [], **kwargs)


def test_base_predicted_length_is_refused():
    with pytest.raises(ValueError, match="trained length head"):
        evaluation.source_conditioning(fixture_model(trained=False), [], oracle_length=False)


def test_real_tiny_model_runs_both_source_paths_without_learning():
    model = fixture_model()
    report = evaluation.source_conditioning(model, [([5], [10]), ([6], [11])], steps=1, rounds=1)
    assert report["paired_examples"] == 2 and report["generated_rows"] == 4
    assert report["readout_calls"] == 2
    assert all(p.grad is None for p in model.parameters())


def test_cli_reports_both_modes_and_keeps_decoded_samples_private(tmp_path, monkeypatch, capsys):
    script = load_script("evaluate")
    script.__file__ = str(tmp_path / "scripts/evaluate.py")
    checkpoint, tokenizer = make_checkpoint(tmp_path, trained=True)
    rows = [{"messages": [{"role": "user", "content": "w0"}], "response": "w1"},
            {"messages": [{"role": "user", "content": "w2"}], "response": "w3"}]
    instructions = write_rows(tmp_path / "valid.jsonl", rows)
    private = tmp_path / ".private/samples.jsonl"
    aggregate = tmp_path / "aggregate.json"
    events = tmp_path / "events"

    def observed(model, ids, valid, **kwargs):
        outcomes = []
        for row, mask in zip(ids, valid):
            prompt = row[mask].tolist()
            # The canonical encoder is called with add_special_tokens=False.
            answer = [tokenizer.token_to_id("w1" if tokenizer.token_to_id("w0") in prompt else "w3")]
            outcomes.append(answer)
        lengths = kwargs["canvas_len"]
        if lengths is None:
            lengths = torch.full((len(ids),), 2, dtype=torch.long)
        canvas = torch.tensor([a + [model.config.eos_id] for a in outcomes])
        return GenerationResult(outcomes, canvas, lengths, [True] * len(ids), readout_calls=4)

    monkeypatch.setattr(evaluation, "generate", observed)
    report = script.main(["--checkpoint", str(checkpoint), "--suites", "conditioning", "--device", "cpu",
                          "--instructions", str(instructions), "--output", str(aggregate),
                          "--conditioning-samples-output", str(private), "--conditioning-language", "ko",
                          "--conditioning-step", "17", "--conditioning-tensorboard-dir", str(events)])
    for mode in ("oracle_length", "predicted_length"):
        assert report["conditioning"][mode]["both_targets_correct_rate"] == 1
        assert "samples" not in report["conditioning"][mode]
    samples = [json.loads(line) for line in private.read_text(encoding="utf-8").splitlines()]
    assert len(samples) == report["conditioning"]["private_samples"]["rows"] == 4
    assert samples[0]["original_reference_text"] in ("w1", "w3")
    assert "w1" not in aggregate.read_text(encoding="utf-8") and "w1" not in capsys.readouterr().out
    assert json.loads(aggregate.read_text(encoding="utf-8")) == report
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    reader = EventAccumulator(str(events)).Reload()
    tag = "valid/ko/conditioning/oracle_length/both_targets_correct_rate"
    assert [(e.step, e.value) for e in reader.Scalars(tag)] == [(17, 1.0)]
    assert reader.Tags()["tensors"] == []  # No prompt, prediction, path or source text events.


@pytest.mark.parametrize("kind", ["public_samples", "existing_report", "existing_samples", "no_suite", "same_output"])
def test_cli_refuses_unsafe_or_existing_output_before_loading(tmp_path, monkeypatch, kind):
    script = load_script("evaluate")
    script.__file__ = str(tmp_path / "scripts/evaluate.py")
    monkeypatch.setattr(script, "load_pretrained", lambda *a, **k: pytest.fail("Must fail before allocation"))
    argv = ["--checkpoint", "unused", "--suites", "conditioning", "--instructions", "unused"]
    private = tmp_path / ".private/samples.jsonl"
    if kind == "public_samples":
        argv += ["--conditioning-samples-output", str(tmp_path / "public.jsonl")]
    elif kind == "existing_report":
        output = tmp_path / "report.json"
        output.write_text("preserve", encoding="utf-8")
        argv += ["--output", str(output)]
    elif kind == "existing_samples":
        private.parent.mkdir()
        private.write_text("preserve", encoding="utf-8")
        argv += ["--conditioning-samples-output", str(private)]
    elif kind == "same_output":
        argv += ["--output", str(private), "--conditioning-samples-output", str(private)]
    else:
        argv[argv.index("conditioning")] = "instructions"
        argv += ["--conditioning-samples-output", str(private)]
    with pytest.raises(SystemExit):
        script.main(argv)


def test_conditioning_input_is_bound_by_release_receipt(tmp_path):
    runner = load_script("run_release")
    rows = tmp_path / "valid.jsonl"
    rows.write_text("{}\n", encoding="utf-8")
    hashes = runner.evaluation_input_hashes(["python", "evaluate.py", "--suites", "conditioning",
                                           "--instructions", str(rows)])
    assert hashes["instructions"][0]["sha256"] == runner.file_sha256(rows)


def test_conditioning_limit_rejected_by_parser_and_missing_input_never_uses_train(monkeypatch):
    script = load_script("evaluate")
    monkeypatch.delenv("NURI_EVAL_INSTRUCTIONS", raising=False)
    monkeypatch.setenv("NURI_IT_DATA", "unused-train.jsonl")
    with pytest.raises(SystemExit, match="conditioning needs --instructions"):
        script.main(["--checkpoint", "unused", "--suites", "conditioning"])
    with pytest.raises(SystemExit):
        script.parse_args(["--checkpoint", "unused", "--suites", "conditioning", "--conditioning-examples", "1025"])


@pytest.mark.parametrize("kwargs", [{"assessment_recipe_config": "fixed.json"},
                                    {"assessment_recipe_config": ""},
                                    {"assessment_recipe_config": True, "holdout_config": "heldout.json"}])
def test_release_rejects_unused_or_invalid_assessment_recipe(kwargs):
    runner = load_script("run_release")
    with pytest.raises(ValueError):
        runner.validate_recipe_settings({"stages": {"dense": {"stage": "base", **kwargs}}})


def test_release_forwards_independent_assessment_only_for_explicit_stage(tmp_path):
    runner = load_script("run_release")
    stage = {"stage": "base", "holdout_config": "heldout.json", "assessment_recipe_config": "fixed.json"}
    plan = {"stages": {"dense": stage}}
    runner.validate_recipe_settings(plan)
    command = runner.train_command("dense", stage, plan, tmp_path)
    assert command[command.index("--assessment-recipe-config") + 1] == "fixed.json"
    with pytest.raises(ValueError):
        runner.validate_recipe_settings({**plan, "train": {"assessment_recipe_config": "fixed.json"}})
