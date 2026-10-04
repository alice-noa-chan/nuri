"""Natural instruction evaluation preserves references and reports its population."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from nuri.checkpoint import save_pretrained
from nuri.config import tiny_config
from nuri.data import chat_prompt_text
from nuri.generation import GenerationResult
from nuri.model import NuriModel

SPECIALS = ["[PAD]", "[MASK]", "[EOS]", "[SEP]", "<|system|>", "<|user|>", "<|assistant|>"]
WORDS = SPECIALS + ["[UNK]"] + [f"w{i}" for i in range(20)]


def load_script(name):
    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"instruction_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_checkpoint(tmp_path, trained=False):
    tokenizer = Tokenizer(models.WordLevel({word: index for index, word in enumerate(WORDS)},
                                           unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens(SPECIALS)
    # A postprocessor makes accidental add_special_tokens=True observable.
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[SEP] $A [EOS]", special_tokens=[("[SEP]", 3), ("[EOS]", 2)])
    folder = tmp_path / "tok"
    folder.mkdir()
    tokenizer.save(str(folder / "tokenizer.json"))
    config = tiny_config(vocab_size=len(WORDS), max_input_len=16, max_response_len=8,
                         length_head_trained=trained)
    checkpoint = tmp_path / "ckpt"
    save_pretrained(NuriModel(config), checkpoint, tokenizer_dir=folder)
    return checkpoint, tokenizer


def write_rows(path, rows):
    # Blank JSONL lines do not count as records, as in the training reader.
    path.write_text("\n\n" + "\n\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def row(response="w1"):
    return {"messages": [{"role": "user", "content": "w0"}], "response": response}


def empty_generation(model, input_ids, attention_mask, **kwargs):
    count = len(input_ids)
    lengths = kwargs["canvas_len"]
    if lengths is None:
        lengths = torch.ones(count, dtype=torch.long)
    canvas = torch.full((count, int(lengths.max())), model.config.eos_id, dtype=torch.long)
    return GenerationResult([[] for _ in range(count)], canvas, lengths, [True] * count)


@pytest.mark.parametrize("trained", [False, True])
def test_full_canonical_file_matches_training_tokens_and_reports_coverage(tmp_path, monkeypatch, trained):
    script = load_script("evaluate")
    checkpoint, tokenizer = make_checkpoint(tmp_path, trained)
    records = [
        {"messages": [{"role": "system", "content": "w0"},
                      {"role": "user", "content": "w1"},
                      {"role": "assistant", "content": "w2"},
                      {"role": "user", "content": " ".join(["w3"] * 20)}],
         "response": "w4 w5"},
        row(""), row(" ".join(["w6"] * 8)), row(" ".join(["w7"] * 7)), row("w8"),
    ]
    instructions = write_rows(tmp_path / "heldout.jsonl", records)
    forwarded, generated_prompts = [], []
    actual_accuracy = script.evaluation.generation_accuracy

    def observed_accuracy(model, examples, **kwargs):
        forwarded.append((list(examples), kwargs.get("oracle_length", False)))
        return actual_accuracy(model, examples, **kwargs)

    def observed_generation(model, input_ids, valid, **kwargs):
        generated_prompts.extend([ids[mask].tolist() for ids, mask in zip(input_ids, valid)])
        return empty_generation(model, input_ids, valid, **kwargs)

    monkeypatch.setattr(script.evaluation, "generation_accuracy", observed_accuracy)
    monkeypatch.setattr(script.evaluation, "generate", observed_generation)
    output = tmp_path / "report.json"
    report = script.main(["--checkpoint", str(checkpoint), "--suites", "instructions",
                          "--instructions", str(instructions), "--examples", "1", "--device", "cpu",
                          "--output", str(output)])
    expected = [(tokenizer.encode(chat_prompt_text(record["messages"]), add_special_tokens=False).ids,
                 tokenizer.encode(record["response"], add_special_tokens=False).ids) for record in records]
    assert [examples for examples, _ in forwarded] == [expected] * (2 if trained else 1)
    assert [oracle for _, oracle in forwarded] == ([True, False] if trained else [True])
    assert expected[1][1] == [] and len(expected[2][1]) == 8  # Empty and over-cap targets remain intact.
    assert expected[0][0].count(2) == 4  # One input EOS per chat turn, with no automatic extra EOS.
    assert expected[0][1] == [tokenizer.token_to_id("w4"), tokenizer.token_to_id("w5")]
    eligible_prompts = [prompt[-16:] for prompt, answer in expected if len(answer) + 1 <= 8]
    assert generated_prompts == eligible_prompts * (2 if trained else 1)
    result = report["instructions"]
    assert result["input_examples"] == 5 and result["oracle_length_only"] is (not trained)
    assert report["oracle_length_only"] is (not trained)
    assert ("predicted_length" in result) is trained
    for mode in ("oracle_length", "predicted_length") if trained else ("oracle_length",):
        assert result[mode]["examples"] == 4 and result[mode]["excluded"] == 1
        assert result[mode]["evaluation_coverage"] == pytest.approx(0.8)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert str(instructions) not in output.read_text(encoding="utf-8")


@pytest.mark.parametrize("trained", [False, True])
@pytest.mark.parametrize("records", [[], [row(" ".join(["w1"] * 8))]])
def test_empty_or_all_over_cap_file_reports_zero_population(tmp_path, monkeypatch, trained, records):
    script = load_script("evaluate")
    checkpoint, _ = make_checkpoint(tmp_path, trained)
    instructions = write_rows(tmp_path / "heldout.jsonl", records)

    def unexpected_generation(*args, **kwargs):
        pytest.fail("Generation must not run without eligible answers")

    monkeypatch.setattr(script.evaluation, "generate", unexpected_generation)
    result = script.main(["--checkpoint", str(checkpoint), "--suites", "instructions",
                          "--instructions", str(instructions), "--device", "cpu"])["instructions"]
    assert result["input_examples"] == len(records)
    for mode in ("oracle_length", "predicted_length") if trained else ("oracle_length",):
        assert result[mode]["examples"] == 0 and result[mode]["excluded"] == len(records)
        assert result[mode]["evaluation_coverage"] == 0.0


def test_environment_input_and_cli_precedence(tmp_path, monkeypatch):
    script = load_script("evaluate")
    checkpoint, _ = make_checkpoint(tmp_path)
    env_input = write_rows(tmp_path / "env.jsonl", [row()])
    cli_input = write_rows(tmp_path / "cli.jsonl", [row(), row("")])
    monkeypatch.setenv("NURI_EVAL_INSTRUCTIONS", str(env_input))
    monkeypatch.setattr(script.evaluation, "generate", empty_generation)
    argv = ["--checkpoint", str(checkpoint), "--suites", "instructions", "--device", "cpu"]
    assert script.parse_args(argv).instructions == str(env_input)
    assert script.main(argv)["instructions"]["input_examples"] == 1
    assert script.parse_args(argv + ["--instructions", str(cli_input)]).instructions == str(cli_input)
    assert script.main(argv + ["--instructions", str(cli_input)])["instructions"]["input_examples"] == 2


def test_missing_flag_never_falls_back_to_training_input(monkeypatch):
    script = load_script("evaluate")
    monkeypatch.delenv("NURI_EVAL_INSTRUCTIONS", raising=False)
    monkeypatch.setenv("NURI_IT_DATA", "unused-training-input.jsonl")
    with pytest.raises(SystemExit, match="instructions needs --instructions.*NURI_EVAL_INSTRUCTIONS"):
        script.main(["--checkpoint", "unused", "--suites", "instructions"])


@pytest.mark.parametrize("directory", [False, True])
def test_unavailable_input_fails_clearly(tmp_path, directory):
    script = load_script("evaluate")
    checkpoint, _ = make_checkpoint(tmp_path)
    instructions = tmp_path / "unavailable"
    if directory:
        instructions.mkdir()
    with pytest.raises(SystemExit, match="instructions input is unavailable or unreadable"):
        script.main(["--checkpoint", str(checkpoint), "--suites", "instructions",
                     "--instructions", str(instructions), "--device", "cpu"])


@pytest.mark.parametrize("invalid", [
    [], {"messages": [{"role": "user", "content": "w0"}]}, row(None),
    {"messages": [], "response": "w1"}, {"messages": "w0", "response": "w1"},
    {"messages": ["w0"], "response": "w1"},
    {"messages": [{"role": "unknown", "content": "w0"}], "response": "w1"},
    {"messages": [{"role": "user", "content": None}], "response": "w1"},
])
def test_malformed_record_fails_without_partial_evaluation(tmp_path, monkeypatch, invalid):
    script = load_script("evaluate")
    checkpoint, _ = make_checkpoint(tmp_path)
    instructions = write_rows(tmp_path / "invalid.jsonl", [row(), invalid])

    def unexpected_accuracy(*args, **kwargs):
        pytest.fail("A malformed input must fail before any instruction evaluation")

    monkeypatch.setattr(script.evaluation, "generation_accuracy", unexpected_accuracy)
    with pytest.raises(SystemExit, match="instructions record 2 needs"):
        script.main(["--checkpoint", str(checkpoint), "--suites", "instructions",
                     "--instructions", str(instructions), "--device", "cpu"])


@pytest.mark.parametrize("payload", [b'{"messages":', b'\xff\xfe'])
def test_invalid_json_or_utf8_fails_with_generic_message(tmp_path, payload):
    script = load_script("evaluate")
    checkpoint, _ = make_checkpoint(tmp_path)
    instructions = tmp_path / "invalid.jsonl"
    instructions.write_bytes(payload)
    with pytest.raises(SystemExit, match="instructions input must contain valid UTF-8 JSONL"):
        script.main(["--checkpoint", str(checkpoint), "--suites", "instructions",
                     "--instructions", str(instructions), "--device", "cpu"])


def test_combined_suites_validate_instructions_before_any_generation(tmp_path, monkeypatch):
    script = load_script("evaluate")
    checkpoint, _ = make_checkpoint(tmp_path)
    instructions = write_rows(tmp_path / "invalid.jsonl", [row(), {"messages": []}])

    def unexpected_accuracy(*args, **kwargs):
        pytest.fail("Validate requested instruction inputs before running another generation suite")

    monkeypatch.setattr(script.evaluation, "generation_accuracy", unexpected_accuracy)
    with pytest.raises(SystemExit, match="instructions record 2 needs"):
        script.main(["--checkpoint", str(checkpoint), "--suites", "reasoning", "instructions",
                     "--instructions", str(instructions), "--examples", "2", "--device", "cpu"])


@pytest.mark.parametrize("trained", [False, True])
def test_instruction_suite_runs_real_generation_on_tiny_checkpoint(tmp_path, trained):
    script = load_script("evaluate")
    checkpoint, _ = make_checkpoint(tmp_path, trained)
    instructions = write_rows(tmp_path / "heldout.jsonl", [row("w1"), row("")])
    result = script.main(["--checkpoint", str(checkpoint), "--suites", "instructions",
                          "--instructions", str(instructions), "--device", "cpu"])["instructions"]
    assert result["input_examples"] == 2 and result["oracle_length"]["examples"] == 2
    assert ("predicted_length" in result) is trained


def test_release_runner_passes_only_optional_heldout_instruction_path(tmp_path):
    runner = load_script("run_release")
    spec = {"evaluate": ["instructions"]}
    plan = {"data": {"instructions": "training.jsonl", "eval_instructions": "heldout.jsonl"}}
    command = runner.evaluate_command("it", spec, plan, tmp_path / "runs")
    assert command[command.index("--instructions") + 1] == "heldout.jsonl"
    assert "training.jsonl" not in command
    del plan["data"]["eval_instructions"]
    command = runner.evaluate_command("it", spec, plan, tmp_path / "runs")
    assert "--instructions" not in command and "training.jsonl" not in command
