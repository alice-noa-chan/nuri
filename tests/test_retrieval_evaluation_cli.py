"""The real retrieval CLI validates its population and preserves one candidate pool."""

import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from nuri.checkpoint import save_pretrained
from nuri.config import tiny_config
from nuri.model import NuriModel
from scripts import evaluate as cli


SPECIALS = ["[PAD]", "[MASK]", "[EOS]", "[SEP]", "<|system|>", "<|user|>", "<|assistant|>"]


def checkpoint(tmp_path):
    words = SPECIALS + ["[UNK]", "alpha", "beta", "gamma", "delta"]
    tokenizer = Tokenizer(models.WordLevel({word: index for index, word in enumerate(words)},
                                           unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens(SPECIALS)
    folder = tmp_path / "tokenizer"
    folder.mkdir()
    tokenizer.save(str(folder / "tokenizer.json"))
    path = tmp_path / "checkpoint"
    save_pretrained(NuriModel(tiny_config(vocab_size=len(words), max_input_len=2)), path,
                    tokenizer_dir=folder)
    return path


def write_pairs(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def run(path, pairs, *extra):
    return cli.main(["--checkpoint", str(path), "--suites", "retrieval", "--device", "cpu",
                     "--pairs", str(pairs), *extra])


def test_real_report_reuses_full_pool_ranks_and_reports_empty_exclusions(tmp_path, monkeypatch, capsys):
    path = checkpoint(tmp_path)
    pairs = write_pairs(tmp_path / "private-input-identity.jsonl", [
        {"query": "alpha beta gamma", "positive": "alpha beta delta"},
        {"query": "gamma", "positive": "delta"},
        {"query": "[SEP]", "positive": "[SEP]"},
        {"query": "", "positive": "alpha"},
    ])
    model_calls, lexical_calls = [], []
    actual_model = cli.evaluation.retrieval_ranks
    actual_lexical = cli.evaluation.lexical_retrieval_ranks

    def model_ranks(model, population, **kwargs):
        model_calls.append((len(population), kwargs))
        return actual_model(model, population, **kwargs)

    def lexical_ranks(population, vocab_size, **kwargs):
        lexical_calls.append((len(population), kwargs))
        return actual_lexical(population, vocab_size, **kwargs)

    monkeypatch.setattr(cli.evaluation, "retrieval_ranks", model_ranks)
    monkeypatch.setattr(cli.evaluation, "lexical_retrieval_ranks", lexical_ranks)
    output = tmp_path / "report.json"
    report = run(path, pairs, "--output", str(output))
    overlap = report["retrieval_overlap"]
    assert report["retrieval_input"] == {"supplied_pairs": 4, "evaluated_pairs": 3, "excluded_empty_pairs": 1}
    assert len(model_calls) == 3 and all(count == 3 for count, _ in model_calls)
    assert lexical_calls == [(3, {"max_len": 2})]
    assert report["retrieval"] == overlap["overall"]["model"]
    assert report["retrieval_lexical_baseline"] == overlap["overall"]["lexical_tfidf"]
    assert sum(bucket["queries"] for bucket in overlap["buckets"]) == 3
    assert all(bucket["candidates"] == 3 for bucket in overlap["buckets"])
    assert overlap["overall"]["summary"]["query_tokens"]["max"] == 2
    assert overlap["overlap_special_ids_excluded"] == list(range(7))
    assert next(bucket for bucket in overlap["buckets"] if bucket["name"] == "empty_lexical")["queries"] == 1
    assert json.loads(output.read_text()) == report
    assert pairs.name not in capsys.readouterr().out
    assert pairs.name not in output.read_text()


@pytest.mark.parametrize("rows", [[], [{"query": "", "positive": "alpha"}]])
def test_empty_evaluation_population_is_explicit_and_never_scored(tmp_path, monkeypatch, rows):
    path = checkpoint(tmp_path)
    pairs = write_pairs(tmp_path / "pairs.jsonl", rows)

    def unexpected(*args, **kwargs):
        pytest.fail("an empty population has no scoring work")

    monkeypatch.setattr(cli.evaluation, "retrieval_ranks", unexpected)
    monkeypatch.setattr(cli.evaluation, "lexical_retrieval_ranks", unexpected)
    report = run(path, pairs)
    assert report["retrieval_input"]["supplied_pairs"] == len(rows)
    assert report["retrieval_input"]["evaluated_pairs"] == 0
    assert report["retrieval_overlap"]["overall"]["candidates"] == 0
    assert all(row["recall@1"] == 0 and row["queries"] == 0 for row in report["retrieval"])


@pytest.mark.parametrize("contents, message", [
    ('{"query": "alpha"}', "string query and positive"),
    ('{"query": 2, "positive": "beta"}', "string query and positive"),
    ('["alpha", "beta"]', "string query and positive"),
    ('not-json', "valid UTF-8 JSONL"),
])
def test_bad_requested_pair_file_fails_before_other_suites(tmp_path, monkeypatch, contents, message):
    path = checkpoint(tmp_path)
    pairs = tmp_path / "private-unreadable-identity.jsonl"
    pairs.write_text(contents)

    def unexpected(*args, **kwargs):
        pytest.fail("other suites must not run before requested input validation")

    monkeypatch.setattr(cli.evaluation, "generation_accuracy", unexpected)
    with pytest.raises(SystemExit, match=message):
        cli.main(["--checkpoint", str(path), "--suites", "reasoning", "retrieval", "--pairs", str(pairs)])


def test_unavailable_pair_input_uses_generic_error(tmp_path):
    path = checkpoint(tmp_path)
    with pytest.raises(SystemExit, match="pairs input is unavailable") as error:
        run(path, tmp_path / "private-unavailable-identity.jsonl")
    assert "identity" not in str(error.value)


def test_environment_input_and_explicit_override(tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    env_pairs = write_pairs(tmp_path / "env.jsonl", [{"query": "alpha", "positive": "beta"}])
    explicit = write_pairs(tmp_path / "explicit.jsonl", [{"query": "beta", "positive": "alpha"}] * 2)
    monkeypatch.setenv("NURI_EVAL_PAIRS", str(env_pairs))
    from_environment = cli.main(["--checkpoint", str(path), "--suites", "retrieval", "--device", "cpu"])
    assert from_environment["retrieval_input"]["supplied_pairs"] == 1
    assert run(path, explicit)["retrieval_input"]["supplied_pairs"] == 2


def test_missing_pair_argument_fails_before_checkpoint_access(monkeypatch):
    monkeypatch.delenv("NURI_EVAL_PAIRS", raising=False)
    with pytest.raises(SystemExit, match="retrieval needs --pairs"):
        cli.main(["--checkpoint", "unavailable", "--suites", "retrieval"])
