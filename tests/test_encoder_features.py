"""Frozen encoder feature geometry, isolation, identity and test reservation."""

import hashlib
import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from nuri.checkpoint import save_pretrained
from nuri.config import tiny_config
from nuri.model import NuriModel
from scripts import encoder_features as features


@pytest.fixture
def private(tmp_path):
    folder = tmp_path / ".private"
    folder.mkdir()
    return folder


def native_checkpoint(private):
    torch.manual_seed(3)
    config = tiny_config(vocab_size=8, hidden_size=8, num_heads=2, ffn_size=16,
                         input_layers=1, readout_layers=1, latent_slots=2,
                         max_input_len=16, max_response_len=8, embed_dim=4)
    model = NuriModel(config).eval()
    # Poison unused heads/core so accidental use cannot look like a valid feature.
    with torch.no_grad():
        model.embed_proj.weight.fill_(float("nan"))
        model.reason_proj.weight.fill_(float("nan"))
        for parameter in model.core.parameters():
            parameter.fill_(float("nan"))
    folder = private / "native"
    save_pretrained(model, folder, dtype=torch.float32)
    tokenizer = Tokenizer(models.WordLevel({"[PAD]": 0, "[MASK]": 1, "[EOS]": 2,
                                            "[UNK]": 3, "hello": 4, "world": 5, "alpha": 6, "beta": 7},
                                           unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(folder / "tokenizer.json"))
    return model, folder


def local_hf_checkpoint(private):
    transformers = pytest.importorskip("transformers")
    folder = private / "reference"
    torch.manual_seed(5)
    config = transformers.BertConfig(vocab_size=9, hidden_size=8, num_hidden_layers=1,
                                      num_attention_heads=2, intermediate_size=16,
                                      max_position_embeddings=16)
    model = transformers.BertModel(config)
    model.save_pretrained(folder)
    tokenizer = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "[CLS]": 2,
                                            "[SEP]": 3, "[MASK]": 4, "hello": 5,
                                            "world": 6, "alpha": 7, "beta": 8}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]",
                                                              special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    wrapped = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]",
                                                    unk_token="[UNK]", cls_token="[CLS]",
                                                    sep_token="[SEP]", mask_token="[MASK]")
    wrapped.save_pretrained(folder)
    return folder


def test_modern_encoder_uses_eager_cpu_without_reference_compilation(private):
    transformers = pytest.importorskip("transformers")
    folder = local_hf_checkpoint(private)
    config = transformers.ModernBertConfig(vocab_size=9, hidden_size=8, intermediate_size=16,
                                           num_hidden_layers=2, num_attention_heads=2,
                                           max_position_embeddings=16, global_attn_every_n_layers=2,
                                           local_attention=4, pad_token_id=0, cls_token_id=2, sep_token_id=3,
                                           bos_token_id=None, eos_token_id=None,
                                           reference_compile=True)
    model = transformers.ModernBertModel(config)
    model.save_pretrained(folder)
    loaded, tokenizer, special, *_, details = features.load_transformers({"path_or_id": folder.name, "revision": "c" * 40}, private)
    result, observed = features.encode_field("transformers", loaded, tokenizer, special,
                                             ["hello", "hello world"], 2, 4, progress=False)
    assert result.shape == (2, 8) and np.isfinite(result).all()
    assert observed["retained_content_tokens"] == [1, 2]
    assert details["attention_implementation"] == "eager"
    assert details["reference_compile"] in (False, "not_exposed")


def test_hf_missing_encoder_weight_rejected_but_unused_pooler_missing_allowed(private):
    from safetensors.torch import load_file

    folder = local_hf_checkpoint(private)
    original = load_file(str(folder / "model.safetensors"))
    without_pooler = {name: value for name, value in original.items() if not name.startswith("pooler.")}
    save_file(without_pooler, str(folder / "model.safetensors"))
    loaded, *_, details = features.load_transformers({"path_or_id": folder.name, "revision": "d" * 40}, private)
    assert loaded.pooler is None
    assert details["loading_info"]["missing_keys"] and all(
        name.startswith("pooler.") for name in details["loading_info"]["missing_keys"])
    without_pooler.pop("embeddings.word_embeddings.weight")
    save_file(without_pooler, str(folder / "model.safetensors"))
    with pytest.raises(ValueError, match="Required encoder weights"):
        features.load_transformers({"path_or_id": folder.name, "revision": "d" * 40}, private)


def make_manifest(private, model, *, paired=False):
    inputs = {}
    for split in ("train", "valid"):
        rows = [{"id": f"{split}-first", "text": "hello", "label": 0, "split": split},
                {"id": f"{split}-second", "text": "hello world alpha beta hello", "label": 1, "split": split}]
        if paired:
            rows[0]["text_pair"] = "world alpha"
            rows[1]["text_pair"] = "beta"
        path = private / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        inputs[split] = {"path": path.name, "sha256": features.file_digest(path)}
    # A reserved split need not be opened or even present during TRAIN/VALID work.
    inputs["test"] = {"path": "reserved.jsonl", "sha256": "0" * 64}
    assessment = private / "assessment.json"
    assessment.write_text('{"schema_version":1,"selection":"valid"}', encoding="utf-8")
    manifest = {"schema_version": 1, "label_count": 2, "model": model, "inputs": inputs,
                "protocol": {"batch_size": 2, "max_length": 4},
                "assessment_protocol": {"path": assessment.name, "sha256": features.file_digest(assessment)}}
    path = private / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def test_content_pooling_excludes_padding_and_special_positions():
    hidden = torch.tensor([[[100., 200.], [2., 4.], [6., 8.], [999., 999.]]])
    assert torch.equal(features.content_mean(hidden, torch.tensor([[False, True, True, False]])),
                       torch.tensor([[4., 6.]]))
    hidden[0, 0] = float("nan")
    assert torch.isfinite(features.content_mean(hidden, torch.tensor([[False, True, True, False]]))).all()
    with pytest.raises(ValueError, match="content token"):
        features.content_mean(hidden, torch.zeros(1, 4, dtype=torch.bool))


def test_native_selective_loading_matches_encoder_without_heads_or_core(private, monkeypatch):
    trained, folder = native_checkpoint(private)
    original = features.safe_open
    read_keys = []

    class SelectedReader:
        def __init__(self, *args, **kwargs):
            self.reader = original(*args, **kwargs)

        def __enter__(self):
            self.reader.__enter__()
            return self

        def __exit__(self, *args):
            return self.reader.__exit__(*args)

        def keys(self):
            return self.reader.keys()

        def get_tensor(self, name):
            read_keys.append(name)
            assert name == "embedding.weight" or name.startswith("encoder.")
            return self.reader.get_tensor(name)

    monkeypatch.setattr(features, "safe_open", SelectedReader)
    monkeypatch.setattr(NuriModel, "__init__", lambda *args: pytest.fail("Full Nuri model must not be instantiated"))
    model, tokenizer, special, _, _, details = features.load_nuri({"path_or_id": folder.name}, private)
    inputs, mask, observed = features.tokenize_texts("nuri", tokenizer, special, model,
                                                     ["hello", "hello world"], 4)
    with torch.inference_mode():
        expected, valid = trained.encode(inputs["input_ids"], inputs["attention_mask"])
        actual = features.content_mean(model(**inputs), mask)
    assert torch.equal(actual, features.content_mean(expected, valid))
    assert torch.isfinite(actual).all() and actual.shape == (2, 8)
    assert set(read_keys) == set(details["loaded_tensor_names"])
    assert not hasattr(model, "core") and not hasattr(model, "embed_proj")
    assert observed["retained_content_tokens"] == [1, 2]


def test_export_prefix_and_seeded_random_control_do_not_load_unused_weights(private):
    trained, folder = native_checkpoint(private)
    weights = {"model." + name: value for name, value in trained.state_dict().items()}
    save_file(weights, str(folder / "model.safetensors"))
    config = trained.config.to_dict()
    (folder / "config.json").write_text(json.dumps({"nuri": config}), encoding="utf-8")
    exported, *_ = features.load_nuri({"path_or_id": folder.name}, private)
    assert torch.equal(exported.embedding.weight, trained.embedding.weight)
    (folder / "model.safetensors").unlink()
    rng = torch.get_rng_state()
    first, *_, first_details = features.load_nuri({"path_or_id": folder.name, "random_initialization_seed": 0}, private)
    second, *_, second_details = features.load_nuri({"path_or_id": folder.name, "random_initialization_seed": 0}, private)
    different, *_, different_details = features.load_nuri({"path_or_id": folder.name, "random_initialization_seed": 1}, private)
    assert torch.equal(rng, torch.get_rng_state())
    assert first_details["selected_tensor_sha256"] == second_details["selected_tensor_sha256"]
    assert first_details["selected_tensor_sha256"] != different_details["selected_tensor_sha256"]
    assert torch.equal(first.embedding.weight, second.embedding.weight)
    assert not torch.equal(first.embedding.weight, different.embedding.weight)


@pytest.mark.parametrize("backend", ["nuri", "transformers"])
def test_pair_cache_geometry_identity_and_truncation(private, backend):
    if backend == "nuri":
        _, folder = native_checkpoint(private)
    else:
        folder = local_hf_checkpoint(private)
    spec = {"backend": backend, "path_or_id": folder.name, "revision": "a" * 40}
    manifest_path, manifest = make_manifest(private, spec, paired=True)
    output = private / "cache"
    report = features.extract_features(manifest_path, output, progress=False)
    model, tokenizer, special, *_ = (features.load_nuri if backend == "nuri" else features.load_transformers)(spec, private)
    u, _ = features.encode_field(backend, model, tokenizer, special, ["hello", "hello world alpha beta hello"], 2, 4, progress=False)
    v, _ = features.encode_field(backend, model, tokenizer, special, ["world alpha", "beta"], 2, 4, progress=False)
    with np.load(output / "train.npz", allow_pickle=False) as cache:
        assert cache.files == ["features", "labels", "row_ids", "metadata_json"]
        assert cache["features"].dtype == np.float32 and cache["features"].shape == (2, 32)
        np.testing.assert_array_equal(cache["features"], features.combine_pair(u, v))
        assert cache["labels"].dtype == np.int64 and cache["row_ids"].dtype.kind == "U"
        assert cache["metadata_json"].shape == () and cache["metadata_json"].dtype.kind == "U"
        metadata = json.loads(str(cache["metadata_json"]))
        assert metadata["raw_input_sha256"] == manifest["inputs"]["train"]["sha256"]
        assert metadata["assessment_protocol_sha256"] == manifest["assessment_protocol"]["sha256"]
        assert metadata["protocol_sha256"] != metadata["assessment_protocol_sha256"]
        assert metadata["row_ids_sha256"] == hashlib.sha256(b'["train-first","train-second"]').hexdigest()
        assert metadata["model_revision"] and metadata["label_count"] == 2
    assert report["used_parameter_count"] == sum(value.numel() for value in model.parameters())
    assert report["splits"]["train"]["observed_lengths"]["text"]["truncated_fields"] == 1
    assert report["splits"]["train"]["observed_lengths"]["text_pair"]["truncated_fields"] == 0
    complete = json.loads((output / "COMPLETE.json").read_text())
    assert complete["report_sha256"] == features.file_digest(output / "report.json")
    assert complete["features"]["train"] == features.file_digest(output / "train.npz")
    assert not (private / "reserved.jsonl").exists()
    with pytest.raises(FileExistsError, match="immutable"):
        features.extract_features(manifest_path, output, progress=False)


def test_hf_unknown_content_is_retained_and_wrappers_pad_excluded(private):
    folder = local_hf_checkpoint(private)
    model, tokenizer, special, *_ = features.load_transformers({"path_or_id": folder.name, "revision": "b" * 40}, private)
    inputs, content, observed = features.tokenize_texts("transformers", tokenizer, special, model,
                                                      ["unseen", "hello world"], 4)
    assert inputs["input_ids"].tolist() == [[2, 1, 3, 0], [2, 5, 6, 3]]
    assert content.tolist() == [[False, True, False, False], [False, True, True, False]]
    assert observed["full_content_tokens"] == observed["retained_content_tokens"] == [1, 2]
    assert model.pooler is None and model.config._attn_implementation == "eager"
    assert not model.training and all(not parameter.requires_grad for parameter in model.parameters())


def test_reserved_test_and_changed_input_bindings_fail_before_model_load(private, monkeypatch):
    _, folder = native_checkpoint(private)
    manifest_path, manifest = make_manifest(private, {"backend": "nuri", "path_or_id": folder.name})
    monkeypatch.setattr(features, "load_nuri", lambda *args: pytest.fail("Refusal must precede model loading"))
    with pytest.raises(ValueError, match="reserved"):
        features.extract_features(manifest_path, private / "test-cache", splits=("test",), progress=False)
    assert not (private / "test-cache").exists()
    with (private / "train.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Input bytes"):
        features.extract_features(manifest_path, private / "changed-cache", progress=False)
    assert not (private / "changed-cache").exists()
    manifest["inputs"]["train"]["sha256"] = features.file_digest(private / "train.jsonl")
    (private / "assessment.json").write_text("{}")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Assessment protocol bytes"):
        features.extract_features(manifest_path, private / "assessment-cache", progress=False)


def test_completion_flag_allows_separate_test_cache_and_cli_reports_counts(private, capsys):
    _, folder = native_checkpoint(private)
    manifest_path, manifest = make_manifest(private, {"backend": "nuri", "path_or_id": folder.name})
    test = private / "reserved.jsonl"
    test.write_text('{"id":"test-only","text":"hello","label":0,"split":"test"}\n')
    manifest["inputs"]["test"]["sha256"] = features.file_digest(test)
    manifest_path.write_text(json.dumps(manifest))
    output = private / "completion-cache"
    assert features.main(["--manifest", str(manifest_path), "--output", str(output), "--splits", "test",
                          "--allow-test-completion", "--no-progress"]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "complete", "splits": {"test": 1}}
    assert not (output / "train.npz").exists()


def test_unicode_row_digest_and_save_contract(private):
    ids = ["first", "두번째"]
    expected = hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert features.row_ids_digest(ids) == expected
    path = private / "saved.npz"
    features.save_features(path, np.ones((2, 3), dtype=np.float32), np.array([0, 1], dtype=np.int64), ids, {"split": "train"})
    with np.load(path, allow_pickle=False) as cache:
        assert cache["row_ids"].tolist() == ids
    with pytest.raises(FileExistsError):
        features.save_features(path, np.ones((2, 3), dtype=np.float32), np.array([0, 1], dtype=np.int64), ids, {})


def test_hf_mutable_revision_rejected_without_network(private):
    with pytest.raises(ValueError, match="immutable"):
        features.hf_folder("unused", "main", private)


def test_new_cache_binds_changed_model_tokenizer_and_feature_protocol(private):
    trained, folder = native_checkpoint(private)
    manifest_path, manifest = make_manifest(private, {"backend": "nuri", "path_or_id": folder.name})
    first = features.extract_features(manifest_path, private / "first", progress=False)
    with torch.no_grad():
        trained.embedding.weight[4].add_(0.5)
    save_pretrained(trained, folder, dtype=torch.float32)
    changed_model = features.extract_features(manifest_path, private / "new-model", progress=False)
    assert first["model_sha256"] != changed_model["model_sha256"]
    assert first["tokenizer_sha256"] == changed_model["tokenizer_sha256"]
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    vocab = tokenizer.get_vocab()
    vocab["hello"], vocab["world"] = vocab["world"], vocab["hello"]
    tokenizer.model = models.WordLevel(vocab, unk_token="[UNK]")
    tokenizer.save(str(folder / "tokenizer.json"))
    changed_tokenizer = features.extract_features(manifest_path, private / "new-tokenizer", progress=False)
    assert changed_tokenizer["model_sha256"] == changed_model["model_sha256"]
    assert changed_tokenizer["tokenizer_sha256"] != changed_model["tokenizer_sha256"]
    manifest["protocol"]["max_length"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    changed_protocol = features.extract_features(manifest_path, private / "new-protocol", progress=False)
    assert changed_protocol["protocol_sha256"] != changed_tokenizer["protocol_sha256"]
    assert changed_protocol["splits"]["train"]["raw_input_sha256"] == first["splits"]["train"]["raw_input_sha256"]


def test_artifact_mutation_during_extraction_cannot_publish_completion(private, monkeypatch):
    _, folder = native_checkpoint(private)
    manifest_path, _ = make_manifest(private, {"backend": "nuri", "path_or_id": folder.name})
    original = features.encode_field

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        with (folder / "tokenizer.json").open("a", encoding="utf-8") as stream:
            stream.write(" ")
        return result

    monkeypatch.setattr(features, "encode_field", mutate)
    output = private / "incomplete"
    with pytest.raises(ValueError, match="changed during extraction"):
        features.extract_features(manifest_path, output, progress=False)
    assert (output / "IN_PROGRESS.json").exists()
    assert not (output / "COMPLETE.json").exists() and not (output / "report.json").exists()


def test_tiny_extractor_to_real_probe_handoff(private):
    from scripts.encoder_probe import run_probe

    _, folder = native_checkpoint(private)
    manifest_path, _ = make_manifest(private, {"backend": "nuri", "path_or_id": folder.name,
                                              "revision": "b" * 64}, paired=True)
    output = private / "features"
    features.extract_features(manifest_path, output, progress=False)
    report = run_probe(output / "train.npz", output / "valid.npz", private / "probe.json",
                       protocol_path=private / "assessment.json", bootstrap_samples=20, progress=False)
    assert report["status"] == "completed"
    assert report["frozen_protocol_sha256"] == features.file_digest(private / "assessment.json")
    assert report["validation_predictions"]["row_ids"] == ["valid-first", "valid-second"]
    assert report["validation_predictions"]["labels"] == [0, 1]
