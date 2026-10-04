import math

import pytest
import torch

from nuri.config import tiny_config
from nuri.model import NuriModel
from tools import benchmark_nuri as benchmark


def config():
    return tiny_config(hidden_size=16, num_heads=2, ffn_size=32,
                       input_layers=1, readout_layers=1)


def packed_run(monkeypatch, checkpointing):
    captured = {}

    def capture_model(model_config):
        model = NuriModel(model_config)
        captured['model'] = model
        attention = model.encoder.blocks[0].attn
        original_forward = attention.forward

        def forward(*args, **kwargs):
            captured['keys'] = kwargs['key_mask'].clone()
            return original_forward(*args, **kwargs)

        attention.forward = forward
        return model

    monkeypatch.setattr(benchmark, 'NuriModel', capture_model)
    report = benchmark.run_benchmark(config(), task='mlm', training=True,
                                     batch_size=2, input_length=8, output_length=4,
                                     pack_mlm=True, packed_document_length=3,
                                     gradient_checkpointing=checkpointing,
                                     seed=7, warmup=0, iterations=1)
    return report, captured


def test_packed_training_isolates_documents_and_preserves_token_accounting(monkeypatch):
    report, captured = packed_run(monkeypatch, False)
    document_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])
    expected = document_ids[:, None] == document_ids[None, :]
    assert torch.equal(captured['keys'], expected.expand(2, -1, -1))
    assert report['pack_mlm'] and report['packed_document_length'] == 3
    assert report['packed_documents_per_row'] == 3
    assert report['input_sequences_per_sample'] == 1
    assert report['input_tokens_per_second'] * report['latency_seconds']['mean'] == pytest.approx(16)
    assert report['output_length'] is None
    gradient = captured['model'].embedding.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_checkpointed_packed_benchmark_keeps_gradients_and_updates(monkeypatch):
    plain_report, plain = packed_run(monkeypatch, False)
    saved_report, saved = packed_run(monkeypatch, True)
    assert not plain_report['gradient_checkpointing']
    assert saved_report['gradient_checkpointing']
    assert saved['model'].encoder.gradient_checkpointing
    assert saved['model'].readout.gradient_checkpointing
    for (name, left), (other_name, right) in zip(plain['model'].named_parameters(), saved['model'].named_parameters()):
        assert name == other_name
        assert torch.allclose(left, right, atol=1e-6), name
        if left.grad is not None:
            assert right.grad is not None and torch.allclose(left.grad, right.grad, atol=1e-6), name
    assert math.isfinite(saved_report['input_tokens_per_second'])


@pytest.mark.parametrize('options,match', [
    ({'training': False, 'gradient_checkpointing': True}, 'training benchmark'),
    ({'training': False, 'pack_mlm': True}, 'Packed MLM'),
    ({'task': 'denoise', 'training': True, 'pack_mlm': True}, 'Packed MLM'),
    ({'task': 'pairs', 'training': True, 'pack_mlm': True}, 'Packed MLM'),
    ({'training': True, 'pack_mlm': True, 'packed_document_length': 0}, 'document length'),
    ({'training': True, 'pack_mlm': True, 'packed_document_length': 8}, 'document length'),
])
def test_invalid_runtime_options_fail_before_model_allocation(monkeypatch, options, match):
    def unexpected_allocation(*args, **kwargs):
        pytest.fail('Invalid options must be rejected before allocating a model')

    monkeypatch.setattr(benchmark, 'NuriModel', unexpected_allocation)
    arguments = dict(task='mlm', batch_size=2, input_length=8, output_length=4)
    arguments.update(options)
    with pytest.raises(ValueError, match=match):
        benchmark.run_benchmark(config(), **arguments)


def test_cli_records_requested_runtime_options(tmp_path):
    output = tmp_path / 'packed.json'
    assert benchmark.main(['--preset', 'tiny', '--task', 'mlm', '--training',
                           '--device', 'cpu', '--input-length', '8', '--output-length', '4',
                           '--pack-mlm', '--packed-document-length', '3', '--gradient-checkpointing',
                           '--warmup', '0', '--iterations', '1', '--output', str(output)]) == 0
    import json
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['pack_mlm'] and report['gradient_checkpointing']
    assert report['packed_documents_per_row'] == 3
