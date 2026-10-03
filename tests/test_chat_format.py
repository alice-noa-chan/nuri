"""The tokenizer's chat template must build exactly the prompts the IT stage trains on."""

import importlib.util
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")

from nuri.data import chat_prompt_text  # noqa: E402


def tokenizer_template() -> str:
    path = Path(__file__).resolve().parent.parent / "tools" / "train_tokenizer.py"
    spec = importlib.util.spec_from_file_location("train_tokenizer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CHAT_TEMPLATE


@pytest.mark.parametrize("messages", [
    [{"role": "user", "content": "Explain RoPE."}],
    [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "안녕하세요"}],
    [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}],
])
def test_chat_template_matches_training_prompt(messages):
    from jinja2 import Template

    rendered = Template(tokenizer_template()).render(messages=messages, add_generation_prompt=True)
    assert rendered == chat_prompt_text(messages)
