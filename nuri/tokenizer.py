"""Connect a trained tokenizer (tools/train_tokenizer.py) to the model.

The model and its tokenizer must agree on two things:
- the vocabulary size (the embedding has one row per token id), and
- the fixed special ids [PAD]=pad_id, [MASK]=mask_id, [EOS]=eos_id.

Every saved checkpoint gets a copy of the tokenizer files, so a checkpoint
folder is self-contained.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import NuriConfig

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
SPECIAL_TOKENS = {"pad_id": "[PAD]", "mask_id": "[MASK]", "eos_id": "[EOS]"}


def tokenizer_folder(path: str | Path) -> Path:
    """Accepts a tokenizer.json file or the folder that holds it."""
    path = Path(path)
    return path if path.is_dir() else path.parent


def load_tokenizer(path: str | Path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(tokenizer_folder(path) / "tokenizer.json"))


def vocab_size(tokenizer) -> int:
    return tokenizer.get_vocab_size(with_added_tokens=True)


def check_tokenizer(tokenizer, config: NuriConfig) -> None:
    """Raise a clear error if the tokenizer does not fit the model config."""
    if vocab_size(tokenizer) != config.vocab_size:
        raise ValueError(f"tokenizer has {vocab_size(tokenizer)} tokens but the model expects {config.vocab_size}")
    for field, token in SPECIAL_TOKENS.items():
        token_id = tokenizer.token_to_id(token)
        if token_id != getattr(config, field):
            raise ValueError(f"{token} has id {token_id} in the tokenizer but {field}={getattr(config, field)}")


def copy_tokenizer_files(source: str | Path, destination: str | Path) -> None:
    source, destination = tokenizer_folder(source), Path(destination)
    if not (source / "tokenizer.json").exists():
        raise FileNotFoundError(f"no tokenizer.json in {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in TOKENIZER_FILES:
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
