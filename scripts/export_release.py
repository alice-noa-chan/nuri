"""Turn a training checkpoint into a Hugging Face model folder ready for upload (ARCHITECTURE.md §15).

Usage
  python scripts/export_release.py --checkpoint runs/nuri-base-it/final --out release/nuri-base-it \
      --name nuri-base-it --stage it --tokens 12e9 --gpu "1x H100 80GB" --gpu-hours 9.5

The folder gets:
  model.safetensors   bf16 weights, names prefixed with "model."
  config.json         Hugging Face config with auto_map; Nuri fields under "nuri"
  *.py                the model code, loaded with trust_remote_code=True
  tokenizer files     copied from the checkpoint (or --tokenizer)
  LICENSE, README.md  MIT license and a model card (training data sources are never named)

After writing, the folder is loaded back (with Nuri's own loader and with
transformers) and its outputs are compared with the original model. If that
check fails, the script exits with an error and the folder must not be uploaded.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nuri.checkpoint import load_pretrained  # noqa: E402
from nuri.model import NuriModel  # noqa: E402
from nuri.tokenizer import copy_tokenizer_files  # noqa: E402

# The model code a release needs; they import each other with relative imports.
CODE_FILES = (
    "config.py",
    "attention_backend.py",
    "liger_backend.py",
    "layers.py",
    "moe.py",
    "encoder.py",
    "core.py",
    "readout.py",
    "model.py",
    "generation.py",
    "configuration_nuri.py",
    "modeling_nuri.py",
)
ARCHITECTURE_BY_STAGE = {
    "base": "NuriForMaskedLM",
    "it": "NuriForMaskedLM",
    "embed": "NuriHFModel",
}


def parameter_counts(model: NuriModel) -> tuple[int, int]:
    """(total, active per token). MoE experts count as top_k + shared per token."""
    total = sum(p.numel() for p in model.parameters())
    moe = model.config.moe
    if moe is None:
        return total, total
    inactive = 0
    for layer in model.moe_layers():
        per_expert = (
            layer.w_gate[0].numel() + layer.w_up[0].numel() + layer.w_down[0].numel()
        )
        inactive += per_expert * (moe.num_experts - moe.top_k)
    return total, total - inactive


def write_folder(
    model: NuriModel, out: Path, stage: str, tokenizer_dir: Path | None
) -> None:
    from nuri.configuration_nuri import NuriHFConfig
    from nuri.modeling_nuri import AUTO_MAP

    out.mkdir(parents=True, exist_ok=True)
    weights = {
        f"model.{name}": t.detach().to("cpu", torch.bfloat16).contiguous()
        for name, t in model.state_dict().items()
    }
    save_file(weights, str(out / "model.safetensors"), metadata={"format": "pt"})

    config = NuriHFConfig.from_nuri_config(
        model.config, architectures=[ARCHITECTURE_BY_STAGE[stage]]
    )
    config.auto_map = AUTO_MAP
    config.save_pretrained(out)

    for name in CODE_FILES:
        shutil.copy2(ROOT / "nuri" / name, out / name)
    shutil.copy2(ROOT / "LICENSE", out / "LICENSE")
    if tokenizer_dir is not None:
        copy_tokenizer_files(tokenizer_dir, out)


def model_card(model: NuriModel, args) -> str:
    total, active = parameter_counts(model)
    c = model.config
    kind = "MoE" if c.moe else "dense"
    params = f"{total / 1e6:.1f}M" + (
        f" total, {active / 1e6:.1f}M active per token" if c.moe else ""
    )
    hours = "not given" if args.gpu_hours is None else f"{args.gpu_hours:g} GPU-hours"
    budget = (
        f"trained on {args.tokens} tokens"
        if args.tokens
        else "trained on an unreported number of tokens"
    )
    usage = {
        "base": "generate_refine(ids, canvas_len=32)  # base models need an explicit answer length",
        "it": "generate_refine(ids, T=4, S=4)  # the length head picks the answer length",
        "embed": 'encode(ids, mode="reasoning", dim=128)  # unit-length sentence embeddings',
    }[args.stage]
    # Instruction-tuned models were trained on chat-formatted prompts ending with the assistant tag.
    if args.stage == "it":
        prompt = (
            'text = tokenizer.apply_chat_template([{"role": "user", "content": "..."}],\n'
            "                                     add_generation_prompt=True, tokenize=False)\n"
            'ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids'
        )
    else:
        prompt = 'ids = tokenizer("...", add_special_tokens=False, return_tensors="pt").input_ids'
    decode = (
        "" if args.stage == "embed" else "\nprint(tokenizer.decode(result.tokens[0]))"
    )
    return f"""---
license: mit
language: [ko, en, ja]
library_name: transformers
tags: [nuri, encoder, masked-generation, latent-reasoning]
---

# {args.name}

Nuri is an encoder-only language model that reasons by updating hidden
latent slots several times with one shared block, and writes text by filling
a masked answer in parallel and refining it, without an autoregressive decoder.

| | |
|---|---|
| stage | {args.stage} |
| type | {kind} |
| parameters | {params} |
| hidden size / heads | {c.hidden_size} / {c.num_heads} |
| input layers / readout layers | {c.input_layers} / {c.readout_layers} |
| latent slots K | {c.latent_slots} |
| max input / answer tokens | {c.max_input_len} / {c.max_response_len} |
| vocabulary | {c.vocab_size} |
| languages | Korean, English (primary), Japanese (some) |
| training tokens | {args.tokens or "not given"} |
| hardware | {args.gpu or "not given"}, {hours} |

## Use

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("{args.name}", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("{args.name}")
{prompt}
result = model.{usage}{decode}
```

`T` is the number of reasoning steps and `S` the number of refinement rounds;
both default to {c.default_depth} and {c.refine_rounds}. The standard `generate()`
does not apply because there is no decoder.

## Limitations

This is a small research model, {budget} ({hours}). Expect factual errors
and weak long-form writing. It has not been evaluated for safety.

## License

MIT
"""


def verify(model: NuriModel, out: Path, stage: str) -> dict:
    """Load the folder back two ways and compare with the original model."""
    model = model.float().eval()
    ids = torch.randint(
        3, model.config.vocab_size, (2, 12), generator=torch.Generator().manual_seed(0)
    )
    report = {}

    reloaded = load_pretrained(out).eval()
    with torch.no_grad():
        x_ref, _ = model.encode(ids)
        x_new, _ = reloaded.encode(ids)
    report["nuri_loader_max_diff"] = float((x_ref - x_new).abs().max())

    from transformers import AutoModelForMaskedLM

    # Loaded in its stored dtype and cast here, which works across transformers versions.
    hf = (
        AutoModelForMaskedLM.from_pretrained(str(out), trust_remote_code=True)
        .float()
        .eval()
    )
    with torch.no_grad():
        logits_ref = model.vocab_logits(x_ref)
        logits_hf = hf(input_ids=ids).logits
    report["hf_logits_max_diff"] = float((logits_ref - logits_hf).abs().max())
    report["hf_argmax_agreement"] = float(
        (logits_ref.argmax(-1) == logits_hf.argmax(-1)).float().mean()
    )

    canvas_len = None if stage == "it" else 8
    with torch.no_grad():
        result = hf.generate_refine(ids[:1], canvas_len=canvas_len)
        embedding = hf.encode(ids, mode="reasoning")
    report["generated_tokens"] = len(result.tokens[0])
    report["embedding_norm"] = float(embedding.norm(dim=-1).mean())

    # bf16 weights shift values slightly; the predicted tokens must still agree.
    report["ok"] = bool(
        report["nuri_loader_max_diff"] < 0.5
        and report["hf_argmax_agreement"] >= 0.9
        and abs(report["embedding_norm"] - 1) < 1e-3
        and int(result.lengths[0]) >= 1
    )
    return report


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="a folder written by save_pretrained (e.g. runs/x/final)",
    )
    p.add_argument("--out", required=True)
    p.add_argument(
        "--name", required=True, help="repository name used in the model card"
    )
    p.add_argument("--stage", choices=sorted(ARCHITECTURE_BY_STAGE), required=True)
    p.add_argument(
        "--tokenizer", help="defaults to the tokenizer files inside --checkpoint"
    )
    p.add_argument(
        "--tokens", help="training tokens, e.g. 12e9 (shown in the model card)"
    )
    p.add_argument("--gpu", help='e.g. "1x H100 80GB"')
    p.add_argument("--gpu-hours", type=float)
    args = p.parse_args(argv)

    checkpoint, out = Path(args.checkpoint), Path(args.out)
    tokenizer_dir = Path(args.tokenizer) if args.tokenizer else checkpoint
    if not (tokenizer_dir / "tokenizer.json").exists() and not tokenizer_dir.is_file():
        print(
            "warning: no tokenizer files found; the release will not include a tokenizer"
        )
        tokenizer_dir = None

    model = load_pretrained(checkpoint)
    if args.stage == "it" and not model.config.length_head_trained:
        sys.exit(
            "the checkpoint is not instruction-tuned (length_head_trained is False)"
        )
    write_folder(model, out, args.stage, tokenizer_dir)
    (out / "README.md").write_text(model_card(model, args), encoding="utf-8")

    report = verify(model, out, args.stage)
    (out.parent / f"{out.name}.verify.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        sys.exit("verification failed: do not upload this folder")


if __name__ == "__main__":
    main()
