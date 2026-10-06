"""Train one release stage of Nuri.

Examples
  # base stage from scratch
  python scripts/train.py --stage base --preset release-dense --corpus $NURI_TEXT_CORPUS \
      --out runs/nuri-base --steps 20000 --batch-size 64

  # instruction tuning from a base checkpoint
  python scripts/train.py --stage it --init runs/nuri-base/final --tokenizer $NURI_TOKENIZER \
      --instructions $NURI_IT_DATA --corpus $NURI_TEXT_CORPUS --out runs/nuri-base-it --steps 3000

  # further base training from checkpoint weights, with a fresh optimizer/schedule
  python scripts/train.py --stage base --init runs/repaired-base --corpus $NURI_TEXT_CORPUS \
      --out runs/continued-base --steps 20000 --batch-size 64

  # continue an interrupted run (same flags plus --resume)
  python scripts/train.py --stage base ... --resume runs/nuri-base/training_state/step-2000.pt

  # embedding training from the dense base checkpoint
  python scripts/train.py --stage embed --init runs/nuri-base/final --tokenizer $NURI_TOKENIZER \
      --pairs $NURI_PAIR_DATA --out runs/nuri-base-embed --steps 3000

Data paths can be given as flags or through the NURI_* environment variables
shown above. File formats:
  corpus        a folder written by nuri.data.write_corpus (tokens.bin + offsets.npy)
  instructions  JSONL: {"messages": [{"role": "user", "content": "..."}], "response": "..."}
  pairs         JSONL: {"query": "...", "positive": "..."}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nuri.checkpoint import load_pretrained, training_state_config  # noqa: E402
from nuri.config import NuriConfig, PRESETS  # noqa: E402
from nuri.data import chat_prompt_text, open_corpus, usable_instruction  # noqa: E402
from nuri.model import NuriModel  # noqa: E402
from nuri.pipeline import base_stage, embedding_stage, instruction_stage, read_jsonl  # noqa: E402
from nuri.recipe import load_recipe_config, recipe_loss_settings, validate_recipe_resume  # noqa: E402
from nuri.tasks import reasoning_examples  # noqa: E402
from nuri.tokenizer import check_tokenizer, load_tokenizer, vocab_size  # noqa: E402
from nuri.trainer import (INTERRUPTED_EXIT_CODE, NONFINITE_EXIT_CODE, NonFiniteTraining,  # noqa: E402
                          HoldoutSet, TrainingInterrupted, TrainSettings, train)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["base", "it", "embed"], required=True)
    p.add_argument("--preset", choices=sorted(PRESETS), default="release-dense", help="model size for scratch Base initialization")
    p.add_argument("--init", help="checkpoint weights for a fresh run (required for it/embed unless --resume); "
                                "--resume takes priority and restores the full run")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="peak LR (default 3e-4 base, 5e-5 it/embed)")
    p.add_argument("--warmup", type=int, default=None, help="warmup steps (default 2%% of steps)")
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--log-every", type=positive_int, default=10,
                   help="record metrics every N optimizer steps (first and last always recorded)")
    p.add_argument("--no-progress", action="store_true", help="disable tqdm bars; keep setup messages and logs")
    p.add_argument("--tensorboard", action="store_true", help="write TensorBoard events to <out>/tensorboard")
    p.add_argument("--tensorboard-dir", help="custom event directory; also enables TensorBoard")
    p.add_argument("--holdout-config", default=os.environ.get("NURI_HOLDOUT_CONFIG"),
                   help="private JSON with explicit valid/test inputs; no training-input fallback")
    p.add_argument("--recipe-config", default=os.environ.get("NURI_RECIPE_CONFIG"),
                   help="optional private versioned JSON for existing task, loss and corruption controls")
    p.add_argument("--assessment-recipe-config", default=os.environ.get("NURI_ASSESSMENT_RECIPE_CONFIG"),
                   help="optional private corruption-only recipe for explicit holdouts; requires --holdout-config")
    p.add_argument("--valid-every", type=nonnegative_int, default=None,
                   help="validation interval in completed updates (0 = completion only); requires --holdout-config")
    p.add_argument("--resume", help="private training_state/*.pt file of an interrupted run with the same flags")
    p.add_argument("--stop-after-completed-steps", type=positive_int,
                   help="save full state and exit 75 at this update, before assessment; below --steps; "
                        "omit on resume to continue the original full schedule")
    p.add_argument("--save-every-minutes", type=float, default=0.0,
                   help="also save the full training state to training_state/latest.pt this often (0 = off); "
                        "SIGTERM always saves it and exits with code 75 so the run can be resumed")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="micro-batches per optimizer step; effective batch = batch size x grad accum")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="recompute block activations in backward: less memory, about one extra forward")
    p.add_argument("--compile", action="store_true", help="torch.compile the model blocks (GPU runs)")
    p.add_argument("--native-gqa", action="store_true",
                   help="let supported SDPA backends use compact GQA heads; benchmark before enabling")
    p.add_argument("--cross-mode-weight", type=float, default=None,
                   help="embed stage: also align reasoning-mode queries with fast-mode documents (0 = off)")
    p.add_argument("--pack-mlm", action="store_true",
                   help="base stage: fill each MLM row with several documents (attention stays inside each one)")
    p.add_argument("--bucket", action="store_true",
                   help="base/it stages: group span and instruction samples of similar length (less padding)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--corpus", nargs="+", default=env_list("NURI_TEXT_CORPUS"),
                   help='corpus folder(s); mix several as "folder=weight" (e.g. ko=35 en=45)')
    p.add_argument("--tokenizer", default=os.environ.get("NURI_TOKENIZER"),
                   help="tokenizer.json or its folder (it/embed default to the copy inside --init)")
    p.add_argument("--instructions", default=os.environ.get("NURI_IT_DATA"))
    p.add_argument("--pairs", default=os.environ.get("NURI_PAIR_DATA"))
    p.add_argument("--reasoning-examples", type=int, default=20000,
                   help="synthetic reasoning examples for the it stage (0 = none)")
    args = p.parse_args(argv)
    args.cross_mode_weight_explicit = args.cross_mode_weight is not None
    if args.cross_mode_weight is None:
        args.cross_mode_weight = 0.0
    if args.stop_after_completed_steps is not None and args.stop_after_completed_steps >= args.steps:
        p.error("--stop-after-completed-steps must be below --steps")
    if args.valid_every is not None and not args.holdout_config:
        p.error("--valid-every requires --holdout-config")
    if args.assessment_recipe_config and not args.holdout_config:
        p.error("--assessment-recipe-config requires --holdout-config")
    return args


def env_list(name: str) -> list[str] | None:
    """An environment variable holding one or more entries separated by os.pathsep."""
    value = os.environ.get(name)
    return [part for part in value.split(os.pathsep) if part] if value else None


def require(value, flag):
    if not value:
        sys.exit(f"missing {flag} (flag or environment variable)")
    return value


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resource_path(spec: str, directory: Path, corpus: bool = False) -> tuple[Path, str]:
    path, separator, weight = spec.rpartition("=") if corpus and "=" in spec else (spec, "", "")
    resolved = (directory / path).resolve()
    return resolved, str(resolved) + (separator + weight if separator else "")


LANGUAGE_SEED_OFFSETS = {"ko": 1009, "en": 2017, "ja": 3011}


def _resolve_holdout_inputs(block: dict, args, directory: Path, resources: list) -> None:
    """Apply the same stage and split checks to aggregate and language inputs."""
    required = "corpus" if args.stage == "base" else "instructions" if args.stage == "it" else "pairs"
    if not block.get(required) or (args.stage == "it" and args.corpus and not block.get("corpus")):
        raise ValueError("each holdout split needs its own stage inputs, including span corpus when trained")
    if "corpus" in block and (not isinstance(block["corpus"], list) or not block["corpus"]):
        raise ValueError("holdout corpus must be a nonempty list of folder specifications")
    for name in ("corpus", "instructions", "reasoning", "pairs"):
        values = block.get(name, []) if name == "corpus" else ([block[name]] if block.get(name) else [])
        resolved_values = []
        for spec in values:
            if not isinstance(spec, str) or not spec:
                raise ValueError("holdout paths must be nonempty strings")
            resolved, value = _resource_path(spec, directory, name == "corpus")
            if not resolved.exists():
                raise ValueError("a configured holdout input does not exist")
            resources.append((name, resolved))
            resolved_values.append(value)
        if values:
            block[name] = resolved_values if name == "corpus" else resolved_values[0]


def load_holdout_config(args) -> dict | None:
    """Validate explicit split paths before allocating the model.

    Optional manifest hashes are preparation bindings, not a replacement for
    the full runtime byte checks of a private deployment preflight.
    """
    if not args.holdout_config:
        return None
    path = Path(args.holdout_config).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"schema_version", "valid_every", "batch_size", "valid", "test",
               "input_manifest", "input_manifest_sha256", "exposure_caveats"}
    if not isinstance(data, dict) or data.get("schema_version") != 1 or set(data) - allowed:
        raise ValueError("holdout config needs schema_version 1 and supported fields")
    data["valid_every"] = args.valid_every if args.valid_every is not None else data.get("valid_every", 0)
    data["batch_size"] = data.get("batch_size", args.batch_size)
    if (type(data["valid_every"]) is not int or data["valid_every"] < 0
            or type(data["batch_size"]) is not int or data["batch_size"] < 1):
        raise ValueError("holdout interval must be nonnegative and batch size positive")
    resources = {"train": [], "valid": [], "test": []}
    for name, values in (("corpus", args.corpus or []), ("instructions", [args.instructions] if args.instructions else []),
                         ("pairs", [args.pairs] if args.pairs else [])):
        for spec in values:
            resolved, _ = _resource_path(spec, Path.cwd(), name == "corpus")
            resources["train"].append((name, resolved))
    for split in ("valid", "test"):
        block = data.get(split)
        keys = {"corpus", "instructions", "reasoning", "pairs", "batches_per_task", "seed", "depth",
                "exposure_caveats", "language_groups"}
        if not isinstance(block, dict) or set(block) - keys:
            raise ValueError("holdout config needs explicit valid and test blocks with supported fields")
        for key, default in (("batches_per_task", 8), ("seed", 0), ("depth", 4)):
            block[key] = block.get(key, default)
            if type(block[key]) is not int or block[key] < (1 if key == "batches_per_task" else 0):
                raise ValueError(f"invalid holdout {key}")
        _resolve_holdout_inputs(block, args, path.parent, resources[split])
        groups = block.get("language_groups", {})
        if (not isinstance(groups, dict) or set(groups) - LANGUAGE_SEED_OFFSETS.keys()
                or ("language_groups" in block and not groups)):
            raise ValueError("language_groups must be a nonempty mapping using ko, en or ja")
        for group in groups.values():
            if not isinstance(group, dict) or set(group) - {"corpus", "instructions", "reasoning", "pairs"}:
                raise ValueError("language groups support only explicit stage input fields")
            _resolve_holdout_inputs(group, args, path.parent, resources[split])
    if set(data["valid"].get("language_groups", {})) != set(data["test"].get("language_groups", {})):
        raise ValueError("valid and test must declare the same language groups")
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if {path for _, path in resources[left]} & {path for _, path in resources[right]}:
            raise ValueError("train, valid and test must use distinct resolved input paths")
    if data.get("input_manifest"):
        manifest_path = (path.parent / data["input_manifest"]).resolve()
        if not data.get("input_manifest_sha256") or _file_sha256(manifest_path) != data["input_manifest_sha256"]:
            raise ValueError("holdout input manifest hash mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        hashes = {}
        for entry in manifest["files"]:
            resource = (manifest_path.parent / entry["path"]).resolve()
            digest = entry["sha256"]
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest) or resource in hashes):
                raise ValueError("input manifest needs unique paths and lowercase SHA256 values")
            hashes[resource] = digest
        signatures = {}
        for split, entries in resources.items():
            signatures[split] = set()
            for name, resource in entries:
                files = (resource / "tokens.bin", resource / "offsets.npy") if name == "corpus" else (resource,)
                if any(file not in hashes for file in files):
                    raise ValueError("holdout input manifest does not cover all selected inputs")
                # Equal-length corpora can have identical offsets without sharing tokens.
                kind = "corpus" if name == "corpus" else "jsonl"
                signatures[split].add((kind, tuple(hashes[file] for file in files)))
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            if signatures[left] & signatures[right]:
                raise ValueError("manifest declares identical content across train, valid or test")
    elif data.get("input_manifest_sha256"):
        raise ValueError("input_manifest_sha256 requires input_manifest")
    data["config_sha256"] = _file_sha256(path)
    return data


def _holdout_examples(path, tokenizer, config, limit, seed, pairs=False):
    """A fixed uniform reservoir of eligible rows, with bounded retained memory."""
    rng, selected, eligible, seen = random.Random(seed), [], 0, 0
    for row in read_jsonl(path):
        seen += 1
        example = (encode(tokenizer, row["query"]), encode(tokenizer, row["positive"])) if pairs else (
            encode(tokenizer, chat_prompt_text(row["messages"])), encode(tokenizer, row["response"]))
        if not (all(example) if pairs else usable_instruction(example, config)):
            continue
        eligible += 1
        if len(selected) < limit:
            selected.append(example)
        else:
            index = rng.randrange(eligible)
            if index < limit:
                selected[index] = example
    return selected, {"source_rows": seen, "eligible_rows": eligible, "selected_pool_rows": len(selected),
                      "excluded_rows": seen - eligible, "selection": "fixed_uniform_reservoir"}


def _build_holdout_tasks(block, args, config, tokenizer, batch_size, count, seed, recipe=None):
    """Build independent task objects; never share sampling state between groups."""
    corpus = open_corpus(block["corpus"]) if block.get("corpus") else None
    selections = {}
    # Assessment visits every available task, independent of training mix draws.
    recipe_options = {"recipe": recipe, "apply_recipe_mix": False} if recipe is not None else {}
    if args.stage == "base":
        tasks, _ = base_stage(corpus, config, batch_size, seed, args.pack_mlm, args.bucket, **recipe_options)
    elif args.stage == "it":
        require(tokenizer, "--tokenizer for holdout instructions")
        limit = batch_size * count * (8 if args.bucket else 1)
        examples, selections["instruction"] = _holdout_examples(block["instructions"], tokenizer, config, limit, seed)
        reasoning = None
        if block.get("reasoning"):
            reasoning, selections["reasoning"] = _holdout_examples(block["reasoning"], tokenizer, config, limit, seed + 1)
            if not reasoning:
                raise ValueError("explicit holdout reasoning input has no eligible rows")
        tasks, _ = instruction_stage(examples, config, batch_size, corpus, reasoning, seed, bucket=args.bucket,
                                     **recipe_options)
    else:
        require(tokenizer, "--tokenizer for holdout pairs")
        pairs, selections["pairs"] = _holdout_examples(block["pairs"], tokenizer, config, batch_size * count, seed, pairs=True)
        tasks, _ = embedding_stage(pairs, config, batch_size, seed, **recipe_options)
    for name, selection in selections.items():
        if name in tasks:
            tasks[name].options.update(selection)
    return tasks


def build_holdouts(data, args, config, tokenizer, recipe=None,
                   assessment_recipe=None) -> tuple[HoldoutSet | None, HoldoutSet | None]:
    if assessment_recipe is not None and (assessment_recipe.task_mix is not None or assessment_recipe.loss):
        raise ValueError("assessment recipe supports corruption only; omit task_mix and nonempty loss")
    if data is None:
        if assessment_recipe is not None:
            raise ValueError("an assessment recipe requires explicit holdout inputs")
        return None, None
    assessed_recipe = assessment_recipe if assessment_recipe is not None else recipe
    sets = []
    for split in ("valid", "test"):
        block, batch_size = data[split], data["batch_size"]
        seed, count = block["seed"], block["batches_per_task"]
        tasks = _build_holdout_tasks(block, args, config, tokenizer, batch_size, count, seed, assessed_recipe)
        for language, group in sorted(block.get("language_groups", {}).items()):
            group_seed = seed + LANGUAGE_SEED_OFFSETS[language]
            group_tasks = _build_holdout_tasks(group, args, config, tokenizer, batch_size, count, group_seed, assessed_recipe)
            tasks.update({f"{language}/{name}": task for name, task in group_tasks.items()})
        metadata = {"config_sha256": data["config_sha256"],
                    "input_manifest_sha256": data.get("input_manifest_sha256"),
                    "exposure_caveats": block.get("exposure_caveats", data.get("exposure_caveats", []))}
        if assessed_recipe is not None:
            metadata["recipe_sha256"] = assessed_recipe.sha256
        if assessment_recipe is not None:
            metadata["assessment_recipe_sha256"] = assessment_recipe.sha256
        sets.append(HoldoutSet(tasks, count, seed, block["depth"], metadata))
    return tuple(sets)


def _recipe_model_config(args, state=None):
    """Read model geometry without allocating its weights or optimizer."""
    if args.resume:
        if state is None:
            state = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
        return NuriConfig.from_dict(state["config"])
    if args.init:
        data = json.loads((Path(args.init) / "config.json").read_text(encoding="utf-8"))
        return NuriConfig.from_dict(data["nuri"] if "nuri" in data else data)
    if args.stage == "base":
        return PRESETS[args.preset]()
    raise ValueError("recipe IT/embed stage requires --init or --resume")


def prepare_recipe(args):
    """Validate an optional recipe and its model geometry before constructing tensors."""
    recipe = load_recipe_config(args.recipe_config, args.stage)
    losses = recipe_loss_settings(recipe, args.cross_mode_weight, args.cross_mode_weight_explicit)
    if recipe is None:
        return None, losses, None
    required = "corpus" if args.stage == "base" else "instructions" if args.stage == "it" else "pairs"
    if not getattr(args, required):
        raise ValueError(f"recipe stage requires --{required}")
    available = {"mlm", "span"} if args.stage == "base" else {"pairs"} if args.stage == "embed" else {"instruction"}
    if args.stage == "it":
        if args.corpus:
            available.add("span")
        if args.reasoning_examples > 0:
            available.add("reasoning")
        if "min_span" in recipe.corruption and not args.corpus:
            raise ValueError("recipe min_span requires the IT span corpus")
    recipe.validate_tasks(available)
    state = None
    if args.resume:
        # mmap keeps optimizer/model storage off the heap during this metadata check.
        state = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
        validate_recipe_resume(recipe, state.get("extra", {}))
    config = _recipe_model_config(args, state)
    recipe.validate_model(config)
    return recipe, losses, config


def prepare_assessment_recipe(args, config=None, holdout_config=None):
    """Validate an opt-in fixed corruption policy and its resume binding before allocation."""
    recipe = load_recipe_config(args.assessment_recipe_config, args.stage)
    if recipe is None:
        return None
    if not args.holdout_config:
        raise ValueError("an assessment recipe requires --holdout-config")
    if recipe.task_mix is not None or recipe.loss:
        raise ValueError("assessment recipe supports corruption only; omit task_mix and nonempty loss")
    if args.stage == "it" and "min_span" in recipe.corruption:
        data = holdout_config if holdout_config is not None else load_holdout_config(args)
        for split in ("valid", "test"):
            block = data[split]
            if any(not group.get("corpus") for group in (block, *block.get("language_groups", {}).values())):
                raise ValueError("assessment min_span requires a corpus in every IT holdout group")
    state = None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
        extra = state.get("extra", {})
        options = extra.get("task_options") if isinstance(extra, dict) else None
        if not isinstance(options, dict) or not options or any(
            not isinstance(task, dict) or task.get("assessment_recipe") != recipe.to_dict()
            or task.get("assessment_recipe_sha256") != recipe.sha256 for task in options.values()
        ):
            raise ValueError("resume requires the same saved assessment recipe; use --init and a fresh output directory")
    recipe.validate_model(config if config is not None else _recipe_model_config(args, state))
    return recipe


def bind_assessment_recipe(tasks, recipe):
    """Bind explicit assessment controls without changing training sampling or losses."""
    if recipe is not None:
        for task in tasks.values():
            task.options = {**task.options, "assessment_recipe": recipe.to_dict(),
                            "assessment_recipe_sha256": recipe.sha256}


def main(argv=None):
    args = parse_args(argv)
    recipe, loss_settings, recipe_config = prepare_recipe(args)
    holdout_config = load_holdout_config(args)
    assessment_recipe = prepare_assessment_recipe(args, recipe_config, holdout_config)
    # Seed before building the model so --seed fully fixes the initial weights.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer_path = args.tokenizer
    if not tokenizer_path and args.init and (Path(args.init) / "tokenizer.json").exists():
        tokenizer_path = args.init  # every checkpoint carries a copy of its tokenizer
    print(f"[{args.stage}] Loading tokenizer and checking vocabulary options...", flush=True)
    tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path else None
    if args.stage == "base" and tokenizer is None:
        print("warning: no --tokenizer; retaining the model vocabulary size and saving no tokenizer files")

    if args.resume:
        print(f"[{args.stage}] Building model from the resume configuration...", flush=True)
        # The interrupted run decides the model shape; its weights are loaded by train().
        model = NuriModel(recipe_config if recipe_config is not None else training_state_config(args.resume))
    elif args.init:
        print(f"[{args.stage}] Loading initial checkpoint weights for a fresh training run...", flush=True)
        model = load_pretrained(args.init)
    elif args.stage == "base":
        print(f"[{args.stage}] Initializing model weights...", flush=True)
        # The embedding needs exactly one row per token id of the real tokenizer.
        overrides = {"vocab_size": vocab_size(tokenizer)} if tokenizer else {}
        model = NuriModel(PRESETS[args.preset](**overrides))
    else:
        print(f"[{args.stage}] Loading initial checkpoint weights...", flush=True)
        model = load_pretrained(require(args.init, "--init"))
    config = model.config
    if args.stage == "base":
        # Base continuation does not calibrate natural answer lengths.
        config.length_head_trained = False
    if tokenizer is not None:
        check_tokenizer(tokenizer, config)
    if args.corpus:
        print(f"[{args.stage}] Opening the token corpus...", flush=True)
    corpus = open_corpus(args.corpus) if args.corpus else None

    print(f"[{args.stage}] Preparing training tasks...", flush=True)
    recipe_options = {"recipe": recipe} if recipe is not None else {}
    if args.stage == "base":
        tasks, mix = base_stage(require(corpus, "--corpus"), config, args.batch_size, args.seed, args.pack_mlm,
                                args.bucket, **recipe_options)
    elif args.stage == "it":
        require(tokenizer, "--tokenizer")

        def tokenize(messages, response):
            return encode(tokenizer, chat_prompt_text(messages)), encode(tokenizer, response)

        with tqdm(read_jsonl(require(args.instructions, "--instructions")),
                  desc="it | tokenizing instructions", unit="example", ascii=True,
                  dynamic_ncols=True, mininterval=0.5, disable=args.no_progress) as progress:
            examples = [tokenize(r["messages"], r["response"]) for r in progress]
        print("[it] Generating structured reasoning examples...", flush=True)
        raw_reasoning = reasoning_examples(args.reasoning_examples, seed=args.seed)
        with tqdm(raw_reasoning, desc="it | tokenizing reasoning examples", unit="example", ascii=True,
                  dynamic_ncols=True, mininterval=0.5, disable=args.no_progress) as progress:
            reasoning = [tokenize(m, a) for m, a in progress]
        tasks, mix = instruction_stage(examples, config, args.batch_size, corpus, reasoning, args.seed,
                                       bucket=args.bucket, **recipe_options)
        if recipe is None:
            config.length_head_trained = True  # legacy stage behavior
        else:
            trains_length = loss_settings.get("length_weight", 0.1) > 0 and any(
                tasks[name].kind == "instruction" for name in mix)
            config.length_head_trained = config.length_head_trained or trains_length
    else:
        require(tokenizer, "--tokenizer")
        with tqdm(read_jsonl(require(args.pairs, "--pairs")),
                  desc="embed | tokenizing pairs", unit="pair", ascii=True,
                  dynamic_ncols=True, mininterval=0.5, disable=args.no_progress) as progress:
            pairs = [(encode(tokenizer, r["query"]), encode(tokenizer, r["positive"])) for r in progress]
        tasks, mix = embedding_stage(pairs, config, args.batch_size, args.seed, **recipe_options)

    bind_assessment_recipe(tasks, assessment_recipe)
    validation, test = build_holdouts(holdout_config, args, config, tokenizer, recipe, assessment_recipe)
    peak_lr = args.lr or (3e-4 if args.stage == "base" else 5e-5)
    warmup = args.warmup if args.warmup is not None else max(1, args.steps // 50)
    tensorboard_dir = args.tensorboard_dir
    if args.tensorboard and not tensorboard_dir:
        tensorboard_dir = str(Path(args.out) / "tensorboard")
    settings = TrainSettings(out_dir=args.out, total_steps=args.steps, peak_lr=peak_lr, warmup_steps=warmup,
                             save_every=args.save_every, log_every=args.log_every, seed=args.seed, mix=mix,
                             stage=args.stage, progress=not args.no_progress, tensorboard_dir=tensorboard_dir,
                             tokenizer_dir=str(tokenizer_path) if tokenizer_path else None,
                             grad_accum=args.grad_accum, gradient_checkpointing=args.gradient_checkpointing,
                             compile=args.compile, native_gqa=args.native_gqa, **loss_settings,
                             save_every_minutes=args.save_every_minutes,
                             valid_every=holdout_config["valid_every"] if holdout_config else 0,
                             stop_after_completed_steps=args.stop_after_completed_steps)
    print(f"stage={args.stage} mix={mix} params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    if recipe is not None:
        print(f"recipe schema=1 sha256={recipe.sha256}", flush=True)
    if assessment_recipe is not None:
        print(f"assessment recipe schema=1 sha256={assessment_recipe.sha256}", flush=True)
    if tensorboard_dir:
        print(f"TensorBoard events: {tensorboard_dir}", flush=True)
    print(f"[{args.stage}] {'Restoring state and continuing' if args.resume else 'Starting'} training...", flush=True)
    try:
        records = train(model, tasks, settings, resume_from=args.resume, validation=validation, test=test)
    except TrainingInterrupted as stop:
        label = "at completed-step boundary" if stop.reason == "step_boundary" else "on request"
        print(f"[{args.stage}] stopped {label} after step {stop.step}; "
              f"resume with --resume {stop.path}", flush=True)
        sys.exit(INTERRUPTED_EXIT_CODE)
    except NonFiniteTraining as error:
        print(f"[{args.stage}] stopped: {error}", flush=True)
        sys.exit(NONFINITE_EXIT_CODE)
    if records:
        print(f"done: last loss {records[-1]['loss']:.4f}, checkpoints in {args.out}")
    else:
        print("nothing to do: the resumed run had already finished")


if __name__ == "__main__":
    main()
