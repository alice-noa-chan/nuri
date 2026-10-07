"""Measure Nuri's actual latency, throughput, and CUDA memory on synthetic inputs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

# Make this file runnable from a checkout without an editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nuri.attention_backend import ATTENTION_BACKENDS, validate_attention_backend
from nuri.config import PRESETS, NuriConfig
from nuri.data import build_pair_batch
from nuri.generation import generate
from nuri.layers import Attention
from nuri.liger_backend import validate_liger
from nuri.losses import (
    VOCAB_LOSS_BACKENDS,
    validate_vocab_loss,
    vocab_sample_mean_ce,
    vocab_token_mean_ce,
)
from nuri.model import NuriModel
from nuri.trainer import TrainSettings, make_optimizer, pair_loss


def run_benchmark(
    config: NuriConfig,
    *,
    task: str = "denoise",
    training: bool = False,
    device: str = "cpu",
    dtype: str = "float32",
    batch_size: int = 1,
    input_length: int = 16,
    output_length: int = 8,
    steps: int | None = None,
    rounds: int = 4,
    warmup: int = 2,
    iterations: int = 5,
    seed: int = 0,
    native_gqa: bool = False,
    attention_backend: str = "sdpa",
    liger_kernels: bool = False,
    vocab_loss_backend: str = "torch",
    vocab_chunk_size: int = 256,
    gradient_checkpointing: bool = False,
    pack_mlm: bool = False,
    packed_document_length: int = 256,
) -> dict:
    if task not in {
        "mlm",
        "denoise",
        "pairs",
        "embed-fast",
        "embed-reasoning",
        "generate",
    }:
        raise ValueError("Unknown benchmark task")
    if training and task not in {"mlm", "denoise", "pairs"}:
        raise ValueError("Training benchmarks support mlm, denoise, and pairs tasks")
    if task == "pairs" and not training:
        raise ValueError("Pairs benchmarks require --training")
    if task == "pairs" and batch_size < 2:
        raise ValueError("Pairs benchmarks require batch_size >= 2")
    if gradient_checkpointing and not training:
        raise ValueError("Gradient checkpointing requires a training benchmark")
    if pack_mlm and (task != "mlm" or not training):
        raise ValueError("Packed MLM requires --training --task mlm")
    if pack_mlm and not 1 <= packed_document_length < input_length:
        raise ValueError(
            "Packed document length must be positive and shorter than the input row"
        )
    if (
        min(batch_size, input_length, output_length, iterations, rounds) < 1
        or warmup < 0
    ):
        raise ValueError(
            "Lengths, batch size, iterations, and rounds must be positive; warmup may be zero"
        )
    if input_length > config.max_input_len or (
        task != "pairs" and output_length > config.max_response_len
    ):
        raise ValueError("Benchmark lengths exceed the model config")
    steps = config.default_depth if steps is None else steps
    if steps < 0:
        raise ValueError("Reasoning steps must be nonnegative")
    if dtype not in {"float32", "bfloat16"}:
        raise ValueError("dtype must be float32 or bfloat16")
    if attention_backend != "sdpa" and dtype != "bfloat16":
        raise ValueError("FlashAttention benchmarks require --dtype bfloat16")
    if vocab_loss_backend != "torch" and task not in {"mlm", "denoise"}:
        raise ValueError("Vocabulary loss options require an mlm or denoise benchmark")
    runtime_device = torch.device(device)
    validate_attention_backend(
        attention_backend, runtime_device, [config.hidden_size // config.num_heads]
    )
    validate_vocab_loss(vocab_loss_backend, vocab_chunk_size, runtime_device)
    if liger_kernels:
        validate_liger(runtime_device)
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable; use --device cpu or install a CUDA PyTorch runtime"
        )
    if (
        runtime_device.type == "cuda"
        and dtype == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("This CUDA device does not support BF16")

    torch.manual_seed(seed)
    model = NuriModel(config).to(runtime_device)
    model.set_native_gqa(native_gqa)
    model.set_attention_backend(attention_backend)
    model.set_liger_kernels(liger_kernels)
    model.set_gradient_checkpointing(gradient_checkpointing)
    model.train(training)
    pair_settings = (
        TrainSettings(out_dir="", total_steps=1, peak_lr=5e-5)
        if task == "pairs"
        else None
    )
    if pair_settings is not None:
        optimizer = make_optimizer(model, pair_settings)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) if training else None
    clean_input = torch.randint(
        3, config.vocab_size, (batch_size, input_length), device=runtime_device
    )
    valid_input = torch.ones_like(clean_input, dtype=torch.bool)
    segments = None
    if pack_mlm:
        # Equal-size synthetic documents exercise the real isolation mask.
        document_ids = (
            torch.arange(input_length, device=runtime_device) // packed_document_length
        )
        segments = document_ids.expand(batch_size, -1)
    if task == "pairs":
        documents = torch.randint(
            3, config.vocab_size, (batch_size, input_length), device=runtime_device
        )
        # Row-aligned labels exercise the real pair loss; random text is not a quality test.
        pairs = list(zip(clean_input.cpu().tolist(), documents.cpu().tolist()))
        pair_batch = {
            key: value.to(runtime_device)
            for key, value in build_pair_batch(pairs, config).items()
        }
    else:
        response = torch.randint(
            3, config.vocab_size, (batch_size, output_length), device=runtime_device
        )
        valid_canvas = torch.ones_like(response, dtype=torch.bool)
        response[:, -1] = config.eos_id

    def autocast_context():
        if dtype == "bfloat16":
            return torch.autocast(device_type=runtime_device.type, dtype=torch.bfloat16)
        return nullcontext()

    def synchronize():
        if runtime_device.type == "cuda":
            torch.cuda.synchronize(runtime_device)

    def one_iteration():
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        gradient_context = nullcontext() if training else torch.inference_mode()
        with gradient_context, autocast_context():
            if task.startswith("embed-"):
                model.embed(clean_input, valid_input, mode=task[6:], steps=steps)
                loss = None
            elif task == "pairs":
                result = pair_loss(
                    model,
                    pair_batch,
                    steps,
                    pair_settings.embed_dims,
                    pair_settings.temperature,
                    pair_settings.cross_mode_weight,
                )
                loss = result["loss"]
            elif task == "generate":
                generate(
                    model,
                    clean_input,
                    valid_input,
                    steps=steps,
                    rounds=rounds,
                    canvas_len=output_length,
                )
                loss = None
            elif task == "mlm":
                loss_mask = torch.rand(clean_input.shape, device=runtime_device) < 0.3
                loss_mask[:, 0] = True
                corrupted = clean_input.masked_fill(loss_mask, config.mask_id)
                x, _ = model.encode(corrupted, valid_input, segments=segments)
                loss = vocab_token_mean_ce(
                    x[loss_mask],
                    model.embedding.weight,
                    clean_input[loss_mask],
                    vocab_loss_backend,
                    vocab_chunk_size,
                )
            else:
                loss_mask = torch.rand(response.shape, device=runtime_device) < 0.5
                loss_mask[:, 0] = True
                canvas = response.masked_fill(loss_mask, config.mask_id)
                x, valid = model.encode(clean_input, valid_input)
                if config.draft_feedback:
                    r = model.reason(x, valid, steps, draft=(canvas, valid_canvas))
                else:
                    r = model.reason(x, valid, steps)
                cache = model.generation_memory(x, valid, r)
                hidden = model.readout_hidden(canvas, valid_canvas, cache)
                loss = vocab_sample_mean_ce(
                    hidden[loss_mask],
                    model.embedding.weight,
                    response,
                    loss_mask,
                    vocab_loss_backend,
                    vocab_chunk_size,
                )
            aux = model.pop_moe_aux()
            if loss is not None and aux["loss"] is not None:
                loss = loss + aux["loss"]
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        return loss

    for _ in range(warmup):
        one_iteration()
    synchronize()
    for layer in model.moe_layers():
        layer.used_backends.clear()
    attention_layers = [
        layer for layer in model.modules() if isinstance(layer, Attention)
    ]
    for layer in attention_layers:
        layer.used_backends.clear()
    if runtime_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime_device)
    elapsed = []
    for _ in range(iterations):
        synchronize()
        start = time.perf_counter()
        loss = one_iteration()
        synchronize()
        elapsed.append(time.perf_counter() - start)
        if loss is not None and not torch.isfinite(loss.detach()):
            raise RuntimeError("Benchmark produced a nonfinite loss")
    total_seconds = sum(elapsed)
    input_sequences_per_sample = 2 if task == "pairs" else 1
    input_tokens = batch_size * input_length * iterations * input_sequences_per_sample
    canvas_tokens = (
        batch_size * output_length * iterations
        if task in {"denoise", "generate"}
        else 0
    )
    sorted_times = sorted(elapsed)
    expert_backends = sorted(
        {backend for layer in model.moe_layers() for backend in layer.used_backends}
    )
    actual_moe_backend = (
        {
            (): "not_run",
            ("loop",): "portable_expert_loop",
            ("grouped",): "grouped_mm",
        }.get(tuple(expert_backends), "mixed_grouped_and_loop")
        if config.moe is not None
        else "dense"
    )
    attention_paths = sorted(
        {path for layer in attention_layers for path in layer.used_backends}
    )
    actual_attention = (
        attention_paths[0]
        if len(attention_paths) == 1
        else "mixed"
        if attention_paths
        else "not_run"
    )
    gqa_paths = sorted(
        {
            "compact_flash_gqa"
            if path.startswith("flash")
            else "native_sdpa_gqa"
            if layer.native_gqa
            else "repeat_kv_gqa"
            for layer in attention_layers
            if layer.num_heads != layer.num_kv_heads
            for path in layer.used_backends
        }
    )
    loss_path = "not_run"
    if task in {"mlm", "denoise"}:
        loss_path = (
            "torch_full_logits"
            if vocab_loss_backend == "torch"
            else "torch_checkpointed_chunks"
            if vocab_loss_backend == "chunked"
            else "liger_fused_linear_mean"
            if task == "mlm"
            else "liger_checkpointed_sample_mean"
        )
    return {
        "input_kind": "synthetic_random_tokens_not_a_quality_evaluation",
        "task": task,
        "training": training,
        "device": str(runtime_device),
        "device_name": torch.cuda.get_device_name(runtime_device)
        if runtime_device.type == "cuda"
        else "CPU",
        "torch_version": torch.__version__,
        "native_gqa": native_gqa,
        "attention_backend_requested": attention_backend,
        "attention_backend": actual_attention,
        "attention_paths_used": attention_paths,
        "liger_kernels": liger_kernels,
        "vocab_loss_backend_requested": vocab_loss_backend,
        "vocab_loss_execution": loss_path,
        "vocab_chunk_size": vocab_chunk_size,
        "gradient_checkpointing": gradient_checkpointing,
        "pack_mlm": pack_mlm,
        "packed_document_length": packed_document_length if pack_mlm else None,
        "packed_documents_per_row": math.ceil(input_length / packed_document_length)
        if pack_mlm
        else None,
        "attention_execution": gqa_paths[0]
        if len(gqa_paths) == 1
        else "mixed_gqa"
        if gqa_paths
        else "mha",
        "gqa_paths_used": gqa_paths,
        "autocast_dtype": dtype,
        "batch_size": batch_size,
        "input_length": input_length,
        "input_sequences_per_sample": input_sequences_per_sample,
        "output_length": output_length if canvas_tokens else None,
        "reasoning_steps": steps
        if task in {"denoise", "generate", "pairs", "embed-reasoning"}
        else 0,
        "refinement_rounds": rounds if task == "generate" else None,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "moe_backend_requested": config.moe.backend
        if config.moe is not None
        else "dense",
        "moe_backend": actual_moe_backend,
        "moe_expert_paths_used": expert_backends,
        "latency_seconds": {
            "mean": statistics.mean(elapsed),
            "median": statistics.median(elapsed),
            "p95": sorted_times[max(0, math.ceil(0.95 * len(sorted_times)) - 1)],
        },
        "input_tokens_per_second": input_tokens / total_seconds,
        "canvas_sequences_tokens_per_second": canvas_tokens / total_seconds
        if canvas_tokens
        else None,
        "generation_canvas_positions_processed": canvas_tokens * rounds
        if task == "generate"
        else canvas_tokens,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(runtime_device)
        if runtime_device.type == "cuda"
        else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(runtime_device)
        if runtime_device.type == "cuda"
        else None,
        "notes": [
            "Training includes forward, backward, clipping, and an AdamW step.",
            "Input throughput counts observed input tokens once per measured iteration.",
            "Pairs training counts query plus document tokens and includes both embedding modes; host batch building is excluded.",
            "Packed MLM uses synthetic equal-size documents and the real isolation mask; actual text lengths and host packing are excluded.",
            "Synthetic corruption selects about 30% of MLM positions and 50% of canvas positions; this is not the trainer's full corruption distribution.",
            "Generation throughput counts each output sequence once, not one copy per refinement round.",
            "Compare tasks, lengths, T, S, batch size, and dtype before comparing throughput.",
            "External FlashAttention uses key padding; query-specific document/local/group masks retain SDPA.",
            "Checkpointed vocabulary chunks trade recomputation for memory; savings and throughput must be measured.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=list(PRESETS), default="tiny")
    parser.add_argument(
        "--config", help="Use a saved executable NuriConfig JSON instead of a preset"
    )
    parser.add_argument(
        "--task",
        choices=[
            "mlm",
            "denoise",
            "pairs",
            "embed-fast",
            "embed-reasoning",
            "generate",
        ],
        default="denoise",
    )
    parser.add_argument("--training", action="store_true")
    parser.add_argument(
        "--native-gqa",
        action="store_true",
        help="use compact GQA heads in supported SDPA",
    )
    parser.add_argument(
        "--attention-backend", choices=ATTENTION_BACKENDS, default="sdpa"
    )
    parser.add_argument("--liger-kernels", action="store_true")
    parser.add_argument(
        "--vocab-loss-backend", choices=VOCAB_LOSS_BACKENDS, default="torch"
    )
    parser.add_argument("--vocab-chunk-size", type=int, default=256)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="recompute encoder/readout activations during backward",
    )
    parser.add_argument(
        "--pack-mlm",
        action="store_true",
        help="benchmark document-isolated synthetic packed MLM training",
    )
    parser.add_argument(
        "--packed-document-length",
        type=int,
        default=256,
        help="tokens per synthetic packed document; must be shorter than the input row",
    )
    parser.add_argument(
        "--moe-backend",
        choices=["loop", "grouped"],
        help="override the configured expert backend",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--input-length",
        type=int,
        default=16,
        help="tokens per input sequence; applies to both sides of pairs",
    )
    parser.add_argument("--output-length", type=int, default=8)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="Optional JSON output file")
    args = parser.parse_args(argv)
    config = NuriConfig.load(args.config) if args.config else PRESETS[args.preset]()
    if args.moe_backend is not None:
        if config.moe is None:
            parser.error("--moe-backend requires a MoE model configuration")
        config.moe.backend = args.moe_backend
    results = run_benchmark(
        config,
        task=args.task,
        training=args.training,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        input_length=args.input_length,
        output_length=args.output_length,
        steps=args.steps,
        rounds=args.rounds,
        warmup=args.warmup,
        iterations=args.iterations,
        seed=args.seed,
        native_gqa=args.native_gqa,
        attention_backend=args.attention_backend,
        liger_kernels=args.liger_kernels,
        vocab_loss_backend=args.vocab_loss_backend,
        vocab_chunk_size=args.vocab_chunk_size,
        gradient_checkpointing=args.gradient_checkpointing,
        pack_mlm=args.pack_mlm,
        packed_document_length=args.packed_document_length,
    )
    text = json.dumps(results, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
