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

from nuri.config import NuriConfig, PRESETS
from nuri.generation import generate
from nuri.losses import sample_mean_ce, token_mean_ce
from nuri.model import NuriModel


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
) -> dict:
    if task not in {"mlm", "denoise", "embed-fast", "embed-reasoning", "generate"}:
        raise ValueError("Unknown benchmark task")
    if training and task not in {"mlm", "denoise"}:
        raise ValueError("Training benchmarks support mlm and denoise tasks")
    if min(batch_size, input_length, output_length, iterations, rounds) < 1 or warmup < 0:
        raise ValueError("Lengths, batch size, iterations, and rounds must be positive; warmup may be zero")
    if input_length > config.max_input_len or output_length > config.max_response_len:
        raise ValueError("Benchmark lengths exceed the model config")
    steps = config.default_depth if steps is None else steps
    if steps < 0:
        raise ValueError("Reasoning steps must be nonnegative")
    if dtype not in {"float32", "bfloat16"}:
        raise ValueError("dtype must be float32 or bfloat16")
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu or install a CUDA PyTorch runtime")
    if runtime_device.type == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support BF16")

    torch.manual_seed(seed)
    model = NuriModel(config).to(runtime_device)
    model.set_native_gqa(native_gqa)
    model.train(training)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) if training else None
    clean_input = torch.randint(3, config.vocab_size, (batch_size, input_length), device=runtime_device)
    valid_input = torch.ones_like(clean_input, dtype=torch.bool)
    response = torch.randint(3, config.vocab_size, (batch_size, output_length), device=runtime_device)
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
            elif task == "generate":
                generate(model, clean_input, valid_input, steps=steps, rounds=rounds, canvas_len=output_length)
                loss = None
            elif task == "mlm":
                loss_mask = torch.rand(clean_input.shape, device=runtime_device) < 0.3
                loss_mask[:, 0] = True
                corrupted = clean_input.masked_fill(loss_mask, config.mask_id)
                x, _ = model.encode(corrupted, valid_input)
                logits = model.mlm_logits(x, loss_mask)
                loss = token_mean_ce(logits, clean_input[loss_mask])
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
                loss = sample_mean_ce(model.vocab_logits(hidden[loss_mask]), response, loss_mask)
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
    input_tokens = batch_size * input_length * iterations
    canvas_tokens = batch_size * output_length * iterations if task in {"denoise", "generate"} else 0
    sorted_times = sorted(elapsed)
    expert_backends = sorted({backend for layer in model.moe_layers() for backend in layer.used_backends})
    actual_moe_backend = {(): "not_run", ("loop",): "portable_expert_loop", ("grouped",): "grouped_mm"}.get(
        tuple(expert_backends), "mixed_grouped_and_loop") if config.moe is not None else "dense"
    return {
        "input_kind": "synthetic_random_tokens_not_a_quality_evaluation",
        "task": task,
        "training": training,
        "device": str(runtime_device),
        "device_name": torch.cuda.get_device_name(runtime_device) if runtime_device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "native_gqa": native_gqa,
        "attention_execution": ("native_sdpa_gqa" if native_gqa else "repeat_kv_gqa")
                               if config.num_kv_heads not in (None, config.num_heads) else "mha",
        "autocast_dtype": dtype,
        "batch_size": batch_size,
        "input_length": input_length,
        "output_length": output_length if canvas_tokens else None,
        "reasoning_steps": steps if task in {"denoise", "generate", "embed-reasoning"} else 0,
        "refinement_rounds": rounds if task == "generate" else None,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "moe_backend_requested": config.moe.backend if config.moe is not None else "dense",
        "moe_backend": actual_moe_backend,
        "moe_expert_paths_used": expert_backends,
        "latency_seconds": {
            "mean": statistics.mean(elapsed),
            "median": statistics.median(elapsed),
            "p95": sorted_times[max(0, math.ceil(0.95 * len(sorted_times)) - 1)],
        },
        "input_tokens_per_second": input_tokens / total_seconds,
        "canvas_sequences_tokens_per_second": canvas_tokens / total_seconds if canvas_tokens else None,
        "generation_canvas_positions_processed": canvas_tokens * rounds if task == "generate" else canvas_tokens,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(runtime_device) if runtime_device.type == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(runtime_device) if runtime_device.type == "cuda" else None,
        "notes": [
            "Training includes forward, backward, clipping, and an AdamW step.",
            "Input throughput counts observed input tokens once per measured iteration.",
            "Generation throughput counts each output sequence once, not one copy per refinement round.",
            "Compare tasks, lengths, T, S, batch size, and dtype before comparing throughput.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=list(PRESETS), default="tiny")
    parser.add_argument("--config", help="Use a saved executable NuriConfig JSON instead of a preset")
    parser.add_argument("--task", choices=["mlm", "denoise", "embed-fast", "embed-reasoning", "generate"], default="denoise")
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--native-gqa", action="store_true", help="use compact GQA heads in supported SDPA")
    parser.add_argument("--moe-backend", choices=["loop", "grouped"], help="override the configured expert backend")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-length", type=int, default=16)
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
        config, task=args.task, training=args.training, device=args.device, dtype=args.dtype,
        batch_size=args.batch_size, input_length=args.input_length, output_length=args.output_length,
        steps=args.steps, rounds=args.rounds, warmup=args.warmup, iterations=args.iterations, seed=args.seed,
        native_gqa=args.native_gqa,
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
