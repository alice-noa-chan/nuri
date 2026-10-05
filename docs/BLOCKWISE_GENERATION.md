# Optional blockwise generation

Native `nuri.generation.generate` accepts `block_size=None` by default. This
keeps the existing full-canvas Mask-Predict policy. Set a positive integer to
compare left-to-right block refinement on the same checkpoint:

```python
from nuri.generation import generate

result = generate(model, input_ids, attention_mask,
                  canvas_len=64, rounds=4, block_size=16)
print(result.readout_calls)  # 16 calls for this four-block canvas
```

Base checkpoints require an explicit `canvas_len`. Instruction-tuned
checkpoints can use their trained length head by omitting it. Lengths still
count EOS, and `max_len` still bounds the canvas.

The existing text inference tool exposes the same option:

```text
python tools/infer_nuri.py --checkpoint <checkpoint-folder> --text "Hello" --canvas-len 64 --rounds 4 --block-size 16
```

`refinement_rounds` in its JSON report means rounds **per block**. `block_size`
records the requested option and `readout_calls` records actual batched
readout forwards. Leaving out `--block-size` preserves full-canvas generation.
An explicit block covering the longest actual canvas gives the same tokens,
confidence updates, history and seeded sampling as the default.

Each block starts fully masked. Every round predicts its current MASK
positions in parallel and stores the chosen token's probability at temperature
one. Before the final round, it remasks the least confident active positions
using that row's actual block length. Visible positions retain confidence from
their last masked prediction. Previously completed blocks are immutable;
future valid blocks remain MASK until their turn. Padding stays PAD and never
participates in filling or confidence ranking. The last block may be shorter,
and different batch rows can finish at different blocks.

This policy retains bidirectional canvas attention: every call still processes
the entire valid canvas, including the frozen prefix and future masks. Only
the active block receives vocabulary projections and token updates. Input
memory X, reasoning state R and the per-layer memory K/V cache are computed
once per request. Reasoning depth T is independent of the number of blocks.
Explicit `block_size` with `draft_feedback=True` is rejected before encoding;
this experiment does not define feedback across block boundaries.

For maximum batch canvas length L, block width K and S rounds per block, the
call count is `S * ceil(L / K)`, compared with S for full-canvas generation.
Rows that have already finished remain in those batched forwards. Full-canvas
attention is recomputed each time, so smaller blocks can cost more time even
though vocabulary projection applies only to the active block. Measure actual
latency and quality; no speed or quality improvement is assumed.

Intermediate EOS does not stop refinement or skip later blocks. After all
blocks finish, the returned answer is cut before the first EOS, exactly as in
the default sampler. Missing EOS and early EOS remain separate diagnostics.
`keep_history=True` stores a full-canvas snapshot after each refinement round,
in left-to-right block order, with future blocks still masked.

This is blockwise Mask-Predict, not an exact LLaDA or absorbing-diffusion
sampler. It uses Nuri's last-masked confidence and can revisit predictions
within the active block. Training corruption, model weights, architecture
defaults and acceptance thresholds are unchanged. Compare it on validation
before selecting any inference setting for a final completion test.
