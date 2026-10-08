# Unattended release run with easy_run.py (D-022)

The release run uses about 7 days on one A100 80GB host that nobody watches
and that has **no internet access**. Everything goes over as one zstd
bundle, the host runs one command, and one output archive comes back. This
page explains the three sides of that flow.

## 1. Prepare (this PC)

1. Collect the offline packages in a private folder (once per dependency
   change). The host matches the environment of our other GPU jobs:
   Linux x86_64, CUDA 12.8 driver, PyTorch 2.10.0+cu128.

   ```text
   <offline>/python/cpython-3.11.*-x86_64-unknown-linux-gnu-install_only.tar.gz
   <offline>/wheels/*.whl          cp311 manylinux wheels, including the CUDA ones
   <offline>/requirements.txt      the pinned top-level packages
   ```

   The top-level packages are torch, numpy, safetensors, tokenizers, tqdm,
   tensorboard and transformers (the export writes Hugging Face classes).
   The install step imports every module of the release path
   (`RELEASE_MODULES` in easy_run.py) and stops at once if one is missing.

   The CPython archive is a relocatable python-build-standalone build
   (check it against the release SHA256SUMS). For the wheels, run
   `pip download` with `--platform manylinux_2_28_x86_64 --python-version
   3.11 --only-binary=:all:` and the PyTorch cu128 index. pip skips
   torch's Linux-only requirements (the `nvidia-*` packages, `triton`,
   `cuda-bindings`) on a non-Linux PC, so download those explicitly, with
   the versions from torch's own metadata.
2. Copy `configs/easy_run.example.json` to a private folder and fill in
   the input paths inside `data/`. Never commit the filled-in file: it
   names private inputs (D-008).
3. Write a private data spec (`{"entries": {"<path inside data/>": "<local
   file or folder>"}}`), commit all code, then build the bundle:

   ```text
   python tools/pack_easy_run.py bundle --config <private easy_run.json> \
       --data-spec <private spec> --offline <offline folder> \
       --out <private folder>/nuri-easy-run.tar.zst
   ```

   It contains the committed tree (git archive HEAD), `easy_run.json`,
   `offline/` and `data/` with `data/SHA256SUMS`, all under one folder
   `nuri/`. It also writes `nuri-easy-run.tar.zst.sha256`. Hand over both
   files.

## 2. Run (GPU host)

```text
sha256sum -c nuri-easy-run.tar.zst.sha256
tar --zstd -xf nuri-easy-run.tar.zst        (or: zstd -dc nuri-easy-run.tar.zst | tar -x)
cd nuri
python3 easy_run.py
```

That is all. The script checks the GPU and the bundle, sets up the
bundled Python 3.11 with the bundled packages (`pip --no-index`), checks
every data file, measures each stage on the real data, plans the steps,
trains, evaluates and exports all five models, and packs the result. It
never uses the network. The stages run in the order base-dense,
it-dense, embed, base-moe, it-moe, so the main (dense) family is
finished first.

If the process or the machine stops, run `python3 easy_run.py` again in
the same folder: finished steps are skipped and training resumes from the
newest saved state (saved every 30 minutes). The budget clock starts at
the first run, so a restart does not add time.

Requirements: Linux x86_64, any `python3` 3.8 or newer to start the
script, `tar` (and `zstd` or a `tar` with `--zstd` to unpack), an NVIDIA
driver for CUDA 12.8 (`nvidia-smi` must work), and about 300 GB of free
disk.

### What happens without supervision

| Situation | Behavior |
|---|---|
| Damaged or incomplete unpacking | The data check names the damaged files and stops; unpack the bundle again |
| Stop during installation | The next run reuses what is installed and reruns the offline pip install |
| Batch does not fit | Calibration halves the micro-batch and doubles accumulation (same effective batch) until the peak stays under 85% of GPU memory |
| Out of memory later | Gradient checkpointing is switched on for unfinished stages and training resumes |
| Other crash | run_release is restarted up to 4 times; each restart resumes |
| Non-finite loss | Training stops without saving that update; the logs say so (needs a person) |
| Run slower than planned | At the deadline (`total_hours - package_hours` after the first start) the running stage saves and stops, and the finished stages are packed |
| `SIGTERM` to easy_run | The training child saves first, then the finished stages are packed |
| Any failure that ends easy_run | The finished stages and all logs are still packed; `status.json` names the reason |

## 3. Bring back

Bring back `easy_run_output.tar`. It contains:

- `models/`: the five release folders (`nuri-base`, `nuri-base-moe`,
  `nuri-base-it`, `nuri-base-moe-it`, `nuri-base-embed`) with tokenizer
  files and verification records;
- `eval/`: evaluation results per stage;
- `logs/<stage>/`: training and holdout logs and TensorBoard events;
- `run_summary.json`, `plan.json`, `state.json` (measured step times,
  chosen micro-batches, planned steps) and `easy_run.log`;
- `easy_run_logs/`: the full console output of every calibration and
  training run;
- `status.json`: `"complete"`, or `"stopped early: ..."` with the reason,
  plus the finished steps, whether the deadline was reached and the
  released models.

The archive is written whenever easy_run ends, also after a failure or a
stop. If `status.json` is not `"complete"` and time is left, run
`python3 easy_run.py` again: it continues and packs again at the end. A
stage stopped at the deadline keeps its training state under
`easy_run_work/runs/<stage>/` but is not exported.

## Budget

`budget.total_hours` (168) minus the time already used and
`budget.reserve_hours` (8, for evaluation, export and packing), times
`budget.safety` (0.9), is split by the stage shares of D-022
(50/30/8/6/6). IT and embedding stages stop at `budget.max_epochs` (3)
passes over their data and their unused time goes to `base-dense`.
Steps are frozen in `plan.json` once written, because the learning-rate
schedule depends on the total step count. `budget.package_hours` (1) is
kept free at the very end: training is stopped at that point even if a
stage is unfinished, so the output archive is written inside the window.
