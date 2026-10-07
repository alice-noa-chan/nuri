# Unattended release run with easy_run.py (D-022)

The release run uses about 7 days on one A100 80GB host that nobody watches.
The project goes over as one archive, the host runs one command, and one
output archive comes back. This page explains the three sides of that flow.

## 1. Prepare (this PC)

1. Write a private data spec (`{"entries": {"<name in archive>": "<local path>"}}`)
   and pack the inputs:

   ```text
   python tools/pack_easy_run.py data --spec <private spec> --out <private folder>/nuri-data.tar
   ```

   The command prints the archive SHA256. Upload the archive anywhere that
   serves it over HTTP(S), ideally with resumable (Range) downloads.
2. Copy `configs/easy_run.example.json` to a private folder, fill in
   `data.url`, `data.sha256` and the input names inside the archive.
   Never commit the filled-in file: it names private inputs (D-008).
3. Commit all code, then pack the code archive. It contains exactly the
   committed tree plus the private config as `easy_run.json`:

   ```text
   python tools/pack_easy_run.py code --config <private easy_run.json> --out <private folder>/nuri-easy-run.tar.gz
   ```

## 2. Run (GPU host)

```text
tar -xzf nuri-easy-run.tar.gz -C nuri && cd nuri
python easy_run.py
```

That is all. The script checks the host, installs the package, downloads
and verifies the data, measures each stage on the real data, plans the
steps, trains, evaluates and exports all five models, and packs the
result. The stages run in the order base-dense, it-dense, embed,
base-moe, it-moe, so the main (dense) family is finished first. If the process or the machine stops, run `python easy_run.py`
again in the same folder: finished steps are skipped, the download
resumes, and training resumes from the newest saved state (saved every
30 minutes). The budget clock starts at the first run, so a restart does
not add time.

Requirements: Linux, Python 3.10 or newer, an NVIDIA driver with a CUDA
build of PyTorch (or network access so pip can install one), `zstd` only
for `.tar.zst` data, and about 300 GB of free disk.

### What happens without supervision

| Situation | Behavior |
|---|---|
| Download breaks | Retried up to 20 times, continuing from the received bytes (also when the connection closes quietly before the end) |
| Wrong or expired URL | An HTTP 4xx answer stops at once with the code; 5xx answers are retried |
| Wrong archive | SHA256 mismatch stops before unpacking; the bad file is set aside so a rerun downloads again |
| Stop right after unpacking | A marker in the data folder records the unpacked archive, so a rerun does not download again |
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
`python easy_run.py` again: it continues and packs again at the end. A
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
