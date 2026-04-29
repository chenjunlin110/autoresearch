# qwen-sft Serial agent protocol

You are the autoresearch agent on the Qwen-3 SFT data-mixture task. You run **alone** on **one GPU** for ~4 hours, iterating SFT data-mix experiments sequentially. This is the Serial baseline against which our parallel framework is compared. It is not a parallel run; do not try to fork.

## What you change

Only the module-level `DATA_MIX` dict in `train.py`. Five buckets (math / code / chat / if / reasoning); weights normalize to 1.0. **Do not** touch model size, seq_len, learning rate, batch size, optimizer, schedule — those are the search baseline.

## What you measure

`val_loss` from `metrics.json` written by `train.py`. Lower is better. Balanced held-out cross-entropy across all five buckets (200 val samples per bucket = 1000 total), so over-weighting one easy bucket does not help — only mixes that produce a generally-stronger model win.

## Per-experiment loop

Each experiment is a fresh edit + run. Per experiment:

1. Decide a new `DATA_MIX` based on past results visible in `results.tsv`. Be deliberate: state in your scratch what hypothesis the change tests.
2. Edit `train.py`'s module-level `DATA_MIX` dict (only that line). Keep the keys exactly `math`, `code`, `chat`, `if`, `reasoning`. Each value `>= 0`. Do not add or remove keys.
3. Stage and commit with a short message: `git add train.py && git commit -m "<exp_id>: <one-line summary>"`.
4. Set `SFT_TIME_BUDGET_SECONDS=1200` and `QWEN_SFT_METRICS_PATH=experiments/<exp_id>/metrics.json` and run the wrapper:
   `bash ../tasks/qwen-sft/worker.sh experiments/<exp_id>` from the sandbox repo root (or use absolute paths).
5. After it exits, read `experiments/<exp_id>/metrics.json`, verify `val_loss` is finite, and append a row to `results.tsv` with `exp_id, val_loss, num_steps, training_seconds, data_mix_normalized`.
6. Decide whether to **keep** this mix as the new HEAD baseline (if `val_loss` improved meaningfully) or **revert** (`git reset --hard HEAD~1`) and try a different direction. The whole point of the Serial loop is that you can advance HEAD whenever an edit wins.
7. Write a 3-5 line `experiments/<exp_id>/note.md` recording: hypothesis, mix vector, val_loss, kept-or-reverted decision.

## Time budget

You have ~4 hours of wall time. Each `worker.sh` call takes ~22 minutes (1200 s training + ~120 s model load + ~50 s eval), so expect ~10-12 experiments. Do not parallelize; this is the Serial loop.

When ~10 minutes of wall remain, stop launching new experiments. Write a `summary.md` at the repo root with:
- Ranked table of all experiments (exp_id, mix, val_loss).
- Which directions paid off (which axes moved val_loss the most).
- Final HEAD baseline mix.
- 3 concrete "next mix" proposals the search did not try.

## Search axes (recap)

```python
DATA_MIX = {
    "math":      0.20,
    "code":      0.20,
    "chat":      0.20,
    "if":        0.20,
    "reasoning": 0.20,
}
```

Five buckets, each in `[0.0, 1.0]`. Trainer normalizes to sum = 1 internally.

Bucket sizes (from `~/.cache/qwen-sft/data/manifest.json`):
- math: train 234,072 / val 200
- code: train 142,075 / val 200
- chat: train 109,540 / val 200
- if:   train  29,780 / val 200  (smallest pool — will repeat under heavy weighting)
- reasoning: train 204,782 / val 200

## Operating constraints

- Only one experiment at a time. If you are tempted to run two `worker.sh` calls in parallel, do not — the GPU pool is one card.
- Do not modify `prepare.py`, `pyproject.toml`, or anything outside `train.py`'s `DATA_MIX` line.
- If a run fails (non-zero exit), record the failure in `results.tsv` with `val_loss=FAIL` and revert the edit. Move on; do not retry the same mix.
- Be patient: each worker is 22 min. Do not pre-decide all 10 experiments up front; read results between runs and adapt.
