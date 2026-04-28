# qwen-sft — SFT data-mix search

A task plugin for the framework. The manager LLM proposes mixture weights over
distinct Tulu-3 SFT domains (math / code / chat / if / reasoning); a worker
fully fine-tunes Qwen3-0.6B on that mixture for ~10 min; the held-out balanced
val loss is reported back; manager iterates.

## What's here

```
tasks/qwen-sft/
├── README.md             this file
├── source/               the SFT trainer (we own this code; not upstream)
│   ├── train.py          full-FT loop with WeightedBucketSampler
│   ├── prepare.py        download + bucket Tulu-3 → ~/.cache/qwen-sft/data/
│   └── pyproject.toml    transformers / datasets / accelerate / flash-attn
├── manager.js            workspace generator + manager/worker prompts
├── create.js             CLI wrapper (called by submit.sbatch)
├── worker.sh             bash wrapper (timeout / flock / early-stop watcher)
├── sandbox.sh            per-experiment git clone of source/
├── submit.sbatch         8-GPU 4h sbatch entrypoint
└── tests/smoke.test.js   workspace-generator regression test
```

## Setup (one-time)

```bash
# 1. Install the source/ venv (uv handles flash-attn no-build-isolation)
cd tasks/qwen-sft/source
uv sync

# 2. Initialize source/ as a git repo so sandbox.sh / direct-executor can
#    clone it at run time.
git init && git add . && git commit -m "qwen-sft source init"

# 3. Download + bucket Tulu-3 into ~/.cache/qwen-sft/data/
uv run python prepare.py
# (smoke variant for plumbing test only: --mode smoke)
```

## Run

```bash
# 8 GPUs × 4h
sbatch tasks/qwen-sft/submit.sbatch

# With Claude Code as the runtime
AUTORESEARCH_AGENT_RUNTIME=claude_cli sbatch \
  --export=ALL,AUTORESEARCH_AGENT_RUNTIME \
  tasks/qwen-sft/submit.sbatch
```

Per-run outputs land at `artifact/qwen-sft/run-<timestamp>/`.

## Search surface

The manager edits exactly one thing: the module-level `DATA_MIX` dict in
`source/train.py`. Five buckets, each in `[0.0, 1.0]`; the trainer
normalizes the dict to sum to 1 internally.

Fixed control variables (do not change between experiments):

| Knob | Value | Notes |
|---|---|---|
| Model | `Qwen/Qwen3-0.6B` | bf16, full FT |
| `seq_len` | 2048 | |
| `micro_batch` | 16 | per-step, single-GPU |
| `grad_accum` | 2 | → effective batch 32 sequences (~65k tokens/step) |
| `learning_rate` | 2e-5 | constant after warmup |
| `warmup_steps` | 50 | linear |
| `weight_decay` | 0.01 | mild regularization |
| `grad_ckpt` | True | enables higher batch on 0.6B |
| Optimizer | AdamW (β=0.9/0.95, fused) | |
| Attention | FlashAttention-2 if installed, else SDPA | |

The framework's `param_patch` execution mode + `constant_replace` edit kind
applies `DATA_MIX` edits via Python AST round-trip, so the manager doesn't
need to worry about whitespace or dict-key ordering — only correctness of
the *current* value (`expected_old_repr`) and its *new* target (`new_repr`).

## Metric

Balanced held-out cross-entropy across all five buckets (200 val samples per
bucket = 1000 total). Lower is better. The canonical final value is written
to `metrics.json` under the `val_loss` key (also `val_loss_per_bucket` for
analysis); the framework's result-validator reads this and the manager
optimizes against it via the compact ledger.

`train.py` also runs an intermediate balanced eval every `EVAL_EVERY_STEPS`
(default 200) so wandb shows val-loss curves per experiment, including
per-bucket breakdowns (`val/math`, `val/code`, `val/chat`, `val/if`,
`val/reasoning`). Useful for spotting whether a winning mix wins on overall
average or whether it's lifting one bucket at the expense of another.

## WandB

Each experiment becomes one wandb run.

- `project = workshop`, `entity = haolong` (override via `WANDB_PROJECT` / `WANDB_ENTITY`)
- `run name = <experiment_id>` (e.g. `exp_0042_math_heavy`) — set automatically by `worker.sh` from the output dir basename
- `run group = qwen-sft-<sbatch timestamp>` — every experiment from one sbatch shares a group
- `tags = qwen-sft,datamix,framework`
- `config` includes the full data mix (raw + normalized), all fixed hyperparameters, model attn impl, and param counts
- Auth comes from `~/.netrc` (`machine api.wandb.ai`) — no extra setup required
- Set `WANDB_DISABLED=1` to skip logging

## Memory

Qwen3-0.6B full FT, bf16, with grad-ckpt + FlashAttention2 fits comfortably
in ~12 GB on a single H200 (144 GB), which leaves huge headroom for activations
at micro-batch 16 / grad-accum 2 / seq 2048. No FSDP / ZeRO needed; each
worker is single-GPU.
