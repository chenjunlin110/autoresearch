#!/usr/bin/env bash
#SBATCH --job-name=qwen-sft-baseline
#SBATCH --partition=main
#SBATCH --account=k2m
#SBATCH --qos=k2m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=04:05:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Karpathy-style single-agent baseline for qwen-sft. ONE Claude agent runs on
# ONE GPU inside a fresh clone of tasks/qwen-sft/source, follows program.md,
# and iterates DATA_MIX edits sequentially for ~4h. Compares directly against
# our 8-GPU manual ALPS framework: same metric, same train.py, same 1200s
# per-experiment budget; only differs in scheduling.
#
# Submit:
#   sbatch tasks/qwen-sft/baseline-submit.sbatch
#
# Output dir:
#   artifact/qwen-sft-baseline/run-<timestamp>/

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  repo_root="$SLURM_SUBMIT_DIR"
else
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
source_root="$repo_root/tasks/qwen-sft/source"

if [[ ! -d "$source_root/.git" ]]; then
  echo "$source_root must be a git repo for the agent to commit experiments." >&2
  exit 1
fi

if [[ ! -f "$HOME/.cache/qwen-sft/data/manifest.json" ]]; then
  echo "~/.cache/qwen-sft/data/manifest.json missing — run prepare.py first:" >&2
  echo "  uv run --project $source_root python $source_root/prepare.py" >&2
  exit 3
fi

cd "$repo_root"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_root="${QWEN_SFT_BASELINE_RUN_ROOT:-$repo_root/artifact/qwen-sft-baseline/run-$timestamp}"
mkdir -p "$run_root"
sandbox_root="$run_root/source"

# Hardlinked clone keeps the agent's edits isolated from the canonical
# source/ baseline while reusing the data caches.
git clone --no-hardlinks -q "$source_root" "$sandbox_root"
cd "$sandbox_root"
git config user.name "qwen-sft-baseline"
git config user.email "qwen-sft-baseline@example.com"
git checkout -q -B "qwen-sft/auto-$timestamp"
cd "$repo_root"

# Same persistent shared cache the framework uses, so cycle-1 of the
# baseline isn't penalized by a cold torch.compile vs the parallel run.
export AUTORESEARCH_SHARED_CACHE_ROOT="${AUTORESEARCH_SHARED_CACHE_ROOT:-$HOME/.cache/autoresearch-shared-cache}"
mkdir -p "$AUTORESEARCH_SHARED_CACHE_ROOT"
export UV_CACHE_DIR="$AUTORESEARCH_SHARED_CACHE_ROOT/uv"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="$AUTORESEARCH_SHARED_CACHE_ROOT/xdg"
export TRITON_CACHE_DIR="$AUTORESEARCH_SHARED_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$AUTORESEARCH_SHARED_CACHE_ROOT/inductor"
mkdir -p "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# wandb defaults so the serial baseline shows up in the same workshop project.
export WANDB_PROJECT="${WANDB_PROJECT:-workshop}"
export WANDB_ENTITY="${WANDB_ENTITY:-haolong}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-qwen-sft-baseline-$timestamp}"
export WANDB_TAGS="qwen-sft,datamix,serial,baseline"

# Agent will set DIRECT_EXECUTOR_REPO_ROOT to the sandbox so worker.sh runs
# train.py from the sandbox (not canonical source/).
export QWEN_SFT_REPO_ROOT="$sandbox_root"

# Manifest so post-run analysis knows what was set.
{
  echo "timestamp=$timestamp"
  echo "hostname=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "run_root=$run_root"
  echo "sandbox=$sandbox_root"
  echo "shared_cache=$AUTORESEARCH_SHARED_CACHE_ROOT"
  echo "branch=qwen-sft/auto-$timestamp"
  echo "wandb_run_group=$WANDB_RUN_GROUP"
} > "$run_root/job.env"

if [[ ! -e /dev/nvidia0 ]]; then
  echo "No GPU at /dev/nvidia0 — sbatch did not land on a GPU node." >&2
  exit 2
fi

# The agent reads program.md itself; this kickoff prompt only handles the
# human-in-the-loop scaffolding (run-tag, branch creation) that program.md
# normally negotiates with a human, plus environment guidance.
read -r -d '' kickoff_prompt <<EOF || true
You are the qwen-sft serial agent. Follow the protocol in \`program.md\` exactly. Two scaffolding steps are already done so you can skip them:

  - The run tag is "auto-${timestamp}". Branch "qwen-sft/auto-${timestamp}" has been created and checked out for you.
  - Initialize \`results.tsv\` with header columns: \`timestamp\trun_id\tval_loss\texit_code\tnum_steps\tdata_mix\` if it does not exist.

Environment setup the wrapper expects (set these before running \`worker.sh\`, or pass them inline per call):

  export SFT_TIME_BUDGET_SECONDS=1200
  export DIRECT_EXECUTOR_REPO_ROOT="$sandbox_root"
  export AUTORESEARCH_SHARED_CACHE_ROOT="$AUTORESEARCH_SHARED_CACHE_ROOT"
  export HF_HOME="$HF_HOME"
  export WANDB_PROJECT=workshop WANDB_ENTITY=haolong WANDB_RUN_GROUP=$WANDB_RUN_GROUP

For each experiment:
  1. Decide a DATA_MIX edit, reason about it briefly.
  2. Edit \`train.py\`'s \`DATA_MIX\` dict.
  3. \`git add train.py && git commit -m "<exp_id>: <one-line>"\`
  4. Make output dir: \`mkdir -p experiments/<exp_id>\`
  5. Set \`QWEN_SFT_METRICS_PATH=$sandbox_root/experiments/<exp_id>/metrics.json\` and \`WANDB_RUN_NAME=<exp_id>\`.
  6. Run \`bash $repo_root/tasks/qwen-sft/worker.sh "$sandbox_root/experiments/<exp_id>"\`.
  7. Read \`metrics.json\`, append to \`results.tsv\`, write \`experiments/<exp_id>/note.md\`.
  8. Decide whether to KEEP this mix as the new HEAD baseline, or revert (\`git reset --hard HEAD~1\`).

Work autonomously. Each experiment is ~22 min wall (1200 s train + load + eval), so target ~10-12 experiments in 4 h. Do not ask questions; if program.md says "confirm with human" treat the answer as "go".

When ~10 minutes of wall time remain, stop launching new \`worker.sh\` calls. Write a single \`summary.md\` at the repo root with: ranked experiment table, axes that moved val_loss the most, final HEAD mix, 3 next-mix proposals.

Read \`program.md\`, \`train.py\`, \`prepare.py\` first to ground yourself. Then begin.
EOF

# Run claude in non-interactive mode, pinned to opus for fair comparison vs
# our framework's manager. The 14400s wall - 5min slack timeout means we exit
# cleanly before slurm SIGTERM.
log_path="$run_root/agent.log"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching claude in $sandbox_root" > "$log_path"

cd "$sandbox_root"
exec timeout --kill-after=30s 14400s claude \
  --print \
  --dangerously-skip-permissions \
  --model opus \
  --add-dir "$sandbox_root" \
  --add-dir "$repo_root/tasks/qwen-sft" \
  --max-budget-usd 30 \
  -- "$kickoff_prompt" \
  >> "$log_path" 2>&1
