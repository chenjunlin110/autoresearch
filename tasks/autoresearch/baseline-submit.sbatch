#!/usr/bin/env bash
#SBATCH --job-name=autoresearch-baseline
#SBATCH --partition=main
#SBATCH --account=k2m
#SBATCH --qos=k2m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=01:05:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Karpathy-style single-agent baseline. ONE Claude agent runs on ONE GPU
# inside a fresh clone of tasks/autoresearch/source, follows program.md,
# and iterates train.py edits sequentially for 1h. Compares directly
# against our 8-GPU parallel framework: same metric, same train.py, same
# 5-min experiment budget; only differs in scheduling.
#
# Submit:
#   sbatch tasks/autoresearch/baseline-submit.sbatch
#
# Output dir:
#   artifact/autoresearch-baseline/run-<timestamp>/

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  repo_root="$SLURM_SUBMIT_DIR"
else
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
source_root="$repo_root/tasks/autoresearch/source"

if [[ ! -d "$source_root/.git" ]]; then
  echo "$source_root must be a git repo for the agent to commit experiments." >&2
  exit 1
fi

cd "$repo_root"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_root="${AUTORESEARCH_BASELINE_RUN_ROOT:-$repo_root/artifact/autoresearch-baseline/run-$timestamp}"
mkdir -p "$run_root"
sandbox_root="$run_root/source"

# Hardlinked clone keeps the agent's edits isolated from the canonical
# source/ baseline while reusing the data caches.
git clone --no-hardlinks -q "$source_root" "$sandbox_root"
cd "$sandbox_root"
git config user.name "autoresearch-baseline"
git config user.email "autoresearch-baseline@example.com"
git checkout -q -B "autoresearch/auto-$timestamp"
cd "$repo_root"

# Same persistent shared cache the framework uses, so cycle-1 of the
# baseline isn't penalized by a cold torch.compile vs the parallel run.
export AUTORESEARCH_SHARED_CACHE_ROOT="${AUTORESEARCH_SHARED_CACHE_ROOT:-$HOME/.cache/autoresearch-shared-cache}"
mkdir -p "$AUTORESEARCH_SHARED_CACHE_ROOT"
export UV_CACHE_DIR="$AUTORESEARCH_SHARED_CACHE_ROOT/uv"
export HF_HOME="$AUTORESEARCH_SHARED_CACHE_ROOT/hf"
export XDG_CACHE_HOME="$AUTORESEARCH_SHARED_CACHE_ROOT/xdg"
export TRITON_CACHE_DIR="$AUTORESEARCH_SHARED_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$AUTORESEARCH_SHARED_CACHE_ROOT/inductor"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Manifest so post-run analysis knows what was set.
{
  echo "timestamp=$timestamp"
  echo "hostname=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "run_root=$run_root"
  echo "sandbox=$sandbox_root"
  echo "shared_cache=$AUTORESEARCH_SHARED_CACHE_ROOT"
  echo "branch=autoresearch/auto-$timestamp"
} > "$run_root/job.env"

if [[ ! -e /dev/nvidia0 ]]; then
  echo "No GPU at /dev/nvidia0 — sbatch did not land on a GPU node." >&2
  exit 2
fi

# The agent reads program.md itself; this kickoff prompt only handles
# the human-in-the-loop scaffolding (run-tag, branch creation) that
# program.md normally negotiates with a human.
read -r -d '' kickoff_prompt <<EOF || true
You are the autoresearch agent. Your task is to follow the protocol in program.md exactly, with two scaffolding steps already done so you can skip them:

  - The run tag is "auto-${timestamp}". A fresh git branch "autoresearch/auto-${timestamp}" has already been created and checked out for you. Skip program.md sections 1 and 2 (run-tag + branch).
  - Initialize results.tsv with just the header row if it doesn't already exist. Then run the unedited baseline as the first experiment per program.md.

Work autonomously and iterate as many experiments as you can within ~1 hour of wall time. Each \`uv run train.py\` takes ~5 minutes (training is internally time-budgeted). Do not ask questions; if program.md says "confirm with human" or similar, treat the answer as "go".

After each experiment, follow program.md exactly: read metrics.json, append to results.tsv, write a short experiment.md note in the experiment dir, then decide what to edit next. Commit each experiment so the lineage is preserved.

When ~5 minutes of wall time remain, stop launching new \`uv run train.py\` calls. Write a single \`summary.md\` at the repo root with: the experiments tried (id, edit, val_bpb), which directions paid off, and what you would do next.

Read program.md, README.md, prepare.py, train.py first to ground yourself. Then begin.
EOF

# Run claude in non-interactive mode with bash + edit tools allowed,
# pinned to opus for fair comparison vs our framework's manager. The
# 3300s timeout is sbatch wall - 5min slack so we get a clean exit
# instead of slurm SIGTERM.
log_path="$run_root/agent.log"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching claude in $sandbox_root" > "$log_path"

cd "$sandbox_root"
exec timeout --kill-after=30s 3300s claude \
  --print \
  --dangerously-skip-permissions \
  --model opus \
  --add-dir "$sandbox_root" \
  --max-budget-usd 20 \
  -- "$kickoff_prompt" \
  >> "$log_path" 2>&1
