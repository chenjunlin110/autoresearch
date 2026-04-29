#!/usr/bin/env bash
#SBATCH --job-name=qwen-sft-explicit
#SBATCH --partition=main
#SBATCH --account=k2m
#SBATCH --qos=k2m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=8
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Explicit-ALPS configuration for qwen-sft (paper Table 2 row 4). Same K=8,
# same 1200 s/eval, same metric, same manager prompt as Manual ALPS — but
# the 4 ALPS scheduling decisions are now CODE rather than prompt:
#   - statistical commit gate auto-fires (autoKeepEnabled=true, τ=0.1..0.5,
#     calibrationRepeats=3 to estimate σ̂ empirically)
#   - stale-winner rebase auto-enqueued (rebaseValidationEnabled=true)
#   - diversity-aware acquisition (quotaDiversityEnabled=true with
#     per-operator and same-value caps)
#   - hazard-driven Q_t (hazardQueueEnabled=true, qMin=1, qMax=8,
#     hazardZeta=2.0, posteriorMinSamples=3)
#
# Submit:
#   AUTORESEARCH_AGENT_RUNTIME=claude_cli sbatch \
#     --export=ALL,AUTORESEARCH_AGENT_RUNTIME tasks/qwen-sft/submit-explicit.sbatch

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  repo_root="$SLURM_SUBMIT_DIR"
else
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
script_dir="$repo_root/tasks/qwen-sft"
runner_root="$repo_root/infra/hpc_agent/runner"

if [[ ! -f "$script_dir/create-explicit.js" ]]; then
  echo "submit-explicit.sbatch: cannot find $script_dir/create-explicit.js" >&2
  exit 1
fi

if [[ ! -f "$HOME/.cache/qwen-sft/data/manifest.json" ]]; then
  echo "submit-explicit.sbatch: ~/.cache/qwen-sft/data/manifest.json missing." >&2
  echo "Run: uv run --project $script_dir/source python $script_dir/source/prepare.py" >&2
  exit 3
fi

cd "$repo_root"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_root="${QWEN_SFT_EXPLICIT_RUN_ROOT:-$repo_root/artifact/qwen-sft-explicit/run-$timestamp}"
mkdir -p "$run_root"

export TBC_HOME="$run_root/tbc_home"
mkdir -p "$TBC_HOME/logs"

export AUTORESEARCH_SHARED_CACHE_ROOT="${AUTORESEARCH_SHARED_CACHE_ROOT:-$HOME/.cache/autoresearch-shared-cache}"
mkdir -p "$AUTORESEARCH_SHARED_CACHE_ROOT"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

export QWEN_SFT_WANDB_GROUP="${QWEN_SFT_WANDB_GROUP:-qwen-sft-explicit-$timestamp}"

{
  echo "timestamp=$timestamp"
  echo "hostname=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "run_root=$run_root"
  echo "tbc_home=$TBC_HOME"
  echo "shared_cache=$AUTORESEARCH_SHARED_CACHE_ROOT"
  echo "wandb_run_group=$QWEN_SFT_WANDB_GROUP"
} > "$run_root/job.env"

if [[ ! -e /dev/nvidia0 ]]; then
  echo "No GPU device at /dev/nvidia0 — sbatch did not land on a GPU node." >&2
  exit 2
fi

node "$script_dir/create-explicit.js" \
  --artifact-root "$run_root" \
  --project-id qwen-sft-explicit \
  --gpu-count 8 \
  --experiment-worker-count 8 \
  --time-budget-seconds 1200

ln -sf "$run_root/projects.yaml" "$TBC_HOME/projects.yaml"

if [[ ! -f "$TBC_HOME/.env" ]]; then
  password="$(head -c 16 /dev/urandom | base64 | tr -d '=+/' | head -c 16)"
  port="${TBC_PORT:-0}"
  cat > "$TBC_HOME/.env" <<EOF
TBC_PASSWORD=$password
TBC_PORT=$port
EOF
fi

exec node "$runner_root/src/server.js" \
  > "$run_root/server.log" 2>&1
