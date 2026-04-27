#!/usr/bin/env bash
#SBATCH --job-name=qwen-sft-runner
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

# Launch the qwen-sft data-mix search task (manager + 8 SFT workers) directly
# on a GPU compute node. The framework server runs in the foreground so the
# sbatch job's lifetime equals the run's lifetime.
#
# Submit with:
#   sbatch tasks/qwen-sft/submit.sbatch
# Override run dir:
#   QWEN_SFT_RUN_ROOT=/path sbatch tasks/qwen-sft/submit.sbatch
# Override agent runtime (codex_cli | claude_cli | api):
#   AUTORESEARCH_AGENT_RUNTIME=claude_cli sbatch --export=ALL,AUTORESEARCH_AGENT_RUNTIME tasks/qwen-sft/submit.sbatch
#
# Run prepare.py once before the first sbatch:
#   uv run --project tasks/qwen-sft/source python tasks/qwen-sft/source/prepare.py

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  repo_root="$SLURM_SUBMIT_DIR"
else
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
script_dir="$repo_root/tasks/qwen-sft"
runner_root="$repo_root/infra/hpc_agent/runner"

if [[ ! -f "$script_dir/create.js" ]]; then
  echo "submit.sbatch: cannot find $script_dir/create.js — repo_root=$repo_root may be wrong." >&2
  exit 1
fi

# Sanity check: prepared data must exist before kicking off the manager.
if [[ ! -f "$HOME/.cache/qwen-sft/data/manifest.json" ]]; then
  echo "submit.sbatch: ~/.cache/qwen-sft/data/manifest.json missing." >&2
  echo "Run: uv run --project $script_dir/source python $script_dir/source/prepare.py" >&2
  exit 3
fi

cd "$repo_root"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_root="${QWEN_SFT_RUN_ROOT:-$repo_root/artifact/qwen-sft/run-$timestamp}"
mkdir -p "$run_root"

# Per-run TBC_HOME so this allocation's server doesn't collide with a
# user-level ~/.hpc_agent instance.
export TBC_HOME="$run_root/tbc_home"
mkdir -p "$TBC_HOME/logs"

# Reuse the same persistent compile-cache as autoresearch — kernels overlap
# (PyTorch + uv + HF), so cycle-1 of qwen-sft can hit warm cache from a
# previous autoresearch run and vice versa.
export AUTORESEARCH_SHARED_CACHE_ROOT="${AUTORESEARCH_SHARED_CACHE_ROOT:-$HOME/.cache/autoresearch-shared-cache}"
mkdir -p "$AUTORESEARCH_SHARED_CACHE_ROOT"

# Reuse user's HF cache so Qwen3-4B isn't re-downloaded.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

{
  echo "timestamp=$timestamp"
  echo "hostname=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "slurm_job_nodelist=${SLURM_JOB_NODELIST:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "run_root=$run_root"
  echo "tbc_home=$TBC_HOME"
  echo "shared_cache=$AUTORESEARCH_SHARED_CACHE_ROOT"
  echo "hf_home=$HF_HOME"
} > "$run_root/job.env"

if [[ ! -e /dev/nvidia0 ]]; then
  echo "No GPU device at /dev/nvidia0 — this sbatch job did not land on a GPU node." >&2
  exit 2
fi

# Materialize the project workspace under $run_root.
node "$script_dir/create.js" \
  --artifact-root "$run_root" \
  --project-id qwen-sft \
  --gpu-count 8 \
  --experiment-worker-count 8 \
  --time-budget-seconds 1800

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
