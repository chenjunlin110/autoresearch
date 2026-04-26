#!/usr/bin/env bash
#SBATCH --job-name=autoresearch-runner
#SBATCH --partition=main
#SBATCH --account=k2m
#SBATCH --qos=k2m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=8
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Launch the autoresearch task (manager + 8 experiment workers) directly on
# a GPU compute node. The framework server runs in the foreground so the
# sbatch job's lifetime equals the run's lifetime.
#
# Submit with:
#   sbatch tasks/autoresearch/submit.sbatch
# Override run dir:
#   AUTORESEARCH_RUN_ROOT=/path sbatch tasks/autoresearch/submit.sbatch
# Override agent runtime (codex_cli | claude_cli | api):
#   AUTORESEARCH_AGENT_RUNTIME=claude_cli sbatch --export=ALL,AUTORESEARCH_AGENT_RUNTIME tasks/autoresearch/submit.sbatch

set -euo pipefail

# Slurm copies the script into a spool dir before executing, so BASH_SOURCE
# points at /var/spool/slurm/<job>/slurm_script — useless for resolving the
# repo. Use SLURM_SUBMIT_DIR (the directory sbatch was invoked from) when
# available; outside Slurm fall back to BASH_SOURCE for ad-hoc runs.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  repo_root="$SLURM_SUBMIT_DIR"
else
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
script_dir="$repo_root/tasks/autoresearch"
runner_root="$repo_root/infra/hpc_agent/runner"

if [[ ! -f "$script_dir/create-full.js" ]]; then
  echo "submit.sbatch: cannot find $script_dir/create-full.js — repo_root=$repo_root may be wrong." >&2
  exit 1
fi

cd "$repo_root"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_root="${AUTORESEARCH_RUN_ROOT:-$repo_root/artifact/autoresearch/run-$timestamp}"
mkdir -p "$run_root"

# Use a per-run TBC_HOME so this allocation's server does not collide with a
# user-level ~/.hpc_agent instance.
export TBC_HOME="$run_root/tbc_home"
mkdir -p "$TBC_HOME/logs"

# Shared compile cache root. Both the direct executor and any LLM-driven
# bash worker that sources tasks/autoresearch/worker.sh will point all of
# their inductor/triton/uv/hf/xdg caches under here, so the first cold
# torch.compile pays for itself across every later worker on this node.
export AUTORESEARCH_SHARED_CACHE_ROOT="$run_root/.shared-cache"
mkdir -p "$AUTORESEARCH_SHARED_CACHE_ROOT"

{
  echo "timestamp=$timestamp"
  echo "hostname=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "slurm_job_nodelist=${SLURM_JOB_NODELIST:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "run_root=$run_root"
  echo "tbc_home=$TBC_HOME"
} > "$run_root/job.env"

# Fail fast if we are not actually on a GPU node.
if [[ ! -e /dev/nvidia0 ]]; then
  echo "No GPU device at /dev/nvidia0 — this sbatch job did not land on a GPU node." >&2
  exit 2
fi

# Materialize the dag-full project workspace under $run_root. The creator
# writes a projects.yaml at $run_root/projects.yaml which the server will
# pick up via TBC_HOME.
node "$script_dir/create-full.js" \
  --artifact-root "$run_root" \
  --project-id autoresearch-dag-full \
  --gpu-count 8 \
  --experiment-worker-count 8 \
  --time-budget-seconds 300

# TBC_HOME expects projects.yaml at its root; link the generated one.
ln -sf "$run_root/projects.yaml" "$TBC_HOME/projects.yaml"

# Pre-populate .env so the CLI doesn't prompt interactively inside sbatch.
if [[ ! -f "$TBC_HOME/.env" ]]; then
  password="$(head -c 16 /dev/urandom | base64 | tr -d '=+/' | head -c 16)"
  port="${TBC_PORT:-0}"
  cat > "$TBC_HOME/.env" <<EOF
TBC_PASSWORD=$password
TBC_PORT=$port
EOF
fi

# Run the server in foreground so the sbatch job's lifetime = the run's lifetime.
exec node "$runner_root/src/server.js" \
  > "$run_root/server.log" 2>&1
