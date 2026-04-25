#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd "$script_dir/../../../.." && pwd)"
default_repo_root="$workspace_root/autoresearch"
repo_root="${AUTORESEARCH_REPO_ROOT:-$default_repo_root}"
venv_root="${AUTORESEARCH_VENV_ROOT:-$default_repo_root}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="${1:-$workspace_root/artifacts/worker-runs/$timestamp}"

mkdir -p "$output_dir"

: "${UV_CACHE_DIR:=$repo_root/.uv-cache}"
: "${HF_HOME:=$repo_root/.hf-home}"
: "${XDG_CACHE_HOME:=$repo_root/.xdg-cache}"
: "${TRITON_CACHE_DIR:=$repo_root/.triton-cache}"
: "${TORCHINDUCTOR_CACHE_DIR:=$repo_root/.torchinductor-cache}"
: "${AUTORESEARCH_METRICS_PATH:=$output_dir/metrics.json}"
: "${OMP_NUM_THREADS:=1}"

export UV_CACHE_DIR
export HF_HOME
export XDG_CACHE_HOME
export TRITON_CACHE_DIR
export TORCHINDUCTOR_CACHE_DIR
export AUTORESEARCH_METRICS_PATH
export OMP_NUM_THREADS
export PYTHONUNBUFFERED=1

mkdir -p \
  "$UV_CACHE_DIR" \
  "$HF_HOME" \
  "$XDG_CACHE_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$(dirname "$AUTORESEARCH_METRICS_PATH")"

if [[ -x "$venv_root/.venv/bin/python" ]]; then
  cmd=("$venv_root/.venv/bin/python" "$repo_root/train.py")
else
  cmd=(uv run --project "$repo_root" python "$repo_root/train.py")
fi

{
  echo "timestamp=$timestamp"
  echo "repo_root=$repo_root"
  echo "workspace_root=$workspace_root"
  echo "venv_root=$venv_root"
  echo "output_dir=$output_dir"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "omp_num_threads=$OMP_NUM_THREADS"
  echo "autoresearch_time_budget_seconds=${AUTORESEARCH_TIME_BUDGET_SECONDS:-300}"
  echo "autoresearch_metrics_path=$AUTORESEARCH_METRICS_PATH"
  printf "command="
  printf "%q " "${cmd[@]}"
  echo
} > "$output_dir/run.env"

set +e
"${cmd[@]}" > "$output_dir/train.log" 2>&1
exit_code=$?
set -e

{
  echo "exit_code=$exit_code"
  echo "output_dir=$output_dir"
  echo "metrics_path=$AUTORESEARCH_METRICS_PATH"
} > "$output_dir/result.txt"

exit "$exit_code"
