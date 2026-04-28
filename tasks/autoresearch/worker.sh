#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# script_dir = <repo>/tasks/autoresearch ; workspace_root = <repo>
workspace_root="$(cd "$script_dir/../.." && pwd)"
default_repo_root="$script_dir/source"
# DIRECT_EXECUTOR_REPO_ROOT (set by infra/hpc_agent/runner/src/direct-executor.js)
# wins over AUTORESEARCH_REPO_ROOT and the default. This is what makes
# `param_patch` experiments actually run from their per-experiment sandbox
# instead of the shared source/ dir — without it, every concurrent worker
# reads the same source/train.py and applied edits are invisible to python.
repo_root="${DIRECT_EXECUTOR_REPO_ROOT:-${AUTORESEARCH_REPO_ROOT:-$default_repo_root}}"
# Venv stays at the canonical source/.venv even when training from a sandbox —
# every sandbox is a clone of source/, so the venv hasn't moved.
venv_root="${AUTORESEARCH_VENV_ROOT:-$default_repo_root}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="${1:-$workspace_root/artifact/autoresearch/worker-runs/$timestamp}"

mkdir -p "$output_dir"

# Shared cache: when AUTORESEARCH_SHARED_CACHE_ROOT is set (the framework's
# direct executor and submit.sbatch both set it), every worker on the node
# points its torchinductor/triton/uv/hf/xdg caches at the same root. Inductor's
# internal graph hashing dedupes across structurally-different code, so the
# first worker pays the ~150s cold-compile and subsequent workers running the
# same — or even similar — train.py hit the warm cache. Without the env, fall
# back to per-sandbox caches so single-shot ad-hoc runs still work.
shared_cache="${AUTORESEARCH_SHARED_CACHE_ROOT:-}"
if [[ -n "$shared_cache" ]]; then
  : "${UV_CACHE_DIR:=$shared_cache/uv}"
  : "${HF_HOME:=$shared_cache/hf}"
  : "${XDG_CACHE_HOME:=$shared_cache/xdg}"
  : "${TRITON_CACHE_DIR:=$shared_cache/triton}"
  : "${TORCHINDUCTOR_CACHE_DIR:=$shared_cache/inductor}"
else
  : "${UV_CACHE_DIR:=$repo_root/.uv-cache}"
  : "${HF_HOME:=$repo_root/.hf-home}"
  : "${XDG_CACHE_HOME:=$repo_root/.xdg-cache}"
  : "${TRITON_CACHE_DIR:=$repo_root/.triton-cache}"
  : "${TORCHINDUCTOR_CACHE_DIR:=$repo_root/.torchinductor-cache}"
fi
: "${AUTORESEARCH_METRICS_PATH:=$output_dir/metrics.json}"
: "${OMP_NUM_THREADS:=1}"
: "${AUTORESEARCH_TIME_BUDGET_SECONDS:=300}"
# Inner-ring training timeout. Anything beyond budget+slack is a runaway, so
# kill the whole training process group rather than waste GPU on it. The
# orchestrator enforces a separate outer ring (Phase 2.2) for the bash
# wrapper itself in case it hangs before/after train.py.
: "${AUTORESEARCH_TRAIN_TIMEOUT_SLACK_SECONDS:=180}"
train_timeout=$((AUTORESEARCH_TIME_BUDGET_SECONDS + AUTORESEARCH_TRAIN_TIMEOUT_SLACK_SECONDS))
# Default compile=on to match the official autoresearch baseline. A worker
# may still pass AUTORESEARCH_DISABLE_COMPILE=1 explicitly to opt out (e.g.
# retrying after a torch.compile error).
: "${AUTORESEARCH_DISABLE_COMPILE:=0}"
export AUTORESEARCH_DISABLE_COMPILE
export AUTORESEARCH_TIME_BUDGET_SECONDS

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
  echo "autoresearch_time_budget_seconds=$AUTORESEARCH_TIME_BUDGET_SECONDS"
  echo "train_timeout_seconds=$train_timeout"
  echo "autoresearch_metrics_path=$AUTORESEARCH_METRICS_PATH"
  printf "command="
  printf "%q " "${cmd[@]}"
  echo
} > "$output_dir/run.env"

# `timeout --kill-after=15s` SIGTERMs the whole training process group at
# `train_timeout`, then SIGKILLs anything that lingered 15s later. Exit 124
# means timeout fired and the child obeyed SIGTERM; 137 means SIGKILL had to
# escalate. Either is "training_timeout" to the validator.
set +e
timeout --kill-after=15s "${train_timeout}s" "${cmd[@]}" > "$output_dir/train.log" 2>&1 &
train_pid=$!

# Manager-driven early stop: if AUTORESEARCH_EARLY_STOP_AFTER_S is set,
# wait that many seconds, then peek at the latest `loss: <number>` line
# train.py prints each step. If the latest loss exceeds the threshold,
# kill the training group early — saves GPU on hypotheses that are
# clearly diverging without waiting for the full budget.
early_stop_triggered=0
early_stop_loss=""
if [[ -n "${AUTORESEARCH_EARLY_STOP_AFTER_S:-}" && -n "${AUTORESEARCH_EARLY_STOP_LOSS_ABOVE:-}" ]]; then
  (
    sleep "$AUTORESEARCH_EARLY_STOP_AFTER_S"
    if kill -0 "$train_pid" 2>/dev/null; then
      latest_loss="$(grep -oE 'loss:[[:space:]]*[0-9]+\.[0-9]+' "$output_dir/train.log" 2>/dev/null \
        | tail -n 1 | grep -oE '[0-9]+\.[0-9]+')"
      if [[ -n "$latest_loss" ]] && awk \
        -v cur="$latest_loss" -v thr="$AUTORESEARCH_EARLY_STOP_LOSS_ABOVE" \
        'BEGIN{exit !(cur+0 > thr+0)}'; then
        echo "early_stop_triggered=1 latest_loss=$latest_loss threshold=$AUTORESEARCH_EARLY_STOP_LOSS_ABOVE" \
          > "$output_dir/early_stop.log"
        kill -TERM "$train_pid" 2>/dev/null || true
        sleep 5
        kill -KILL "$train_pid" 2>/dev/null || true
      fi
    fi
  ) &
  watcher_pid=$!
fi

wait "$train_pid"
exit_code=$?
set -e

# If the watcher fired, signal it via the early_stop.log file and treat
# it as a deliberate "abort", not a runaway timeout.
if [[ -f "$output_dir/early_stop.log" ]]; then
  early_stop_triggered=1
  early_stop_loss="$(grep -oE 'latest_loss=[0-9.]+' "$output_dir/early_stop.log" \
    | head -1 | cut -d= -f2)"
fi

# Best-effort cleanup of the watcher subshell if training finished first.
if [[ -n "${watcher_pid:-}" ]]; then
  kill "$watcher_pid" 2>/dev/null || true
  wait "$watcher_pid" 2>/dev/null || true
fi

timed_out=0
if [[ "$exit_code" == "124" || "$exit_code" == "137" ]]; then
  timed_out=1
fi

{
  echo "exit_code=$exit_code"
  echo "output_dir=$output_dir"
  echo "metrics_path=$AUTORESEARCH_METRICS_PATH"
  echo "timed_out=$timed_out"
  echo "train_timeout_seconds=$train_timeout"
  echo "early_stop_triggered=$early_stop_triggered"
  if [[ -n "$early_stop_loss" ]]; then
    echo "early_stop_loss=$early_stop_loss"
  fi
} > "$output_dir/result.txt"

if [[ -n "${AUTORESEARCH_RESULTS_PATH:-}" ]]; then
  val_bpb="$(python3 -c "import json; d=json.load(open('$AUTORESEARCH_METRICS_PATH')); print(d.get('val_bpb',''))" 2>/dev/null || true)"
  # Serialize append across concurrent workers; the lock is on a sidecar
  # `.lock` file so it survives the data file being created inside the crit.
  lock_file="${AUTORESEARCH_RESULTS_PATH}.lock"
  : > "$lock_file" 2>/dev/null || true
  (
    flock -w 30 9 || exit 0
    if [[ ! -f "$AUTORESEARCH_RESULTS_PATH" ]]; then
      printf 'timestamp\trun_id\tval_bpb\texit_code\toutput_dir\n' > "$AUTORESEARCH_RESULTS_PATH"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$timestamp" \
      "$(basename "$output_dir")" \
      "${val_bpb:-}" \
      "$exit_code" \
      "$output_dir" >> "$AUTORESEARCH_RESULTS_PATH"
  ) 9>"$lock_file"
fi

exit "$exit_code"
