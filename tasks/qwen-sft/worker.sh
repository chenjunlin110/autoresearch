#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# script_dir = <repo>/tasks/qwen-sft ; workspace_root = <repo>
workspace_root="$(cd "$script_dir/../.." && pwd)"
default_repo_root="$script_dir/source"
# DIRECT_EXECUTOR_REPO_ROOT wins so per-experiment sandboxes actually get
# picked up by python; otherwise concurrent experiments race over the same
# source/train.py and edits applied to the sandbox are never read.
repo_root="${DIRECT_EXECUTOR_REPO_ROOT:-${QWEN_SFT_REPO_ROOT:-$default_repo_root}}"
venv_root="${QWEN_SFT_VENV_ROOT:-$default_repo_root}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="${1:-$workspace_root/artifact/qwen-sft/worker-runs/$timestamp}"

mkdir -p "$output_dir"

# Shared cache layout matches autoresearch's so cycle-1 of either task
# reuses inductor / triton kernels compiled by the other. Same env name
# (AUTORESEARCH_SHARED_CACHE_ROOT) so the framework's submit.sbatch
# variants both work without per-task aliasing.
shared_cache="${AUTORESEARCH_SHARED_CACHE_ROOT:-}"
if [[ -n "$shared_cache" ]]; then
  : "${UV_CACHE_DIR:=$shared_cache/uv}"
  : "${HF_HOME:=${HF_HOME:-$shared_cache/hf}}"
  : "${XDG_CACHE_HOME:=$shared_cache/xdg}"
  : "${TRITON_CACHE_DIR:=$shared_cache/triton}"
  : "${TORCHINDUCTOR_CACHE_DIR:=$shared_cache/inductor}"
else
  : "${UV_CACHE_DIR:=$repo_root/.uv-cache}"
  : "${HF_HOME:=$HOME/.cache/huggingface}"
  : "${XDG_CACHE_HOME:=$repo_root/.xdg-cache}"
  : "${TRITON_CACHE_DIR:=$repo_root/.triton-cache}"
  : "${TORCHINDUCTOR_CACHE_DIR:=$repo_root/.torchinductor-cache}"
fi
: "${QWEN_SFT_METRICS_PATH:=$output_dir/metrics.json}"
: "${OMP_NUM_THREADS:=2}"
: "${SFT_TIME_BUDGET_SECONDS:=1800}"
# Inner-ring training timeout: budget + slack. Slack is wider than autoresearch's
# (300s vs 180s) because Qwen3-4B model loading + first compile can take 90-120s
# before training begins; we don't want to kill those.
: "${SFT_TRAIN_TIMEOUT_SLACK_SECONDS:=300}"
train_timeout=$((SFT_TIME_BUDGET_SECONDS + SFT_TRAIN_TIMEOUT_SLACK_SECONDS))

export SFT_TIME_BUDGET_SECONDS
export UV_CACHE_DIR
export HF_HOME
export XDG_CACHE_HOME
export TRITON_CACHE_DIR
export TORCHINDUCTOR_CACHE_DIR
export QWEN_SFT_METRICS_PATH
export OMP_NUM_THREADS
export PYTHONUNBUFFERED=1

mkdir -p \
  "$UV_CACHE_DIR" \
  "$HF_HOME" \
  "$XDG_CACHE_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$(dirname "$QWEN_SFT_METRICS_PATH")"

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
  echo "sft_time_budget_seconds=$SFT_TIME_BUDGET_SECONDS"
  echo "train_timeout_seconds=$train_timeout"
  echo "qwen_sft_metrics_path=$QWEN_SFT_METRICS_PATH"
  printf "command="
  printf "%q " "${cmd[@]}"
  echo
} > "$output_dir/run.env"

# `timeout --kill-after=15s` SIGTERMs the training process group at
# train_timeout; SIGKILLs anything still alive 15s later.
set +e
timeout --kill-after=15s "${train_timeout}s" "${cmd[@]}" > "$output_dir/train.log" 2>&1 &
train_pid=$!

# Manager-driven early stop: if AUTORESEARCH_EARLY_STOP_AFTER_S is set, peek
# at train.log for the latest `loss: X.XX` line train.py prints; if it
# exceeds the threshold, kill training early.
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

if [[ -f "$output_dir/early_stop.log" ]]; then
  early_stop_triggered=1
  early_stop_loss="$(grep -oE 'latest_loss=[0-9.]+' "$output_dir/early_stop.log" \
    | head -1 | cut -d= -f2)"
fi

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
  echo "metrics_path=$QWEN_SFT_METRICS_PATH"
  echo "timed_out=$timed_out"
  echo "train_timeout_seconds=$train_timeout"
  echo "early_stop_triggered=$early_stop_triggered"
  if [[ -n "$early_stop_loss" ]]; then
    echo "early_stop_loss=$early_stop_loss"
  fi
} > "$output_dir/result.txt"

if [[ -n "${QWEN_SFT_RESULTS_PATH:-}" ]]; then
  val_loss="$(python3 -c "import json; d=json.load(open('$QWEN_SFT_METRICS_PATH')); print(d.get('val_loss',''))" 2>/dev/null || true)"
  lock_file="${QWEN_SFT_RESULTS_PATH}.lock"
  : > "$lock_file" 2>/dev/null || true
  (
    flock -w 30 9 || exit 0
    if [[ ! -f "$QWEN_SFT_RESULTS_PATH" ]]; then
      printf 'timestamp\trun_id\tval_loss\texit_code\toutput_dir\n' > "$QWEN_SFT_RESULTS_PATH"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$timestamp" \
      "$(basename "$output_dir")" \
      "${val_loss:-}" \
      "$exit_code" \
      "$output_dir" >> "$QWEN_SFT_RESULTS_PATH"
  ) 9>"$lock_file"
fi

exit "$exit_code"
