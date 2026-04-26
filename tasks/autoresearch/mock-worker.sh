#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-artifacts/mock-autoresearch-worker}"
mkdir -p "$output_dir"

metrics_path="${AUTORESEARCH_METRICS_PATH:-$output_dir/metrics.json}"
mkdir -p "$(dirname "$metrics_path")"

run_name="$(basename "$output_dir")"
seed=0
case "$run_name" in
  *2*|*wave2*) seed=2 ;;
  *3*|*wave3*) seed=3 ;;
  *) seed=1 ;;
esac

val_bpb="$(awk -v seed="$seed" 'BEGIN { printf "%.6f", 1.250000 - seed * 0.010000 }')"
training_seconds="${AUTORESEARCH_TIME_BUDGET_SECONDS:-10}"

cat > "$metrics_path" <<JSON
{
  "val_bpb": $val_bpb,
  "training_seconds": $training_seconds,
  "total_seconds": $training_seconds,
  "peak_vram_mb": 1024.0,
  "mfu_percent": 1.0,
  "total_tokens_M": 1.0,
  "num_steps": 10,
  "mock": true
}
JSON

cat > "$output_dir/result.txt" <<EOF
exit_code=0
output_dir=$output_dir
metrics_path=$metrics_path
EOF

echo "mock autoresearch complete: val_bpb=$val_bpb metrics=$metrics_path"
