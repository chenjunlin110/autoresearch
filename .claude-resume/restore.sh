#!/usr/bin/env bash
# Drop this repo's plan + Claude session into ~/.claude/ so `claude --resume`
# picks up the autoresearch HPC agent redesign exactly where it left off.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

claude_home="${CLAUDE_HOME:-$HOME/.claude}"
plans_dir="$claude_home/plans"

# Claude Code derives the per-project session directory from the absolute path
# to the repo by replacing every "/" with "-".
hashed_cwd="$(printf '%s' "$repo_root" | sed 's|/|-|g')"
project_dir="$claude_home/projects/$hashed_cwd"

mkdir -p "$plans_dir" "$project_dir"

cp "$script_dir/redesign-plan.md" "$plans_dir/autoresearch-hpc-agent-kind-zephyr.md"
cp "$script_dir/session.jsonl" "$project_dir/fd6465ef-7937-4ad1-8942-08b698990432.jsonl"

cat <<EOF
Imported:
  plan      -> $plans_dir/autoresearch-hpc-agent-kind-zephyr.md
  session   -> $project_dir/fd6465ef-7937-4ad1-8942-08b698990432.jsonl

Resume with:
  claude --resume fd6465ef-7937-4ad1-8942-08b698990432
EOF
