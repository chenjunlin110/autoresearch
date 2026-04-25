#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd "$script_dir/../../../.." && pwd)"
source_root="${1:-$workspace_root/autoresearch}"
dest_root="${2:-}"
source_label="${3:-$source_root}"
branch_name="${4:-}"

if [[ -z "$dest_root" ]]; then
  echo "usage: $0 <source_root> <dest_root> [source_label] [branch_name]" >&2
  exit 1
fi

mkdir -p "$(dirname "$dest_root")"

if [[ ! -d "$source_root/.git" ]]; then
  echo "error: $source_root is not a git repository" >&2
  exit 1
fi

git clone -q --no-hardlinks "$source_root" "$dest_root"

cat > "$dest_root/.autoresearch-sandbox.json" <<EOF
{
  "source_root": "$(printf '%s' "$source_root" | sed 's/"/\\"/g')",
  "source_label": "$(printf '%s' "$source_label" | sed 's/"/\\"/g')"
}
EOF

(
  cd "$dest_root"
  git config user.name "autoresearch-sandbox"
  git config user.email "autoresearch-sandbox@example.com"
  git add .autoresearch-sandbox.json
  git commit -q -m "annotate sandbox source $source_label" >/dev/null 2>&1 || true

  if [[ -n "$branch_name" ]]; then
    git checkout -q -B "$branch_name"
  fi
)
