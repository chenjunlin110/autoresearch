#!/usr/bin/env bash
# HPC Agent — one-time setup script
# Usage: bash setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== HPC Agent Setup ==="
echo ""

# ── Runner setup ──────────────────────────────────────────────────────────────
echo "[1/2] Preparing runner..."
cd "$SCRIPT_DIR/runner"
node scripts/setup.js

echo "[2/2] Installing runner dependencies..."
npm install

if [ ! -f .env ]; then
  cp .env.example .env
  echo "      Created runner/.env"
fi

cd "$SCRIPT_DIR"

# ── Root .env ─────────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
  cp .env.example .env
  echo "      Created .env"
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To start:"
echo "  Terminal 1: cd runner && npm start"
echo ""
echo "Default local runtime is codex_cli. API keys are optional unless you switch a project to agentRuntime=api."
