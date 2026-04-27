# Search-Task Framework Runner

The runtime under `runner/` is a generic concurrent search-task scheduler. It
runs an LLM manager that proposes experiments and dispatches them either via
a deterministic direct executor (no per-task LLM cost) or a worker-LLM
fallback path, all under one shared GPU allocation.

Current capabilities:
- single-manager + worker-pool orchestration
- DAG scheduler with `worker_class` pooled dispatch + `priority` ranking
- two execution paths the manager picks per task:
  - `param_patch` — structured `edits[]` applied by the direct executor with
    no worker LLM (constant_replace with Python AST normalization,
    regex_replace with mandatory match-count, block_replace, unified_diff)
  - `code_edit` / `llm_repair` — worker LLM rewrites code in natural language
- GPU token admission + orphan grant sweep on restart
- watermark-gated live replan (manager only wakes when ready+running queue
  drops below `1.5 × maxConcurrentWorkers`)
- manager-issued task kill (`<!-- KILL_TASKS -->`)
- three concentric timeout rings: in-wrapper `timeout` around train.py, a
  900s outer hard cap on the orchestrator side, manager-decided per-task
  early-stop on training loss
- per-task event trace + per-cycle aggregate report
- compact experiment ledger that surfaces `rationale → outcome` to the
  manager so it can pattern-match its own past hypotheses
- canonical result validation from `result.txt` + `metrics.json`; logs
  LLM-claim/canonical mismatch as instrumentation
- nvidia-smi drain probe after a kill before reissuing freed tokens
- persistent shared `torch.compile` cache at
  `$HOME/.cache/autoresearch-shared-cache/` so cycle-1 of run N reuses
  kernels compiled by run N-1
- agent runtimes: `codex_cli` (default), `claude_cli`, `api`

## Setup

```bash
bash setup.sh
```

This checkout is self-contained; no external upstream checkout is required.

## Running a task

Each task plugin lives at `<repo>/tasks/<name>/` and supplies its own sbatch
entrypoint. Submit a task as:

```bash
sbatch tasks/<name>/submit.sbatch
```

For the autoresearch task specifically:

```bash
# default runtime is codex_cli
sbatch tasks/autoresearch/submit.sbatch

# or use Claude Code as the manager + worker runtime
AUTORESEARCH_AGENT_RUNTIME=claude_cli sbatch \
  --export=ALL,AUTORESEARCH_AGENT_RUNTIME \
  tasks/autoresearch/submit.sbatch

# Karpathy-style 1-GPU single-agent baseline (for comparison)
sbatch tasks/autoresearch/baseline-submit.sbatch
```

Each run writes to `artifact/<task>/run-<timestamp>/` (gitignored). Logs land
in `artifact/<task>/run-<timestamp>/server.log`.

Notes:

- Codex sandbox cannot open Slurm stream sockets on this cluster, so the
  runner executes directly on the GPU compute node via sbatch — there is no
  login-node + `srun --overlap` attach path.
- Provider API keys are only needed for projects that set `agentRuntime: api`.
- Scheduler design: `runner/docs/SCHEDULER.md`
- Codex runtime notes: `runner/docs/CODEX_LOCAL_RUNTIME.md`
- Claude Code runtime notes: `runner/docs/CLAUDE_LOCAL_RUNTIME.md`
- Resource validation plan: `runner/docs/RESOURCE_ORCHESTRATION_TEST_PLAN.md`

## Adding a new task

See [`docs/architecture.md`](../../docs/architecture.md) for the five-piece
task plugin contract, or run the `/new-task` skill from Claude Code to
scaffold a task directory from the autoresearch template.
