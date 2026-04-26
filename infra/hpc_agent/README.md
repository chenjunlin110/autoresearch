# Search-Task Framework Runner

The runtime under `runner/` is a generic concurrent search-task scheduler. It
runs an LLM manager that dispatches work to a pool of LLM workers under one
shared GPU allocation.

Current capabilities:
- single-manager + worker-pool orchestration
- DAG scheduler with `worker_class` pooled dispatch + `priority` ranking
- GPU token admission + orphan grant sweep on restart
- live replan triggered by worker completion
- manager-issued task kill (`<!-- KILL_TASKS -->`)
- per-task event trace + per-cycle aggregate report
- canonical result validation from `result.txt` + `metrics.json`
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
