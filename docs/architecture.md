# Architecture

This repo is split into a **framework** (the runner + scheduler) and **tasks** (specific search problems plugged into it). The autoresearch hyperparameter / architecture search is the first task; the framework is structured so a new task can be added without touching the runtime.

## Three concerns, three locations

| Concern | Location | What lives there |
|---|---|---|
| Framework | `infra/hpc_agent/runner/` | Manager LLM dispatch, worker pool, direct (non-LLM) executor, scheduler, GPU grant lifecycle, live replan, kill, retries, walltime admission, canonical result validation, per-task event trace, cycle reports, structured-edit primitives, persistent compile cache. None of it knows about train.py. |
| Task | `tasks/<name>/` | The runnable code, the wrapper script, the manager + worker prompts, the workspace generator. For autoresearch: `tasks/autoresearch/`. |
| Runs | `artifact/<task>/run-<timestamp>/` (gitignored) | Generated per-run outputs: project DB, state.json, agent skill files, sandbox repos, metrics.json, train.log. Never committed. |

The persistent `torch.compile` cache is one level up at `$HOME/.cache/autoresearch-shared-cache/` so it survives across sbatch runs.

## The framework's contract with a task

A task plugs in by providing five things:

1. **Code that runs end-to-end.** Anything that runs once with a fixed time budget and emits a metric. For autoresearch: `tasks/autoresearch/source/{train.py, prepare.py}` (`prepare.py` is the immutable evaluation harness).
2. **A wrapper script.** Bash (or any shell) that takes an output directory, runs one experiment, and writes:
   - `result.txt` with at minimum `exit_code=<int>`
   - `metrics.json` with at least the metric the manager optimizes (a finite number)
   For autoresearch: `tasks/autoresearch/worker.sh`.
3. **A manager prompt.** Markdown describing the search surface, the metric, the keep/discard semantics, and the TASK_GRAPH schema (including `execution_mode`, `edits[]`, `rationale`, `early_stop`). The framework supplies the TASK_GRAPH protocol; the task teaches the LLM what edits make sense.
4. **A worker prompt.** Markdown telling LLM workers (the `code_edit` fallback path) how to apply edits and run the wrapper.
5. **A workspace generator.** A Node.js function that materializes `config.yaml`, `projects.yaml`, `state.json` (`isPaused: false`), and the per-worker skill files. For autoresearch: `tasks/autoresearch/manager-full.js`.

The `/new-task` slash command scaffolds these from the autoresearch template.

The framework supplies for free:
- 8-way concurrent execution under one Slurm allocation
- GPU token grants with orphan-sweep on restart
- Live replan via watermark gate (no per-completion wake-up storm)
- Manager-issued task kill (`<!-- KILL_TASKS -->`)
- Per-task retry with re-grant
- Walltime admission gate
- Three concentric timeout rings (in-wrapper `timeout`, orchestrator hard cap, abort signal)
- Canonical result validation from `result.txt + metrics.json`
- Structured-edit primitives (`constant_replace`, `regex_replace`, `block_replace`, `unified_diff`) with Python AST normalization
- Direct executor that applies `param_patch` edits without a worker LLM
- Per-task event trace (`task-events/<id>.json`)
- Compact experiment ledger surfacing `rationale → outcome` to the manager
- Persistent shared `torch.compile` cache

## Framework runtime data flow

```
Slurm sbatch
  └─ exec node infra/hpc_agent/runner/src/server.js
       └─ ProjectRunner.runLoop()
            └─ runSingleManagerCycle(config)
                  └─ runAgent(manager) → claude / codex CLI
                       parses TASK_GRAPH + KILL_TASKS from manager output
                  └─ executeSchedule(plan)
                        └─ _executeTaskGraph(plan):
                              while running:
                                pick ready task by (priority, critPath, utility, gpuDemand, runtime)
                                walltime admission: skip if remaining < estRT + 2min
                                _tryGrantWorkerResources → DB INSERT token_grant

                                if step.executionMode === 'param_patch':
                                  _startDirectExecutorTask:
                                    runDirectTask:
                                      prepareDirectSandbox: clone + checkout SHA + apply
                                                            edits[] (4 kinds, AST-normalized)
                                                            + ast.parse-check + commit
                                      runDirectWorker: spawn wrapper detached (own pgroup),
                                                       hard-cap timer, abortSignal, validate
                                else:
                                  _runWorkerStepWithRetries:
                                    runAgent(worker) → claude / codex CLI
                                       worker LLM reads task body, runs sandbox.sh +
                                       worker.sh; orchestrator records canonical truth +
                                       logs LLM-claim mismatch

                                _releaseWorkerGrant on .finally; nvidia-smi drain probe
                                if killed-by-timeout

                              watermark gate: if ready+running < 1.5×maxConcurrent,
                                              call manager again (live replan)
                              on graph drain: build cycle report
```

The orchestrator is **single Node.js process**, **single SQLite handle** (cached, WAL). Workers are detached process groups (`detached: true`) so SIGTERM/SIGKILL on the pgid takes down bash + python descendants.

## Persistence

| Data | Where | Why |
|---|---|---|
| GPU token grants, requests, leases | `<projectDir>/project.db` (SQLite, WAL) | Crash-recoverable; orphan-swept on startup |
| Task graph runtime state | in-memory `plan._runtime` + `<projectDir>/state.json` (50ms-debounced) | Recoverable across sbatch restart |
| Per-task event trace | `<projectDir>/task-events/<task_id>.json` | One file per task; appended on each lifecycle event |
| Per-cycle report | `<projectDir>/cycle-reports/cycle-<ts>.json` | Aggregated time accounting + duty cycle |
| Per-experiment artifacts | `experiments/<exp_id>/{train.log, metrics.json, result.txt, failure.json, sandbox/repo/.git}` | Reproducibility; canonical task outcome |
| Roll-up | `experiments/results.tsv` (flock-protected) | One row per worker exit |
| Shared compile cache | `$HOME/.cache/autoresearch-shared-cache/{inductor,triton,uv,hf,xdg}` | Cross-run warm cache |

## Concurrency model

- **One Node.js event loop** does orchestration, no threads.
- **N=8 worker tasks in parallel** under one Slurm allocation. Each is either:
  - A direct-executor task (a detached `bash worker.sh`) — no LLM in the worker path, applied edits committed deterministically.
  - An LLM-worker task (a detached `claude --print` or `codex exec`) — used for `code_edit` and `llm_repair` modes only.
- **Each worker spawns 1 bash + 1 python** descendant in the same process group. `process.kill(-pgid, ...)` flushes the whole subtree on cancel.
- **DB**: single cached connection; `better-sqlite3` (synchronous), WAL.
- **State writes**: 50ms debounced.
- **Live replan**: fired async on every worker completion but gated by the watermark (default `1.5 × maxConcurrentWorkers = 12`). Below the threshold the manager is woken; above it, completed task ids queue FIFO into the next replan.

## Manager primitives

The manager LLM is given exactly these knobs in its prompt:

| Primitive | How emitted | Runtime behavior |
|---|---|---|
| Schedule N tasks | `<!-- TASK_GRAPH -->{tasks:[…]}<!-- /TASK_GRAPH -->` | parsed → executed |
| Pick execution mode | `execution_mode: "param_patch" \| "code_edit" \| "llm_repair"` | param_patch → direct executor (no LLM); else → worker LLM |
| Structured edits | `edits: [{file, kind, …}]` | applied deterministically by the direct executor |
| Provide reasoning trace | `rationale: "…"` per task | surfaced in next ledger so manager can pattern-match its own past hypotheses |
| Pick parent commit | `base_ref: <SHA \| HEAD>` | resolved server-side; SHA-pinned in the executor |
| Express ordering | `priority: int` per task | scheduler tiebreaker (dominant in practice) |
| Express expected runtime | `estimated_runtime_seconds: int` | rank tiebreaker; walltime admission input |
| Dependencies | `depends_on: [...]` / `depends_on_tags: [...]` | DAG gate before dispatch |
| Mark a barrier | `replan_after: true` | manager called again after the barrier drains |
| Pin a worker | `agent: "maya"` instead of `worker_class: "experiment_runner"` | bypass the worker pool |
| Abort tasks | `<!-- KILL_TASKS -->["exp_0042"]<!-- /KILL_TASKS -->` | SIGTERM worker pgid, mark cancelled, free token |
| Early-stop diverging runs | `early_stop: {check_at_seconds, abort_if_loss_above}` | wrapper watcher SIGTERMs training if loss exceeds threshold |

Everything else (which experiments to try, when to fork from which parent, when to declare a dead branch, what threshold "diverging" means) is up to the LLM.

## Why this shape, briefly

- **Trust the LLM, dumb-down the runner.** Manager has full prior-experiment context (now with rationale), runner enforces feasibility (token availability, dep satisfaction, walltime, hard caps). No bandit / Bayesian / diversity heuristic in runtime.
- **Two execution paths, manager picks.** `param_patch` is the cheap default (no worker LLM, AST-normalized constant replacement, deterministic git lineage). `code_edit` is the LLM fallback for free-form rewrites.
- **Per-experiment git sandbox.** Every non-baseline experiment lives on its own branch in its own clone. Lineage = git history. The direct executor SHA-pins the parent commit so a re-run with the same id is rejected (`experiment_id_already_used`) instead of silently overwriting.
- **Canonical result validation in the orchestrator, not in worker stdout.** The orchestrator parses `result.txt + metrics.json` itself; for direct-executor tasks this IS the source of truth. For LLM-worker tasks the orchestrator still logs LLM-claim/canonical mismatch as instrumentation.
- **Single allocation, no per-worker job submit.** All workers run inside the same Slurm node, share the same allocation, contend for an in-process GPU token pool. Multi-node is deferred until single-node duty cycle is reliably ≥75%.
- **Shared compile cache that survives sbatch boundaries.** `torch.compile`'s cold-compile cost is ~150-200s. Pinning the cache to `$HOME` means cycle-1 of run N reuses kernels compiled by run N-1.
