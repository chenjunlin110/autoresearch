# Architecture

This repo is split into a **framework** (the runner + scheduler) and **tasks** (specific search problems plugged into it). Today the only landed task is the autoresearch hyperparameter / architecture search; the framework is structured so a new task can be added without touching the runtime.

## Three concerns, three locations

| Concern | Location | What lives there |
|---|---|---|
| Framework | `infra/hpc_agent/runner/` | Manager LLM dispatch, worker pool, scheduler, GPU grant lifecycle, live replan, kill, retries, walltime admission, canonical result validation, per-task event trace, cycle reports. None of it knows about train.py. |
| Task | `autoresearch/`<br>+ task-specific files inside `infra/hpc_agent/runner/` | The runnable code (`train.py`, `prepare.py`), the manager prompt, the worker prompt, the wrapper script, the workspace generator. A planned cleanup moves all task-specific files into `tasks/<name>/`. |
| Runs | `artifacts/` (gitignored), `experiments/<exp_id>/` (gitignored) | Generated per-run outputs: project DB, state.json, agent skill files, sandbox repos, metrics.json, train.log. Never committed. |

## The framework's contract with a task

A task plugs in by providing five things:

1. **Code that runs end-to-end.** Anything that can be invoked once with a fixed time budget and emits a metric. For autoresearch: `autoresearch/train.py` + the immutable harness in `prepare.py`.
2. **A wrapper script.** Bash (or any shell) that takes an output directory, runs one experiment, and writes:
   - `result.txt` containing at minimum `exit_code=<int>`
   - `metrics.json` containing at least the metric the manager optimizes (a finite number)
   For autoresearch: `infra/hpc_agent/runner/scripts/run-autoresearch-worker.sh`.
3. **A manager prompt.** Markdown describing the search surface, the metric, the keep/discard semantics, and the TASK_GRAPH schema. The framework already supplies the TASK_GRAPH protocol; the task's job is to teach the LLM what edits make sense.
4. **A worker prompt.** Markdown telling worker LLMs how to apply manager-described edits, run the wrapper, and report. The framework already supplies the resource-grant block injection; the task's job is to spell out the specific shell idioms.
5. **A workspace generator.** A small Node.js function that materializes the per-task workspace: `config.yaml`, `projects.yaml`, `state.json` (`isPaused: false`), per-worker skill files, and the manager + worker prompts. For autoresearch: `infra/hpc_agent/runner/src/autoresearch-dag-full.js`.

That's the entire plugin surface. Once a task supplies these five, the framework gives it for free:
- 8-way concurrent execution under one Slurm allocation
- GPU token grants with orphan-sweep on restart
- Live replan as soon as any worker finishes (no wave barriers)
- Manager-issued task kill (`<!-- KILL_TASKS -->`)
- Per-task retry with re-grant
- Walltime admission gate
- Canonical result validation from `result.txt + metrics.json`
- Per-task event trace and per-cycle aggregate report

## Framework runtime data flow

```
Slurm sbatch
  └─ exec node infra/hpc_agent/runner/src/server.js
       └─ ProjectRunner.runLoop()
            └─ runSingleManagerCycle(config)
                  └─ runAgent(manager)         → claude / codex CLI
                       parses TASK_GRAPH + KILL_TASKS from manager output
                  └─ executeSchedule(plan)
                        └─ _executeTaskGraph(plan):
                              while running:
                                pick ready task by (priority, critPath, utility, gpuDemand, runtime)
                                walltime admission: skip if remaining < estRT + 2min
                                _tryGrantWorkerResources → DB INSERT token_grant
                                _runWorkerStepWithRetries:
                                   buildManagedWorkerTask injects grant_id, gpu_ordinals
                                   runAgent(worker) → claude / codex CLI
                                      worker LLM reads task body, runs:
                                        prepare-autoresearch-sandbox.sh
                                        git checkout -B <exp_id>; edit; commit
                                        bash run-autoresearch-worker.sh <output_dir>
                                          → uv run train.py
                                          → metrics.json + result.txt + flock results.tsv
                                   _recordCanonicalValidation: parse files, log mismatch
                                _releaseWorkerGrant on .finally
                              if liveReplanOnTaskComplete and others still running:
                                _maybeLiveReplanTaskGraph → manager LLM appends tasks
                              on graph drain: cycle finalizes, build cycle report
```

The orchestrator is **single Node.js process**, **single SQLite handle** (cached, WAL). Concurrency is async I/O on the event loop, not threads. Task workers are detached process groups (`setsid`) so SIGTERM on the LLM CLI takes down its bash + python descendants.

## Persistence

| Data | Where | Why |
|---|---|---|
| GPU token grants, requests, leases | `<projectDir>/project.db` (SQLite, WAL) | Crash-recoverable; orphan-swept on startup |
| Task graph runtime state | in-memory `plan._runtime` + `<projectDir>/state.json` (50ms-debounced) | Recoverable across sbatch restart |
| Per-task event trace | `<projectDir>/task-events/<task_id>.json` | One file per task; appended on each lifecycle event |
| Per-cycle report | `<projectDir>/cycle-reports/cycle-<ts>.json` | Aggregated time accounting + duty cycle |
| LLM prompt + response (debug) | `<projectDir>/responses/<agent>.log` | Audit and debugging only — not source of truth |
| Per-experiment artifacts | `experiments/<exp_id>/{train.log, metrics.json, result.txt, sandbox/repo/.git}` | Reproducibility; canonical task outcome |
| Roll-up | `experiments/results.tsv` (flock-protected) | One row per worker exit |

## Concurrency model

- **One Node.js event loop** does orchestration, no threads.
- **N=8 worker LLMs in parallel** (each a detached `claude --print` or `codex exec` subprocess).
- **Each worker spawns 1 bash + 1 python (train.py)** descendant in the same process group.
- **Cancel propagation**: `process.kill(-pid, 'SIGTERM')` on the LLM kills bash + python descendants.
- **DB**: single cached connection; better-sqlite3 (synchronous), WAL.
- **State writes**: 50ms debounced.
- **Live replan**: fired async on every worker completion; queued behind any in-flight replan (FIFO of triggering task ids).

## Manager primitives

The manager LLM is given exactly these knobs in its prompt:

| Primitive | How emitted | Runtime behavior |
|---|---|---|
| Schedule N tasks | `<!-- TASK_GRAPH -->{tasks:[…]}<!-- /TASK_GRAPH -->` | parsed → executed |
| Express ordering | `priority: int` per task | scheduler tiebreaker (dominant in practice) |
| Express expected runtime | `estimated_runtime_seconds: int` | rank tiebreaker; walltime admission gate input |
| Dependencies | `depends_on: [...]` / `depends_on_tags: [...]` | DAG gate before dispatch |
| Mark a barrier | `replan_after: true` | manager called again after the barrier drains |
| Pin a worker | `agent: "maya"` instead of `worker_class: "experiment_runner"` | bypass the worker pool |
| Abort tasks | `<!-- KILL_TASKS -->["exp_0042"]<!-- /KILL_TASKS -->` | SIGTERM worker pgid, mark cancelled, free token |

Everything else (which experiments to try, when to fork from which parent, when to declare a dead branch) is up to the LLM.

## Why this shape, briefly

- **Trust the LLM, dumb-down the runner.** Manager has full prior-experiment context, runner enforces feasibility (token availability, dep satisfaction, walltime). No bandit / Bayesian / diversity heuristic in runtime.
- **Per-experiment git sandbox.** Every non-baseline experiment lives on its own branch in its own clone of the searchable code. Lineage = git history. Keep/discard = the manager forks future experiments from a kept parent's HEAD or skips a discarded branch.
- **Canonical result validation in the orchestrator, not in worker stdout.** Since v1.5 the orchestrator parses `result.txt + metrics.json` itself; worker stdout is debug only. (Phase 3 of the redesign upgrades this from "log mismatch" to "enforce canonical truth".)
- **Single allocation, no per-worker job submit.** All workers run inside the same Slurm node, share the same allocation, and contend for an in-process GPU token pool. Multi-node scheduling is deferred until single-node duty cycle is ≥75% (today: 34%).

## Where the redesign goes next

The approved redesign plan is in [`.claude-resume/redesign-plan.md`](../.claude-resume/redesign-plan.md). Phase 0 (lease + blocked log) and Phase 1 (event trace, walltime gate, result validator, cycle report) are landed. The biggest remaining lever is **Phase 4: shared content-hashed `torch.compile` cache**, which is the actual throughput unlock — but it can only safely deploy after **Phase 3** introduces the deterministic `param_patch` executor (which gives you a stable AST hash to key the cache on).

## Future cleanup: explicit `tasks/` directory

A planned next step that reduces the autoresearch-specific surface in the framework directory:

```text
runner/                       # was: infra/hpc_agent/runner/
  src/                        # framework only
  scripts/                    # framework only
  tests/                      # framework only

tasks/
  autoresearch/
    train.py                  # was: autoresearch/train.py
    prepare.py
    program.md
    manager.js                # was: infra/hpc_agent/runner/src/autoresearch-dag-full.js
    worker.sh                 # was: infra/hpc_agent/runner/scripts/run-autoresearch-worker.sh
    submit.sbatch             # was: infra/hpc_agent/runner/scripts/submit-autoresearch-runner.sbatch
  <future-task>/
    ...
```

Deferred until the v1.5 redesign is functional, since it touches every config path and would need cluster verification.
