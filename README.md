# Concurrent Search-Task Framework

A small framework for running **complex search tasks** under a shared compute allocation, where the search policy is an LLM and the search execution runs in parallel across many GPUs.

The framework gives you:

1. A **manager LLM** that proposes experiments, given prior results.
2. A **worker pool** plus a **deterministic direct executor** for parameter-patch tasks (no worker LLM, no per-task LLM cost).
3. A **DAG scheduler** with priority, dependencies, watermark-gated live replan, kill primitive, walltime + GPU admission, and three concentric timeout rings.
4. A **per-experiment git sandbox** so concurrent edits don't step on each other; experiment lineage is git history.
5. A **persistent `torch.compile` cache** shared across workers and across sbatch runs.

The first concrete task is [autoresearch](https://github.com/karpathy/autoresearch) (LLM-driven hyperparameter + architecture search on a small GPT trainer). The framework is intended for any search problem that fits the shape:

- there is a body of "main code" that runs and emits a metric
- the search proposes edits to that code (a hyperparameter, a code change, a config)
- you want many edits running concurrently with the manager picking the next batch from results

Examples beyond autoresearch:
- hyperparameter sweeps decoupled from autoresearch's specific knobs
- data-mix search (manager proposes mixture weights → worker runs eval → result feeds next proposal)
- prompt / system-prompt search
- compiler / kernel autotuning where each candidate runs a benchmark

## Repository layout

```text
repo/
  infra/hpc_agent/runner/        # framework: scheduler, runtime, manager/worker dispatch
    src/
      server.js                  # ProjectRunner, scheduler loop, live replan, kill, grants
      orchestration-utils.js     # TASK_GRAPH parser (incl. execution_mode + edits[])
      scheduler.js               # ranking + worker-class binding (pure functions)
      resource-orchestration.js  # SQLite token grant lifecycle
      agent-runner.js            # claude_cli / codex_cli / api runtime adapters
      direct-executor.js         # non-LLM execution path for param_patch tasks
      edits.js + edits-ast.py    # 4-kind structured edits with Python AST normalization
      task-events.js             # per-task lifecycle event log
      result-validator.js        # canonical truth from result.txt + metrics.json
      experiment-ledger.js       # compact "rationale → outcome" memory for the manager
      slurm-walltime.js          # walltime admission gate
      cycle-report.js            # aggregate per-cycle telemetry
      gpu-probe.js               # nvidia-smi drain probe after kill
    tests/                       # 92 unit + integration tests
    agent/                       # framework-level manager/worker rules

  tasks/
    autoresearch/                # the autoresearch task plugin
      source/                    # the executable code: train.py, prepare.py, program.md
      manager-full.js            # workspace generator + manager/worker prompts
      worker.sh                  # bash wrapper (timeout + flock + early-stop watcher)
      sandbox.sh                 # per-experiment git clone (LLM-worker path)
      submit.sbatch              # 8-GPU framework run
      baseline-submit.sbatch     # 1-GPU Karpathy-style baseline run
      tests/                     # task-level smoke test

  artifact/                      # gitignored: per-run outputs (project DB, sandboxes, logs)

  docs/
    architecture.md              # framework architecture
    notes/                       # design notes
```

## Adding a new task

A task plugin needs:

1. **Code that runs end-to-end on its own.** A directory with the executable code and an immutable evaluation harness. For autoresearch this is `tasks/autoresearch/source/{train.py, prepare.py}`.
2. **A wrapper script** that runs one experiment and writes `result.txt` (`exit_code=...`) + `metrics.json` (`{the_metric: <number>}`). For autoresearch: `tasks/autoresearch/worker.sh`.
3. **A manager prompt** describing the search surface, the metric, and how to translate ideas into TASK_GRAPH entries (with `execution_mode`, `edits[]`, `rationale`).
4. **A worker prompt** for the LLM-worker fallback path (`code_edit` mode).
5. **A workspace generator** that materializes `config.yaml`, `projects.yaml`, `state.json`, manager/worker skill files. For autoresearch: `tasks/autoresearch/manager-full.js`.

The framework supplies the scheduler, GPU grant lifecycle, kill, replan, retry, walltime admission, canonical result validation, per-task event tracing, the direct executor, the structured-edit primitives, and the persistent compile cache — none of it is autoresearch-specific.

The `/new-task` slash command in this repo's `.claude/skills/` scaffolds a fresh task plugin from the autoresearch template.

## Quick start

```bash
git clone <this-repo> && cd autoresearch && git checkout framework

cd infra/hpc_agent/runner && npm install && npm test    # 92 tests, ~40s

# Launch the autoresearch task on Slurm: 8 GPUs × 2h, claude_cli runtime
cd ../../..
AUTORESEARCH_AGENT_RUNTIME=claude_cli sbatch --time=02:00:00 \
  --export=ALL,AUTORESEARCH_AGENT_RUNTIME \
  tasks/autoresearch/submit.sbatch

# Or compare against the original Karpathy single-agent design: 1 GPU × 1h
sbatch tasks/autoresearch/baseline-submit.sbatch
```

Per-run outputs land under `artifact/autoresearch/run-<timestamp>/`. The persistent `torch.compile` cache lives at `$HOME/.cache/autoresearch-shared-cache/` so each fresh sbatch reuses kernels from prior runs (cycle-1 is warm).

## Status

The redesign described in `docs/notes/` (six phases: observability, timeouts, direct executor + structured edits, shared compile cache, watermark replan + compact ledger, refactor) is **landed**. End-to-end verified on 1h sbatch:

- ≥90% of manager-emitted tasks dispatch via the direct executor (no worker LLM)
- ~50% reduction in manager Opus calls per worker completion (watermark gate)
- Canonical truth-mismatch instrumentation in place
- 92 / 92 tests passing

See `docs/architecture.md` for the framework contract and runtime data flow.
