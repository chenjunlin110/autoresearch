# Repository Architecture

This repository now separates three concerns that were previously mixed together at the root:

1. Workload: the actual `autoresearch` task definition and training code.
2. Harness: the generic multi-agent runtime used to execute workloads.
3. Runs: generated artifacts, experiment outputs, and orchestration state.

## Layout

```text
repo/
  autoresearch/
    README.md
    prepare.py
    train.py
    program.md
    results.tsv
    pyproject.toml
    uv.lock

  infra/
    hpc_agent/
      runner/
      docs/

  artifacts/
    ...

  docs/
    architecture.md
    notes/
```

## Meaning of Each Layer

### `autoresearch/`

This directory is the `autoresearch` workload itself:

- `prepare.py`: fixed data prep and evaluation utilities. Do not modify during experiments.
- `train.py`: the file the research agent edits.
- `program.md`: baseline single-agent experiment loop.
- `results.tsv`: official single-agent run ledger.

### `infra/hpc_agent/`

This is the harness layer. It is not the workload. It provides:

- manager/worker orchestration
- task graphs
- resource grants
- live replanning
- worker execution wrappers
- generated multi-agent project templates

Anything under `infra/hpc_agent/runner/` should be treated as runtime infrastructure, not as part of the core `autoresearch` training code.

### `artifacts/`

This directory contains generated run state and outputs:

- manager project workspaces
- experiment outputs
- logs
- plots
- temporary orchestration state

These files are products of a run, not source-of-truth workload definitions.

### `docs/notes/`

This directory contains human notes and migration history:

- `CHANGELOG.md`
- `LOCAL_MULTIAGENT_PLAN.md`
- `PROGRESS_CHECKLIST.md`

These documents are useful context but are not runtime entrypoints.

## Design Intent

The intended composition is:

- `autoresearch` defines the task.
- `hpc_agent` executes and coordinates that task.
- `artifacts/` records what happened during execution.

That separation makes it easier to reason about:

- what is being researched
- what infrastructure is running the research
- what outputs were produced by past runs
