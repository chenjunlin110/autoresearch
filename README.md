# Repository Layout

This repository now separates the workload from the harness:

```text
repo/
  autoresearch/
    README.md
    program.md
    prepare.py
    train.py
    results.tsv

  infra/
    hpc_agent/
      runner/

  artifacts/
  docs/
```

- `autoresearch/` is the workload: training code, experiment protocol, and baseline ledger.
- `infra/hpc_agent/` is the orchestration harness: manager/worker runtime, scheduler, and resource control.
- `artifacts/` contains generated run outputs.
- `docs/` contains architecture notes and migration history.

Start with [autoresearch/README.md](/mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/README.md) for the workload itself, or [docs/architecture.md](/mnt/weka/home/junlin.chen/workspace/autoresearch/docs/architecture.md) for the repo-level structure.
