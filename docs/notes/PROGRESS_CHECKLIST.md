# Progress Checklist

## Current Snapshot

- Date: 2026-04-24
- Focus: single-allocation, resource-aware, manager/worker orchestration for `autoresearch`
- Runtime baseline: one shared Slurm allocation, overlap worker steps, GPU token grants
- Local agent runtime: `codex_cli` by default, `api` as compatibility fallback
- Architecture: runner-only (`backend` / frontend / diamond removed)

## Milestone Status

### M1. Shared Allocation Baseline

- [x] Define shared-allocation architecture and execution semantics
- [x] Persist shared allocation lease state in project SQLite
- [x] Seed project-scoped GPU token inventory
- [x] Expose resource orchestration status in runner state
- [x] Add CLI support for allocation / token lifecycle operations
- [x] Add `HpcSubmit` shared-allocation probe path
- [x] Add `HpcSubmit` shared-allocation worker-step attach path
- [x] Validate real `srun --jobid <JOB> --overlap` execution on a fresh 8-GPU allocation
- [x] Validate deterministic GPU placement via `CUDA_VISIBLE_DEVICES`
- [x] Re-run shared-allocation probe / attach after runner-only cleanup
- [x] Re-run one real `autoresearch` worker task through `shared_step`

Status: complete for current runner-only baseline

### M2. Single Manager + Multiple Workers

- [x] Add a dedicated single-manager orchestration mode
- [x] Bypass the current `athena -> ares -> apollo -> themis` phase chain in this mode
- [x] Define manager prompt behavior for resource arbitration
- [x] Define worker prompt behavior for resource request / release flow
- [x] Support one manager issuing multiple worker assignments in one cycle
- [x] Replace sequential worker execution with a true worker-pool execution model
- [x] Run worker agents through local `codex_cli` runtime without API keys
- [x] Add DAG-style task-graph orchestration with dependency-aware ready-queue scheduling
- [x] Add `worker_class` task dispatch so ready tasks bind to idle worker instances
- [x] Add graph-drain refill so the manager can continue optimization waves instead of stopping after one summary

Status: code path implemented; scheduler now supports pooled worker classes and repeated manager waves after graph drain

### M3. Token-Based GPU Scheduler

- [x] Persist token request / grant / release lifecycle
- [x] Enforce exclusive GPU-token assignment in the DB layer
- [x] Validate grant-aware worker-step launch against the project DB
- [x] Resolve granted GPU tokens into worker-visible GPU ordinals
- [x] Add backfill queue semantics for GPU-constrained parallel worker batches
- [x] Add scheduler ranking by priority, critical path, utility, GPU demand, and estimated runtime
- [x] Split pure scheduler policy into `src/scheduler.js` with focused tests
- [ ] Add CPU token inventory
- [ ] Add RAM-aware admission control
- [ ] Add queued waiting-task scheduling
- [ ] Add manager-driven deny / reprioritize flow
- [ ] Add automatic release hooks tied to orchestrator task lifecycle

Status: partial; DB/control-plane complete, scheduler policy incomplete

### M4. Autoresearch Experiment Interface

- [x] Add cluster-safe compile fallback via `AUTORESEARCH_DISABLE_COMPILE=1`
- [x] Validate real `autoresearch` runs inside shared overlap steps
- [x] Add structured config overrides for experiments
- [x] Emit structured results such as `metrics.json`
- [x] Add stable run directory layout owned by the orchestrator
- [x] Add repeatable 8-GPU full-run script with metrics summary
- [x] Add direct `sbatch` wrapper for non-sandboxed full 8-GPU runs
- [ ] Add helper scripts for summarizing experiment history

Status: mostly complete for single-run orchestration; history summarization still pending

### M5. End-to-End Closed Loop

- [x] Manager proposes experiments
- [x] Workers request resources through the manager
- [x] Scheduler admits work based on available resources
- [x] Workers execute and release resources
- [x] Result analyst updates summaries / leaderboards
- [x] Manager consumes results and schedules the next iteration

Status: minimal DAG closed loop validated with one live `autoresearch` experiment, one analyst task, same-cycle replan, and `PROJECT_COMPLETE`

## Validation Checklist

- [x] `infra/hpc_agent/runner` unit tests pass
- [x] New orchestration-utils unit tests pass
- [x] New codex-cli runtime unit tests pass
- [x] Direct `runAgentWithCodexCLI()` smoke passes locally
- [x] Real runner project smoke passes with `agentRuntime: codex_cli` and no API key
- [x] Formal live manager/worker smoke passes with `agentRuntime: codex_cli`
- [x] Shared-allocation probe path tested against real Slurm job `1575847`
- [x] Shared-allocation worker-step path tested against real Slurm job `1575847`
- [x] 8-way `autoresearch` concurrency tested on one shared allocation
- [x] Shared-allocation probe path re-tested against fresh Slurm job `1575902`
- [x] Concurrent `shared_step` probe re-tested against fresh Slurm job `1575902`
- [x] Single-worker `autoresearch` shared-step re-tested against fresh Slurm job `1575902`
- [x] Direct `autoresearch` live run succeeds on the 8-GPU node
- [x] `run-autoresearch-worker.sh` GPU smoke writes `metrics.json` and `result.txt`
- [x] Corrected full 8-GPU direct launch succeeds with 8/8 workers
- [x] `TASK_GRAPH` parser and DAG scheduling tests pass
- [x] Manager-generated `autoresearch` manager/worker end-to-end orchestration test
- [x] `autoresearch` DAG smoke templates generate repo-relative output dirs without duplicated artifact prefixes
- [x] `worker_class` task parsing and critical-path score tests pass
- [x] Scheduler module tests pass for worker-class inference, binding, and rank selection
- [x] Previous local logs/artifacts cleared before full-run attempt
- [x] Confirm current Codex tool sandbox blocks both `srun` and `sbatch` stream sockets
- [ ] Token leak / orphaned-process recovery test
- [ ] Worker queue fairness test

## Immediate Next Steps

- [x] Re-run live 8-GPU `autoresearch` with 8 pooled experiment runners
- [ ] Add event-driven graph append while long-running workers are still active
- [ ] Add manager prompt + workflow for deny / reprioritize decisions
- [ ] Add CPU tokens alongside GPU tokens
