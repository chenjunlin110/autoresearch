# Single-Allocation Resource-Aware Multi-Agent Orchestration Plan

## Objective

Build a local-control-plane orchestration mode with these properties:

- The system acquires one shared Slurm allocation for the project.
- One manager agent coordinates multiple worker agents.
- Workers do not submit independent jobs by default.
- Workers enter the shared allocation with `--overlap` and explicit GPU binding.
- Resource assignment is token-based and manager-controlled.
- The system can drive `autoresearch` experiments as managed worker tasks.

## Status Snapshot

As of 2026-04-24:

- `hpc_agent` has been simplified to a runner-only architecture.
- Legacy `backend/`, frontend/monitor code, and diamond-specific code have been removed.
- M1 shared-allocation baseline is implemented and validated on live Slurm allocations.
- M2 single-manager, multi-worker orchestration is implemented for the generic runtime path and validated with live `codex_cli` smoke.
- M3 now has a pure scheduler policy module for worker-class binding and DAG ready-queue selection.
- `worker_class` lets the manager submit pooled work without hard-coding a specific agent.
- Graph-drain refill lets the manager continue optimization waves after successful DAG completion.
- `autoresearch` now supports env-driven experiment overrides plus structured `metrics.json` output.
- The next unfinished milestone is a full live 8-GPU pooled `autoresearch` rerun under the current scheduler.

## Scope

In scope for the first implementation:

- Single shared Slurm allocation per project.
- One manager + many workers execution model.
- Token-based resource requests and grants.
- GPU-aware worker scheduling on an 8-GPU node.
- Local-on-node experiment execution for `autoresearch`.
- Structured experiment results and experiment history.

Out of scope for the first implementation:

- Per-worker independent remote job submission as the main path.
- Globus Compute integration in the default path.
- Multi-node distributed training.
- Complex UI redesign.

## Current State

The current `hpc_agent` implementation is now aligned with the local shared-allocation direction:

1. The real orchestrator lives in `infra/hpc_agent/runner/`.
2. `single_manager` bypasses the legacy `athena -> ares -> apollo -> themis` phase chain.
3. Worker schedules can run concurrently through `parallel` groups or DAG ready queues.
4. `TASK_GRAPH` supports dependencies, produced tags, `worker_class`, priority, utility, estimated runtime, and replan barriers.
5. `HpcSubmit` supports shared-allocation probe/attach flows.
6. Resource lease / token lifecycle exists in the project DB and GPU token admission is enforced by the runner.
7. `autoresearch/train.py` accepts environment-driven overrides and emits structured metrics.

Remaining design gaps:

1. Manager policy is still prompt-driven; deny / reprioritize decisions are not a first-class API yet.
2. CPU and RAM are not enforced as tokens.
3. The scheduler refills after a graph drains, but it does not yet append new manager work while long tasks are still running.
4. Recovery tests for orphaned processes and token leaks are still pending.

## Design Decisions

### 1. Use `runner/` as the core runtime

The local multi-agent system should be built on top of `infra/hpc_agent/runner/`.

Reason:

- `runner/src/server.js` already owns project state, agent prompts, reports, issues, PR records, and orchestration.
- The current shared-allocation runtime, token DB, and Slurm attach path already live there.

### 2. Introduce a single-manager orchestration mode

Add a new runtime mode, for example:

- `orchestration.mode: shared_allocation_manager_workers`

In this mode:

- There is exactly one manager role.
- There is a pool of worker roles.
- The current multi-phase manager chain is bypassed.
- The manager is the resource arbiter.

### 3. Replace job-per-worker submission with allocation + overlap execution

The execution model should be:

- acquire one Slurm allocation for the project
- keep the allocation alive as a project-scoped lease
- workers run commands inside that allocation with `srun --jobid <jobid> --overlap ...`
- worker launch includes explicit GPU placement via manager-granted tokens

This is still local-first in control-plane terms:

- one orchestrator
- one shared node allocation
- no remote job fan-out per worker

### 4. Introduce token-based resource management

Workers do not directly decide which GPU they use.

Each worker submits a resource request to the manager. The manager grants or rejects requests based on:

- available GPU tokens
- current active workers
- expected run duration
- worker priority
- experiment value
- optional fairness rules

For the first implementation, the token model should be simple:

- one exclusive token per GPU
- 8 total GPU tokens on the target node
- a worker receives one or more explicit GPU IDs
- the worker must release the token on completion or failure

Future extension:

- memory-class tokens
- fractional GPU tokens
- CPU / RAM tokens
- preemption

### 5. Add an allocation manager as a first-class runtime component

There should be a project-scoped allocation manager responsible for:

- acquiring the shared Slurm allocation
- storing the active `job_id`
- checking allocation liveness
- launching overlapped worker tasks
- cleaning up on shutdown
- exposing allocation state to the scheduler and status APIs

### 6. Make scheduling resource-aware

Every worker task should be admitted through a scheduler that understands:

- CPU slots
- RAM budget
- GPU token ownership
- GPU assignment
- Optional API/model concurrency budget

The manager decides policy. The scheduler enforces feasibility.

### 7. Make `autoresearch` orchestration-friendly

`autoresearch` needs a stable experiment interface:

- structured config input
- structured result output
- stable artifact paths
- explicit run IDs

Without that, agents will keep mutating source constants and parsing ad hoc logs.

## Execution Semantics

### Submit Semantics

`submit` should no longer mean:

- each worker creates its own Slurm job

Instead, `submit` should mean:

- the project acquires or refreshes one shared allocation lease

For example:

- manager or allocation manager obtains one job allocation
- all worker compute tasks attach to that allocation

### Worker Execution Semantics

Default worker compute path:

1. worker prepares a request
2. manager evaluates request against current token state
3. manager grants token set
4. worker launches inside the shared allocation with overlap
5. worker releases tokens on exit

Concrete launch form:

```bash
srun --jobid <JOB_ID> --overlap bash -lc 'CUDA_VISIBLE_DEVICES=<GPU_IDS> <COMMAND>'
```

The exact wrapper may change, but the semantics should stay the same:

- same allocation
- overlap execution
- deterministic GPU binding

### Token Protocol

Minimum request fields:

- `worker`
- `task_type`
- `num_gpus`
- `estimated_duration_minutes`
- `priority`
- `reason`

Minimum grant fields:

- `grant_id`
- `worker`
- `gpu_ids`
- `token_ids`
- `lease_job_id`
- `expires_at`

Release reasons:

- `completed`
- `failed`
- `cancelled`
- `expired`

### Initial Token Model

For the first implementation:

- 8 physical GPU tokens
- token `gpu0` through `gpu7`
- exclusive assignment
- one worker can hold one or more tokens
- no fractional sharing in v1

This keeps the first scheduler simple and auditable.

## Target Architecture

### A. Orchestrator

`infra/hpc_agent/runner/src/server.js`

Responsibilities:

- project loop
- manager run
- worker queue
- allocation state
- task admission control
- run state persistence

### B. Agent Runtime

`infra/hpc_agent/runner/src/agent-runner.js`

Responsibilities:

- model loop
- local tool execution
- task tool dispatch
- cancellation and timeout

### C. Allocation Manager

Current layout:

- lease and token state: `infra/hpc_agent/runner/src/resource-orchestration.js`
- shared allocation attach/probe: `infra/hpc_agent/runner/src/hpc-tool.js`

Responsibilities:

- store active Slurm job ID
- seed and account GPU tokens
- enter allocation with overlap
- expose resource summary to the runner

Planned:

- extract allocation liveness / renewal into a dedicated allocation-manager module if this grows beyond the current control-plane helpers

### D. Scheduler

Current layout:

- `infra/hpc_agent/runner/src/scheduler.js`
- `infra/hpc_agent/runner/src/orchestration-utils.js`

Responsibilities:

- infer worker classes from worker metadata
- bind `worker_class` tasks to idle worker instances
- rank ready DAG tasks by priority, critical path, utility, GPU demand, and estimated runtime
- enforce GPU feasibility through token availability before launch

Planned:

- CPU/RAM tokens
- aging / starvation prevention
- event-driven DAG append while workers are still running

### E. Worker Launcher

Current layout:

- grant-aware task injection: `infra/hpc_agent/runner/src/orchestration-utils.js`
- process execution: `infra/hpc_agent/runner/src/agent-runner.js`
- shared Slurm step attach: `infra/hpc_agent/runner/src/hpc-tool.js`

Responsibilities:

- transform a manager grant into a concrete worker command
- run worker command via overlap inside the active allocation
- inject `CUDA_VISIBLE_DEVICES`
- capture stdout / stderr / exit status
- release tokens on exit

### F. Autoresearch Adapter

Current layout:

- `autoresearch/train.py` with config overrides
- `infra/hpc_agent/runner/scripts/run-autoresearch-worker.sh`
- per-run artifact directories with `metrics.json`, `result.txt`, `train.log`, and `run.env`

Responsibilities:

- start a run from a structured config
- emit structured metrics
- keep reproducible artifacts

## Proposed Worker Roles

For the first implementation:

1. `manager`
   - plans work
   - inspects current results
   - receives worker resource requests
   - grants GPU tokens
   - decides priorities under resource limits

2. `experiment_runner`
   - launches `autoresearch` training/eval jobs
   - requests tokens from manager
   - runs inside the shared allocation
   - records artifacts and status

3. `result_analyst`
   - reads completed run outputs
   - updates leaderboard / summary
   - recommends next experiments

4. `code_worker`
   - modifies code/config/scripts when the manager decides the system itself needs changes

4. `code_worker`
   - modifies code/config/scripts when the manager decides the system itself needs changes

The first implementation does not need many worker types. It needs correct coordination plus correct token accounting.

## Milestones

### M1. Shared Allocation Baseline

Goal:

- Acquire one shared Slurm allocation for the project.
- Track that allocation in project state.
- Run one manager and one worker against that allocation.

Deliverables:

- project-scoped allocation lease in SQLite
- project-scoped GPU token inventory and request / grant / release lifecycle
- `HpcSubmit` `shared_probe` path
- `HpcSubmit` `shared_step` path using `--overlap`
- local state exposure for active `job_id` and token summary

Acceptance:

- the system can probe a live allocation and launch a worker inside it with `--overlap`
- a granted worker sees only its assigned GPU ordinals
- `autoresearch` can run as a worker task inside the shared allocation

Status:

- Complete for the runner-only baseline.
- Validated on fresh shared allocations `1575847` and `1575902`.

### M2. Single Manager + Multiple Workers

Goal:

- Collapse the current phase-manager structure into a single-manager workflow for this mode.
- Support multiple workers per manager cycle.
- Route all worker launch decisions through the manager.

Deliverables:

- manager prompt/profile for local orchestration mode
- worker pool discovery for this mode
- schedule format that can express multiple worker runs
- resource request / grant schema

Acceptance:

- one manager can assign at least three workers in a single cycle

### M3. Token-Based GPU Scheduler

Goal:

- Ensure worker runs are admitted by available tokens inside the shared allocation.

Deliverables:

- 8-GPU token inventory
- per-task token request format
- grant / deny / release flow
- queue for waiting tasks
- GPU assignment and release logic

Acceptance:

- tasks that cannot obtain tokens are queued instead of started
- granted tasks receive deterministic GPU assignments
- tokens are released reliably on process exit

### M4. Autoresearch Experiment Interface

Goal:

- Make `autoresearch` runnable as a structured experiment task.

Deliverables:

- config override mechanism for `train.py`
- structured output file such as `metrics.json`
- stable experiment directory layout

Acceptance:

- a worker can launch an `autoresearch` experiment from a config file and produce structured metrics

### M5. End-to-End Closed Loop

Goal:

- Manager proposes experiments.
- Workers request tokens and execute under resource limits.
- Results are analyzed and fed back into the next cycle.

Deliverables:

- experiment queue
- leaderboard / summary artifact
- manager prompt tuned for experiment iteration
- token-aware experiment request policy

Acceptance:

- the system can complete at least one full local optimization loop without manual intervention

## File-Level Implementation Status

### `infra/hpc_agent/runner/src/server.js`

Implemented:

- shared-allocation orchestration mode
- single-manager mode
- active run tracking for concurrent workers
- DAG runtime state and manager replan handoff

### `infra/hpc_agent/runner/src/agent-runner.js`

Implemented:

- keep current tool loop
- run agents through `codex_cli` or API runtime
- keep existing file/tool sandbox rules

### `infra/hpc_agent/runner/src/hpc-tool.js`

Implemented:

- shared allocation probe
- grant-aware overlap worker step execution
- deterministic `CUDA_VISIBLE_DEVICES` from GPU token names

### Runtime files

Implemented:

- `src/resource-orchestration.js`
- `src/orchestration-utils.js`
- `src/scheduler.js`
- `src/autoresearch-dag-smoke.js`

Still planned:

- dedicated allocation liveness manager if needed
- dedicated worker launcher module if launch logic grows further

### `autoresearch/train.py`

Implemented:

- support env-based overrides
- stop relying on source edits for experiments
- emit structured metrics at the end of a run

### `autoresearch/`

Implemented:

- experiment artifact directory conventions
- runner wrapper for launching experiments

Still planned:

- helper scripts for summarizing experiment history

## Persistence

Keep the current project DB for:

- issues
- comments
- TBC PRs
- reports

Add runtime tables for the new mode as needed:

- `allocation_leases`
- `task_runs`
- `token_requests`
- `token_grants`
- `resource_claims`
- `experiment_runs`

These should live in the existing project SQLite database so orchestration state lives beside the rest of the project data.

## Main Risks

1. Concurrent active-agent tracking now exists, but status and recovery still need hardening around long-running worker pools.

2. DAG and parallel execution exist, but fairness policy is still basic.

3. Shared-step execution is still process/blocking at the worker-task level, so event-driven graph append while workers are running is not implemented yet.

4. `autoresearch` has env overrides and structured metrics, but experiment-history summarization is still thin.

5. Allocation liveness is now critical state.
   If the shared Slurm allocation dies, all active workers must be failed or retried cleanly.

6. GPU token leaks would deadlock the system.
   Token release must be tied to worker process lifecycle.

7. Local GPU scheduling can conflict with local LLM/tool workloads.
   The scheduler must keep model/tool concurrency separate from experiment resource claims.

## Success Criteria

This project is successful when:

1. It can acquire and maintain one shared Slurm allocation for a project.
2. It can run one manager and multiple workers in that allocation.
3. Worker runs are admitted according to token availability.
4. Workers can be attached into the shared node with deterministic GPU binding.
5. `autoresearch` runs can be launched, tracked, and summarized through the orchestrator.
6. The manager can use completed results to decide the next experiment cycle.

## Recommended Next Build Order

Build in this order:

1. live 8-GPU pooled `autoresearch` rerun with the current `worker_class` scheduler
2. event-driven manager append while long-running workers are still active
3. token leak / orphaned-process recovery
4. CPU token inventory
5. RAM-aware admission control
6. experiment-history summarization helpers

This order keeps the system runnable after every phase.
