# Resource Orchestration Test Plan

## Goal

Validate the first local resource-orchestration slice for `infra/hpc_agent/runner`:

- one shared allocation per project
- manager-visible GPU token inventory
- worker token request / grant / release lifecycle
- project status exposure for allocation and token state

This document started as the control-plane test plan. It now also records the first real Slurm validation for shared-allocation execution.

Current runtime context:

- `hpc_agent` is now runner-only for this plan
- the legacy backend/frontend path is removed
- M1 validation should therefore target `runner` plus live Slurm behavior
- current scheduler semantics are documented in `SCHEDULER.md`

## Scope

### In scope

- SQLite schema creation and idempotent migration
- token seeding for a fixed GPU pool
- allocation lease create / update / release
- token request create / list
- token grant create
- token release and token reuse
- runner status summary built from persisted state
- CLI behavior for resource orchestration commands
- basic DAG scheduler admission against GPU token availability

### Out of scope

- preemption, oversubscription, and fractional GPU allocation
- automatic token denial / reprioritization logic
- advanced fairness policy such as aging and starvation prevention
- CPU/RAM token admission control

## Assumptions

- initial deployment uses 8 exclusive GPU tokens: `gpu0` .. `gpu7`
- a project has at most one active shared allocation lease in M1
- a token can be assigned to at most one active grant at a time
- the manager is the only actor allowed to grant or deny requests
- workers release tokens explicitly when their task completes

## Test Matrix

### 1. Schema bootstrap

Expected result:

- resource tables exist after opening the project DB
- repeated bootstrap does not fail or duplicate seeded tokens
- initial GPU token count matches configured `gpuCount`

Coverage:

- automated unit test

### 2. Allocation lease lifecycle

Expected result:

- creating a lease persists `job_id`, `node_name`, `state`, and resource metadata
- updating the lease replaces the active lease snapshot without duplicating rows
- releasing a lease marks it inactive and clears the active lease from summaries

Coverage:

- automated unit test

### 3. Token request lifecycle

Expected result:

- worker can create a pending request for one or more GPU tokens
- request metadata persists actor, count, rationale, and timestamps
- request listing returns newest requests first

Coverage:

- automated unit test
- CLI integration test

### 4. Token grant lifecycle

Expected result:

- manager can grant a pending request with concrete token names
- granted tokens become unavailable to subsequent grants
- grant records include manager, worker, request id, token ids, and lease/job context

Coverage:

- automated unit test
- CLI integration test

### 5. Token release lifecycle

Expected result:

- releasing a grant marks the grant inactive
- released tokens become available again
- repeated release of the same grant fails clearly

Coverage:

- automated unit test
- CLI integration test

### 6. Status summary

Expected result:

- project status exposes:
  - orchestration mode
  - configured GPU pool size
  - active lease summary
  - available / granted token counts
  - pending request count
  - active grant count
- status summary remains valid after restart by reading persisted state

Coverage:

- automated unit test

## Manual Verification

After automated tests pass:

1. create a disposable project DB
2. seed an 8-GPU pool
3. create one shared allocation lease
4. create requests for two workers
5. grant disjoint GPU tokens
6. inspect status summary
7. release one grant
8. confirm the token becomes available again

## Real Slurm Validation

Validated on April 23, 2026 against a fresh shared allocation:

- shared job: `1575847`
- node: `fs-mbz-gpu-275`
- allocation shape: `--nodes=1 --ntasks=1 --cpus-per-task=8 --gpus=8`

### Runtime findings

- do not trust Slurm job TRES alone; probe the live step environment
- worker steps must use `srun --jobid <JOB> --overlap --ntasks=1 --cpus-per-task=1`
- manual `CUDA_VISIBLE_DEVICES=<gpu_ordinal>` assignment mapped cleanly to distinct physical GPUs
- on this cluster, `torch.compile` failed on compute nodes unless `AUTORESEARCH_DISABLE_COMPILE=1` was set

### 8-way smoke test

Using eight overlap steps with manual `CUDA_VISIBLE_DEVICES=0..7`:

- each worker saw exactly one visible GPU
- `nvidia-smi` showed eight Python processes on eight distinct H200 GPUs

Artifacts:

- `artifacts/slurm-8gpu-smoke-20260423T191221Z/`

### 8-way autoresearch run

Using eight overlap steps, one CPU per step, `AUTORESEARCH_DISABLE_COMPILE=1`, and `uv run train.py`:

- all eight workers completed successfully
- mean `training_seconds`: `300.94`
- mean `total_seconds`: `469.09`
- mean `val_bpb`: `1.224259`
- mean `mfu_percent`: `5.67`
- mean `num_steps`: `189.0`
- one worker lagged (`gpu4`, `172` steps, `val_bpb=1.238939`), so scheduler policy should track stragglers rather than assuming uniform throughput

Artifacts:

- `artifacts/slurm-8gpu-autoresearch-20260423T191355Z/`

### Fresh M1 rerun after runner-only cleanup

Validated again on April 23, 2026 after backend/frontend removal:

- shared job: `1575902`
- node: `fs-mbz-gpu-875`
- artifact dir: `artifacts/m1-validation-20260423T202218Z/`

Rerun checks:

- `shared_probe` completed and reported:
  - `SLURM_STEP_GPUS=0,1,2,3,4,5,6,7`
  - `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
  - `torch_cuda_device_count=8`
- two concurrent `shared_step` probes completed with disjoint grants:
  - worker `probe-a` grant `gpu0` -> `CUDA_VISIBLE_DEVICES=0`, `torch_cuda_device_count=1`
  - worker `probe-b` grant `gpu1` -> `CUDA_VISIBLE_DEVICES=1`, `torch_cuda_device_count=1`
- one real `shared_step` `uv run train.py` completed with:
  - `val_bpb=1.034209`
  - `training_seconds=300.3`
  - `total_seconds=338.8`
  - `peak_vram_mb=114019.8`
  - `mfu_percent=20.12`
  - `num_steps=644`
- token-release cleanup returned all GPU tokens to the available pool in the project DB

### Execution recipe captured in code

The runner now has two shared-allocation execution paths in `src/hpc-tool.js`:

- `mode: "shared_probe"` to inspect the live allocation and visible GPU set
- `mode: "shared_step"` to attach a worker step to an existing shared allocation using granted GPU tokens

## Exit Criteria

M1 is acceptable when:

- all automated tests in this plan pass
- CLI commands behave correctly on a disposable DB
- project status exposes a consistent allocation / token summary
- no existing runner behavior regresses in the touched paths

Current result:

- M1 exit criteria are satisfied for the runner-only shared-allocation baseline.
