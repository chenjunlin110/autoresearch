# Changelog

## 2026-04-24

### Added

- Added a checked-in `autoresearch` DAG smoke workspace generator:
  - `infra/hpc_agent/runner/src/autoresearch-dag-smoke.js`
  - `infra/hpc_agent/runner/scripts/create-autoresearch-dag-smoke.js`
  - regression test coverage in `infra/hpc_agent/runner/tests/autoresearch-dag-smoke.test.js`
- Added `infra/hpc_agent/runner/src/scheduler.js` as the pure scheduling policy module for worker-class binding and DAG ready-queue ranking.
- Added `infra/hpc_agent/runner/tests/scheduler.test.js` to cover worker-class inference, idle-worker binding, and scheduler ranking.
- Added `infra/hpc_agent/runner/docs/SCHEDULER.md` to document the current manager/worker scheduling flow and known limits.
- Added `infra/hpc_agent/runner/scripts/run-full-autoresearch-8gpu.sh` for repeatable full 8-GPU overlap runs and metrics summarization.
- Added `infra/hpc_agent/runner/scripts/submit-full-autoresearch-8gpu.sbatch` for direct Slurm batch submission from a non-sandboxed shell.

### Changed

- Updated the `autoresearch` DAG smoke templates to use repo-relative experiment output directories such as `exp1` instead of embedding artifact-root absolute paths into manager instructions.
- Updated manager guidance to prefer repo-relative run directories for `autoresearch` tasks.
- Updated DAG scheduling toward a CPU-scheduler-style model:
  - task graph nodes may now use `worker_class` instead of fixed `agent`
  - the runtime binds ready tasks to idle worker instances at dispatch time
  - scheduler ranking now considers priority, critical-path length, utility, GPU fit, and estimated runtime
  - successful task-graph drain can trigger another manager pass via `refillOnGraphDrain`
  - the `autoresearch` smoke generator now creates one experiment-runner worker per GPU by default
- Moved worker-class inference and task selection out of `server.js` so orchestration policy is easier to test independently.
- Updated root HPC Agent docs and env examples so `codex_cli` is clearly the default no-key runtime.
- Cleared previous local run logs/artifacts before the next full-run attempt.
- Verified that this Codex tool sandbox blocks Slurm stream sockets for both `srun` and `sbatch`; full GPU execution must be launched from the outer allocation shell.
- Fixed the full 8-GPU runner to prefer direct local launch when already inside the allocation node, because `srun --overlap` steps on job `1575972` exposed only `SLURM_STEP_GPUS=0,1,2,3`.
- Validated a corrected full 8-GPU `autoresearch` run:
  - artifact dir: `artifacts/autoresearch-full-8gpu-direct-20260424T015450Z/`
  - exit status: `0`
  - successful workers: `8/8`
  - mean `val_bpb`: `1.224895`
  - mean `training_seconds`: `301.09`
  - mean `total_seconds`: `470.31`
  - mean `mfu_percent`: `5.66`
  - mean `num_steps`: `188.875`

## 2026-04-23

### Added

- Added resource orchestration control-plane support in `infra/hpc_agent/runner`:
  - SQLite tables for allocation leases, resource tokens, token requests, token grants, and grant-token mappings
  - project status snapshot plumbing for resource orchestration state
  - `tbc-db` commands for allocation upsert, token request, token grant, token release, and related listing commands
- Added shared-allocation execution support in `infra/hpc_agent/runner/src/hpc-tool.js`:
  - `mode: "shared_probe"` to inspect the live Slurm step environment
  - `mode: "shared_step"` to launch a worker step inside an existing shared allocation
  - grant-aware validation against the project DB
  - manual `CUDA_VISIBLE_DEVICES` assignment derived from granted GPU tokens
- Added automated tests for:
  - resource orchestration schema and token lifecycle
  - shared allocation step input resolution
  - shared probe / shared step command construction
- Added real-Slurm validation notes to `infra/hpc_agent/runner/docs/RESOURCE_ORCHESTRATION_TEST_PLAN.md`.
- Added `infra/hpc_agent/runner/scripts/shared-step-probe.py` as a small helper for repeatable shared-step validation.
- Added M2 orchestration helpers in `infra/hpc_agent/runner`:
  - `src/orchestration-utils.js` for orchestration config and richer schedule parsing
  - `agent/managers/manager.md` for dedicated single-manager mode
  - support for `parallel` worker groups and per-step `resources`
  - grant-aware task injection for shared-allocation worker runs
- Added local Codex CLI agent runtime support in `infra/hpc_agent/runner`:
  - `runAgentWithCodexCLI()` in `src/agent-runner.js`
  - `agentRuntime: codex_cli` runtime selection in `src/server.js`
  - Codex CLI invocation helpers and unit tests
  - runtime documentation in `runner/docs/CODEX_LOCAL_RUNTIME.md`
- Added a worker-oriented `autoresearch` execution wrapper:
  - `infra/hpc_agent/runner/scripts/run-autoresearch-worker.sh`
  - stable per-run output directories with `train.log`, `metrics.json`, `result.txt`, and `run.env`
  - workspace-local runtime caches for `uv`, Hugging Face, Triton, and TorchInductor

### Changed

- Simplified `hpc_agent` to a runner-only architecture centered on shared-allocation execution.
- Updated `infra/hpc_agent/runner/src/server.js` to pass project DB path, agent name, shared working directory, shared job id, and shared environment defaults into `HpcSubmit`.
- Updated `train.py` with a compile fallback controlled by `AUTORESEARCH_DISABLE_COMPILE=1` so compute-node runs can bypass `torch.compile` / Triton failures on this cluster.
- Updated `prepare.py` and `train.py` to accept environment-driven experiment overrides:
  - `AUTORESEARCH_TIME_BUDGET_SECONDS`
  - `AUTORESEARCH_CACHE_DIR`
  - model and optimizer overrides such as `AUTORESEARCH_DEPTH`, `AUTORESEARCH_TOTAL_BATCH_SIZE`, and related learning-rate knobs
- Updated `train.py` to emit structured metrics via `AUTORESEARCH_METRICS_PATH`.
- Organized historical training logs under `artifacts/` and added log summaries / indexes.
- Updated `infra/hpc_agent/runner/src/server.js` with:
  - `single_manager` orchestration mode
  - concurrent worker-batch execution for `parallel` schedule steps
  - active run tracking beyond a single `currentAgent`
  - automatic GPU token grant/release around worker runs that request `resources.gpus`
- Updated shared manager/worker prompt rules to describe `parallel` groups and resource grants.
- Updated `infra/hpc_agent/runner` setup defaults so local projects use `agentRuntime: codex_cli` by default.
- Updated preflight logic so only `agentRuntime: api` projects require a non-empty key pool.
- Updated prompt scaffolding so `codex_cli` workers use shell commands directly instead of assuming JSON API tools.
- Updated single-manager manager/worker prompts and injected task context so granted workers see:
  - available worker roster in the manager context
  - granted GPU ordinals
  - a direct `run-autoresearch-worker.sh` command recipe when the worker is already on the allocation node
- Updated the single-manager parallel scheduler with GPU-aware backfill:
  - parallel batches no longer block on the queue head when a smaller runnable experiment fits the free GPUs
  - the scheduler now chooses the largest runnable GPU task that fits current availability
  - permanently unschedulable GPU steps still fail fast
- Updated manager guidance to make fan-out/fan-in explicit:
  - independent experiments go in one `parallel` block
  - result-dependent analyst / patch steps run after that block
- Added DAG-style orchestration support:
  - managers may now emit `<!-- TASK_GRAPH -->`
  - task nodes support `id`, `depends_on`, `depends_on_tags`, `produces_tags`, `priority`, and `replan_after`
  - the runtime maintains a ready queue instead of relying only on array order
  - completed `replan_after` nodes hand control back to the manager immediately within the same cycle
  - unresolved or deadlocked pending nodes are marked `blocked` instead of hanging silently

### Removed

- Removed legacy `hpc_agent/backend/`.
- Removed the old frontend / monitor bundle.
- Removed diamond-specific paths and code.

### Validated

- Confirmed `codex --version` is available locally:
  - `codex-cli 0.123.0`
- Confirmed direct `runAgentWithCodexCLI()` smoke succeeds locally:
  - prompt: `Reply with exactly: CODEX_CLI_OK`
  - result: `CODEX_CLI_OK`
- Confirmed a fresh 8-GPU shared allocation exposes all 8 GPUs at runtime:
  - shared job: `1575847`
  - node: `fs-mbz-gpu-275`
- Confirmed overlap worker steps must use:
  - `srun --jobid <JOB_ID> --overlap --ntasks=1 --cpus-per-task=1`
- Confirmed manual `CUDA_VISIBLE_DEVICES=<gpu_ordinal>` maps cleanly to distinct physical GPUs in the shared allocation.
- Completed a real 8-way `autoresearch` run inside one shared allocation:
  - artifact dir: `artifacts/slurm-8gpu-autoresearch-20260423T191355Z/`
  - mean `training_seconds`: `300.94`
  - mean `total_seconds`: `469.09`
  - mean `val_bpb`: `1.224259`
  - mean `mfu_percent`: `5.67`
  - mean `num_steps`: `189.0`
- Re-ran M1 after backend/frontend removal on a fresh shared allocation:
  - shared job: `1575902`
  - node: `fs-mbz-gpu-875`
  - artifact dir: `artifacts/m1-validation-20260423T202218Z/`
  - `shared_probe` saw `8` visible GPUs in-step
  - two concurrent `shared_step` probes saw exactly one visible GPU each (`gpu0`, `gpu1`)
  - one `shared_step` `uv run train.py` completed successfully on `gpu2`
  - single-worker `autoresearch` metrics:
    - `val_bpb`: `1.034209`
    - `training_seconds`: `300.3`
    - `total_seconds`: `338.8`
    - `peak_vram_mb`: `114019.8`
    - `mfu_percent`: `20.12`
    - `num_steps`: `644`
  - all granted GPU tokens were released cleanly back to the DB
- Ran a live no-key `codex_cli` project smoke through the actual runner:
  - temp project id: `codex-cli-smoke`
  - runtime: `agentRuntime: codex_cli`
  - no provider API key configured
  - resumed single-manager schedule executed worker `smoke`
  - runner log showed `Using Codex CLI runner for smoke (model: mid)`
  - worker created `runtime-smoke.txt` with content `CODEX_CLI_PROJECT_OK`
  - worker response log recorded success and the written file path
- Ran a real GPU smoke for the new `autoresearch` worker wrapper:
  - artifact dir: `artifacts/autoresearch-worker-smoke-20260423T223020Z/`
  - `AUTORESEARCH_TIME_BUDGET_SECONDS=10`
  - `CUDA_VISIBLE_DEVICES=0`
  - wrapper exit code: `0`
  - structured metrics file written successfully
  - smoke metrics:
    - `val_bpb`: `1.877070`
    - `training_seconds`: `10.1`
    - `total_seconds`: `48.4`
    - `peak_vram_mb`: `114019.8`
    - `mfu_percent`: `20.82`
    - `num_steps`: `32`

### Known Gaps

- Manager-side resource arbitration policy is still prompt-driven and simple; there is no deny / reprioritize policy yet.
- CPU tokens and broader resource accounting are not implemented yet.
- The new M2 runtime path still needs a live end-to-end manager-generated `autoresearch` schedule validation pass.
