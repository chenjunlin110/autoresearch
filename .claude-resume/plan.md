# Autoresearch HPC Agent — v1.5 Architecture Redesign

## Context

Today's system gets ~34% GPU duty cycle on 8×H200 over a 2h Slurm allocation: 62 successful experiments out of a theoretical 192. A reviewer audit + throughput trace confirms three real systemic problems and refutes one false alarm:

| Issue | Verdict | File:line |
|---|---|---|
| `producedTags` added on failure | **false** — code already gates on `if (success)` | `server.js:2619-2630` |
| `success` = LLM CLI exit 0, never reads `metrics.json` / `result.txt` | **true** | `server.js:3033`, `agent-runner.js:1469,1601` |
| No per-task walltime admission gate (only cycle-level brake) | **true** | `server.js:2737-2739` |
| Live replan watermark | **partial** — guards exist, default `liveReplanMinIntervalSeconds=0` makes it eager | `server.js:2174-2176` |

The throughput audit (`autoresearch-runner-20260425T191901Z`) shows the 60% overhead per task (458s of overhead vs 315s of actual training) is **not** mostly LLM thinking. The breakdown is:

| Cost | Time | Fixable? |
|---|---|---|
| **cold `torch.compile`** | 150–200s | yes, with shared content-hashed cache |
| python + FA3 kernel imports | ~20s | partial (warm import server) |
| weka git clone | 3–8s | yes, hardlinked cache |
| LLM agent thinking (worker) | 30–60s | yes, by removing LLM from default path |
| misc | 3–5s | — |

That reframes the redesign: **the executor change is the means; the shared compile cache is the actual throughput unlock.** You can only safely deploy a content-hash cache when edits are deterministic (no LLM) — so the executor must come first, but pure executor without cache is only a ~2× win, not the 5× the GPU budget allows.

Three secondary failures from the same audit also have to be addressed: (1) 3 of 65 workers got stuck for 60 min hitting the Claude CLI envelope timeout — the bash subprocess has no inner timeout; (2) once a process group leaks, the GPU token stays leased; (3) `AUTORESEARCH_RESULTS_PATH` TSV append already has flock (verified in `run-autoresearch-worker.sh`) but per-experiment artifact locking is not consistent.

## Goals

1. Single-allocation GPU duty cycle: 34% → ≥75%.
2. P99 task wall time < 360s (vs current ~770s mean).
3. Eliminate the 60-min stuck-worker class entirely.
4. Manager LLM cost shrinks by ≥50% via context compaction + queue watermark.
5. Worker LLM cost shrinks by ≥80% (only invoked on `param_patch` failure).
6. Orchestrator becomes the canonical truth source for task success — not LLM stdout.
7. Plumbing for multi-node scheduling stays clean (don't add new SQLite contention; events become the join key).

## Architecture (v1.5)

```
                          ┌───────────────────────────────┐
                          │  Manager LLM (Opus, high)     │
                          │  - reads compact ledger        │
                          │  - emits strict-JSON TASK_GRAPH│
                          │    with execution_mode+edits[] │
                          │  - emits priority, kill, replan│
                          └────────────┬──────────────────┘
                                       │ schema-validated
                                       ▼
       ┌───────────────────────────────────────────────────────┐
       │  Node.js Orchestrator (single process)                │
       │                                                       │
       │  Manager Loop ── feasibility ── ledger compaction      │
       │      │                                                │
       │      ▼                                                │
       │  Scheduler                                            │
       │  - watermark replan (refill iff ready+running<W)       │
       │  - walltime admission (don't start if won't finish)    │
       │  - duplicate experiment-signature dedup                │
       │  - GPU token grant (existing DB layer)                 │
       │  - 3 timeout rings (sandbox 60s / train 480s / hard 900s)│
       │      │                                                │
       │      ▼                                                │
       │  ┌────────────────────────┐  ┌─────────────────────┐  │
       │  │ Direct Executor        │  │ LLM Worker (repair) │  │
       │  │ default path           │  │ exception path      │  │
       │  │ - SHA-pin parent       │  │ - Haiku             │  │
       │  │ - apply edits[] (AST)  │  │ - context = failure  │  │
       │  │ - git commit           │  │   .json + diff      │  │
       │  │ - bash wrapper         │  │ - one shot, no       │  │
       │  │ - no LLM               │  │   recursive escalate │  │
       │  └────────────┬───────────┘  └──────────┬──────────┘  │
       │               │                         │             │
       │               └────────────┬────────────┘             │
       │                            ▼                          │
       │  Canonical Result Validator                           │
       │  - parse result.txt: exit_code == 0                   │
       │  - parse metrics.json: val_bpb is finite              │
       │  - hash artifact, write metadata.json                 │
       │  - emit task_event { completed | failed | needs_repair }│
       │      │                                                │
       │      ▼                                                │
       │  Compact Ledger (top-K + recent + failed clusters)     │
       │      │                                                │
       │      └──> back to Manager                             │
       └───────────────────────────────────────────────────────┘
                       │
                       ▼ (one process group per task, setsid)
            ┌────────────────────────────┐
            │ bash run-autoresearch-     │
            │   worker.sh <output_dir>   │
            │ - SHARED hash-keyed        │
            │   TORCHINDUCTOR_CACHE_DIR  │
            │ - shared TRITON / HF / UV  │
            │ - timeout 480 wraps train  │
            │ - flock for results.tsv    │
            └────────────┬───────────────┘
                         ▼
                    train.py (5min budget)
                    writes metrics.json + result.txt
```

### What stays the same
- Single Node.js process orchestrator; SQLite (cached single connection, WAL).
- Per-experiment git sandbox under `experiments/<id>/sandbox/repo`.
- Slurm sbatch entrypoint; `dataDir` mechanism in `projects.yaml`; initial `state.json` (isPaused: false) — both already in place.
- DAG semantics: `worker_class`, `depends_on`, `depends_on_tags`, `produces_tags`, `replan_after`, `priority`.
- Live replan + KILL_TASKS protocol (already added).
- Shared 8-GPU token pool with lease lifecycle.

### What changes

**1. TASK_GRAPH schema gains `execution_mode` + `edits[]`**

```json
{
  "id": "exp_0042_betas_097",
  "execution_mode": "param_patch",
  "base_ref": "exp_0009_betas_09_095",   // SHA OR experiment_id; resolved server-side to a commit SHA
  "edits": [
    {"file": "train.py", "kind": "constant_replace",
     "name": "ADAM_BETAS", "expected_old_repr": "(0.9, 0.95)", "new_repr": "(0.9, 0.97)"},
    {"file": "train.py", "kind": "regex_replace",
     "pattern": "adam_betas=\\(0\\.9, 0\\.95\\)", "replacement": "adam_betas=(0.9, 0.97)",
     "expected_count": 1}
  ],
  "resources": {"gpus": 1, "cpus": 1},
  "priority": 9,
  "estimated_runtime_seconds": 320,
  "produces_tags": ["metrics:exp_0042_betas_097"],
  "rationale": "Manager-visible WHY (one sentence)."
}
```

Four edit kinds:
- `constant_replace` — module-level Python assignment; AST-normalized comparison so `2**19`, `524288`, `1<<19` all match.
- `regex_replace` — for in-line tuples, function default kwargs, dependent strings; `expected_count` mandatory, fail loud on mismatch.
- `block_replace` — `{anchor_regex, end_regex, new_text}` for multi-line bodies.
- `unified_diff` — escape hatch with full patch text.

`execution_mode: "code_edit"` keeps current LLM-worker path for genuinely free-form rewrites (architecture changes, optimizer rewrites). `execution_mode: "llm_repair"` is auto-emitted by the orchestrator on failure of a `param_patch`, never by the manager directly.

**2. Direct executor replaces LLM worker for `param_patch`**

New module `src/direct-executor.js`:
- SHA-pin parent commit (refuse branch refs).
- Idempotent branch creation: if `<id>` already exists, verify its parent SHA matches; otherwise refuse with `experiment_id_already_used` and force the manager to pick a new id.
- AST-normalize edits before comparing/applying: `ast.parse + ast.unparse` round-trip via Python subprocess (cheap, ~30ms).
- Post-patch sanity: `python -c 'import ast; ast.parse(...)'` to catch syntax breaks before training.
- Atomic edit-or-rollback (`git reset --hard` if any edit fails).
- Spawn worker with `setsid` (own process group), so SIGTERM on the pgid takes down bash + python + any leaked descendants.

**3. Shared content-hashed compile cache (the actual throughput lever)**

Modify `run-autoresearch-worker.sh`:
- Compute `cache_key = sha256(ast_unparse(train.py) || ast_unparse(prepare.py) || gpu_sku)`. The Python AST normalization makes constant-only changes (`DEPTH=8 → DEPTH=9`) NOT change the cache key when they don't affect the compute graph; structural changes (loop body rewrite, attention restructure) DO change it.
- Set `TORCHINDUCTOR_CACHE_DIR=$shared/inductor/$cache_key`, `TRITON_CACHE_DIR=$shared/triton/$cache_key`.
- Pin `UV_CACHE_DIR` and `HF_HOME` to a single shared, read-mostly dir (currently per-experiment, wasted 8 ways).
- Cold compile (~150-200s) is paid only on the first member of an equivalence class; warm path is ~5-10s.

This is the change with the largest expected throughput impact. Independent of removing the LLM, this alone would compress most experiments from ~470s to ~320s.

**4. Canonical result validation**

Replace "success = LLM CLI exit code" with:
```js
function validateExperimentResult(outputDir) {
  const resultTxt = readResultTxt(outputDir);   // parses exit_code=
  const metrics = readMetricsJson(outputDir);    // parses val_bpb, num_steps, ...
  if (!resultTxt) return { success: false, reason: "missing result.txt" };
  if (resultTxt.exit_code !== 0) return { success: false, reason: `train exit_code=${resultTxt.exit_code}` };
  if (!metrics || typeof metrics.val_bpb !== 'number' || !Number.isFinite(metrics.val_bpb)) {
    return { success: false, reason: "missing or invalid val_bpb" };
  }
  return { success: true, val_bpb: metrics.val_bpb, training_seconds: metrics.training_seconds };
}
```

Worker stdout becomes debug-only. The orchestrator's task state transition reads from these files, so silent worker LLM hallucinations ("I succeeded!") can't leak into the dependency graph.

**5. Three concentric timeout rings**

Enforced by orchestrator + bash, not by LLM CLI envelope:

| Ring | Limit | Action |
|---|---|---|
| Sandbox+edit | 60s | abort task, mark `failed` (NOT escalated to llm_repair — infra issue) |
| Train (`timeout` wrapping `bash run-autoresearch-worker.sh`) | `TIME_BUDGET + 180s` (default 480s) | SIGTERM pgid, mark `failed` with `train_timeout` reason |
| Total task hard cap | 900s | SIGTERM → SIGKILL after 30s, free GPU token, emit project-level alert if >2 in last 10 tasks (cluster sickness signal) |

CUDA cleanup: after SIGKILL, run `nvidia-smi --query-compute-apps=pid --format=csv,noheader` for the freed GPU; if anything still owns it, wait 5s and re-check before reissuing the token.

**6. Walltime admission gate**

Before any `_startScheduledWorker`:
```js
const slurmDeadline = process.env.SLURM_JOB_END_TIME ? Date.parse(process.env.SLURM_JOB_END_TIME) : null;
if (slurmDeadline) {
  const remainingMs = slurmDeadline - Date.now();
  const requiredMs = 1000 * (task.estimatedRuntimeSeconds + 120);  // 2min slack
  if (remainingMs < requiredMs) {
    return { started: false, reason: `not enough Slurm walltime: ${remainingMs}ms < ${requiredMs}ms` };
  }
}
```

When Slurm time is too short, manager is signaled to stop dispatching and start a `worker_class: analyst` finalization task instead.

**7. Watermark live replan**

`_executeTaskGraph` calls `startLiveReplan` only when:
```
ready_count + running_count < target_watermark
```
where `target_watermark = max_concurrent_workers * 1.5` (default 8 × 1.5 = 12). Default `liveReplanMinIntervalSeconds: 30` (was 0).

Manager prompt updated to emit larger plans (12-16 tasks, not exactly N=GPU count) so the queue stays warm without per-completion replanning. Manager LLM share of total time should drop from ~5% to ≤2% while opus call count drops from 44 to ~10-15 over a 2h run.

**8. Failure escalation contract (`llm_repair` is automatic)**

Direct executor failure → write `failure.json`:
```json
{
  "reason": "constant_replace name=DEPTH not found at module scope; closest match: DEPTH at line 489 inside build_model_config local",
  "attempted_edits": [...],
  "parent_commit": "abc123",
  "nearby_symbols": ["DEPTH", "depth", "DEPTHS"],
  "grep_context": ["line 489: DEPTH = 8"]
}
```

Orchestrator auto-emits a sibling `execution_mode: "llm_repair"` task with `worker_class: experiment_runner_repair`, `depends_on` failure.json. The repair task receives the *parent commit's train.py* (not HEAD's) and the structured `attempted_edits`, NOT the manager's natural-language hypothesis (avoids the repair LLM rationalizing). Cap repairs at 1 per task; second failure → `failed` with full diagnostics.

Edit-confidence report: when last 20 tasks have <70% `param_patch` success rate, the manager prompt addendum injects "your last N param_patch attempts failed; review train.py's current state before emitting more edits."

**9. Compact experiment ledger**

Manager prompt context per replan:
- `top_k`: 10 experiments by val_bpb (pinned)
- `recent`: 10 most recent terminal experiments
- `running`: list of in-flight (id + parent + dispatched_at)
- `failed_clusters`: grouped by failure reason (e.g., "compile error in attention path: 3 tasks", "OOM with batch_size=2**20: 2 tasks")
- `lineage_summary`: which branches are kept, which are dead
- `walltime_remaining_seconds`
- `axes_explored`: deduplicated list of {(constant, value)} tried

Each completed experiment stored as a compact record (~200 chars each), not the full `experiment.md`. With 100+ experiments the prompt stays roughly constant.

**10. Per-task timing trace**

Orchestrator records and writes to `metadata.json`:
```json
{
  "experiment_id": "...",
  "base_commit": "...",
  "final_commit": "...",
  "diff_hash": "...",
  "execution_mode": "param_patch",
  "params": {...},
  "gpu_token": "gpu3",
  "hostname": "fs-mbz-gpu-489",
  "attempt": 1,
  "events": [
    {"name": "created",       "t": 1234567890.123},
    {"name": "granted",       "t": 1234567890.456, "grant_id": 17},
    {"name": "executor_start","t": 1234567890.789},
    {"name": "edits_applied", "t": 1234567891.234},
    {"name": "train_start",   "t": 1234567892.000},
    {"name": "train_finished","t": 1234568192.000},
    {"name": "validated",     "t": 1234568193.500},
    {"name": "released",      "t": 1234568194.000}
  ],
  "metrics_sha256": "..."
}
```

Aggregate report at end of cycle: GPU duty cycle, manager-share, executor-share, mean per-task setup/compile/train/tail.

## Files to modify

| Concern | File |
|---|---|
| TASK_GRAPH schema + edit-kinds parsing | `infra/hpc_agent/runner/src/orchestration-utils.js` (extend `parseTaskGraphDocument` to accept `execution_mode` + `edits[]`; reject `param_patch` with malformed edits) |
| Worker dispatch branch | `infra/hpc_agent/runner/src/server.js` (in `_startScheduledWorker`, branch on `step.executionMode`; new `_runDirectExecutor` method calling new module) |
| Direct executor module (NEW) | `infra/hpc_agent/runner/src/direct-executor.js` (sandbox + edit + commit + bash spawn + validation) |
| Edit-application module (NEW) | `infra/hpc_agent/runner/src/edits.js` (4 kinds, AST normalize via subprocess) |
| Cache strategy | `infra/hpc_agent/runner/scripts/run-autoresearch-worker.sh` (compute cache key, pin shared cache dirs, `setsid`, `timeout 480` wrapper) |
| Canonical result reader | `infra/hpc_agent/runner/src/result-validator.js` (NEW; reads `result.txt` + `metrics.json`) |
| Walltime admission | `infra/hpc_agent/runner/src/server.js:_executeTaskGraph` (gate before `_startScheduledWorker`) |
| Watermark replan | `infra/hpc_agent/runner/src/server.js:_executeTaskGraph` (gate `startLiveReplan` on `ready+running < watermark`) |
| Failure escalation | `infra/hpc_agent/runner/src/server.js` (auto-append `llm_repair` sibling task; write `failure.json`) |
| Manager prompt | `infra/hpc_agent/runner/src/autoresearch-dag-full.js:renderAutoresearchFullManagerProgram` (3 worked examples; cost telegraphing; lineage compaction instructions) |
| Compact ledger builder | `infra/hpc_agent/runner/src/server.js:_buildSingleManagerContext` (replace full-history dump with top-K + recent + clusters) |
| metadata.json + timing trace | `infra/hpc_agent/runner/src/direct-executor.js` (write per task) |
| Smoke variant | `infra/hpc_agent/runner/src/autoresearch-dag-smoke.js` (mirror schema additions) |
| Tests | `infra/hpc_agent/runner/tests/edits.test.js` (NEW; 4 edit kinds incl. AST normalize), `tests/direct-executor.test.js` (NEW), `tests/orchestration-utils.test.js` (extend with `execution_mode` + bad edits) |

## Existing utilities to reuse
- `prepare-autoresearch-sandbox.sh` (clone-from-parent — direct executor calls this directly, not via LLM).
- `parseTaskGraphDocument` (`orchestration-utils.js`) — already validates ids, dedups tags; just extend.
- `_killTaskGraphTasks` (`server.js`) — kill primitive already present.
- `_sweepOrphanedGrants` (`server.js`) — orphan recovery on startup, already there.
- `_tryGrantWorkerResources` / `_releaseWorkerGrant` — token grant lifecycle, no change.
- DB schema (`resource-orchestration.js`) — no change for v1.5; multi-node migration deferred.
- Live replan plumbing (`_maybeLiveReplanTaskGraph`) — keep, but gate by watermark.
- `parseKillTasksDocument` — already added; reused.

## Implementation phases (single-node, in order; each phase shippable on its own)

### Phase 1 — Observability + Walltime Gate (no behavior change beyond protection)
1. Add per-task timing trace via `metadata.json` writer.
2. Add walltime admission gate in `_startScheduledWorker`.
3. Add canonical result validator (parse `result.txt` + `metrics.json`); start logging both the LLM's claim and the canonical truth so we measure mismatch frequency before switching the source of truth.
4. Add aggregate cycle report at end of `runSingleManagerCycle`.

**Verification**: submit a 30-min sbatch; confirm `metadata.json` written for every task; mismatch counter logged; no walltime overruns past Slurm deadline.

### Phase 2 — Timeout Rings + Process-Group Hygiene
1. Wrap `run-autoresearch-worker.sh` body in `timeout 480 …` and `setsid`.
2. In orchestrator, hard cap any task at 900s wall (SIGTERM → SIGKILL pgid).
3. Add CUDA cleanup probe (`nvidia-smi --query-compute-apps`) before reissuing a freed token.
4. Verify `results.tsv` flock still in place after recent reverts.

**Verification**: artificial test: deliberately hang a worker (e.g., `sleep 7200` injected into prompt). Confirm task killed at 900s, GPU token freed, no zombie pids.

### Phase 3 — Edits Module + Direct Executor (default for `param_patch`)
1. Build `edits.js` with 4 kinds + Python AST normalize subprocess.
2. Build `direct-executor.js` end-to-end (clone → checkout → apply → commit → spawn → validate).
3. Extend `parseTaskGraphDocument` to accept `execution_mode` + `edits[]`; reject malformed.
4. Server dispatch branches on `executionMode === 'param_patch'` → direct executor; else → existing LLM worker path.
5. Update manager prompt: 3 worked examples (constant_replace, unified_diff, code_edit). Reject `code_edit` tasks whose body sniffs as `constant_replace`-shaped (regex sniff `change\s+\w+\s*=\s*[\d.()"\-]+\s+to`); push correction back to manager.
6. Failure escalation: write `failure.json` and auto-emit `llm_repair` sibling.

**Verification**: submit 2h sbatch with claude_cli; confirm:
- ≥80% of tasks emitted by manager carry `execution_mode: "param_patch"`.
- direct executor mean elapsed ≤ 320s (vs ~770s today).
- ≥1 `param_patch` failure → `llm_repair` escalation → eventual success.
- `code_edit` path still works for one architecture-change experiment.

### Phase 4 — Shared Compile Cache (the big throughput unlock)
1. In `run-autoresearch-worker.sh`, compute `cache_key = sha256(ast_unparse(train.py) || ast_unparse(prepare.py) || gpu_sku)`.
2. Point `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` to `$shared_cache/$kind/$cache_key`.
3. Pin `UV_CACHE_DIR` and `HF_HOME` to a single read-mostly path.
4. Add a "first-cold-compile" warm-up task at the start of cycle 1: a no-edit baseline that primes the cache, then 7 parallel runs hit warm. (Optional but valuable: cuts cold-compile cost from 8× to 1×.)

**Verification**: per-task `metadata.json.events` shows `train_started - executor_start ≤ 30s` for ≥80% of tasks (warm cache), vs 150-200s for the first member of a new compile equivalence class.

### Phase 5 — Watermark Live Replan + Compact Ledger
1. Default `liveReplanMinIntervalSeconds: 30`; trigger replan only when `ready + running < watermark`.
2. Manager prompt asks for 12–16 tasks per emission, not 8.
3. Replace `_buildSingleManagerContext` full-history dump with compact ledger.
4. Edit-confidence addendum injected when last-20 success rate < 70%.

**Verification**: 2h run shows ≤15 manager opus calls (vs 44 today); ledger payload ≤ 8KB.

### Phase 6 — Refactor (no behavior change, code health)
1. Split `server.js` into modules per the GPT review's tree (`manager_loop.js`, `task_graph_executor.js`, `worker_runner.js`, `state_store.js`, `experiment_store.js`, `live_replan.js`).
2. Keep single Node.js process, single SQLite handle.

## Verification

End-to-end success criteria after Phase 4:

```
sbatch --time=02:00:00 (claude_cli)
  ├── ≥120 successful experiments (vs 62 today)
  ├── GPU duty cycle ≥75% (from metadata.json events)
  ├── 0 tasks killed by 900s hard cap during normal operation
  ├── ≥80% of tasks via param_patch direct executor
  ├── manager LLM share ≤2% of wall time (vs 5% today)
  ├── worker LLM invocations ≤20 (only repair + code_edit), vs ~65 today
  └── canonical-vs-LLM mismatch counter shows 0 false-success
```

Tests added:
- `tests/edits.test.js`: 4 kinds × {AST normalize, expected_count mismatch, file-not-found, syntax-break-after-patch}.
- `tests/direct-executor.test.js`: end-to-end with a small fixture train.py; verifies SHA pin, idempotent branch, atomic rollback, validate output.
- `tests/orchestration-utils.test.js`: extend with `execution_mode` parsing + malformed `edits[]` rejection + `code_edit` sniff regex.

Run via `npm test` from `infra/hpc_agent/runner/`. Phase-by-phase live validation: dispatch a 30-60min sbatch after each phase, inspect `metadata.json` aggregates + the orchestrator log.

## What this plan deliberately does NOT do

1. **Multi-node scheduling** — token model needs `node_id × gpu_index` and SQLite stops being adequate. Defer until single-node ≥75% duty cycle is real.
2. **Bandit / Bayesian planner** — LLM manager remains the science brain. The reviewer was right that adding a side-channel optimizer right now mixes concerns.
3. **Postgres / Redis migration** — single-node SQLite + state.json is fine. Move when multi-node forces it.
4. **Manager retraining / fine-tuning** — prompt-only changes; no model changes.
5. **train.py modifications** — `train.py` stays untouched as the user's experimental surface. All edits stay in per-experiment sandboxes.
6. **Replacing claude_cli/codex_cli with API runtime** — keep current runtime selector. Direct executor sits *below* the runtime in the dispatch tree, so runtime choice still matters only for LLM workers (manager + repair + code_edit).

## Phase 0 (do before phase 1, 30 min, low risk)

Re-apply two reverted fixes that the audit confirmed are still needed:
1. `_getParallelGpuAvailability` honoring `grantRequiresLease: false`.
2. `_tryGrantWorkerResources` honoring `grantRequiresLease: false`.
3. Blocked-task log in `_executeTaskGraph` (so silent block bugs are visible in future runs).

These have been re-applied at least twice and reverted at least twice; the plan should land them as durable code that can survive a clean checkout, not patched in-flight.
