# Concurrent Search-Task Framework

A small framework for running **complex search tasks** under a shared compute allocation, where the search policy is an LLM and the search execution runs in parallel across many GPUs.

The framework gives you four things:

1. A **manager LLM** that proposes experiments, given prior results.
2. A **worker pool** that runs experiments in parallel on GPU tokens granted by the runtime.
3. A **DAG scheduler** with priority, dependencies, live replanning, kill primitive, and per-task walltime/GPU admission.
4. A **per-experiment git sandbox** so concurrent edits to the searchable code don't step on each other; the experiment lineage is recorded as git history.

The first concrete task built on top of it is [autoresearch](https://github.com/karpathy/autoresearch) (LLM-driven hyperparameter + architecture search on a small GPT trainer). The same framework is intended for any task that fits the shape:

- there is a body of "main code" that runs and emits a metric
- the search proposes edits to that code (a hyperparameter change, a code change, a config change)
- you want many edits running concurrently and the manager picking the next batch from results

Examples beyond autoresearch:
- hyperparameter sweeps (decoupled from autoresearch's specific knobs)
- data-mix search (manager proposes mixture weights → worker runs eval → result feeds next proposal)
- prompt / system-prompt search
- preprocessing-recipe search
- compiler / kernel autotuning where each candidate runs a benchmark

## Repository layout

```text
repo/
  infra/hpc_agent/runner/        # the framework: scheduler, runtime, manager/worker dispatch
    src/
      server.js                  # ProjectRunner, scheduler loop, live replan, kill, grants
      orchestration-utils.js     # TASK_GRAPH parser, priority/criticalPath ranking, KILL_TASKS
      scheduler.js               # ranking + worker-class binding (pure functions)
      resource-orchestration.js  # SQLite token grant lifecycle
      agent-runner.js            # claude_cli / codex_cli / api runtime adapters
      task-events.js             # per-task event log → metadata.json
      result-validator.js        # canonical truth from result.txt + metrics.json
      slurm-walltime.js          # walltime admission gate
      cycle-report.js            # aggregate per-cycle telemetry
      autoresearch-dag-full.js   # autoresearch task plugin (see "Tasks" below)
    scripts/
      run-autoresearch-worker.sh # autoresearch task wrapper
      submit-autoresearch-runner.sbatch
    tests/                       # 52+ unit tests covering the framework
    agent/                       # framework-level manager/worker rules

  autoresearch/                  # the first task: nanochat-style training code
    train.py                     # the file workers edit per experiment
    prepare.py                   # immutable dataset / tokenizer / eval harness
    program.md                   # original single-agent autoresearch protocol
    results.tsv                  # baseline ledger

  docs/
    architecture.md              # framework architecture (concurrent-search abstraction)
    notes/                       # design notes, plans, post-mortems

  .claude-resume/                # bundle to resume a Claude Code redesign session
                                 # (plan, transcript, restore.sh)
```

## Adding a new task

A task plugin needs:

1. **Code that can run end-to-end on its own.** A directory with the executable code and an immutable evaluation harness. For autoresearch this is `autoresearch/train.py + prepare.py`.
2. **A wrapper script** that runs one experiment and writes `result.txt` (`exit_code=`) + `metrics.json` (`{the_metric: <number>}`) into a per-experiment output dir. For autoresearch this is `tasks/autoresearch/worker.sh`.
3. **A manager prompt** that describes the search surface, the metric, and how to translate ideas into TASK_GRAPH entries. For autoresearch this is rendered by `autoresearch-dag-full.js:renderAutoresearchFullManagerProgram`.
4. **A worker prompt** that tells worker LLMs how to apply edits and run the wrapper. Same renderer.
5. **A workspace generator** that materializes `config.yaml`, `projects.yaml`, `state.json`, manager/worker skill files for the task. `create-autoresearch-dag-full.js` is the example.

That's the full surface. The framework supplies the scheduler, GPU grant lifecycle, kill, replan, retry, walltime admission, canonical result validation, and per-task event tracing — none of it is autoresearch-specific.

A planned cleanup will move the autoresearch-specific files (`autoresearch-dag-full.js`, `run-autoresearch-worker.sh`, `submit-autoresearch-runner.sbatch`) into `tasks/autoresearch/` so the framework / task split becomes explicit at the directory level. See [docs/architecture.md](docs/architecture.md).

## Quick start

To run the autoresearch task:

```bash
# clone the framework + the autoresearch task code
git clone <this-repo> && cd autoresearch && git checkout framework

# install runner dependencies
cd infra/hpc_agent/runner && npm install

# verify the framework
npm test                                                           # unit tests

# launch on Slurm (8 GPUs × 2h)
cd ../../..
AUTORESEARCH_AGENT_RUNTIME=claude_cli sbatch --time=02:00:00 \
  --export=ALL,AUTORESEARCH_AGENT_RUNTIME \
  tasks/autoresearch/submit.sbatch
```

## Status

The framework is mid-redesign (v1.5). The approved plan lives in [.claude-resume/redesign-plan.md](.claude-resume/redesign-plan.md). Phase 0 (lease admission gate, blocked-task log) and Phase 1 (per-task event trace, walltime gate, canonical result validator, cycle report) are landed. Phases 2–6 (timeout rings, deterministic param-patch executor, shared compile cache, watermark replan, compact ledger, module split) remain.

To resume the redesign session on a new machine:

```bash
git clone <this-repo>
cd autoresearch
git checkout framework
bash .claude-resume/restore.sh
claude --resume fd6465ef-7937-4ad1-8942-08b698990432
```
