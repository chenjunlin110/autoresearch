import { mkdirSync, writeFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const RUNNER_ROOT = path.resolve(__dirname, '..');
const REPO_ROOT = path.resolve(RUNNER_ROOT, '../../..');
const DEFAULT_WORKER_SCRIPT_PATH = path.join(RUNNER_ROOT, 'scripts', 'run-autoresearch-worker.sh');
const DEFAULT_SANDBOX_SCRIPT_PATH = path.join(RUNNER_ROOT, 'scripts', 'prepare-autoresearch-sandbox.sh');
const DEFAULT_WORKLOAD_ROOT = path.join(REPO_ROOT, 'autoresearch');

function toPosix(relativePath) {
  return String(relativePath || '').split(path.sep).join('/');
}

export function buildAutoresearchDagFullPaths({
  artifactRoot,
  projectId = 'autoresearch-dag-full',
  experimentName = 'experiments',
  experimentWorkerCount = 8,
} = {}) {
  if (!artifactRoot) throw new Error('artifactRoot is required');
  if (!projectId) throw new Error('projectId is required');
  if (!experimentName) throw new Error('experimentName is required');

  const root = path.resolve(artifactRoot);
  const projectRoot = path.join(root, 'local', projectId);
  const repoRoot = path.join(projectRoot, 'repo');
  const expOutputDir = path.join(repoRoot, experimentName);
  const workerCount = Math.max(1, Number.parseInt(experimentWorkerCount, 10) || 1);
  const experimentWorkerSkillPaths = Array.from({ length: workerCount }, (_, index) => (
    path.join(projectRoot, 'skills', 'workers', `exp_runner_${index}.md`)
  ));

  return {
    artifactRoot: root,
    projectId,
    projectRoot,
    repoRoot,
    expOutputDir,
    resultsPath: path.join(expOutputDir, 'results.tsv'),
    experimentName,
    readmePath: path.join(repoRoot, 'README.md'),
    managerProgramPath: path.join(repoRoot, 'program.md'),
    workerProgramPath: path.join(repoRoot, 'worker_program.md'),
    configPath: path.join(projectRoot, 'config.yaml'),
    projectsYamlPath: path.join(root, 'projects.yaml'),
    experimentWorkerSkillPaths,
    mayaSkillPath: path.join(projectRoot, 'skills', 'workers', 'maya.md'),
  };
}

export function renderAutoresearchDagFullReadme({
  experimentName = 'experiments',
  workerScriptPath,
  sandboxScriptPath,
  timeBudgetSeconds = 300,
  gpuCount = 8,
} = {}) {
  if (!workerScriptPath) throw new Error('workerScriptPath is required');
  if (!sandboxScriptPath) throw new Error('sandboxScriptPath is required');
  const experimentsPerWave = Math.max(1, Number.parseInt(gpuCount, 10) || 1);
  const backlogTarget = experimentsPerWave * 2;

  return `# Autoresearch Continuous Goal

This is a full manager-driven autoresearch run, not a smoke test. It adapts the top-level autoresearch workflow to an ${experimentsPerWave}-GPU hpc_agent setup: the manager designs experiments, dispatches GPU workers, reads real measured feedback, and keeps planning new experiments while the allocation is available.

Read the operating programs before planning or running work:

- Manager program: \`program.md\`
- Worker program: \`worker_program.md\`

The run should continue indefinitely until a human stops the project. Do not emit \`PROJECT_COMPLETE\` just because one wave finished or progress slowed down.

Core objective:
- Minimize validation \`val_bpb\`; lower is better.
- Use a fixed wall-clock training budget of \`AUTORESEARCH_TIME_BUDGET_SECONDS=${timeBudgetSeconds}\`.
- Keep \`AUTORESEARCH_DISABLE_COMPILE=1\`.
- Compare only runs that use the same data, tokenizer, validation set, and evaluation metric.

Required orchestration shape:
- Use \`<!-- TASK_GRAPH -->\`.
- First launch a diverse research portfolio, not identical replicas.
- Use at least ${experimentsPerWave} GPU-backed experiment slots initially; you may split them across mixed resource shapes such as \`{"gpus": 1, "cpus": 1}\`, \`{"gpus": 2, "cpus": 2}\`, \`{"gpus": 4, "cpus": 4}\`, or a larger \`{"gpus": 8, "cpus": 8}\` task if the hypothesis truly requires it.
- Every task must request an explicit \`resources\` object. GPU experiments should usually request matched CPU guidance. CPU-only analysis tasks should request \`{"gpus": 0, "cpus": N}\`.
- Every experiment writes \`experiment.md\`, \`result.txt\`, \`metrics.json\`, and \`train.log\` under \`${experimentName}/<experiment_id>/\`.
- Output paths are relative to the current project repository root. Do not prefix them with \`repo/\`.
- Each experiment task must produce a unique metrics tag.
- The manager may schedule CPU-only analysis tasks, but continuous live replanning is the primary control loop.

Live replanning contract:
- Every worker completion returns its GPU token and triggers manager replanning while other workers continue running.
- On a live replanning call, inspect the completed result, still-running tasks, ready/pending tasks, previous artifacts, and the best-so-far results.
- Emit a \`TASK_GRAPH\` containing only additional tasks to append.
- Do not repeat existing task ids.
- New experiment tasks must not depend on currently-running task ids or tags those tasks have not produced yet.
- New tasks may depend on completed task ids, already-produced tags, and prior artifacts.
- Do not wait for all current workers to finish before planning useful independent work.
- Do not emit \`PROJECT_COMPLETE\` during this continuous run.

Experiment rules:
- Run \`${workerScriptPath}\`.
- Use the granted GPU ordinals from the resource grant block; never hard-code GPU ids.
- Keep the fixed time budget and compile setting unchanged.
- You may either:
  1. run pure override experiments using \`AUTORESEARCH_*\` environment variables, or
  2. run code experiments by creating an isolated git-managed experiment repo with \`${sandboxScriptPath}\`, checking out an experiment branch, editing only that repo's \`train.py\`, and then running the worker with \`AUTORESEARCH_REPO_ROOT=<experiment_repo>\`.
- Do not modify the shared root \`train.py\` during concurrent runs.
- Do not modify \`prepare.py\`, data, tokenizer, validation set, or evaluation harness, either in the shared repo or in an experiment repo.
- Prefer small, interpretable \`train.py\` changes. Record the exact change in the experiment's \`experiment.md\`.
- Every code experiment should name its parent explicitly: either the shared baseline repo or a previously kept experiment repo.
- Keep/discard semantics for code experiments are lineage-based and decided by the manager: a code path is "kept" only if the manager decides later experiments should inherit from that branch; otherwise later experiments should fork from an earlier kept parent instead of extending the discarded branch.
- Seed-only changes are controls or variance checks; do not treat them as new research variants unless a promising config exists.

Allowed search surface:
- Any isolated \`AUTORESEARCH_*\` override already supported by \`train.py\`
- Any isolated edit to experiment-repo \`train.py\` that keeps the fixed-budget evaluation comparable
- Architecture changes, optimizer changes, schedule changes, attention/window changes, simplifications, and combinations of previous near-misses are all allowed if they stay within the fixed 5-minute budget

Resource planning guidance:
- Use \`1 GPU\` tasks for broad local search, ablations, and cheap controls.
- Use \`2 GPU\` or \`4 GPU\` tasks only when the hypothesis genuinely needs that shape.
- Use an \`8 GPU\` task only for high-conviction experiments whose scientific value clearly outweighs the opportunity cost.
- Use CPU-only tasks for analysis, summarization, lineage reconstruction, and queue maintenance.
- Keep the allocation warm, but do not force every queued task to be \`1 GPU\` if a larger task is the right next step.

Still disallowed:
- Editing \`prepare.py\`
- Editing the metric or validation harness
- Installing new packages
- Writing back experiment-specific code edits into the shared repo during a concurrent run

Safe first portfolio:
- \`${experimentName}/0001_baseline\`: no extra overrides
- \`${experimentName}/0002_depth8\`: \`AUTORESEARCH_DEPTH=8\`
- \`${experimentName}/0003_depth10\`: \`AUTORESEARCH_DEPTH=10\`
- \`${experimentName}/0004_aspect56\`: \`AUTORESEARCH_ASPECT_RATIO=56\`
- \`${experimentName}/0005_aspect72\`: \`AUTORESEARCH_ASPECT_RATIO=72\`
- \`${experimentName}/0006_head64\`: \`AUTORESEARCH_HEAD_DIM=64\`
- \`${experimentName}/0007_matrix_lr_003\`: \`AUTORESEARCH_MATRIX_LR=0.03\`
- \`${experimentName}/0008_warmdown05\`: \`AUTORESEARCH_WARMDOWN_RATIO=0.5\`, \`AUTORESEARCH_FINAL_LR_FRAC=0.05\`

When extending the queue, exploit the best-so-far configuration, test nearby alternatives, include controls when useful, and keep every new task interpretable.
`;
}

export function renderAutoresearchFullManagerProgram({
  experimentName = 'experiments',
  workerScriptPath,
  sandboxScriptPath,
  workloadRoot = DEFAULT_WORKLOAD_ROOT,
  timeBudgetSeconds = 300,
  gpuCount = 8,
} = {}) {
  if (!workerScriptPath) throw new Error('workerScriptPath is required');
  if (!sandboxScriptPath) throw new Error('sandboxScriptPath is required');
  const experimentsPerWave = Math.max(1, Number.parseInt(gpuCount, 10) || 1);
  const backlogTarget = experimentsPerWave * 2;

  return `# autoresearch continuous manager program

You are the research manager for a full hpc_agent autoresearch run. This is not a smoke test. Keep the research loop running until a human stops it.

This program mirrors the top-level autoresearch \`program.md\`, adapted for a multi-agent shared-allocation setup. Unlike the earlier restricted version, this run may explore both environment overrides and real \`train.py\` edits. Because workers run concurrently, any code change must happen inside a per-experiment git-managed repo, never in the shared root.

## Standing setup

Every manager pass starts by reading:

1. \`README.md\`
2. \`program.md\`
3. \`worker_program.md\`
4. Existing \`${experimentName}/**/experiment.md\`, \`${experimentName}/**/metrics.json\`, \`${experimentName}/**/result.txt\`, and any analysis files.

Also use the original workload context at \`${workloadRoot}/README.md\`, \`program.md\`, \`prepare.py\`, and \`train.py\` when you need to understand the training code and official search surface.

## Objective

Minimize \`val_bpb\`; lower is better. Treat \`training_seconds\`, \`num_steps\`, throughput, VRAM, and failures as supporting evidence only.

Every experiment must use:

- \`AUTORESEARCH_TIME_BUDGET_SECONDS=${timeBudgetSeconds}\`
- \`AUTORESEARCH_DISABLE_COMPILE=1\`

Never use a longer budget as an improvement.

## Continuous loop

Use \`<!-- TASK_GRAPH -->\` for planning.

The runtime supports live replanning while a task graph is still running. Every time a worker finishes and returns its GPU token, the manager can be called to inspect the completed result, the still-running tasks, pending/ready tasks, and prior artifacts, then append new tasks to the active graph while other workers continue running.

The desired behavior is:

1. Keep the GPU queue warm.
2. Maintain about ${experimentsPerWave}-${backlogTarget} ready or soon-ready GPU-backed task slots when there are enough useful ideas. Mix \`1/2/4/8 GPU\` task sizes deliberately instead of assuming every experiment is \`1 GPU\`.
3. When a task finishes, analyze that result immediately against prior results.
4. Append useful independent experiments without waiting for all current tasks to finish.
5. Avoid unbounded duplicate replicas; use seeds only as explicit controls after promising configs exist.
6. Periodically write or update \`${experimentName}/analysis.md\` or schedule CPU-only \`maya\` analysis tasks for factual summaries, but do not use analysis barriers that stop the live loop.
7. For code experiments, maintain a clear lineage of kept branches versus discarded branches. Do not continue building on a code-edit branch that failed unless you are explicitly debugging that failure mode.
8. Use resource-aware planning. Multi-GPU experiments and higher-CPU tasks are allowed when justified, and should be weighed against the opportunity cost of occupying shared resources.

Do not emit \`PROJECT_COMPLETE\`. If progress stalls, broaden the search, test simpler variants, combine previous near-misses, run controlled replications, or inspect \`train.py\` for new isolated knobs. The run is manually stopped by the human.

## Live replanning contract

When called during a running graph:

- Emit a \`TASK_GRAPH\` containing only additional tasks to append.
- Do not repeat existing task ids.
- New experiment tasks must not depend on currently-running task ids or tags that currently-running tasks have not produced yet.
- New tasks may depend on completed task ids, already-produced tags, and prior artifacts.
- If there are no ready or pending GPU-backed tasks already queued, append enough independent work to keep the allocation warm. In practice, emit enough ready or soon-ready tasks to cover at least ${experimentsPerWave} GPU slots unless there are fewer than ${experimentsPerWave} scientifically useful independent next steps.
- Do not size the append-only graph only to the currently available token count; more workers may finish while you are planning, and queued independent tasks should be ready as tokens return.
- Use larger \`resources.gpus\` values only when the hypothesis justifies waiting for multiple tokens at once.
- Do not emit \`PROJECT_COMPLETE\`.

The runner will skip appended tasks that depend on currently-running tasks or their not-yet-produced tags.

## Experiment task requirements

Each experiment task must:

1. Use \`worker_class: "experiment_runner"\`.
2. Request an explicit \`resources\` object. Default to \`{"gpus": 1, "cpus": 1}\`, but use larger GPU counts or CPU counts when the experiment genuinely needs them.
3. Use a unique id, for example \`exp_0009_head64_aspect56\`.
4. Write outputs under \`${experimentName}/<experiment_id>\`.
5. Use paths relative to the current project repository root; do not prefix output paths with \`repo/\`.
6. Produce a unique metrics tag, for example \`metrics:exp_0009_head64_aspect56\`.
7. State the hypothesis and exact override set in the task text.
8. Tell the worker to read \`worker_program.md\`.
9. Tell the worker to use \`${workerScriptPath}\`.
10. Tell the worker to use the granted GPU ordinals, not hard-coded GPU ids.
11. Tell the worker to write \`experiment.md\` before running.
12. If the experiment edits code, tell the worker to create a git-managed experiment repo with \`${sandboxScriptPath}\`, first run \`git checkout -B <experiment_id>\`, edit only that repo's \`train.py\`, and run with \`AUTORESEARCH_REPO_ROOT=<experiment_repo>\`.
13. If the experiment edits code, state the parent explicitly: either "parent: shared baseline repo" or "parent: exp_XXXX repo/branch".
14. If the experiment edits code, require the worker to make at least one git commit in that experiment repo describing the code change before running.
15. Require the worker to write only a local \`suggested_status\` recommendation in \`experiment.md\` for code experiments.
16. The manager is responsible for writing the final \`final_status\` decision in analysis or follow-up planning.
17. Tell the worker not to wrap the launcher in \`timeout\`; the training script enforces the fixed budget and the wrapper must be allowed to write \`result.txt\` and \`metrics.json\`.
18. Tell the worker to verify \`result.txt\` reports \`exit_code=0\` and \`metrics.json\` exists.

Use \`agent: "maya"\` only for CPU-only analysis tasks. Do not mark analysis tasks with \`replan_after: true\` in the continuous run unless you intentionally want to drain running work first.

When deciding task sizes:

1. Prefer many independent \`1 GPU\` tasks while frontier quality is uncertain.
2. Escalate to \`2 GPU\` or \`4 GPU\` only when you are testing a hypothesis that intrinsically requires that shape.
3. Treat \`8 GPU\` tasks as expensive commitments. Schedule them intentionally and explain why they dominate smaller alternatives.
4. For CPU-heavy analysis or artifact-processing tasks, request the needed \`cpus\` explicitly even when \`gpus\` is \`0\`.

## Allowed experiment surfaces

1. Pure override experiments using \`AUTORESEARCH_*\`:
   - Architecture/model size: \`AUTORESEARCH_DEPTH\`, \`AUTORESEARCH_ASPECT_RATIO\`, \`AUTORESEARCH_HEAD_DIM\`, \`AUTORESEARCH_WINDOW_PATTERN\`
   - Batch shape: \`AUTORESEARCH_TOTAL_BATCH_SIZE\`, \`AUTORESEARCH_DEVICE_BATCH_SIZE\`
   - Optimization: \`AUTORESEARCH_EMBEDDING_LR\`, \`AUTORESEARCH_UNEMBEDDING_LR\`, \`AUTORESEARCH_MATRIX_LR\`, \`AUTORESEARCH_SCALAR_LR\`, \`AUTORESEARCH_WEIGHT_DECAY\`, \`AUTORESEARCH_ADAM_BETAS\`
   - Schedule: \`AUTORESEARCH_WARMUP_RATIO\`, \`AUTORESEARCH_WARMDOWN_RATIO\`, \`AUTORESEARCH_FINAL_LR_FRAC\`
   - Control/replication only: \`AUTORESEARCH_SEED\`

2. Isolated code experiments:
   - Edit only experiment-repo \`train.py\`
   - Keep the same metric, validation data, tokenizer, and 300-second budget
   - Prefer compact, interpretable changes over broad rewrites
   - Use this for official-style search directions such as attention/window changes, optimizer restructuring, architecture simplifications, or combinations not exposed as env vars
   - Treat the experiment repo git history as the branch state for keep/discard: later code experiments should inherit only from code paths the manager has decided to keep

Constraints:

- Integer knobs must be positive integers.
- \`AUTORESEARCH_ASPECT_RATIO\` is not a multiplier; use integer values such as 48, 56, 64, 72, or 80.
- \`AUTORESEARCH_WINDOW_PATTERN\` may contain only \`S\` and \`L\`.
- \`AUTORESEARCH_ADAM_BETAS\` must be two comma-separated floats.
- \`AUTORESEARCH_TOTAL_BATCH_SIZE\` must be divisible by \`AUTORESEARCH_DEVICE_BATCH_SIZE * 2048\`.
- Do not change data, tokenizer, validation set, metric, \`prepare.py\`, or the evaluation harness.
- Do not modify the shared root \`train.py\`; all code edits must stay inside per-experiment repos.

## Initial portfolio

The first plan must be a diverse research portfolio, not identical replicas. Include at most one baseline/control.

If unsure, start with:

- \`${experimentName}/0001_baseline\`: no extra overrides
- \`${experimentName}/0002_depth8\`: \`AUTORESEARCH_DEPTH=8\`
- \`${experimentName}/0003_depth10\`: \`AUTORESEARCH_DEPTH=10\`
- \`${experimentName}/0004_aspect56\`: \`AUTORESEARCH_ASPECT_RATIO=56\`
- \`${experimentName}/0005_aspect72\`: \`AUTORESEARCH_ASPECT_RATIO=72\`
- \`${experimentName}/0006_head64\`: \`AUTORESEARCH_HEAD_DIM=64\`
- \`${experimentName}/0007_matrix_lr_003\`: \`AUTORESEARCH_MATRIX_LR=0.03\`
- \`${experimentName}/0008_warmdown05\`: \`AUTORESEARCH_WARMDOWN_RATIO=0.5\`, \`AUTORESEARCH_FINAL_LR_FRAC=0.05\`

If you want a larger ready backlog, add controlled nearby variants such as depth 7/11, aspect 48/80, matrix LR 0.025/0.035, warmdown 0.6 with final LR 0.02, or \`AUTORESEARCH_WINDOW_PATTERN=SL\`.
`;
}

export function renderAutoresearchFullWorkerProgram({
  workerScriptPath,
  sandboxScriptPath,
  timeBudgetSeconds = 300,
} = {}) {
  if (!workerScriptPath) throw new Error('workerScriptPath is required');
  if (!sandboxScriptPath) throw new Error('sandboxScriptPath is required');

  return `# autoresearch worker program

You are a worker in a full hpc_agent autoresearch run. Follow the manager's assigned task exactly.

Shared rules:

- Use shell commands directly.
- Work only on the output directory assigned in your task.
- Do not modify \`prepare.py\`, the data, tokenizer, validation set, or evaluation metric.
- Do not invent GPU ids. If your task includes a resource grant block, extract \`gpu_ordinals\` and \`recommended_cpus_per_task\`.
- Use the granted GPU ordinals as \`CUDA_VISIBLE_DEVICES\`. If more than one GPU ordinal is granted, preserve the full comma-separated list.
- Never edit the shared root \`train.py\` during a concurrent run. If code edits are requested, create a git-managed experiment repo under the output directory and edit only that copy.
- For code experiments, treat the experiment repo git history as your local branch state. Make real commits there so the manager can inspect lineage and decide keep/discard.

Experiment runner protocol:

1. Read the task and identify the output directory, hypothesis, whether this is an override-only or code-edit experiment, and the exact \`AUTORESEARCH_*\` override set.
2. Create the output directory.
3. If the task requests code changes, create an isolated git-managed experiment repo under \`<output_dir>/sandbox/repo\` using \`${sandboxScriptPath}\`. If the task names a parent repo, use that repo as the source root so you inherit the kept code path.
4. If the task requests code changes, your first command inside that repo must be \`git checkout -B <experiment_id>\`. Verify the branch exists before editing.
5. If the task requests code changes, edit only that repo's \`train.py\`. Keep \`prepare.py\` unchanged.
6. If the task requests code changes, make a git commit in that repo describing the exact code change before running.
7. Write \`experiment.md\` in the output directory with the experiment id, hypothesis, explicit parent, exact overrides, whether code edits were used, a brief change summary, granted GPU ordinals, requested resources, branch name, commit hash, and timestamp.
8. Run \`${workerScriptPath}\` with that output directory. For code experiments, set \`AUTORESEARCH_REPO_ROOT=<output_dir>/sandbox/repo\`.

Always set:

- \`AUTORESEARCH_TIME_BUDGET_SECONDS=${timeBudgetSeconds}\`
- \`AUTORESEARCH_DISABLE_COMPILE=1\`

If the task provides additional \`AUTORESEARCH_*\` overrides, set them exactly, except do not change the fixed time budget.

Do not wrap the launcher in \`timeout\`, \`timeout(1)\`, or any other external wall-clock killer. The training code enforces \`AUTORESEARCH_TIME_BUDGET_SECONDS\`; the wrapper still needs time after training exits to write \`result.txt\` and \`metrics.json\`.

The wrapper writes \`run.env\`, \`train.log\`, \`result.txt\`, and \`metrics.json\`.

After training:

1. Verify \`result.txt\` exists.
2. Verify \`result.txt\` reports \`exit_code=0\`.
3. Verify \`metrics.json\` exists.
4. For code experiments, add only a short \`suggested_status\` recommendation to \`experiment.md\`: \`keep\`, \`discard\`, or \`control-only\`, plus one sentence of justification. This is advisory only.
5. For code experiments, also record \`git rev-parse --abbrev-ref HEAD\` and \`git rev-parse --short HEAD\` in \`experiment.md\`.
6. Report \`val_bpb\`, \`training_seconds\`, \`num_steps\`, \`peak_vram_mb\`, \`depth\`, \`seed\`, and the relevant config fields.

If the run crashes, report the failure and include the last useful error lines from \`train.log\`. Do not silently retry with a different config unless the task explicitly allows it.

Analysis worker protocol:

1. Read requested \`experiment.md\`, \`result.txt\`, and \`metrics.json\` files.
2. Rank successful runs by lower \`val_bpb\`.
3. Use \`training_seconds\`, \`num_steps\`, throughput, VRAM, and failures as supporting evidence.
4. Include each run's recorded config from \`metrics.json\`.
5. Write the requested analysis file.
6. Recommend concrete follow-up experiments under the same fixed budget.
`;
}

export function renderAutoresearchFullExperimentWorkerSkill({
  workerScriptPath,
  sandboxScriptPath,
  timeBudgetSeconds = 300,
  resultsPath = null,
} = {}) {
  if (!workerScriptPath) throw new Error('workerScriptPath is required');
  if (!sandboxScriptPath) throw new Error('sandboxScriptPath is required');
  const resultsEnv = resultsPath ? ` and \`AUTORESEARCH_RESULTS_PATH=${resultsPath}\`` : '';

  return `---
reports_to: manager
worker_class: experiment_runner
role: Autoresearch Experiment Runner
model: mid
---

You run full-budget autoresearch experiments.

Rules:
- Before doing anything else, read \`worker_program.md\` in the repository root and follow it.
- Use shell commands directly.
- If the task includes a resource grant block, extract \`gpu_ordinals\` and \`recommended_cpus_per_task\`.
- Respect the full requested resource shape. A task may legitimately ask for multiple GPUs or extra CPUs.
- Use \`${workerScriptPath}\`.
- If the task requests code edits, create the git-managed experiment repo with \`${sandboxScriptPath}\`, first run \`git checkout -B <experiment_id>\`, and edit only that repo's \`train.py\`.
- Always set \`AUTORESEARCH_TIME_BUDGET_SECONDS=${timeBudgetSeconds}\` and \`AUTORESEARCH_DISABLE_COMPILE=1\`${resultsEnv}.
- If the task asks for additional \`AUTORESEARCH_*\` overrides, set them exactly, except do not change the fixed time budget.
- Use the exact output directory requested by the manager.
- Write \`experiment.md\` before running.
- For code experiments, create repo git commits and record only a local \`suggested_status\`; the manager decides the final branch status.
- Do not wrap the launcher in \`timeout\`; let the wrapper finish and write \`result.txt\` and \`metrics.json\`.
- Verify \`result.txt\` and \`metrics.json\` after the run.
- Report the key metrics briefly.
`;
}

export function renderAutoresearchFullAnalysisWorkerSkill() {
  return `---
reports_to: manager
worker_class: analyst
role: Autoresearch Result Analyst
model: mid
---

You read completed autoresearch experiment outputs and write factual summaries.

Rules:
- Before doing anything else, read \`worker_program.md\` in the repository root and follow its analysis worker protocol.
- Use shell commands directly.
- Read requested \`metrics.json\`, \`result.txt\`, and \`experiment.md\` files.
- Compare experiments by lower \`val_bpb\`.
- Use \`training_seconds\`, \`num_steps\`, VRAM, and failures as supporting evidence.
- Do not invent metrics.
- Do not recommend a longer time budget as an improvement.
`;
}

export function renderAutoresearchDagFullConfig({ gpuCount = 8 } = {}) {
  const count = Number(gpuCount) || 0;
  return `agentRuntime: codex_cli
cycleIntervalMs: 0
orchestration:
  mode: single_manager
  manager: manager
  maxConcurrentWorkers: ${Math.max(1, count)}
  maxManagerPasses: 1000000
  refillOnGraphDrain: true
  liveReplanOnTaskComplete: true
  liveReplanMinIntervalSeconds: 0
  targetGpuUtilization: 1
  defaultWorkerResources:
    gpus: 0
    cpus: 1
resourceOrchestration:
  enabled: true
  gpuCount: ${count}
  tokenPrefix: gpu
  grantRequiresLease: false
`;
}

export function renderProjectsYaml({ projectId, repoRoot }) {
  return `projects:
  ${projectId}:
    path: ${repoRoot}
`;
}

export function createAutoresearchDagFullWorkspace(options = {}) {
  const paths = buildAutoresearchDagFullPaths(options);
  const workerScriptPath = options.workerScriptPath || DEFAULT_WORKER_SCRIPT_PATH;
  const sandboxScriptPath = options.sandboxScriptPath || DEFAULT_SANDBOX_SCRIPT_PATH;
  const workloadRoot = options.workloadRoot || DEFAULT_WORKLOAD_ROOT;
  const timeBudgetSeconds = Number(options.timeBudgetSeconds ?? 300);
  const gpuCount = Number(options.gpuCount ?? 8);
  const experimentWorkerCount = Number(options.experimentWorkerCount ?? gpuCount ?? 8);

  mkdirSync(path.dirname(paths.readmePath), { recursive: true });
  mkdirSync(path.dirname(paths.experimentWorkerSkillPaths[0]), { recursive: true });
  mkdirSync(paths.expOutputDir, { recursive: true });

  writeFileSync(paths.readmePath, renderAutoresearchDagFullReadme({
    experimentName: paths.experimentName,
    workerScriptPath,
    sandboxScriptPath,
    timeBudgetSeconds,
    gpuCount,
  }), 'utf8');
  writeFileSync(paths.managerProgramPath, renderAutoresearchFullManagerProgram({
    experimentName: paths.experimentName,
    workerScriptPath,
    sandboxScriptPath,
    workloadRoot,
    timeBudgetSeconds,
    gpuCount,
  }), 'utf8');
  writeFileSync(paths.workerProgramPath, renderAutoresearchFullWorkerProgram({
    workerScriptPath,
    sandboxScriptPath,
    timeBudgetSeconds,
  }), 'utf8');
  for (const workerSkillPath of paths.experimentWorkerSkillPaths.slice(0, experimentWorkerCount)) {
    writeFileSync(workerSkillPath, renderAutoresearchFullExperimentWorkerSkill({
      workerScriptPath,
      sandboxScriptPath,
      timeBudgetSeconds,
      resultsPath: toPosix(paths.resultsPath),
    }), 'utf8');
  }
  writeFileSync(paths.mayaSkillPath, renderAutoresearchFullAnalysisWorkerSkill(), 'utf8');
  writeFileSync(paths.configPath, renderAutoresearchDagFullConfig({ gpuCount }), 'utf8');
  writeFileSync(paths.projectsYamlPath, renderProjectsYaml({
    projectId: paths.projectId,
    repoRoot: toPosix(paths.repoRoot),
  }), 'utf8');

  return paths;
}
