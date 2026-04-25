import { mkdirSync, writeFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const RUNNER_ROOT = path.resolve(__dirname, '..');
const DEFAULT_WORKER_SCRIPT_PATH = path.join(RUNNER_ROOT, 'scripts', 'run-autoresearch-worker.sh');

function toPosix(relativePath) {
  return String(relativePath || '').split(path.sep).join('/');
}

export function buildAutoresearchDagSmokePaths({
  artifactRoot,
  projectId = 'autoresearch-dag-smoke',
  experimentName = 'exp1',
  experimentWorkerCount = 8,
} = {}) {
  if (!artifactRoot) {
    throw new Error('artifactRoot is required');
  }
  if (!projectId) {
    throw new Error('projectId is required');
  }
  if (!experimentName) {
    throw new Error('experimentName is required');
  }

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
    experimentName,
    readmePath: path.join(repoRoot, 'README.md'),
    managerProgramPath: path.join(repoRoot, 'program.md'),
    workerProgramPath: path.join(repoRoot, 'worker_program.md'),
    configPath: path.join(projectRoot, 'config.yaml'),
    projectsYamlPath: path.join(root, 'projects.yaml'),
    aliceSkillPath: experimentWorkerSkillPaths[0],
    experimentWorkerSkillPaths,
    mayaSkillPath: path.join(projectRoot, 'skills', 'workers', 'maya.md'),
  };
}

export function renderAutoresearchDagSmokeReadme({
  experimentName = 'exp1',
  workerScriptPath,
  timeBudgetSeconds = 10,
  minimumIterations = 2,
  maximumIterations = 3,
  gpuCount = 8,
} = {}) {
  if (!workerScriptPath) {
    throw new Error('workerScriptPath is required');
  }
  const experimentsPerWave = Math.max(1, Number.parseInt(gpuCount, 10) || 1);
  const backlogTarget = experimentsPerWave * 2;
  const minWaves = Math.max(1, Number.parseInt(minimumIterations, 10) || 1);
  const maxWaves = Math.max(minWaves, Number.parseInt(maximumIterations, 10) || minWaves);

  return `# Autoresearch DAG Smoke Goal

This is a live smoke for the iterative DAG scheduler with GPU-backed autoresearch runs. It should exercise the same research behavior described by the top-level autoresearch README: the manager designs short model/training experiments, runs them for a fixed wall-clock training budget, compares them using \`val_bpb\` where lower is better, and uses the feedback to choose the next experiment wave.

Read the operating programs before planning or running work:

- Manager program: \`program.md\`
- Worker program: \`worker_program.md\`

The README is the goal statement. \`program.md\` is the manager's standing research protocol. \`worker_program.md\` is the protocol that every experiment and analysis worker must follow.

The goal is complete when all of these are true:
1. At least ${minWaves} experiment waves have completed.
2. Each worker completion returns its resource token and triggers live manager replanning while other workers continue running. The manager should inspect the finished result, the still-running tasks, and prior artifacts, then append only independent experiments that do not depend on currently running task ids or tags that currently running tasks have not produced yet.
3. Each experiment directory has \`result.txt\` reporting \`exit_code=0\` and a \`metrics.json\`.
4. Each wave has an \`analysis.txt\` that compares experiments by \`val_bpb\`, \`training_seconds\`, \`num_steps\`, and the recorded config.
5. The manager emits \`<!-- PROJECT_COMPLETE -->\` only after the minimum iteration count is reached and the analysis says no useful next wave remains, or after ${maxWaves} waves have completed.

Smoke iteration cap:
- This is a smoke test, not an unbounded overnight run. Run at least ${minWaves} waves and at most ${maxWaves} waves.
- If wave ${maxWaves} completes and the analysis still sees plausible next experiments, do not launch wave ${maxWaves + 1}; emit \`PROJECT_COMPLETE\` and include the recommended next wave in the completion message or final summary.
- Never exceed ${maxWaves} experiment waves in this smoke.

Required orchestration shape:
- Use \`<!-- TASK_GRAPH -->\`.
- First run one full GPU experiment wave under \`${experimentName}/<experiment_id>\`.
- The first wave must be a research portfolio, not identical replicates: include one baseline/control at most, then distinct variants with different hypotheses and different non-seed \`AUTORESEARCH_*\` overrides.
- Do not schedule copies of the same command in the same wave unless they are explicitly labeled control/seed replicates.
- Keep the GPU queue warm: if there are ${experimentsPerWave} GPUs, schedule enough ready experiment tasks to start the allocation. If the runner calls you for live replanning after a task completes, append only new independent tasks to the active graph. Do not wait for the original wave barrier.
- Experiment tasks must request \`resources: {"gpus": 1, "cpus": 1}\`.
- Experiment tasks must produce tags for their metrics.
- Each experiment task must state a short hypothesis and exact override set in the task text, and must ask the worker to write that hypothesis/override set to \`experiment.md\` in the output directory before running.
- Then run one analyst task that depends on all experiment task ids and produced tags.
- Mark the analyst task with \`replan_after: true\`.
- After replanning, schedule the next wave unless the minimum iteration count has been reached and the analyst found no useful follow-up.
- Also stop after ${maxWaves} completed waves even if a useful next wave exists; summarize that next wave instead of scheduling it.
- Do not emit \`PROJECT_COMPLETE\` after the first successful wave.

Autoresearch experiment rules:
- Keep \`AUTORESEARCH_TIME_BUDGET_SECONDS=${timeBudgetSeconds}\` fixed for all experiments and all waves. Do not use a longer budget as the next-wave "improvement"; that breaks comparability.
- Optimize for lower \`val_bpb\`. Use \`training_seconds\`, \`num_steps\`, throughput, and failures/OOMs as supporting evidence, not as the primary metric.
- Do not change the data, tokenizer, validation set, or evaluation metric.
- Do not use seed-only changes as research variants. \`AUTORESEARCH_SEED\` may be used for a control replicate only after a promising config has been found.
- Prefer controlled changes that are easy to interpret. Avoid changing many unrelated knobs in one task unless the hypothesis explains why.
- Every wave after the first must be based on the previous wave analysis: exploit the best config, test one or two nearby alternatives, and include a baseline or best-so-far control if there is room.

Allowed \`AUTORESEARCH_*\` knobs for isolated smoke experiments:
- Architecture/model size: \`AUTORESEARCH_DEPTH\`, \`AUTORESEARCH_ASPECT_RATIO\`, \`AUTORESEARCH_HEAD_DIM\`, \`AUTORESEARCH_WINDOW_PATTERN\`
- Batch shape: \`AUTORESEARCH_TOTAL_BATCH_SIZE\`, \`AUTORESEARCH_DEVICE_BATCH_SIZE\`
- Optimization: \`AUTORESEARCH_EMBEDDING_LR\`, \`AUTORESEARCH_UNEMBEDDING_LR\`, \`AUTORESEARCH_MATRIX_LR\`, \`AUTORESEARCH_SCALAR_LR\`, \`AUTORESEARCH_WEIGHT_DECAY\`, \`AUTORESEARCH_ADAM_BETAS\`
- Schedule: \`AUTORESEARCH_WARMUP_RATIO\`, \`AUTORESEARCH_WARMDOWN_RATIO\`, \`AUTORESEARCH_FINAL_LR_FRAC\`
- Reproducibility/control only: \`AUTORESEARCH_SEED\`

Value constraints:
- Integer knobs must be positive integers: \`AUTORESEARCH_DEPTH\`, \`AUTORESEARCH_ASPECT_RATIO\`, \`AUTORESEARCH_HEAD_DIM\`, \`AUTORESEARCH_TOTAL_BATCH_SIZE\`, \`AUTORESEARCH_DEVICE_BATCH_SIZE\`, \`AUTORESEARCH_SEED\`.
- \`AUTORESEARCH_ASPECT_RATIO\` is not a multiplier. The default is 64. Use nearby integer values such as 48, 56, 64, 72, or 80. Never use decimals like 0.5, and never use 2 to mean "2x width".
- \`AUTORESEARCH_WINDOW_PATTERN\` may contain only \`S\` and \`L\`, for example \`S\`, \`SL\`, \`LS\`, or \`SSSL\`.
- \`AUTORESEARCH_ADAM_BETAS\` must be two comma-separated floats, for example \`0.8,0.95\`.
- \`AUTORESEARCH_TOTAL_BATCH_SIZE\` must be divisible by \`AUTORESEARCH_DEVICE_BATCH_SIZE * 2048\`.

If unsure, use this safe first-wave portfolio because every override is valid and each task changes one interpretable thing:
- \`wave1/baseline\`: no extra overrides
- \`wave1/depth8\`: \`AUTORESEARCH_DEPTH=8\`
- \`wave1/depth10\`: \`AUTORESEARCH_DEPTH=10\`
- \`wave1/aspect56\`: \`AUTORESEARCH_ASPECT_RATIO=56\`
- \`wave1/aspect72\`: \`AUTORESEARCH_ASPECT_RATIO=72\`
- \`wave1/head64\`: \`AUTORESEARCH_HEAD_DIM=64\`
- \`wave1/matrix_lr_003\`: \`AUTORESEARCH_MATRIX_LR=0.03\`
- \`wave1/warmdown05\`: \`AUTORESEARCH_WARMDOWN_RATIO=0.5\`, \`AUTORESEARCH_FINAL_LR_FRAC=0.05\`

Experiment command requirements:
- Use \`${workerScriptPath}\`
- Use output dirs such as \`${experimentName}/baseline\`, \`${experimentName}/depth8\`, and \`${experimentName}/matrix_lr_003\` for the first wave; use clear repo-relative directories such as \`wave2/best_plus_lr\` for later waves.
- The first wave metrics paths should be under \`${experimentName}/<experiment_id>/metrics.json\`.
- Always use \`AUTORESEARCH_TIME_BUDGET_SECONDS=${timeBudgetSeconds}\` and \`AUTORESEARCH_DISABLE_COMPILE=1\`.
- Use the granted GPU ordinals from the resource grant block, not a hard-coded GPU id
- Pass the manager's experiment-specific config changes as \`AUTORESEARCH_*\` environment variables in the worker command.

Keep iterating until the minimum iteration count is reached and the analysis says there is no useful next experiment under the fixed time budget, but never go past ${maxWaves} waves in this smoke.
`;
}

export function renderAutoresearchManagerProgram({
  experimentName = 'exp1',
  workerScriptPath,
  timeBudgetSeconds = 10,
  minimumIterations = 2,
  maximumIterations = 3,
  gpuCount = 8,
} = {}) {
  if (!workerScriptPath) {
    throw new Error('workerScriptPath is required');
  }
  const experimentsPerWave = Math.max(1, Number.parseInt(gpuCount, 10) || 1);
  const backlogTarget = experimentsPerWave * 2;
  const minWaves = Math.max(1, Number.parseInt(minimumIterations, 10) || 1);
  const maxWaves = Math.max(minWaves, Number.parseInt(maximumIterations, 10) || minWaves);

  return `# autoresearch manager program

You are the research manager for this hpc_agent autoresearch smoke. Your job is to run a small but real autonomous research loop: design experiments, dispatch GPU workers, read measured feedback, and design the next wave.

This program intentionally mirrors the top-level autoresearch \`program.md\`, but adapted for a multi-agent shared-allocation setup. In the original workflow one agent edits \`train.py\` directly. In this smoke, workers run concurrently, so the editable \`train.py\` knobs are exposed as isolated \`AUTORESEARCH_*\` environment overrides. Treat those overrides as the experiment design surface.

## Setup

Every manager pass starts by reading:

1. \`README.md\` — the goal statement.
2. \`program.md\` — this manager program.
3. \`worker_program.md\` — the protocol workers must follow.
4. Existing \`wave*/analysis.txt\`, \`wave*/*/metrics.json\`, \`wave*/*/result.txt\`, and \`wave*/*/experiment.md\` files.

Do not assume the previous pass finished. Inspect artifacts first, then decide whether to emit a new \`TASK_GRAPH\` or \`PROJECT_COMPLETE\`.

## Research objective

The goal is simple: find a lower \`val_bpb\` under a fixed wall-clock training budget. Lower \`val_bpb\` is better.

Use \`training_seconds\`, \`num_steps\`, throughput, VRAM, and failures as supporting evidence. Do not optimize those metrics directly if \`val_bpb\` gets worse.

## Time budget

Every experiment in every wave must use:

- \`AUTORESEARCH_TIME_BUDGET_SECONDS=${timeBudgetSeconds}\`
- \`AUTORESEARCH_DISABLE_COMPILE=1\`

Do not use a longer time budget as an improvement. That breaks comparability.

## Scope

You may vary only isolated \`AUTORESEARCH_*\` overrides supported by \`train.py\`.

Allowed knobs:

- Architecture/model size: \`AUTORESEARCH_DEPTH\`, \`AUTORESEARCH_ASPECT_RATIO\`, \`AUTORESEARCH_HEAD_DIM\`, \`AUTORESEARCH_WINDOW_PATTERN\`
- Batch shape: \`AUTORESEARCH_TOTAL_BATCH_SIZE\`, \`AUTORESEARCH_DEVICE_BATCH_SIZE\`
- Optimization: \`AUTORESEARCH_EMBEDDING_LR\`, \`AUTORESEARCH_UNEMBEDDING_LR\`, \`AUTORESEARCH_MATRIX_LR\`, \`AUTORESEARCH_SCALAR_LR\`, \`AUTORESEARCH_WEIGHT_DECAY\`, \`AUTORESEARCH_ADAM_BETAS\`
- Schedule: \`AUTORESEARCH_WARMUP_RATIO\`, \`AUTORESEARCH_WARMDOWN_RATIO\`, \`AUTORESEARCH_FINAL_LR_FRAC\`
- Control/replication only: \`AUTORESEARCH_SEED\`

Value constraints:

- Integer knobs must be positive integers: \`AUTORESEARCH_DEPTH\`, \`AUTORESEARCH_ASPECT_RATIO\`, \`AUTORESEARCH_HEAD_DIM\`, \`AUTORESEARCH_TOTAL_BATCH_SIZE\`, \`AUTORESEARCH_DEVICE_BATCH_SIZE\`, \`AUTORESEARCH_SEED\`.
- \`AUTORESEARCH_ASPECT_RATIO\` is not a multiplier. The default is 64. Use nearby integer values such as 48, 56, 64, 72, or 80. Never use decimals like 0.5, and never use 2 to mean "2x width".
- \`AUTORESEARCH_WINDOW_PATTERN\` may contain only \`S\` and \`L\`, for example \`S\`, \`SL\`, \`LS\`, or \`SSSL\`.
- \`AUTORESEARCH_ADAM_BETAS\` must be two comma-separated floats, for example \`0.8,0.95\`.
- \`AUTORESEARCH_TOTAL_BATCH_SIZE\` must be divisible by \`AUTORESEARCH_DEVICE_BATCH_SIZE * 2048\`.

Do not change data, tokenizer, validation set, metric, \`prepare.py\`, or the evaluation harness.

## Wave protocol

Use \`<!-- TASK_GRAPH -->\` for every wave.

The runtime supports live replanning while a task graph is still running. Every time a worker finishes and returns its resource token, the manager can be called to inspect the new result, the still-running tasks, and prior artifacts, then append new tasks to the active graph while other workers continue running.

Therefore each wave should contain:

1. At least ${experimentsPerWave} one-GPU experiment tasks, each with \`resources: {"gpus": 1, "cpus": 1}\`.
2. Prefer a ready backlog of about ${backlogTarget} experiments when you have enough plausible ideas, so early-finishing GPUs immediately pick up queued work.
3. Use live replanning calls after each completed task to append more independent experiments while workers are still running.
4. One analysis task that depends on the intended checkpoint set of experiment tasks and produced metric tags.
5. \`replan_after: true\` on the analysis task.

Use \`worker_class: "experiment_runner"\` for experiment workers. Use \`agent: "maya"\` for the analysis worker.

Live replanning contract: when called during a running graph, emit a \`TASK_GRAPH\` containing only additional tasks to append. Do not repeat existing task ids. New experiment tasks must not depend on currently-running task ids or tags that currently-running tasks have not produced yet; they may depend on completed task ids, already-produced tags, and prior artifacts. Do not emit \`PROJECT_COMPLETE\` while workers are still running.

When live-appended experiment tasks are merged, the runner skips tasks that depend on currently-running task ids or their not-yet-produced tags, and automatically extends any pending \`replan_after\` analysis barrier so that the barrier waits for accepted new experiments too. You still need to give each appended task a unique id, output directory, hypothesis, and exact override set.

## First wave

The first wave must be a research portfolio, not identical replicates. Include at most one baseline/control, then distinct variants with different hypotheses and different non-seed overrides. For an ${experimentsPerWave}-GPU allocation, schedule at least ${experimentsPerWave} experiments; prefer ${backlogTarget} if you have enough safe variants.

Use output directories under \`${experimentName}/<experiment_id>\`.

If unsure, use this safe first-wave portfolio:

- \`${experimentName}/baseline\`: no extra overrides
- \`${experimentName}/depth8\`: \`AUTORESEARCH_DEPTH=8\`
- \`${experimentName}/depth10\`: \`AUTORESEARCH_DEPTH=10\`
- \`${experimentName}/aspect56\`: \`AUTORESEARCH_ASPECT_RATIO=56\`
- \`${experimentName}/aspect72\`: \`AUTORESEARCH_ASPECT_RATIO=72\`
- \`${experimentName}/head64\`: \`AUTORESEARCH_HEAD_DIM=64\`
- \`${experimentName}/matrix_lr_003\`: \`AUTORESEARCH_MATRIX_LR=0.03\`
- \`${experimentName}/warmdown05\`: \`AUTORESEARCH_WARMDOWN_RATIO=0.5\`, \`AUTORESEARCH_FINAL_LR_FRAC=0.05\`

If you need a ${backlogTarget}-experiment first-wave backlog, extend that safe set with controlled nearby variants such as:

- \`${experimentName}/depth7\`: \`AUTORESEARCH_DEPTH=7\`
- \`${experimentName}/depth11\`: \`AUTORESEARCH_DEPTH=11\`
- \`${experimentName}/aspect48\`: \`AUTORESEARCH_ASPECT_RATIO=48\`
- \`${experimentName}/aspect80\`: \`AUTORESEARCH_ASPECT_RATIO=80\`
- \`${experimentName}/matrix_lr_0035\`: \`AUTORESEARCH_MATRIX_LR=0.035\`
- \`${experimentName}/matrix_lr_0025\`: \`AUTORESEARCH_MATRIX_LR=0.025\`
- \`${experimentName}/warmdown06\`: \`AUTORESEARCH_WARMDOWN_RATIO=0.6\`, \`AUTORESEARCH_FINAL_LR_FRAC=0.02\`
- \`${experimentName}/window_sl\`: \`AUTORESEARCH_WINDOW_PATTERN=SL\`

## Later waves

Every later wave must be based on measured feedback from the previous analysis:

1. Exploit the current best config.
2. Test nearby alternatives or combinations with clear hypotheses.
3. Include one baseline or best-so-far control if useful.
4. Use seed-only runs only as controls/replicates after a promising config exists.

Do not repeat an entire previous wave. Do not schedule identical configs in the same wave unless they are explicitly labeled as seed/control replicates.

## Worker task contract

Every experiment task you emit must tell the worker to:

1. Read \`worker_program.md\` before running.
2. Use \`${workerScriptPath}\`.
3. Use the granted GPU ordinal from the resource grant block, not a hard-coded GPU id.
4. Write \`experiment.md\` in the output directory before running.
5. Use the exact fixed budget and override set.
6. Verify \`result.txt\` has \`exit_code=0\` and \`metrics.json\` exists.

Every analysis task must tell \`maya\` to:

1. Read \`worker_program.md\`.
2. Read all requested \`experiment.md\`, \`result.txt\`, and \`metrics.json\` files.
3. Rank by lower \`val_bpb\`.
4. Recommend one concrete next wave under the same fixed budget, or state why to stop.

## Iteration cap

This is a smoke test, not an unbounded overnight run.

- Run at least ${minWaves} waves.
- Run at most ${maxWaves} waves.
- If wave ${maxWaves} completes and the analysis still suggests a next experiment, do not launch wave ${maxWaves + 1}. Emit \`PROJECT_COMPLETE\` and include the recommended next wave in the message or final summary.

## Completion

Emit \`PROJECT_COMPLETE\` only when:

1. At least ${minWaves} waves completed, and analysis says no useful next wave remains; or
2. ${maxWaves} waves completed, even if analysis still proposes follow-up work.

Use:

\`\`\`
<!-- PROJECT_COMPLETE -->
{"success": true, "message": "short factual reason"}
<!-- /PROJECT_COMPLETE -->
\`\`\`
`;
}

export function renderAutoresearchWorkerProgram({
  workerScriptPath,
  timeBudgetSeconds = 10,
} = {}) {
  if (!workerScriptPath) {
    throw new Error('workerScriptPath is required');
  }

  return `# autoresearch worker program

You are a worker in the hpc_agent autoresearch smoke. Follow the manager's assigned task exactly, but use this document as your standing protocol.

## Shared rules

- Use shell commands directly.
- Work only on the output directory assigned in your task.
- Prefer repo-relative output directories.
- Do not modify \`prepare.py\`, the data, tokenizer, validation set, or evaluation metric.
- Do not invent GPU ids. If your task includes a resource grant block, extract \`gpu_ordinals\` and \`recommended_cpus_per_task\` from it.
- Use the granted GPU ordinal as \`CUDA_VISIBLE_DEVICES\`.

## Experiment runner protocol

Before running training:

1. Read your task and identify the output directory, hypothesis, and exact \`AUTORESEARCH_*\` override set.
2. Create the output directory.
3. Write \`experiment.md\` in that directory with:
   - experiment id / output directory
   - hypothesis
   - exact overrides
   - granted GPU ordinal
4. Run \`${workerScriptPath}\` with that output directory.

Always set:

- \`AUTORESEARCH_TIME_BUDGET_SECONDS=${timeBudgetSeconds}\`
- \`AUTORESEARCH_DISABLE_COMPILE=1\`

If the task provides additional \`AUTORESEARCH_*\` overrides, set them exactly, except do not change the fixed time budget.

Do not wrap the launcher in \`timeout\`, \`timeout(1)\`, or any other external wall-clock killer. The training code enforces \`AUTORESEARCH_TIME_BUDGET_SECONDS\`; the wrapper still needs time after training exits to write \`result.txt\` and \`metrics.json\`.

The wrapper writes:

- \`run.env\`
- \`train.log\`
- \`result.txt\`
- \`metrics.json\`

After training:

1. Verify \`result.txt\` exists.
2. Verify \`result.txt\` reports \`exit_code=0\`.
3. Verify \`metrics.json\` exists.
4. Report \`val_bpb\`, \`training_seconds\`, \`num_steps\`, \`depth\`, \`seed\`, and the relevant config fields.

If the run crashes, report the failure and include the last useful error lines from \`train.log\`. Do not silently retry with a different config unless the task explicitly allows it.

## Analysis worker protocol

When assigned analysis:

1. Read every requested \`experiment.md\`, \`result.txt\`, and \`metrics.json\`.
2. Confirm which runs succeeded and which failed.
3. Rank successful runs by lower \`val_bpb\`.
4. Use \`training_seconds\`, \`num_steps\`, throughput, and failures as supporting evidence.
5. Include each run's recorded config from \`metrics.json\`; also mention \`depth\` and \`seed\`, because they are top-level metrics fields.
6. Flag unintended duplicate full configs. Seed/control replicates are allowed when clearly labeled.
7. Write the requested \`analysis.txt\`.
8. Recommend one concrete next wave of \`AUTORESEARCH_*\` overrides under the same fixed budget, or explain why to stop.

Never recommend a longer time budget as an improvement.
`;
}

export function renderAutoresearchExperimentWorkerSkill({
  workerScriptPath,
  timeBudgetSeconds = 10,
} = {}) {
  if (!workerScriptPath) {
    throw new Error('workerScriptPath is required');
  }

return `---
reports_to: manager
worker_class: experiment_runner
role: Autoresearch Experiment Runner
model: mid
---

You run short deterministic autoresearch experiments.

Rules:
- Before doing anything else, read \`worker_program.md\` in the repository root and follow it.
- Use shell commands directly.
- If the task includes a resource grant block, extract \`gpu_ordinals\` and \`recommended_cpus_per_task\` from it.
- Use \`${workerScriptPath}\`.
- Always set \`AUTORESEARCH_TIME_BUDGET_SECONDS=${timeBudgetSeconds}\` and \`AUTORESEARCH_DISABLE_COMPILE=1\`.
- If the task asks for additional \`AUTORESEARCH_*\` overrides, set them in the command environment exactly, except do not change the fixed time budget.
- Write outputs to the exact output directory requested by the task.
- Prefer repo-relative output directories such as \`exp1\` over absolute artifact-root paths.
- If the task includes a hypothesis and override set, write them to \`experiment.md\` in the output directory before running.
- Do not wrap the launcher in \`timeout\`; let the wrapper finish and write \`result.txt\` and \`metrics.json\`.
- After the run, verify \`result.txt\` and \`metrics.json\` exist.
- Report the key metrics briefly.
`;
}

export function renderAutoresearchAnalysisWorkerSkill() {
return `---
reports_to: manager
worker_class: analyst
role: Autoresearch Result Analyst
model: mid
---

You read completed autoresearch experiment outputs and write short factual summaries.

Rules:
- Before doing anything else, read \`worker_program.md\` in the repository root and follow its analysis worker protocol.
- Use shell commands directly.
- Read the requested \`metrics.json\` files.
- Write concise repo files only when the task asks for them.
- Compare experiments by lower \`val_bpb\`, using \`training_seconds\`, \`num_steps\`, and failures as supporting evidence.
- Read each run's \`config\` from \`metrics.json\` and flag duplicate configs in the same wave unless they were explicitly requested control replicates.
- When asked to support iteration, recommend one concrete next wave of \`AUTORESEARCH_*\` overrides that changes model/training knobs under the same fixed time budget.
- Do not recommend a longer time budget as an improvement.
- Do not invent metrics that are not present in the run outputs.
`;
}

export function renderAutoresearchDagSmokeConfig({ gpuCount = 8 } = {}) {
  const count = Number(gpuCount) || 0;
  return `agentRuntime: codex_cli
cycleIntervalMs: 0
orchestration:
  mode: single_manager
  manager: manager
  maxConcurrentWorkers: ${Math.max(1, count)}
  maxManagerPasses: 8
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

export function createAutoresearchDagSmokeWorkspace(options = {}) {
  const paths = buildAutoresearchDagSmokePaths(options);
  const workerScriptPath = options.workerScriptPath || DEFAULT_WORKER_SCRIPT_PATH;
  const timeBudgetSeconds = Number(options.timeBudgetSeconds ?? 10);
  const gpuCount = Number(options.gpuCount ?? 8);
  const experimentWorkerCount = Number(options.experimentWorkerCount ?? gpuCount ?? 8);

  mkdirSync(path.dirname(paths.readmePath), { recursive: true });
  mkdirSync(path.dirname(paths.aliceSkillPath), { recursive: true });
  mkdirSync(paths.expOutputDir, { recursive: true });

  writeFileSync(
    paths.readmePath,
    renderAutoresearchDagSmokeReadme({
      experimentName: paths.experimentName,
      workerScriptPath,
      timeBudgetSeconds,
      minimumIterations: Number(options.minimumIterations ?? 2),
      maximumIterations: Number(options.maximumIterations ?? 3),
      gpuCount,
    }),
    'utf8',
  );
  writeFileSync(
    paths.managerProgramPath,
    renderAutoresearchManagerProgram({
      experimentName: paths.experimentName,
      workerScriptPath,
      timeBudgetSeconds,
      minimumIterations: Number(options.minimumIterations ?? 2),
      maximumIterations: Number(options.maximumIterations ?? 3),
      gpuCount,
    }),
    'utf8',
  );
  writeFileSync(
    paths.workerProgramPath,
    renderAutoresearchWorkerProgram({
      workerScriptPath,
      timeBudgetSeconds,
    }),
    'utf8',
  );
  for (const workerSkillPath of paths.experimentWorkerSkillPaths) {
    writeFileSync(
      workerSkillPath,
      renderAutoresearchExperimentWorkerSkill({
        workerScriptPath,
        timeBudgetSeconds,
      }),
      'utf8',
    );
  }
  writeFileSync(paths.mayaSkillPath, renderAutoresearchAnalysisWorkerSkill(), 'utf8');
  writeFileSync(paths.configPath, renderAutoresearchDagSmokeConfig({ gpuCount }), 'utf8');
  writeFileSync(
    paths.projectsYamlPath,
    renderProjectsYaml({
      projectId: paths.projectId,
      repoRoot: toPosix(paths.repoRoot),
    }),
    'utf8',
  );

  return paths;
}
