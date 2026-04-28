import { mkdirSync, writeFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// __dirname here is <repo>/tasks/qwen-sft
const TASK_ROOT = __dirname;
const REPO_ROOT = path.resolve(TASK_ROOT, '../..');
const DEFAULT_WORKER_SCRIPT_PATH = path.join(TASK_ROOT, 'worker.sh');
const DEFAULT_SANDBOX_SCRIPT_PATH = path.join(TASK_ROOT, 'sandbox.sh');
const DEFAULT_WORKLOAD_ROOT = path.join(TASK_ROOT, 'source');

function toPosix(relativePath) {
  return String(relativePath || '').split(path.sep).join('/');
}

export function buildQwenSftPaths({
  artifactRoot,
  projectId = 'qwen-sft',
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
    path.join(projectRoot, 'skills', 'workers', `sft_runner_${index}.md`)
  ));

  return {
    artifactRoot: root,
    projectId,
    projectRoot,
    repoRoot,
    expOutputDir,
    resultsPath: path.join(expOutputDir, 'results.tsv'),
    // Reuse the same persistent shared cache as autoresearch — both are
    // PyTorch + uv + HF, kernel sets overlap, so cycle-1 of this task can
    // also reuse already-compiled inductor / triton kernels from prior runs.
    sharedCacheRoot: process.env.AUTORESEARCH_SHARED_CACHE_ROOT
      || path.join(process.env.HOME || root, '.cache', 'autoresearch-shared-cache'),
    experimentName,
    readmePath: path.join(repoRoot, 'README.md'),
    managerProgramPath: path.join(repoRoot, 'program.md'),
    workerProgramPath: path.join(repoRoot, 'worker_program.md'),
    configPath: path.join(projectRoot, 'config.yaml'),
    projectsYamlPath: path.join(root, 'projects.yaml'),
    statePath: path.join(projectRoot, 'state.json'),
    experimentWorkerSkillPaths,
    mayaSkillPath: path.join(projectRoot, 'skills', 'workers', 'maya.md'),
  };
}

// Headless sbatch runs need the project to start unpaused.
export function renderInitialProjectState() {
  return JSON.stringify({
    cycleCount: 0,
    epochCount: 0,
    completedAgents: [],
    currentCycleId: null,
    currentSchedule: null,
    isPaused: false,
    phase: 'idle',
    isComplete: false,
    completionSuccess: false,
    completionMessage: null,
    resourceOrchestration: null,
    lastUpdated: new Date().toISOString(),
  }, null, 2);
}

export function renderQwenSftReadme({
  experimentName = 'experiments',
  workerScriptPath,
  sandboxScriptPath,
  workloadRoot = DEFAULT_WORKLOAD_ROOT,
  timeBudgetSeconds = 600,
  gpuCount = 8,
} = {}) {
  if (!workerScriptPath) throw new Error('workerScriptPath is required');
  if (!sandboxScriptPath) throw new Error('sandboxScriptPath is required');

  return `# Qwen-SFT Data-Mix Search

This run drives a manager + worker-pool search over **SFT data-mixture weights** for fine-tuning Qwen3-0.6B on Tulu-3. Each experiment fully fine-tunes the model for ${timeBudgetSeconds}s on the chosen mix, then reports balanced held-out \`val_loss\`.

Read these before planning:
- \`program.md\` — manager protocol
- \`worker_program.md\` — what each worker does
- The trainer at \`${workloadRoot}/train.py\` — module-level \`DATA_MIX\` dict is the only thing that should change between experiments
- Source bucket sizes at \`~/.cache/qwen-sft/data/manifest.json\`

## Goal

Lowest balanced \`val_loss\` (lower is better) under a fixed wall-clock SFT budget of ${timeBudgetSeconds}s per experiment. The val set is 200 samples per bucket × 5 buckets = 1000 balanced samples, so upweighting any single bucket does NOT improve the metric — only the actual data mix that produces a generally-stronger model wins.

## What changes between experiments

Only \`DATA_MIX\` in \`${workloadRoot}/train.py\`. Five buckets (math / code / chat / if / reasoning); weights normalize to 1.0. Don't touch model size, seq len, learning rate, or anything else — those are the search baseline.

The runner clones \`${workloadRoot}\` to a per-experiment sandbox, applies your \`DATA_MIX\` edit via the framework's \`param_patch\` execution mode (no worker LLM, no per-task LLM cost), commits, and runs \`${workerScriptPath} <output_dir>\`.
`;
}

export function renderQwenSftManagerProgram({
  experimentName = 'experiments',
  workerScriptPath,
  sandboxScriptPath,
  workloadRoot = DEFAULT_WORKLOAD_ROOT,
  timeBudgetSeconds = 600,
  gpuCount = 8,
} = {}) {
  if (!workerScriptPath) throw new Error('workerScriptPath is required');
  if (!sandboxScriptPath) throw new Error('sandboxScriptPath is required');
  const experimentsPerWave = Math.max(1, Number.parseInt(gpuCount, 10) || 1);
  const backlogTarget = experimentsPerWave * 2;

  return `# qwen-sft manager program

You are the research manager for a **data-mix search** over Tulu-3 SFT mixtures applied to Qwen3-0.6B full fine-tuning. The framework dispatches your tasks to a pool of GPU workers; you propose new \`DATA_MIX\` weights and read back \`val_loss\`.

## Goal

Minimize balanced held-out \`val_loss\` under a fixed ${timeBudgetSeconds}s SFT budget per experiment. The val set is 200 samples per bucket across all five buckets (math, code, chat, if, reasoning), so the metric is *gameable-resistant*: a mix that overfits one easy bucket will lose on the others.

## What you change

Only \`DATA_MIX\` in \`${workloadRoot}/train.py\`. Everything else (model size = Qwen3-0.6B, seq_len = 2048, micro_batch = 16, grad_accum = 1, learning_rate = 1e-5, warmup = 50, weight_decay = 0) is fixed across experiments. Don't propose to change them via \`code_edit\` — those are the controlled axes.

The five buckets and their patterns are documented in \`~/.cache/qwen-sft/data/manifest.json\`. Inspect it before cycle 1 so you know each bucket's size and what's in it.

## How you dispatch

Output \`<!-- TASK_GRAPH -->\` blocks. Default to \`execution_mode: "param_patch"\` (the framework runs the edit without a worker LLM). Each task:

- \`worker_class: "experiment_runner"\`
- \`execution_mode: "param_patch"\` — required for cheap dispatch
- \`base_ref: "HEAD"\` — leave as default
- \`resources: {"gpus": 1, "cpus": 2}\` — Qwen3-0.6B full FT fits in <12 GB; 1 H200 has tons of headroom
- Unique \`id\` like \`exp_0042_math_heavy\`
- Unique \`produces_tags\` like \`["metrics:exp_0042_math_heavy"]\`
- A \`task\` body that's a one-line description (it's logged but not executed)
- A \`rationale\`: one sentence on why this mix should help
- An \`edits\` array with exactly ONE \`constant_replace\` on \`DATA_MIX\` (see below)
- \`estimated_runtime_seconds: ${timeBudgetSeconds + 60}\`

### The \`DATA_MIX\` edit shape

\`\`\`json
{"file":"train.py","kind":"constant_replace","name":"DATA_MIX",
 "expected_old_repr":"{'math': 0.20, 'code': 0.20, 'chat': 0.20, 'if': 0.20, 'reasoning': 0.20}",
 "new_repr":"{'math': 0.40, 'code': 0.20, 'chat': 0.20, 'if': 0.10, 'reasoning': 0.10}"}
\`\`\`

The framework normalizes both sides via Python AST so dict-key order and whitespace don't matter — you just need the *current* value (read \`train.py\` if you're not sure) and your new value. Both must be parseable Python literals.

Constraints:
- All five keys must be present.
- All values must be \`>= 0\`. Do not normalize manually — the trainer normalizes the dict to sum to 1 internally, so e.g. \`{'math': 2, 'code': 1, ...}\` is equivalent to \`{'math': 2/n, 'code': 1/n, ...}\`. Stick to decimals between 0.0 and 1.0 for readability.
- Don't add new keys (no new buckets exist).

### Worked TASK_GRAPH

\`\`\`json
<!-- TASK_GRAPH -->
{"tasks":[
  {"id":"exp_0001_baseline","worker_class":"experiment_runner",
   "execution_mode":"param_patch","base_ref":"HEAD",
   "resources":{"gpus":1,"cpus":2},"priority":3,
   "estimated_runtime_seconds":${timeBudgetSeconds + 60},
   "rationale":"Uniform 0.2/0.2/0.2/0.2/0.2 baseline so we have a regression line.",
   "task":"baseline uniform mix",
   "produces_tags":["metrics:exp_0001_baseline"],
   "edits":[
     {"file":"train.py","kind":"constant_replace","name":"DATA_MIX",
      "expected_old_repr":"{'math': 0.20, 'code': 0.20, 'chat': 0.20, 'if': 0.20, 'reasoning': 0.20}",
      "new_repr":"{'math': 0.20, 'code': 0.20, 'chat': 0.20, 'if': 0.20, 'reasoning': 0.20}"}
   ]},
  {"id":"exp_0002_math_heavy","worker_class":"experiment_runner",
   "execution_mode":"param_patch","base_ref":"HEAD",
   "resources":{"gpus":1,"cpus":2},"priority":2,
   "estimated_runtime_seconds":${timeBudgetSeconds + 60},
   "rationale":"Tulu's natural math share is small (~12%); double it to test if math is starved at uniform.",
   "task":"math-heavy mix",
   "produces_tags":["metrics:exp_0002_math_heavy"],
   "edits":[
     {"file":"train.py","kind":"constant_replace","name":"DATA_MIX",
      "expected_old_repr":"{'math': 0.20, 'code': 0.20, 'chat': 0.20, 'if': 0.20, 'reasoning': 0.20}",
      "new_repr":"{'math': 0.40, 'code': 0.15, 'chat': 0.15, 'if': 0.15, 'reasoning': 0.15}"}
   ]}
]}
<!-- /TASK_GRAPH -->
\`\`\`

## Priority and killing tasks

Each task may carry an integer \`priority\` (default 0; higher runs first). Use it when one experiment is much more decisive than the others queued.

To abort a still-running experiment that's been invalidated by a more recent result:

\`\`\`
<!-- KILL_TASKS -->
["exp_0042"]
<!-- /KILL_TASKS -->
\`\`\`

The runner SIGTERMs the worker (taking down its train.py via process group) and frees the GPU. Don't kill experiments that are >80% through their budget — the cost is already paid.

## Rationale is mandatory — it is your memory

Every task you emit MUST include a \`rationale\` field stating, in one sentence, **what mix property you are testing and why you expect it to help**. Examples:
- \`"rationale": "math=0.4 to test if doubling math from uniform reduces val_loss when reasoning naturally co-trains with it"\`
- \`"rationale": "if-heavy (0.4) since instruction-following typically generalizes; chat down to 0.05 to free budget"\`
- \`"rationale": "code+if pair test — both rely on structured outputs; if their gains are additive we want exp_0010's mix"\`

Why this is non-negotiable: the runtime context you receive next time will replay each completed experiment as **rationale + edit_summary → val_loss**. Without rationale you literally cannot tell, on the next call, whether a winning mix won by signal or noise — the rationale is your reasoning trace across cycles.

## When tasks fail

A failed \`param_patch\` writes \`${experimentName}/<id>/failure.json\` with the structured reason. Most common: \`expected_old_repr\` didn't match because the previous (kept) experiment shifted \`DATA_MIX\` already and your read is stale. Re-read \`train.py\` (or the parent's \`train.py\`) before emitting more edits in the same neighborhood.

## The loop

You will be called continuously. On each call:

1. **Read your past hypotheses next to their results** in the runtime context. Look for *patterns*: which axis (math, code, chat, if, reasoning) actually moved \`val_loss\`? Which "should have helped" mix didn't? Disqualify dead axes; double down on live ones.
2. If you need more detail than the ledger one-liners, read \`${experimentName}/<id>/metrics.json\` directly — it includes the \`data_mix_normalized\` you actually ran.
3. Decide what to try next. **Exploit when you have a winner**: emit a fan-out around it (e.g., winner is math=0.4 → try math=0.35, math=0.45, plus math=0.4 combined with the next-best axis). **Explore when nothing's separating from baseline**: shift to a different bucket-pair you haven't probed.
4. Append new tasks to the live TASK_GRAPH. Each task carries a \`rationale\`. Do not repeat task ids. Tasks must not depend on still-running ids or unproduced tags.
5. Keep around ${experimentsPerWave}–${backlogTarget} ready/pending GPU tasks queued so tokens never sit idle.

Don't emit \`PROJECT_COMPLETE\`. If you run out of ideas: combine prior near-misses, try sharp / extreme mixes (e.g., code+math each at 0.45, others 0.0333), or revisit any single-bucket dominance hypotheses.

## First wave

\`AUTORESEARCH_SHARED_CACHE_ROOT\` persists across sbatch runs, so the first wave usually hits a warm \`torch.compile\` cache from prior runs (autoresearch and qwen-sft share the same cache directory — kernels overlap).

Cycle 1: dispatch ${experimentsPerWave} \`param_patch\` tasks in parallel from t=0. Mix should include:
- One uniform baseline (\`{0.2, 0.2, 0.2, 0.2, 0.2}\`) — gives the regression line
- One per-bucket "heavy" mix (each bucket as the dominant ~0.4 share, others ~0.15) — five tasks, gives a coarse signal on which bucket matters most
- One or two "two-bucket" combinations (e.g., math+code each 0.4) — tests pairwise interactions

Don't make cycle 1 a fine LR-sweep around uniform — that explores the smallest dimension first.
`;
}

export function renderQwenSftWorkerProgram({
  workerScriptPath,
  sandboxScriptPath,
  timeBudgetSeconds = 600,
} = {}) {
  if (!workerScriptPath) throw new Error('workerScriptPath is required');
  if (!sandboxScriptPath) throw new Error('sandboxScriptPath is required');

  return `# qwen-sft worker program

You execute one Qwen-SFT data-mix experiment. The framework's \`param_patch\` direct executor handles the typical case (apply \`DATA_MIX\` edit, run \`worker.sh\`, return). This program is for the LLM-worker fallback path used by \`code_edit\` / \`llm_repair\` execution modes.

## Loop

1. Read the task. It has the output directory, the parent repo, and the manager's intent.
2. \`${sandboxScriptPath} <parent_repo> <output_dir>/sandbox/repo\` to clone the parent. \`cd\` in, \`git checkout -B <experiment_id>\`. Apply the edit (if \`code_edit\`, do whatever the manager described; if \`llm_repair\`, fix the failure described in \`failure.json\`). \`git add -A && git commit\`.
3. Write \`experiment.md\` in the output directory with: experiment id, hypothesis, parent, exact code edit, granted GPU ordinals, branch, commit hash.
4. Set \`SFT_TIME_BUDGET_SECONDS=${timeBudgetSeconds}\` and \`CUDA_VISIBLE_DEVICES=<granted_gpu_ordinals>\`.
5. Run \`${workerScriptPath} <output_dir>\`. Do NOT wrap in \`timeout\` — the wrapper enforces the budget itself.
6. After it exits, verify \`result.txt\` reports \`exit_code=0\` and \`metrics.json\` exists with a finite \`val_loss\`.
7. Report \`val_loss\`, \`training_seconds\`, \`num_steps\`, and the data-mix you actually ran.

## On crashes

Report the last useful lines of \`train.log\`. Don't silently change the config and re-run.

## Analysis tasks

If the manager schedules you with \`worker_class: "analyst"\`, just read the requested \`metrics.json\` / \`experiment.md\` files, rank by \`val_loss\` (lower is better), and write the requested analysis file. No GPU.
`;
}

export function renderQwenSftExperimentWorkerSkill({
  workerScriptPath,
  sandboxScriptPath,
  timeBudgetSeconds = 600,
  resultsPath = null,
} = {}) {
  if (!workerScriptPath) throw new Error('workerScriptPath is required');
  if (!sandboxScriptPath) throw new Error('sandboxScriptPath is required');
  const resultsEnv = resultsPath ? ` and \`QWEN_SFT_RESULTS_PATH=${resultsPath}\`` : '';

  return `---
reports_to: manager
worker_class: experiment_runner
role: Qwen-SFT Experiment Runner
model: low
---

You run one Qwen-SFT data-mix experiment per task. Read \`worker_program.md\` first and follow it. Use the granted GPU ordinals from your task's resource grant. Always set \`SFT_TIME_BUDGET_SECONDS=${timeBudgetSeconds}\`${resultsEnv}.
`;
}

export function renderQwenSftAnalysisWorkerSkill() {
  return `---
reports_to: manager
worker_class: analyst
role: Qwen-SFT Result Analyst
model: mid
---

You read completed Qwen-SFT experiment outputs and write factual summaries.

Rules:
- Before doing anything else, read \`worker_program.md\` in the repository root and follow its analysis worker protocol.
- Use shell commands directly.
- Read requested \`metrics.json\`, \`result.txt\`, and \`experiment.md\` files.
- Compare experiments by lower \`val_loss\`.
- Use \`training_seconds\`, \`num_steps\`, peak VRAM, and the actual \`data_mix_normalized\` as supporting evidence.
- Do not invent metrics.
- Do not recommend a longer time budget as an improvement.
`;
}

export function renderQwenSftConfig({
  gpuCount = 8,
  agentRuntime = 'codex_cli',
  directExecutor = null,
} = {}) {
  const runtime = ['codex_cli', 'claude_cli', 'api'].includes(agentRuntime) ? agentRuntime : 'codex_cli';
  const count = Number(gpuCount) || 0;
  const directBlock = directExecutor ? renderDirectExecutorBlock(directExecutor) : '';
  return `agentRuntime: ${runtime}
cycleIntervalMs: 0
orchestration:
  mode: single_manager
  manager: manager
  maxConcurrentWorkers: ${Math.max(1, count)}
  maxManagerPasses: 1000000
  refillOnGraphDrain: true
  liveReplanOnTaskComplete: true
  liveReplanMinIntervalSeconds: 30
  targetGpuUtilization: 1
  defaultWorkerResources:
    gpus: 0
    cpus: 1
resourceOrchestration:
  enabled: true
  gpuCount: ${count}
  tokenPrefix: gpu
  grantRequiresLease: false
${directBlock}`;
}

/**
 * Render the optional `directExecutor:` block. Wires the framework's
 * non-LLM execution path to this task plugin so manager-emitted
 * `param_patch` tasks can run without a worker LLM.
 */
function renderDirectExecutorBlock({
  sourceRepoPath,
  wrapperScript,
  sandboxRoot,
  outputRoot,
  resultsPath = null,
  sharedCacheRoot = null,
  timeBudgetSeconds = 600,
  hardCapSeconds = 2700,
}) {
  const envEntries = [];
  if (resultsPath) envEntries.push(`    QWEN_SFT_RESULTS_PATH: ${resultsPath}`);
  if (sharedCacheRoot) envEntries.push(`    AUTORESEARCH_SHARED_CACHE_ROOT: ${sharedCacheRoot}`);
  // Reuse the existing HF cache so Qwen3-0.6B isn't re-downloaded per run.
  const hfHome = path.join(process.env.HOME || '', '.cache', 'huggingface');
  if (process.env.HOME) envEntries.push(`    HF_HOME: ${hfHome}`);
  const envLines = envEntries.length
    ? `  envOverrides:\n${envEntries.join('\n')}\n`
    : '';
  return `directExecutor:
  enabled: true
  sourceRepoPath: ${sourceRepoPath}
  wrapperScript: ${wrapperScript}
  sandboxRoot: ${sandboxRoot}
  outputRoot: ${outputRoot}
  metricKey: val_loss
  timeBudgetSeconds: ${timeBudgetSeconds}
  hardCapSeconds: ${hardCapSeconds}
${envLines}`;
}

export function renderProjectsYaml({ projectId, repoRoot, dataDir = null }) {
  const dataDirLine = dataDir ? `    dataDir: ${dataDir}\n` : '';
  return `projects:
  ${projectId}:
    path: ${repoRoot}
${dataDirLine}`;
}

export function createQwenSftWorkspace(options = {}) {
  const paths = buildQwenSftPaths(options);
  const workerScriptPath = options.workerScriptPath || DEFAULT_WORKER_SCRIPT_PATH;
  const sandboxScriptPath = options.sandboxScriptPath || DEFAULT_SANDBOX_SCRIPT_PATH;
  const workloadRoot = options.workloadRoot || DEFAULT_WORKLOAD_ROOT;
  const timeBudgetSeconds = Number(options.timeBudgetSeconds ?? 600);
  const gpuCount = Number(options.gpuCount ?? 8);
  const experimentWorkerCount = Number(options.experimentWorkerCount ?? gpuCount ?? 8);
  const agentRuntime = options.agentRuntime || 'codex_cli';

  mkdirSync(path.dirname(paths.readmePath), { recursive: true });
  mkdirSync(path.dirname(paths.experimentWorkerSkillPaths[0]), { recursive: true });
  mkdirSync(paths.expOutputDir, { recursive: true });
  mkdirSync(paths.sharedCacheRoot, { recursive: true });

  writeFileSync(paths.readmePath, renderQwenSftReadme({
    experimentName: paths.experimentName,
    workerScriptPath,
    sandboxScriptPath,
    workloadRoot,
    timeBudgetSeconds,
    gpuCount,
  }), 'utf8');
  writeFileSync(paths.managerProgramPath, renderQwenSftManagerProgram({
    experimentName: paths.experimentName,
    workerScriptPath,
    sandboxScriptPath,
    workloadRoot,
    timeBudgetSeconds,
    gpuCount,
  }), 'utf8');
  writeFileSync(paths.workerProgramPath, renderQwenSftWorkerProgram({
    workerScriptPath,
    sandboxScriptPath,
    timeBudgetSeconds,
  }), 'utf8');
  for (const workerSkillPath of paths.experimentWorkerSkillPaths.slice(0, experimentWorkerCount)) {
    writeFileSync(workerSkillPath, renderQwenSftExperimentWorkerSkill({
      workerScriptPath,
      sandboxScriptPath,
      timeBudgetSeconds,
      resultsPath: toPosix(paths.resultsPath),
    }), 'utf8');
  }
  writeFileSync(paths.mayaSkillPath, renderQwenSftAnalysisWorkerSkill(), 'utf8');
  writeFileSync(paths.configPath, renderQwenSftConfig({
    gpuCount,
    agentRuntime,
    directExecutor: {
      sourceRepoPath: toPosix(workloadRoot),
      wrapperScript: toPosix(workerScriptPath),
      sandboxRoot: toPosix(paths.expOutputDir),
      outputRoot: toPosix(paths.expOutputDir),
      resultsPath: toPosix(paths.resultsPath),
      sharedCacheRoot: toPosix(paths.sharedCacheRoot),
      timeBudgetSeconds,
      hardCapSeconds: timeBudgetSeconds + 900,
    },
  }), 'utf8');
  writeFileSync(paths.projectsYamlPath, renderProjectsYaml({
    projectId: paths.projectId,
    repoRoot: toPosix(paths.repoRoot),
    dataDir: toPosix(paths.projectRoot),
  }), 'utf8');
  writeFileSync(paths.statePath, renderInitialProjectState(), 'utf8');

  return paths;
}
