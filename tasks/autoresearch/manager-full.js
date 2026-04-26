import { mkdirSync, writeFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// __dirname here is <repo>/tasks/autoresearch
const TASK_ROOT = __dirname;
const REPO_ROOT = path.resolve(TASK_ROOT, '../..');
const DEFAULT_WORKER_SCRIPT_PATH = path.join(TASK_ROOT, 'worker.sh');
const DEFAULT_SANDBOX_SCRIPT_PATH = path.join(TASK_ROOT, 'sandbox.sh');
const DEFAULT_WORKLOAD_ROOT = path.join(TASK_ROOT, 'source');

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
    sharedCacheRoot: path.join(root, '.shared-cache'),
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

// Headless sbatch runs need the project to start unpaused. ProjectRunner
// otherwise treats a missing state.json as "new project, paused by default"
// (a UI convention) and never enters the manager loop.
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

export function renderAutoresearchDagFullReadme({
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

  return `# Autoresearch run

This run drives the autoresearch experiment loop on a multi-GPU shared allocation. The single-agent script described in \`autoresearch/program.md\` is replaced by a manager + worker pool: the manager designs experiments, workers run them in parallel. The science stays the same — edit \`train.py\`, train for the fixed budget, keep what lowers \`val_bpb\`.

Read these before planning:
- \`program.md\` — manager protocol (how to dispatch workers)
- \`worker_program.md\` — what each worker does
- The official \`${workloadRoot}/program.md\` and \`${workloadRoot}/train.py\` — same research goal, full single-agent context

Goal: minimize \`val_bpb\` under \`AUTORESEARCH_TIME_BUDGET_SECONDS=${timeBudgetSeconds}\` (matches the official autoresearch budget). \`torch.compile\` stays on.

Run continuously, ${experimentsPerWave} GPU tokens at a time, until a human stops the project. Don't emit \`PROJECT_COMPLETE\`.
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

  return `# autoresearch manager program

You are the research manager. You are running the autoresearch loop from \`${workloadRoot}/program.md\` — same goal, same metric, same time budget. Read that file. The research is the same. The only delta is that you dispatch experiments to a worker pool instead of running them yourself.

## Goal

Lowest \`val_bpb\` under \`AUTORESEARCH_TIME_BUDGET_SECONDS=${timeBudgetSeconds}\`. Treat \`training_seconds\`, \`num_steps\`, throughput, VRAM, and failures as supporting evidence only. \`torch.compile\` stays on. Don't stop. Keep iterating until a human halts the project.

## What you change

\`train.py\` only — everything in there is fair game (architecture, optimizer, schedule, batch shape, LR, init, attention pattern, etc.). Don't touch \`prepare.py\`, the metric, the data, or the evaluation harness.

Each experiment runs in its own per-experiment git sandbox cloned from \`${workloadRoot}\`. The runner clones, applies your edit, commits, and runs train.py — you just describe what changes.

## Pick the cheapest execution_mode that fits

Every task carries an \`execution_mode\`. Pick the cheapest one that can express the edit; it controls cost and reliability.

- **\`"param_patch"\`** — DEFAULT. The runner applies a structured \`edits[]\` array directly (no worker LLM, no cost). Use this whenever the change is "swap one or more constants / parameters / short literal regions in train.py". This covers nearly all sweeps: depth, width, head_dim, batch shape, LR, betas, weight decay, attention window, warmup/warmdown shape. **Aim for ≥80% of your tasks to be \`param_patch\`.**
- **\`"code_edit"\`** — falls back to an LLM worker that rewrites code in natural language. Use only when the change is genuinely free-form: architecture restructure, swap optimizer family, add/remove a layer type, rewrite the attention path. Costs an LLM call per task; reserve it.
- **\`"llm_repair"\`** — same as \`code_edit\` but specifically for fixing a failed \`param_patch\`. The worker LLM gets the original \`failure.json\` and the parent commit's \`train.py\`. Emit one of these only when a \`param_patch\` task you care about failed because of an unparseable edit and you want the LLM to land it. Cap at one repair per failed experiment; if it fails again, move on.

## How to write \`edits[]\` (param_patch)

Four kinds, in order of preference:

1. **\`constant_replace\`** — module-level Python assignment. Compares values via Python AST so \`524288\`, \`2**19\`, and \`1<<19\` all match.
   \`\`\`json
   {"file":"train.py","kind":"constant_replace","name":"ADAM_BETAS",
    "expected_old_repr":"(0.9, 0.95)","new_repr":"(0.9, 0.97)"}
   \`\`\`
2. **\`regex_replace\`** — for in-line tuples, dependent string literals, or list elements. \`expected_count\` is mandatory; if your pattern doesn't match exactly that many times, the edit fails loudly instead of corrupting the file.
   \`\`\`json
   {"file":"train.py","kind":"regex_replace",
    "pattern":"head_dim=128","replacement":"head_dim=64","expected_count":1}
   \`\`\`
3. **\`block_replace\`** — multi-line region between two anchors (inclusive of both anchor lines).
   \`\`\`json
   {"file":"train.py","kind":"block_replace",
    "anchor_regex":"^# warmup schedule$","end_regex":"^# end warmup$",
    "new_text":"# warmup schedule\\nwarmup = linear(0.01, 0.1, 100)\\n# end warmup"}
   \`\`\`
4. **\`unified_diff\`** — escape hatch. Pass a full \`git apply\`-compatible diff in the \`diff\` field. Use only when the other three can't express the change.

Multiple edits to the same task are applied atomically; if any one fails, none take effect. Read \`train.py\` (or the parent experiment's \`train.py\`) before emitting edits — if your \`expected_old_repr\` is wrong, the task fails before training starts.

## Worked TASK_GRAPH

\`\`\`json
<!-- TASK_GRAPH -->
{"tasks":[
  {"id":"exp_0042_betas_097","worker_class":"experiment_runner",
   "execution_mode":"param_patch","base_ref":"HEAD",
   "resources":{"gpus":1,"cpus":1},"priority":3,
   "estimated_runtime_seconds":320,
   "rationale":"Higher beta_2 may help under our larger batch.",
   "task":"sweep adam_betas to (0.9, 0.97)",
   "produces_tags":["metrics:exp_0042_betas_097"],
   "edits":[
     {"file":"train.py","kind":"constant_replace","name":"ADAM_BETAS",
      "expected_old_repr":"(0.9, 0.95)","new_repr":"(0.9, 0.97)"}
   ]},
  {"id":"exp_0043_arch_swap_attn","worker_class":"experiment_runner",
   "execution_mode":"code_edit",
   "resources":{"gpus":1,"cpus":1},"priority":2,
   "estimated_runtime_seconds":340,
   "task":"In train.py, replace the existing scaled-dot-product attention with sliding-window attention of window size 128, applied uniformly across all layers. Output dir: \`${experimentName}/exp_0043_arch_swap_attn\`."}
]}
<!-- /TASK_GRAPH -->
\`\`\`

Common fields for both modes:
- \`worker_class: "experiment_runner"\`
- \`resources\`: usually \`{"gpus": 1, "cpus": 1}\`. Use larger only when the hypothesis really needs it.
- Unique \`id\` like \`exp_0042_betas_097\`.
- Unique \`produces_tags\` (lets analysts and downstream tasks depend on this experiment).
- \`base_ref\` (param_patch only): a SHA or branch name to fork from. Default \`HEAD\`. Use a prior winning experiment's id if you want to stack edits on it.
- \`rationale\`: one short sentence so the next replan can read why.
- Optional: \`agent: "maya"\` with \`worker_class: "analyst"\` for CPU-only analysis writeups (no execution_mode needed; analysts always go through the LLM path).

Don't include env-override-only experiments. \`train.py\` reads almost no \`AUTORESEARCH_*\` env vars.

## Priority and killing tasks

Each task may carry an integer \`priority\` (default 0; higher runs first when more than one task is ready). Use it to express "this experiment is more worth a GPU right now than the others in the queue".

If a task is still running and you've already decided its result won't matter (e.g., it's stacked on a branch you've now discarded, or a more recent result invalidates it), you can abort it. Emit a \`<!-- KILL_TASKS -->\` block alongside (or instead of) the next \`TASK_GRAPH\`:

\`\`\`
<!-- KILL_TASKS -->
["exp_0042", "exp_0099"]
<!-- /KILL_TASKS -->
\`\`\`

The runner will SIGTERM the worker LLM (which takes its bash + train.py descendants with it) and free the GPU token. Don't kill experiments that are about to finish (>80% through the budget); the cost is already paid.

## Keep/discard via git lineage

After each result lands, decide: was this branch's edit a win? If yes, future experiments fork from that branch (set \`base_ref\` to its experiment id). If no, fork from the previous kept parent (skip the dead branch). Don't keep stacking edits on a discarded branch.

## When tasks fail

A failed \`param_patch\` task writes \`${experimentName}/<id>/failure.json\` with the structured reason — typically "expected_old_repr didn't match" (your read of train.py was stale) or "no module-level assignment to <name>" (the symbol you targeted lives inside a function). Read it before emitting more edits in the same neighborhood. If the fix is obvious (correct \`expected_old_repr\`), emit a corrected \`param_patch\` with a new id. If the change is genuinely free-form, switch to \`llm_repair\` or just \`code_edit\` for a fresh attempt.

## The loop

You will be called continuously. Every time a worker finishes and returns its GPU token, the runner calls you again with the latest state. On each call:

1. Read prior \`${experimentName}/**/experiment.md\`, \`metrics.json\`, \`result.txt\`, and any \`failure.json\` to see what's been tried, what won, and what failed.
2. Decide what to try next.
3. Append new tasks to the live TASK_GRAPH. Do not repeat existing task ids. New tasks must not depend on still-running task ids or unproduced tags.
4. Keep around ${experimentsPerWave}-${backlogTarget} ready/pending GPU tasks queued so tokens never sit idle.

Don't emit \`PROJECT_COMPLETE\`. If you run out of ideas: re-read \`${workloadRoot}/train.py\` for fresh angles, combine previous near-misses, try more radical architecture changes, or read papers referenced in the code.

## First wave (prime the compile cache)

The first \`torch.compile\` of the day costs ~150s; subsequent compiles against the shared \`AUTORESEARCH_SHARED_CACHE_ROOT\` finish in seconds. Pay that cost once, not 8× in parallel:

1. Emit ONE no-edit \`param_patch\` baseline first (use a no-op edit like rewriting an unused constant to itself, or just \`code_edit\` with body "no edit; baseline run") at high priority.
2. Make every other cycle-1 experiment \`depends_on: ["exp_0001_baseline"]\` so they don't dispatch until the baseline returns.
3. From cycle 2 onward, full parallelism is fine — caches are warm.

Beyond that warm-up rule, cycle 1 should still be a diverse portfolio across architectural axes. Don't make it a pure LR sweep — that explores the smallest dimension and is what previous runs wasted hours on.
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

You execute one experiment from the manager's task. The science is the same as \`autoresearch/program.md\`: edit \`train.py\`, train for the fixed budget, report metrics. The only delta is that you work inside a per-experiment git repo so workers don't step on each other.

## Loop

1. Read the task. It has the output directory, the parent repo, and the exact code edit (file, constant, old → new).
2. If the task is the baseline (no code edit), run the shared baseline repo and skip to step 6.
3. Otherwise: \`${sandboxScriptPath} <parent_repo> <output_dir>/sandbox/repo\` to clone the parent. \`cd\` in, \`git checkout -B <experiment_id>\`, apply the edit to \`train.py\` (find the constant, change the value, leave everything else untouched), then \`git add -A && git commit -m "<experiment_id>: <one-line>"\`.
4. Write \`experiment.md\` in the output directory with: experiment id, hypothesis, parent, exact code edit, granted GPU ordinals, branch, commit hash.
5. Set \`AUTORESEARCH_REPO_ROOT=<output_dir>/sandbox/repo\`.
6. Set \`AUTORESEARCH_TIME_BUDGET_SECONDS=${timeBudgetSeconds}\` and \`CUDA_VISIBLE_DEVICES=<granted_gpu_ordinals>\`. Don't set any other \`AUTORESEARCH_*\` env vars.
7. Run \`${workerScriptPath} <output_dir>\`. Don't wrap it in \`timeout\` — the training script enforces the budget itself.
8. After it exits, verify \`result.txt\` reports \`exit_code=0\` and \`metrics.json\` exists.
9. Report \`val_bpb\`, \`training_seconds\`, \`num_steps\`, and the config fields you changed.

## Compile

\`torch.compile\` is on by default. If \`train.log\` shows a torch.compile error and you're retrying the same experiment, you may set \`AUTORESEARCH_DISABLE_COMPILE=1\` for that one retry and note it in \`experiment.md\`. Otherwise leave it alone.

## On crashes

Report the last useful lines of \`train.log\`. Don't silently change the config and re-run.

## Analysis tasks

If the manager schedules you with \`worker_class: "analyst"\`, just read the requested \`experiment.md\` / \`metrics.json\` files, rank by \`val_bpb\`, and write the requested analysis file. No GPU work.
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
model: low
---

You run one autoresearch experiment per task. Read \`worker_program.md\` first and follow it. Use the granted GPU ordinals from your task's resource grant. Always set \`AUTORESEARCH_TIME_BUDGET_SECONDS=${timeBudgetSeconds}\`${resultsEnv}.
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

export function renderAutoresearchDagFullConfig({
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
 * non-LLM execution path (see `infra/hpc_agent/runner/src/direct-executor.js`)
 * to this task plugin's source repo, wrapper, and per-experiment sandbox
 * roots so manager-emitted `param_patch` tasks can run without a worker
 * LLM.
 *
 * @param {Object} cfg
 * @param {string} cfg.sourceRepoPath  the canonical workload repo
 * @param {string} cfg.wrapperScript   bash wrapper that runs train.py
 * @param {string} cfg.sandboxRoot     each task clones into <sandboxRoot>/<id>
 * @param {string} cfg.outputRoot      where the wrapper writes result.txt etc.
 * @param {string=} cfg.resultsPath    appended to envOverrides as AUTORESEARCH_RESULTS_PATH
 * @param {number=} cfg.timeBudgetSeconds
 * @param {number=} cfg.hardCapSeconds
 * @return {string}
 */
function renderDirectExecutorBlock({
  sourceRepoPath,
  wrapperScript,
  sandboxRoot,
  outputRoot,
  resultsPath = null,
  sharedCacheRoot = null,
  timeBudgetSeconds = 300,
  hardCapSeconds = 900,
}) {
  const envEntries = [];
  if (resultsPath) envEntries.push(`    AUTORESEARCH_RESULTS_PATH: ${resultsPath}`);
  if (sharedCacheRoot) envEntries.push(`    AUTORESEARCH_SHARED_CACHE_ROOT: ${sharedCacheRoot}`);
  const envLines = envEntries.length
    ? `  envOverrides:\n${envEntries.join('\n')}\n`
    : '';
  return `directExecutor:
  enabled: true
  sourceRepoPath: ${sourceRepoPath}
  wrapperScript: ${wrapperScript}
  sandboxRoot: ${sandboxRoot}
  outputRoot: ${outputRoot}
  metricKey: val_bpb
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

export function createAutoresearchDagFullWorkspace(options = {}) {
  const paths = buildAutoresearchDagFullPaths(options);
  const workerScriptPath = options.workerScriptPath || DEFAULT_WORKER_SCRIPT_PATH;
  const sandboxScriptPath = options.sandboxScriptPath || DEFAULT_SANDBOX_SCRIPT_PATH;
  const workloadRoot = options.workloadRoot || DEFAULT_WORKLOAD_ROOT;
  const timeBudgetSeconds = Number(options.timeBudgetSeconds ?? 300);
  const gpuCount = Number(options.gpuCount ?? 8);
  const experimentWorkerCount = Number(options.experimentWorkerCount ?? gpuCount ?? 8);
  const agentRuntime = options.agentRuntime || 'codex_cli';

  mkdirSync(path.dirname(paths.readmePath), { recursive: true });
  mkdirSync(path.dirname(paths.experimentWorkerSkillPaths[0]), { recursive: true });
  mkdirSync(paths.expOutputDir, { recursive: true });
  mkdirSync(paths.sharedCacheRoot, { recursive: true });

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
  writeFileSync(paths.configPath, renderAutoresearchDagFullConfig({
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
      hardCapSeconds: timeBudgetSeconds + 600,
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
