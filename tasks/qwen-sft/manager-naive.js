// Naive-parallel variant of the qwen-sft manager. Same RAILS runtime, same
// K=8, same direct executor, same metric — but the manager's primitives are
// stripped to match the paper's "Naive parallel" definition:
//   - lineage disabled (no KEEP_EXPERIMENT primitive in the prompt)
//   - no rationale-as-memory framing (manager isn't told the ledger replays
//     past rationales next cycle)
//   - no source-file injection (train.py text isn't embedded in the prompt)
//   - no live replan; manager is only re-called when the graph drains
//     (refillOnGraphDrain) so it must propose fresh batches in lockstep
//
// All other infrastructure stays (param_patch + constant_replace, 5-bucket
// search surface, val_loss metric, baseline_repeat noise estimation, kill
// primitives, walltime gate). The point is to hold everything constant
// except the four ALPS scheduling decisions surfaced via prompt.

import { mkdirSync, writeFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import {
  buildQwenSftPaths,
  renderInitialProjectState,
  renderQwenSftWorkerProgram,
  renderQwenSftExperimentWorkerSkill,
  renderQwenSftAnalysisWorkerSkill,
  renderProjectsYaml,
} from './manager.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TASK_ROOT = __dirname;
const DEFAULT_WORKER_SCRIPT_PATH = path.join(TASK_ROOT, 'worker.sh');
const DEFAULT_SANDBOX_SCRIPT_PATH = path.join(TASK_ROOT, 'sandbox.sh');
const DEFAULT_WORKLOAD_ROOT = path.join(TASK_ROOT, 'source');

function toPosix(relativePath) {
  return String(relativePath || '').split(path.sep).join('/');
}

export function renderNaiveReadme({
  experimentName = 'experiments',
  workerScriptPath,
  sandboxScriptPath,
  workloadRoot = DEFAULT_WORKLOAD_ROOT,
  timeBudgetSeconds = 1200,
  gpuCount = 8,
} = {}) {
  return `# Qwen-SFT Naive Parallel Search

This run is the **naive-parallel baseline** for the qwen-sft data-mix search. The manager dispatches K=${gpuCount} experiments per batch, all forking from the same fixed baseline (uniform-mix HEAD). There is no \`KEEP_EXPERIMENT\` primitive, no lineage advance, no rationale-as-memory: each batch is independent.

This intentionally matches the paper's "Naive parallel" configuration. Compare against the Manual ALPS pilot (same task, same per-experiment 1200 s budget, same metric) to see what lineage + rationale ledger buy.

## What changes per experiment

Only \`DATA_MIX\` in \`${workloadRoot}/train.py\`. Five buckets (math / code / chat / if / reasoning); weights normalize to 1.0.

## Wall budget

${gpuCount} GPUs × 4 h. Each experiment takes ~22 min wall (1200 s training + load + eval). Manager is only re-called when the graph drains, so each batch is dispatched in lockstep: 8 dispatched → wait for all 8 to finish → manager proposes 8 more.
`;
}

export function renderNaiveManagerProgram({
  experimentName = 'experiments',
  workerScriptPath,
  workloadRoot = DEFAULT_WORKLOAD_ROOT,
  timeBudgetSeconds = 1200,
  gpuCount = 8,
} = {}) {
  const experimentsPerWave = Math.max(1, Number.parseInt(gpuCount, 10) || 1);

  return `# qwen-sft naive-parallel manager program

You are the manager for a **naive-parallel** data-mix search over Tulu-3 SFT mixtures applied to Qwen3-0.6B full fine-tuning. The framework dispatches your tasks to a pool of GPU workers; you propose new \`DATA_MIX\` weights and read back \`val_loss\`.

This is **not** a lineage-aware run. Specifically:
- You have **no \`KEEP_EXPERIMENT\` primitive**. The repository HEAD never moves; every experiment forks from the original uniform-mix baseline.
- You are called **once per batch** (after all ${experimentsPerWave} workers in the previous batch finish), not on individual completions.
- You will not be shown a memory of past rationales. Treat each batch as if it is the next move in a parallel sweep — no cumulative reasoning across waves is required.

## Goal

Minimize balanced held-out \`val_loss\` under a fixed ${timeBudgetSeconds} s SFT budget per experiment. The val set is 200 samples per bucket across all five buckets (math, code, chat, if, reasoning), so the metric is gameable-resistant: a mix that overfits one easy bucket loses on the others.

## What you change

Only \`DATA_MIX\` in \`${workloadRoot}/train.py\`. Everything else (model = Qwen3-0.6B, seq_len = 2048, micro_batch = 16, grad_accum = 2, effective batch = 32, learning_rate = 2e-5, warmup = 50, weight_decay = 0.01) is fixed.

The five buckets and their patterns are documented in \`~/.cache/qwen-sft/data/manifest.json\`.

## How you dispatch

Each batch, output a single \`<!-- TASK_GRAPH -->\` block with **exactly ${experimentsPerWave}** experiments. Each task:

- \`worker_class: "experiment_runner"\`
- \`execution_mode: "param_patch"\`
- \`base_ref: "HEAD"\` (always — HEAD does not move in this configuration)
- \`resources: {"gpus": 1, "cpus": 2}\`
- Unique \`id\` like \`exp_b3_w0_math_heavy\` (\`b\` = batch number, \`w\` = within-batch index)
- Unique \`produces_tags\` like \`["metrics:exp_b3_w0_math_heavy"]\`
- One-line \`task\` description
- An \`edits\` array with exactly ONE \`constant_replace\` on \`DATA_MIX\`
- \`estimated_runtime_seconds: ${timeBudgetSeconds + 60}\`

### The \`DATA_MIX\` edit shape

\`\`\`json
{"file":"train.py","kind":"constant_replace","name":"DATA_MIX",
 "expected_old_repr":"{'math': 0.20, 'code': 0.20, 'chat': 0.20, 'if': 0.20, 'reasoning': 0.20}",
 "new_repr":"{'math': 0.40, 'code': 0.20, 'chat': 0.20, 'if': 0.10, 'reasoning': 0.10}"}
\`\`\`

Constraints:
- All five keys must be present.
- All values must be \`>= 0\`. Trainer normalizes to sum = 1 internally; decimal values in \`[0.0, 1.0]\` are easiest to read.
- Don't add new keys.
- Because HEAD is fixed, \`expected_old_repr\` is **always** \`"{'math': 0.20, 'code': 0.20, 'chat': 0.20, 'if': 0.20, 'reasoning': 0.20}"\`.

### Worked TASK_GRAPH (batch 1)

\`\`\`json
<!-- TASK_GRAPH -->
{"tasks":[
  {"id":"exp_b1_w0_baseline","worker_class":"experiment_runner",
   "execution_mode":"param_patch","base_ref":"HEAD",
   "resources":{"gpus":1,"cpus":2},"priority":2,
   "estimated_runtime_seconds":${timeBudgetSeconds + 60},
   "task":"baseline uniform mix",
   "produces_tags":["metrics:exp_b1_w0_baseline"],
   "edits":[
     {"file":"train.py","kind":"constant_replace","name":"DATA_MIX",
      "expected_old_repr":"{'math': 0.20, 'code': 0.20, 'chat': 0.20, 'if': 0.20, 'reasoning': 0.20}",
      "new_repr":"{'math': 0.20, 'code': 0.20, 'chat': 0.20, 'if': 0.20, 'reasoning': 0.20}"}
   ]},
  {"id":"exp_b1_w1_math_heavy","worker_class":"experiment_runner",
   "execution_mode":"param_patch","base_ref":"HEAD",
   "resources":{"gpus":1,"cpus":2},"priority":2,
   "estimated_runtime_seconds":${timeBudgetSeconds + 60},
   "task":"math-heavy mix",
   "produces_tags":["metrics:exp_b1_w1_math_heavy"],
   "edits":[
     {"file":"train.py","kind":"constant_replace","name":"DATA_MIX",
      "expected_old_repr":"{'math': 0.20, 'code': 0.20, 'chat': 0.20, 'if': 0.20, 'reasoning': 0.20}",
      "new_repr":"{'math': 0.40, 'code': 0.15, 'chat': 0.15, 'if': 0.15, 'reasoning': 0.15}"}
   ]}
]}
<!-- /TASK_GRAPH -->
\`\`\`

## When tasks fail

A failed \`param_patch\` writes \`${experimentName}/<id>/failure.json\`. Most common: \`expected_old_repr\` didn't match because you typed it wrong. Use the canonical uniform-mix string above.

## The loop

You will be called once per batch, after the previous ${experimentsPerWave} experiments all finish. Each call:

1. Read the latest batch's \`val_loss\` results in your runtime context. Look at the table; do not assume any cross-wave memory.
2. Decide the next batch of ${experimentsPerWave} mixes. Independent of past batches' detailed reasoning — propose what looks promising given the current observed table.
3. Append a \`TASK_GRAPH\` with exactly ${experimentsPerWave} tasks. Do not repeat any task id from any prior batch.

Don't emit \`PROJECT_COMPLETE\`. Continue dispatching ${experimentsPerWave}-batch waves until the wall budget is exhausted.

## First batch

Cycle 1: dispatch ${experimentsPerWave} \`param_patch\` tasks in parallel from t=0. Mix should include:
- One uniform baseline (\`{0.2, 0.2, 0.2, 0.2, 0.2}\`)
- Several single-bucket "heavy" mixes (each bucket as the dominant ~0.4 share)
- Optional pair-emphasis mixes
`;
}

export function renderNaiveConfig({
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
  liveReplanOnTaskComplete: false
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

function renderDirectExecutorBlock({
  sourceRepoPath,
  wrapperScript,
  sandboxRoot,
  outputRoot,
  resultsPath = null,
  sharedCacheRoot = null,
  timeBudgetSeconds = 1200,
  hardCapSeconds = 2100,
}) {
  const envEntries = [];
  if (resultsPath) envEntries.push(`    QWEN_SFT_RESULTS_PATH: ${resultsPath}`);
  if (sharedCacheRoot) envEntries.push(`    AUTORESEARCH_SHARED_CACHE_ROOT: ${sharedCacheRoot}`);
  const hfHome = path.join(process.env.HOME || '', '.cache', 'huggingface');
  if (process.env.HOME) envEntries.push(`    HF_HOME: ${hfHome}`);
  const wandbGroup = process.env.QWEN_SFT_WANDB_GROUP
    || `qwen-sft-naive-${new Date().toISOString().replace(/[-:T.Z]/g, '').slice(0, 14)}`;
  envEntries.push(`    WANDB_PROJECT: ${process.env.WANDB_PROJECT || 'workshop'}`);
  envEntries.push(`    WANDB_ENTITY: ${process.env.WANDB_ENTITY || 'haolong'}`);
  envEntries.push(`    WANDB_RUN_GROUP: ${wandbGroup}`);
  envEntries.push(`    WANDB_TAGS: qwen-sft,datamix,naive-parallel`);
  envEntries.push(`    EVAL_EVERY_STEPS: ${process.env.EVAL_EVERY_STEPS || '20'}`);
  envEntries.push(`    SFT_TIME_BUDGET_SECONDS: ${timeBudgetSeconds}`);
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
  fallbackSigma: 0.005
  calibrationRepeats: 0
  autoKeepEnabled: false
  manualStaleKeepPolicy: block
  autoEnqueueHeadValidation: false
  quotaDiversityEnabled: false
  alwaysAllowValidation: false
  searchAxes:
    - name: BUCKET_MIX
      role: primary
${envLines}`;
}

export function createNaiveWorkspace(options = {}) {
  const paths = buildQwenSftPaths(options);
  const workerScriptPath = options.workerScriptPath || DEFAULT_WORKER_SCRIPT_PATH;
  const sandboxScriptPath = options.sandboxScriptPath || DEFAULT_SANDBOX_SCRIPT_PATH;
  const workloadRoot = options.workloadRoot || DEFAULT_WORKLOAD_ROOT;
  const timeBudgetSeconds = Number(options.timeBudgetSeconds ?? 1200);
  const gpuCount = Number(options.gpuCount ?? 8);
  const experimentWorkerCount = Number(options.experimentWorkerCount ?? gpuCount ?? 8);
  const agentRuntime = options.agentRuntime || 'codex_cli';

  mkdirSync(path.dirname(paths.readmePath), { recursive: true });
  mkdirSync(path.dirname(paths.experimentWorkerSkillPaths[0]), { recursive: true });
  mkdirSync(paths.expOutputDir, { recursive: true });
  mkdirSync(paths.sharedCacheRoot, { recursive: true });

  writeFileSync(paths.readmePath, renderNaiveReadme({
    experimentName: paths.experimentName,
    workerScriptPath,
    sandboxScriptPath,
    workloadRoot,
    timeBudgetSeconds,
    gpuCount,
  }), 'utf8');
  writeFileSync(paths.managerProgramPath, renderNaiveManagerProgram({
    experimentName: paths.experimentName,
    workerScriptPath,
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
  writeFileSync(paths.configPath, renderNaiveConfig({
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
