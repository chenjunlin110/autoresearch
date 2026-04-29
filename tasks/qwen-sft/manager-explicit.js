// Explicit-ALPS variant of the qwen-sft manager. Same prompt content as
// the manual ALPS pilot (manager.js) — what changes is the framework's
// directExecutor config: the 4 ALPS scheduling decisions become code
// rather than prompt:
//
//   1. Statistical commit gate (Eq. 6) → autoKeepEnabled=true,
//      gateTauMin=0.1, gateTauMax=0.5, calibrationRepeats=3,
//      fallbackSigma=8e-5 (matches our n=4 estimate from the manual pilot).
//   2. Stale-winner rebase (§4.6)     → rebaseValidationEnabled=true,
//      autoEnqueueHeadValidation=true, manualStaleKeepPolicy=block.
//   3. Diversity-aware acquisition (Eq. 5) → quotaDiversityEnabled=true,
//      maxPerOperatorPerWake=4, maxSameOperatorValuePerWake=1.
//   4. Hazard-driven adaptive parallelism (Eq. 4) → hazardQueueEnabled=true,
//      qMin=1, qMax=8, hazardZeta=2.0, posteriorMinSamples=3
//      (Q only kicks in once posterior has 3+ samples per axis).
//
// The manager prompt is reused unchanged so we hold the proposer constant
// across Manual vs Explicit. The DIFFERENCE is in framework behavior:
// the gate fires automatically, hazard adapts queue depth, diversity
// caps the per-batch dispatch, and stale winners get auto-rebased.

import { mkdirSync, writeFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import {
  buildQwenSftPaths,
  renderInitialProjectState,
  renderQwenSftReadme,
  renderQwenSftManagerProgram,
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

export function renderExplicitConfig({
  gpuCount = 8,
  agentRuntime = 'codex_cli',
  directExecutor = null,
} = {}) {
  const runtime = ['codex_cli', 'claude_cli', 'api'].includes(agentRuntime) ? agentRuntime : 'codex_cli';
  const count = Number(gpuCount) || 0;
  const directBlock = directExecutor ? renderExplicitDirectExecutorBlock(directExecutor) : '';
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

function renderExplicitDirectExecutorBlock({
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
    || `qwen-sft-explicit-${new Date().toISOString().replace(/[-:T.Z]/g, '').slice(0, 14)}`;
  envEntries.push(`    WANDB_PROJECT: ${process.env.WANDB_PROJECT || 'workshop'}`);
  envEntries.push(`    WANDB_ENTITY: ${process.env.WANDB_ENTITY || 'haolong'}`);
  envEntries.push(`    WANDB_RUN_GROUP: ${wandbGroup}`);
  envEntries.push(`    WANDB_TAGS: qwen-sft,datamix,explicit-alps`);
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
  # ===== Explicit ALPS — all 4 scheduling decisions promoted to code =====
  # 1. Statistical commit gate (Eq. 6): auto-fire KEEP when Δ > τ_t · σ̂.
  fallbackSigma: 0.0001          # > our n=4 estimate (8e-5) — gate cautious until calibration
  calibrationRepeats: 3          # 3 baseline repeats to estimate σ̂ empirically
  autoKeepEnabled: true          # gate AUTO-fires KEEP_EXPERIMENT
  gateTauMin: 0.1                # late-run threshold (most aggressive)
  gateTauMax: 0.5                # early-run threshold (more conservative)
  # 2. Stale-winner rebase (§4.6): auto re-evaluate stale candidates on HEAD.
  rebaseValidationEnabled: true
  manualStaleKeepPolicy: block   # forbid manual KEEP from a stale base
  autoEnqueueHeadValidation: true
  alwaysAllowValidation: true
  # 3. Diversity-aware acquisition (Eq. 5): cap per-operator and same-value.
  quotaDiversityEnabled: true
  maxPerOperatorPerWake: 4
  maxSameOperatorValuePerWake: 1
  # 4. Hazard-driven Q_t (Eq. 4): adaptive queue depth from operator posterior.
  hazardQueueEnabled: true
  qMin: 1
  qMax: 8
  hazardZeta: 2.0
  posteriorMinSamples: 3         # warm up before hazard kicks in (avoid Q→qMin)
  searchAxes:
    - name: BUCKET_MIX
      role: primary
    - name: LR
      role: primary
    - name: WARMUP_RATIO
      role: schedule
    - name: WEIGHT_DECAY
      role: primary
${envLines}`;
}

export function createExplicitWorkspace(options = {}) {
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

  // Reuse the same prompts as Manual ALPS — proposer is constant across configs.
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
  writeFileSync(paths.configPath, renderExplicitConfig({
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
