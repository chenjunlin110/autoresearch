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
    // Persist the compile cache outside the per-run dir so cycle-1 of
    // a fresh sbatch reuses kernels compiled by prior runs. Honour the
    // sbatch's AUTORESEARCH_SHARED_CACHE_ROOT env if set so both paths
    // point at the same dir.
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

## Optional: early-stop a clearly-failing experiment

Each task may carry an \`early_stop\` field that tells the wrapper to abort training partway through if the live training loss is above a threshold you set. This saves GPU on experiments diverging fast — useful for aggressive LR sweeps or radical architecture tries.

\`\`\`json
"early_stop": {"check_at_seconds": 90, "abort_if_loss_above": 4.0}
\`\`\`

Both keys are required when present; the wrapper waits \`check_at_seconds\` into training, peeks at the latest \`loss: X.XX\` from \`train.log\`, and SIGTERMs training if it exceeds \`abort_if_loss_above\`. **Note that this is the *training loss* (printed every step), not val_bpb (only computed at the end).** Pick the threshold by reference to the recent baseline's loss curve at the same point — re-read \`${experimentName}/exp_*/train.log\` for typical loss-at-90s values before setting one. Conservative (high) thresholds = rarely fire; aggressive (low) thresholds = save GPU but risk killing slow-converging-but-eventual-winners. **Default: don't set \`early_stop\` unless you have a specific reason to.**

When triggered, the experiment ends with \`exit_code != 0\` and \`early_stop_triggered=1\` in \`result.txt\`. The validator marks it failed; you'll see it in failure clusters next cycle.

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
- \`base_ref\` (param_patch only): a ref in the **shared baseline source repo only** — almost always just leave it as \`HEAD\` (the default). Per-experiment sandboxes are isolated clones, so a child experiment cannot fork from another experiment's id at the git level. To "stack" edits on a prior winning experiment, either include the prior winner's edits + your new edit in a single \`edits[]\` array (multiple edits are applied atomically), or just emit a fresh \`code_edit\` task that describes the cumulative change.
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

## Cumulative lineage via \`KEEP_EXPERIMENT\` (this is your most important primitive)

By default every \`param_patch\` task with \`base_ref: "HEAD"\` clones from the **shared baseline** \`source/\`, not from another experiment's branch. To make wins compound, you must explicitly tell the framework which experiments to keep — like Karpathy's serial agent advances its branch on each win:

\`\`\`
<!-- KEEP_EXPERIMENT -->
[
  {"id": "exp_0010_aspect_96", "reason": "aspect=96 beat baseline by -0.0014"},
  {"id": "exp_0019_compile_on", "reason": "compile=on cut val_bpb by 0.06 vs no-compile baseline"}
]
<!-- /KEEP_EXPERIMENT -->
\`\`\`

What this does:
1. The framework re-applies that experiment's \`edits[]\` to \`tasks/autoresearch/source/\` and commits.
2. \`source/\`'s HEAD now reflects the kept change.
3. Every subsequent \`param_patch\` task with \`base_ref: "HEAD"\` clones from this advanced HEAD — so its edit STACKS on top.
4. The next manager-context you receive will show a **Kept lineage** section listing every kept experiment and the current source HEAD SHA.

**On every result, decide keep or discard — don't sit on it.** This is the same principle as Karpathy's serial harness: each completed experiment either advances the branch or gets discarded, immediately. We dispatch in parallel so several results land at once, but the rule is the same — when a new completion appears in your ledger, your next response should already say what you're doing with it. If \`val_bpb\` improved over current source HEAD: emit \`KEEP_EXPERIMENT\`. If it didn't: leave it alone (the experiment is recorded; nothing more to do). Don't "wait one more wave to confirm" — that lag is the single biggest reason a parallel framework can underperform a serial one: every experiment dispatched between a win and an eventual KEEP started from the wrong baseline and is wasted.

The cost of an early KEEP is small. If it later turns out the win was a fluke, just KEEP a better experiment on top — the lineage stacks, nothing is "rolled back". You're describing a chain, not picking a final winner.

Multiple wins along independent axes can go in a single \`KEEP_EXPERIMENT\` block.

**Don't KEEP**: regressions, crashes, no-edit baselines (the source already has them), or wins so small you'd describe them yourself as "could be the seed".

**ALPS auto-keep (when \`autoKeepEnabled: true\` in config):** the runtime auto-applies KEEP_EXPERIMENT for any candidate that beats current HEAD by more than the noise-aware gate threshold τ·σ̂ (logged as \`[gate] WOULD_AUTO_KEEP\` / \`AUTO_KEEP\`). When auto-keep is on, your manual \`KEEP_EXPERIMENT\` is reserved for sub-threshold wins where you have a strong qualitative reason. **Stale manual KEEPs are blocked**: if your KEEP target's baseline is no longer current HEAD (visible in the picked event's \`picked_base_commit\`), the runtime refuses the keep — re-dispatch the same edits with \`base_ref: "HEAD"\` to re-validate on the new baseline. To dispatch a no-edit baseline_repeat (which re-establishes HEAD metric so the gate can engage), set \`execution_mode: "baseline_repeat"\` and \`base_ref: "HEAD"\` with no \`edits\`.

When to UN-KEEP (rewind): there's no automatic primitive for this — if a kept experiment turns out to have been a fluke, you can effectively rewind by KEEPing the LATER version of that file (i.e., emit a \`code_edit\` task that produces the desired source state and then KEEP it, or just live with the slight regression and find new wins on top).

\`KEEP_EXPERIMENT\` runs **before** any new TASK_GRAPH in the same response is parsed, so you can KEEP an old experiment and immediately emit new tasks that build on it in the same manager response.

## Old "git lineage" advice

Earlier prompts said you could fork from prior winners by setting \`base_ref\` to an experiment id — that does **not** work because per-experiment sandboxes are isolated git clones. Use \`KEEP_EXPERIMENT\` (above) instead. \`base_ref\` should always be \`"HEAD"\` (or omitted; \`HEAD\` is the default).

## When tasks fail

A failed \`param_patch\` task writes \`${experimentName}/<id>/failure.json\` with the structured reason — typically "expected_old_repr didn't match" (your read of train.py was stale) or "no module-level assignment to <name>" (the symbol you targeted lives inside a function). Read it before emitting more edits in the same neighborhood. If the fix is obvious (correct \`expected_old_repr\`), emit a corrected \`param_patch\` with a new id. If the change is genuinely free-form, switch to \`llm_repair\` or just \`code_edit\` for a fresh attempt.

## Rationale is mandatory — it is your memory

Every task you emit MUST include a \`rationale\` field stating, in one sentence, **what you are testing and why you expect it to help**. Examples:
- \`"rationale": "wider aspect ratio (96 vs 64) better matches the FFN dim at this depth"\`
- \`"rationale": "lr=3e-3 is conservative; if 6e-3 also lands we have a longer LR plateau to exploit"\`
- \`"rationale": "depth=12 to test whether the depth=10 win extrapolates"\`

Why this is non-negotiable: the runtime context you receive next time will replay each completed experiment as **rationale → outcome**. Without rationale you literally cannot tell, on the next call, whether \`exp_0010_aspect_96\` won by luck or because aspect ratio is real signal. The rationale is your reasoning trace across cycles.

## The loop

You will be called continuously. Every time the runtime queue runs shallow, the runner calls you again with the latest state. On each call:

1. **Read your past hypotheses next to their results** in the runtime context the runner just gave you (top-K + recent). Look for *patterns*: which axis (depth, aspect, LR, beta_2, attention) actually moved \`val_bpb\`? Which "should have helped" hypothesis didn't? Disqualify dead axes; double down on live ones.
2. **Read the current \`source/train.py\`** that the runner injected into your context — this reflects all KEPT experiments so far. Compare it against what the ledger says was tried. Is there an obvious next edit you can see in the source (e.g., a default flag like \`AUTORESEARCH_DISABLE_COMPILE="1"\` waiting to be flipped, an obviously suboptimal constant)? Those are often the cheapest wins.
3. **KEEP clearly-winning experiments** before emitting more tasks. If \`exp_0010\` beat the current source HEAD by a margin above noise, emit a \`<!-- KEEP_EXPERIMENT -->\` block in this same response so the next batch of tasks builds on it.
4. If you need more detail than the ledger one-liners give, read \`${experimentName}/<id>/metrics.json\`, \`failure.json\`, or the per-experiment \`train.py\` (its diff vs the source baseline).
5. Decide what to try next based on the patterns. **Exploit when you have a winner**: emit a fan-out around that winner's axis (e.g. winner is aspect=96 → try aspect=80, 112, plus aspect=96 combined with the next-best axis). **Explore when nothing's separating from baseline**: shift to a different axis you haven't explored.

   **Don't fixate on one axis.** If your last several waves are all \`<axis>_X\` combinations of the same recent winner, you've stopped exploring. Before queuing yet another combination on the live axis, glance back at the ledger: which axes have you NOT touched yet at all? Which have you touched but only at the extremes (e.g. tried 0.04 and 0.06 but not 0.05)? The interior of an under-sampled range often beats yet another mixture on the over-sampled one.
6. Append new tasks to the live TASK_GRAPH. Each task carries a \`rationale\`. Tasks' \`expected_old_repr\` should match the **current** source/train.py state (which reflects all KEPT experiments) — not the original baseline. Do not repeat existing task ids. New tasks must not depend on still-running task ids or unproduced tags.
7. Keep around ${experimentsPerWave}-${backlogTarget} ready/pending GPU tasks queued so tokens never sit idle. Better to over-emit than under-emit: extra tasks just queue.

Don't emit \`PROJECT_COMPLETE\`. If you run out of ideas: re-read \`${workloadRoot}/train.py\` for fresh angles, combine previous near-misses, try more radical architecture changes, or read papers referenced in the code.

## First wave

\`AUTORESEARCH_SHARED_CACHE_ROOT\` persists across sbatch runs, so the first wave usually hits a warm \`torch.compile\` cache from prior days. Even on a fresh cache (first run on a new node), 8 workers compiling in parallel only costs ~200s of wall time, not 8× — so just dispatch \${experimentsPerWave} tasks in parallel from t=0.

Cycle 1 should be a diverse portfolio across architectural axes — depth, width, head_dim, attention pattern, optimizer, schedule shape. Don't make it a pure LR sweep — that explores the smallest dimension and is what previous runs wasted hours on. Include one no-edit \`param_patch\` baseline (e.g. rewrite an unused constant to itself) so you have a regression line.
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
  liveReplanMinIntervalSeconds: 15
  # Wake the manager whenever ready+running drops below 1.0× max workers
  # (i.e. on every completion). Pairs with KEEP_EXPERIMENT lineage so the
  # manager can act on each result individually instead of batching feedback.
  liveReplanWatermarkRatio: 1.0
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
  contextFiles = ['train.py'],
}) {
  const envEntries = [];
  if (resultsPath) envEntries.push(`    AUTORESEARCH_RESULTS_PATH: ${resultsPath}`);
  if (sharedCacheRoot) envEntries.push(`    AUTORESEARCH_SHARED_CACHE_ROOT: ${sharedCacheRoot}`);
  const envLines = envEntries.length
    ? `  envOverrides:\n${envEntries.join('\n')}\n`
    : '';
  const ctxLines = (contextFiles || [])
    .map((f) => `    - ${f}`)
    .join('\n');
  const ctxBlock = ctxLines ? `  managerContextFiles:\n${ctxLines}\n` : '';
  return `directExecutor:
  enabled: true
  sourceRepoPath: ${sourceRepoPath}
  wrapperScript: ${wrapperScript}
  sandboxRoot: ${sandboxRoot}
  outputRoot: ${outputRoot}
  metricKey: val_bpb
  timeBudgetSeconds: ${timeBudgetSeconds}
  hardCapSeconds: ${hardCapSeconds}
  # ALPS knobs. Phase 2B: autoKeepEnabled=true after Phase 2A dry-run
  # (job 1582493) confirmed [gate] decisions match expected behavior —
  # τ decay, sign convention, HEAD-metric transfer all correct.
  fallbackSigma: 0.0005
  calibrationRepeats: 3
  autoKeepEnabled: true
  manualStaleKeepPolicy: block
  autoEnqueueHeadValidation: true
  quotaDiversityEnabled: false
  maxPerOperatorPerWake: 4
  maxSameOperatorValuePerWake: 1
  alwaysAllowValidation: true
  searchAxes:
    - name: ASPECT_RATIO
      role: primary
    - name: DEPTH
      role: primary
    - name: MATRIX_LR
      role: primary
    - name: EMBEDDING_LR
      role: primary
    - name: WARMDOWN_RATIO
      role: schedule
    - name: WEIGHT_DECAY
      role: primary
${envLines}${ctxBlock}`;
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
