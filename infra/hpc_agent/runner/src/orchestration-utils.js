const DEFAULT_WORKER_RESOURCES = Object.freeze({
  gpus: 0,
  cpus: 1,
});

const DEFAULT_ORCHESTRATION = Object.freeze({
  mode: 'phase_managers',
  manager: 'manager',
  maxConcurrentWorkers: 8,
  maxManagerPasses: 8,
  maxWallClockSeconds: null,
  refillOnGraphDrain: true,
  liveReplanOnTaskComplete: false,
  liveReplanMinIntervalSeconds: 0,
  liveReplanWatermarkRatio: 1.5,
  targetGpuUtilization: 1,
  defaultWorkerResources: DEFAULT_WORKER_RESOURCES,
});

function coercePositiveInteger(value, fallback) {
  if (typeof value === 'number' && Number.isInteger(value) && value > 0) return value;
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number.parseInt(value, 10);
    if (Number.isInteger(parsed) && parsed > 0) return parsed;
  }
  return fallback;
}

function coerceNonNegativeInteger(value, fallback) {
  if (typeof value === 'number' && Number.isInteger(value) && value >= 0) return value;
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number.parseInt(value, 10);
    if (Number.isInteger(parsed) && parsed >= 0) return parsed;
  }
  return fallback;
}

function coerceNonNegativeFloat(value, fallback) {
  if (typeof value === 'number' && Number.isFinite(value) && value >= 0) return value;
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed) && parsed >= 0) return parsed;
  }
  return fallback;
}

function coerceBoolean(value, fallback) {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (['true', '1', 'yes', 'on'].includes(normalized)) return true;
    if (['false', '0', 'no', 'off'].includes(normalized)) return false;
  }
  return fallback;
}

function normalizeVisibility(value) {
  if (value === 'blind' || value === 'focused' || value === 'full') return value;
  return 'full';
}

// `baseline_repeat` is a no-edit harness call used by the noise estimator
// (and by auto HEAD-validation after a manual multi-keep). It gets a fresh
// PRNG seed via the wrapper's *_SEED env var; its observed-metric spread
// across repeats is the seed-noise floor σ̂.
const VALID_EXECUTION_MODES = new Set(['code_edit', 'param_patch', 'llm_repair', 'baseline_repeat']);

function normalizeExecutionMode(value) {
  if (value === undefined || value === null) return 'code_edit';
  if (typeof value !== 'string') return null;
  const normalized = value.trim();
  return VALID_EXECUTION_MODES.has(normalized) ? normalized : null;
}

const VALID_EDIT_KINDS = new Set([
  'constant_replace',
  'regex_replace',
  'block_replace',
  'unified_diff',
]);

/**
 * Validate `edits[]` shape per kind. Returns `null` if valid, or a short
 * human-readable reason. Each kind's required fields must be present;
 * extras are ignored. Schema validation runs at parse time so a malformed
 * edit fails the whole TASK_GRAPH (cheap to fix; expensive in flight).
 *
 * @param {unknown} edits
 * @return {string|null}
 */
/**
 * Normalize an optional `early_stop` field on a task. Returns:
 *   - the normalized object on success
 *   - `null` when the field is unset (no early-stop)
 *   - `false` when the field was provided but malformed (caller should reject)
 *
 * Schema: `{check_at_seconds: int>0, abort_if_loss_above: finite number}`.
 * Both keys are required if `early_stop` is present at all — half-set
 * configs would be footguns.
 *
 * @param {unknown} input
 * @return {Object|null|false}
 */
function normalizeEarlyStop(input) {
  if (input === undefined || input === null) return null;
  if (typeof input !== 'object' || Array.isArray(input)) return false;
  const checkAt = input.check_at_seconds ?? input.checkAtSeconds;
  const abortAbove = input.abort_if_loss_above ?? input.abortIfLossAbove;
  if (!Number.isInteger(checkAt) || checkAt <= 0) return false;
  if (typeof abortAbove !== 'number' || !Number.isFinite(abortAbove)) return false;
  return { checkAtSeconds: checkAt, abortIfLossAbove: abortAbove };
}

function validateEdits(edits) {
  if (!Array.isArray(edits)) return 'edits must be an array';
  for (let i = 0; i < edits.length; i += 1) {
    const edit = edits[i];
    if (!edit || typeof edit !== 'object' || Array.isArray(edit)) {
      return `edits[${i}] is not an object`;
    }
    if (typeof edit.file !== 'string' || !edit.file.trim()) {
      return `edits[${i}] missing "file"`;
    }
    if (!VALID_EDIT_KINDS.has(edit.kind)) {
      return `edits[${i}].kind "${edit.kind}" is not one of ${[...VALID_EDIT_KINDS].join(', ')}`;
    }
    if (edit.kind === 'constant_replace') {
      if (typeof edit.name !== 'string' || !edit.name.trim()) return `edits[${i}] constant_replace missing "name"`;
      if (typeof edit.expected_old_repr !== 'string') return `edits[${i}] constant_replace missing "expected_old_repr"`;
      if (typeof edit.new_repr !== 'string') return `edits[${i}] constant_replace missing "new_repr"`;
    } else if (edit.kind === 'regex_replace') {
      if (typeof edit.pattern !== 'string') return `edits[${i}] regex_replace missing "pattern"`;
      if (typeof edit.replacement !== 'string') return `edits[${i}] regex_replace missing "replacement"`;
      if (!Number.isInteger(edit.expected_count) || edit.expected_count < 1) {
        return `edits[${i}] regex_replace missing positive integer "expected_count"`;
      }
    } else if (edit.kind === 'block_replace') {
      if (typeof edit.anchor_regex !== 'string') return `edits[${i}] block_replace missing "anchor_regex"`;
      if (typeof edit.end_regex !== 'string') return `edits[${i}] block_replace missing "end_regex"`;
      if (typeof edit.new_text !== 'string') return `edits[${i}] block_replace missing "new_text"`;
    } else if (edit.kind === 'unified_diff') {
      if (typeof edit.diff !== 'string' || !edit.diff.trim()) return `edits[${i}] unified_diff missing "diff"`;
    }
  }
  return null;
}

function normalizeStringList(value) {
  if (Array.isArray(value)) {
    return [...new Set(value
      .map((item) => (typeof item === 'string' ? item.trim() : ''))
      .filter(Boolean))];
  }
  if (typeof value === 'string' && value.trim()) {
    return [value.trim()];
  }
  return [];
}

export function normalizeWorkerClass(value) {
  if (typeof value !== 'string') return null;
  const normalized = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return normalized || null;
}

function parseAgentStep(step) {
  if (!step || typeof step !== 'object' || Array.isArray(step)) return null;
  if (typeof step.agent === 'string' && step.agent.trim()) {
    return {
      agent: step.agent.trim(),
      task: step.task,
      visibility: step.visibility,
      resources: step.resources,
      retries: step.retries,
    };
  }

  const keys = Object.keys(step).filter((key) => key !== 'delay' && !key.startsWith('_'));
  if (keys.length !== 1) return null;
  const agent = keys[0];
  const value = step[agent];
  if (typeof agent !== 'string' || !agent.trim()) return null;
  if (typeof value === 'string') {
    return { agent: agent.trim(), task: value };
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  return {
    agent: agent.trim(),
    task: value.task,
    visibility: value.visibility,
    resources: value.resources,
    retries: value.retries,
  };
}

/**
 * Normalize the optional `directExecutor:` block in `config.yaml`. When
 * a task plugin enables it, the framework dispatches `param_patch` tasks
 * to {@link runDirectTask} instead of an LLM worker.
 *
 * Returns `null` when the plugin opted out (`enabled: false` or block
 * missing); callers that see a `param_patch` task without config should
 * surface a clear "directExecutor not configured" error.
 *
 * @param {unknown} input
 * @return {Object|null}
 */
export function normalizeDirectExecutorConfig(input) {
  const source = input && typeof input === 'object' && !Array.isArray(input) ? input : null;
  if (!source) return null;
  if (source.enabled === false) return null;
  const required = ['sourceRepoPath', 'wrapperScript', 'sandboxRoot', 'outputRoot'];
  for (const key of required) {
    if (typeof source[key] !== 'string' || !source[key].trim()) return null;
  }
  const env = source.envOverrides && typeof source.envOverrides === 'object' && !Array.isArray(source.envOverrides)
    ? Object.fromEntries(
        Object.entries(source.envOverrides)
          .filter(([, value]) => typeof value === 'string' || typeof value === 'number')
          .map(([key, value]) => [key, String(value)]),
      )
    : {};
  // Files in source/ to inject into the manager's per-cycle context.
  // Lets the manager actually read the *current* (post-KEEP_EXPERIMENT)
  // train.py instead of just one-line ledger summaries — closes the
  // "code-reading" gap that gave Karpathy's serial agent its first
  // big win in our verification run.
  let contextFiles = [];
  if (Array.isArray(source.managerContextFiles)) {
    contextFiles = source.managerContextFiles
      .filter((entry) => typeof entry === 'string' && entry.trim())
      .map((entry) => entry.trim());
  }

  // ALPS scheduler config. searchAxes is the registry of operator names
  // the task plugin expects the manager to explore. Without it the
  // runtime can't tell "axis never tested" from "axis touched". role is
  // free-form (e.g. "primary", "schedule"); the runtime preserves it
  // for the manifest renderer but doesn't act on it.
  let searchAxes = [];
  if (Array.isArray(source.searchAxes)) {
    for (const entry of source.searchAxes) {
      if (typeof entry === 'string' && entry.trim()) {
        searchAxes.push({ name: entry.trim(), role: 'primary' });
      } else if (entry && typeof entry === 'object' && typeof entry.name === 'string' && entry.name.trim()) {
        searchAxes.push({
          name: entry.name.trim(),
          role: typeof entry.role === 'string' && entry.role.trim() ? entry.role.trim() : 'primary',
        });
      }
    }
  }

  // manualStaleKeepPolicy: "block" (default) refuses a manual KEEP whose
  // baseCommit is not current HEAD. "warn" applies it but marks
  // metricKnown=false and logs a banner.
  const rawPolicy = typeof source.manualStaleKeepPolicy === 'string'
    ? source.manualStaleKeepPolicy.trim().toLowerCase()
    : '';
  const manualStaleKeepPolicy = rawPolicy === 'warn' ? 'warn' : 'block';

  return {
    enabled: true,
    sourceRepoPath: source.sourceRepoPath.trim(),
    wrapperScript: source.wrapperScript.trim(),
    sandboxRoot: source.sandboxRoot.trim(),
    outputRoot: source.outputRoot.trim(),
    metricKey: typeof source.metricKey === 'string' && source.metricKey.trim()
      ? source.metricKey.trim() : 'val_bpb',
    timeBudgetSeconds: coercePositiveInteger(source.timeBudgetSeconds, 300),
    hardCapSeconds: coercePositiveInteger(source.hardCapSeconds, 900),
    envOverrides: env,
    managerContextFiles: contextFiles,
    searchAxes,
    // Per-task noise floor used when calibration hasn't produced an
    // estimate yet. Logged on every commit-gate decision via the
    // sigmaHatSource field so the user can audit which value was in
    // effect.
    fallbackSigma: coerceNonNegativeFloat(source.fallbackSigma, 0.0005),
    // How many baseline_repeat tasks to dispatch at the start of cycle
    // 1 to estimate the noise floor from observed seed-to-seed spread.
    // Set to 0 for tasks where each evaluation is too expensive to
    // afford 3× repeats up front.
    calibrationRepeats: coerceNonNegativeInteger(source.calibrationRepeats, 0),
    // Master switch: when true, the runtime applies KEEP_EXPERIMENT
    // automatically for any candidate that passes the safety rules and
    // the gate. False keeps the gate in dry-run logging mode only.
    autoKeepEnabled: coerceBoolean(source.autoKeepEnabled, false),
    manualStaleKeepPolicy,
    // When metricKnown becomes false (multi-keep block, stale manual
    // keep, etc.) the runtime enqueues a baseline_repeat at HIGHEST
    // priority to re-validate HEAD before the gate re-engages.
    autoEnqueueHeadValidation: coerceBoolean(source.autoEnqueueHeadValidation, true),
    // Cross-axis diversity filter (Phase 4); off by default until
    // Phase 2A logs are vetted.
    quotaDiversityEnabled: coerceBoolean(source.quotaDiversityEnabled, false),
    maxPerOperatorPerWake: coercePositiveInteger(source.maxPerOperatorPerWake, 4),
    maxSameOperatorValuePerWake: coercePositiveInteger(source.maxSameOperatorValuePerWake, 1),
    alwaysAllowValidation: coerceBoolean(source.alwaysAllowValidation, true),
  };
}

export function normalizeWorkerResources(input = {}, fallback = DEFAULT_WORKER_RESOURCES) {
  const source = input && typeof input === 'object' && !Array.isArray(input) ? input : {};
  return {
    gpus: coerceNonNegativeInteger(source.gpus, fallback.gpus),
    cpus: coercePositiveInteger(source.cpus, fallback.cpus),
  };
}

export function normalizeOrchestrationConfig(input = {}) {
  const source = input && typeof input === 'object' && !Array.isArray(input) ? input : {};
  let mode = typeof source.mode === 'string' && source.mode.trim()
    ? source.mode.trim()
    : DEFAULT_ORCHESTRATION.mode;
  if (mode === 'shared_allocation_manager_workers') mode = 'single_manager';
  if (mode === 'single-manager') mode = 'single_manager';

  const manager = typeof source.manager === 'string' && source.manager.trim()
    ? source.manager.trim()
    : DEFAULT_ORCHESTRATION.manager;

  return {
    mode,
    manager,
    maxConcurrentWorkers: coercePositiveInteger(source.maxConcurrentWorkers, DEFAULT_ORCHESTRATION.maxConcurrentWorkers),
    maxManagerPasses: coercePositiveInteger(source.maxManagerPasses, DEFAULT_ORCHESTRATION.maxManagerPasses),
    maxWallClockSeconds: coercePositiveInteger(source.maxWallClockSeconds, DEFAULT_ORCHESTRATION.maxWallClockSeconds),
    refillOnGraphDrain: coerceBoolean(source.refillOnGraphDrain, DEFAULT_ORCHESTRATION.refillOnGraphDrain),
    liveReplanOnTaskComplete: coerceBoolean(source.liveReplanOnTaskComplete, DEFAULT_ORCHESTRATION.liveReplanOnTaskComplete),
    liveReplanMinIntervalSeconds: coerceNonNegativeInteger(source.liveReplanMinIntervalSeconds, DEFAULT_ORCHESTRATION.liveReplanMinIntervalSeconds),
    // 1.5× by default holds the manager back so it sees waves of results;
    // tasks that benefit from per-result feedback (small effect sizes,
    // depth-of-iteration > breadth) can drop this to 1.0× to wake the
    // manager on every completion.
    liveReplanWatermarkRatio: coerceNonNegativeFloat(source.liveReplanWatermarkRatio, DEFAULT_ORCHESTRATION.liveReplanWatermarkRatio),
    targetGpuUtilization: Math.min(1, coerceNonNegativeFloat(source.targetGpuUtilization, DEFAULT_ORCHESTRATION.targetGpuUtilization)),
    defaultWorkerResources: normalizeWorkerResources(source.defaultWorkerResources, DEFAULT_WORKER_RESOURCES),
  };
}

export function normalizeScheduleStep(step, defaultWorkerResources = DEFAULT_WORKER_RESOURCES) {
  if (!step || typeof step !== 'object' || Array.isArray(step)) return null;

  if (Object.prototype.hasOwnProperty.call(step, 'delay')) {
    return Object.keys(step).length === 1 && typeof step.delay === 'number'
      ? { type: 'delay', delay: step.delay }
      : null;
  }

  if (Array.isArray(step.parallel)) {
    const items = step.parallel.map((item) => normalizeScheduleStep(item, defaultWorkerResources));
    if (items.length === 0 || items.some((item) => item?.type !== 'agent')) return null;
    return { type: 'parallel', steps: items };
  }

  const parsedAgentStep = parseAgentStep(step);
  if (!parsedAgentStep) return null;
  if (!Object.prototype.hasOwnProperty.call(parsedAgentStep, 'task')) return null;
  if (typeof parsedAgentStep.task !== 'string' || !parsedAgentStep.task.trim()) return null;

  return {
    type: 'agent',
    agent: parsedAgentStep.agent,
    task: parsedAgentStep.task.trim(),
    visibility: normalizeVisibility(parsedAgentStep.visibility),
    resources: normalizeWorkerResources(parsedAgentStep.resources, defaultWorkerResources),
    retries: coerceNonNegativeInteger(parsedAgentStep.retries, 2),
  };
}

// Manager may emit `<!-- KILL_TASKS --> [...] <!-- /KILL_TASKS -->` alongside
// (or instead of) a TASK_GRAPH to abort tasks whose results are no longer
// worth the GPU time. Returns a deduplicated array of task ids, or [] when
// the block is absent or malformed.
export function parseKillTasksDocument(resultText) {
  const match = String(resultText || '').match(/<!--\s*KILL_TASKS\s*-->\s*(\[[\s\S]*?\])\s*<!--\s*\/KILL_TASKS\s*-->/);
  if (!match) return [];
  let raw;
  try { raw = JSON.parse(match[1]); } catch { return []; }
  if (!Array.isArray(raw)) return [];
  const seen = new Set();
  for (const item of raw) {
    if (typeof item !== 'string') continue;
    const id = item.trim();
    if (id) seen.add(id);
  }
  return [...seen];
}

/**
 * Parse a `<!-- KEEP_EXPERIMENT --> [{"id": "...", "reason": "..."}] <!-- /KEEP_EXPERIMENT -->`
 * block from the manager's output. Each kept entry tells the framework to
 * fast-forward `tasks/<task>/source/`'s HEAD to the named experiment's
 * commit, so subsequent `param_patch` tasks with `base_ref: "HEAD"`
 * inherit its edits cumulatively (Karpathy-style branch advancing on top
 * of our parallel dispatch).
 *
 * Accepts either an array of objects `[{"id": "...", "reason": "..."}]`
 * or an array of bare strings `["exp_0010", ...]` (rationale logged
 * empty in the latter form). Forgiving on parse — missing block returns
 * `[]`, malformed JSON returns `[]`.
 *
 * @param {string|null|undefined} resultText
 * @return {Array<{id: string, reason: string}>}
 */
export function parseKeepExperimentDocument(resultText) {
  const match = String(resultText || '').match(/<!--\s*KEEP_EXPERIMENT\s*-->\s*(\[[\s\S]*?\])\s*<!--\s*\/KEEP_EXPERIMENT\s*-->/);
  if (!match) return [];
  let raw;
  try { raw = JSON.parse(match[1]); } catch { return []; }
  if (!Array.isArray(raw)) return [];
  const seen = new Map();
  for (const item of raw) {
    if (typeof item === 'string') {
      const id = item.trim();
      if (id && !seen.has(id)) seen.set(id, { id, reason: '' });
    } else if (item && typeof item === 'object' && !Array.isArray(item)) {
      const id = typeof item.id === 'string' ? item.id.trim() : '';
      if (!id) continue;
      const reason = typeof item.reason === 'string' ? item.reason.trim() : '';
      if (!seen.has(id)) seen.set(id, { id, reason });
    }
  }
  return [...seen.values()];
}

export function parseScheduleDocument(resultText, defaultWorkerResources = DEFAULT_WORKER_RESOURCES) {
  const match = String(resultText || '').match(/<!--\s*SCHEDULE\s*-->\s*([\[{][\s\S]*?[\]}])\s*<!--\s*\/SCHEDULE\s*-->/);
  if (!match) return null;
  try {
    const raw = JSON.parse(match[1]);
    if (!Array.isArray(raw)) return null;
    const steps = raw.map((step) => normalizeScheduleStep(step, defaultWorkerResources));
    if (steps.some((step) => step === null)) return null;
    return { _steps: steps };
  } catch {
    return null;
  }
}

export function parseTaskGraphDocument(resultText, defaultWorkerResources = DEFAULT_WORKER_RESOURCES, options = {}) {
  const errors = Array.isArray(options.errors) ? options.errors : null;
  const pushError = (msg) => { if (errors) errors.push(msg); };

  const match = String(resultText || '').match(/<!--\s*TASK_GRAPH\s*-->\s*({[\s\S]*?})\s*<!--\s*\/TASK_GRAPH\s*-->/);
  if (!match) return null;
  let raw;
  try {
    raw = JSON.parse(match[1]);
  } catch (err) {
    pushError(`task_graph JSON parse failed: ${err.message}`);
    return null;
  }
  if (!raw || typeof raw !== 'object' || Array.isArray(raw) || !Array.isArray(raw.tasks)) {
    pushError('task_graph root must be an object with a "tasks" array');
    return null;
  }

  const seenIds = new Set();
  const tagProducers = new Map();
  const tasks = raw.tasks.map((task, idx) => {
    const pos = `tasks[${idx}]`;
    if (!task || typeof task !== 'object' || Array.isArray(task)) {
      pushError(`${pos}: not an object`);
      return null;
    }
    const id = typeof task.id === 'string' && task.id.trim() ? task.id.trim() : null;
    const agent = typeof task.agent === 'string' && task.agent.trim() ? task.agent.trim() : null;
    const workerClass = normalizeWorkerClass(task.worker_class ?? task.workerClass ?? task.class);
    const work = typeof task.task === 'string' && task.task.trim() ? task.task.trim() : null;
    if (!id) { pushError(`${pos}: missing "id"`); return null; }
    if (!agent && !workerClass) { pushError(`${pos} (${id}): missing "agent" or "worker_class"`); return null; }
    if (!work) { pushError(`${pos} (${id}): missing "task"`); return null; }
    if (seenIds.has(id)) { pushError(`${pos} (${id}): duplicate id`); return null; }
    seenIds.add(id);
    const producesTags = normalizeStringList(task.produces_tags ?? task.producesTags);
    for (const tag of producesTags) {
      if (tagProducers.has(tag)) {
        pushError(`${pos} (${id}): tag "${tag}" already produced by "${tagProducers.get(tag)}"`);
        return null;
      }
      tagProducers.set(tag, id);
    }
    const executionMode = normalizeExecutionMode(task.execution_mode ?? task.executionMode);
    if (executionMode === null) {
      pushError(`${pos} (${id}): invalid execution_mode (must be code_edit, param_patch, llm_repair, or baseline_repeat)`);
      return null;
    }
    let edits = null;
    const editsProvided = task.edits !== undefined && task.edits !== null;
    if (executionMode === 'param_patch' && (!editsProvided || (Array.isArray(task.edits) && task.edits.length === 0))) {
      pushError(`${pos} (${id}): execution_mode=param_patch requires non-empty "edits"`);
      return null;
    }
    if (executionMode === 'baseline_repeat' && editsProvided && Array.isArray(task.edits) && task.edits.length > 0) {
      pushError(`${pos} (${id}): execution_mode=baseline_repeat cannot have edits; the harness runs unmodified on a fresh seed`);
      return null;
    }
    if (editsProvided && Array.isArray(task.edits) && task.edits.length > 0) {
      const editError = validateEdits(task.edits);
      if (editError) {
        pushError(`${pos} (${id}): ${editError}`);
        return null;
      }
      edits = task.edits;
    }
    const baseRef = typeof (task.base_ref ?? task.baseRef) === 'string'
      && (task.base_ref ?? task.baseRef).trim()
      ? (task.base_ref ?? task.baseRef).trim()
      : null;
    const earlyStop = normalizeEarlyStop(task.early_stop ?? task.earlyStop);
    if (earlyStop === false) {
      pushError(`${pos} (${id}): early_stop must be {check_at_seconds:int>0, abort_if_loss_above:number}`);
      return null;
    }
    return {
      id,
      type: 'agent',
      agent,
      workerClass,
      task: work,
      executionMode,
      edits,
      baseRef,
      earlyStop,
      rationale: typeof task.rationale === 'string' ? task.rationale.trim() : null,
      visibility: normalizeVisibility(task.visibility),
      resources: normalizeWorkerResources(task.resources, defaultWorkerResources),
      retries: coerceNonNegativeInteger(task.retries, 2),
      dependsOn: normalizeStringList(task.depends_on ?? task.dependsOn),
      dependsOnTags: normalizeStringList(task.depends_on_tags ?? task.dependsOnTags),
      producesTags,
      priority: coerceNonNegativeInteger(task.priority, 0),
      utility: coerceNonNegativeFloat(task.utility, 1),
      estimatedRuntimeSeconds: coercePositiveInteger(task.estimated_runtime_seconds ?? task.estimatedRuntimeSeconds, 0),
      replanAfter: task.replan_after === true || task.replanAfter === true,
    };
  });

  if (tasks.some((task) => task === null)) return null;

  const additionalKnownTaskIds = Array.isArray(options.additionalKnownTaskIds)
    ? options.additionalKnownTaskIds
    : [];
  const knownIds = new Set([
    ...tasks.map((task) => task.id),
    ...additionalKnownTaskIds.map((id) => String(id || '').trim()).filter(Boolean),
  ]);
  for (const task of tasks) {
    for (const dependencyId of task.dependsOn) {
      if (dependencyId === task.id) { pushError(`${task.id}: depends_on references itself`); return null; }
      if (!knownIds.has(dependencyId)) { pushError(`${task.id}: depends_on "${dependencyId}" not in graph`); return null; }
    }
  }

  const taskStates = Object.fromEntries(tasks.map((task) => [task.id, {
    status: 'pending',
    attempts: 0,
    startedAt: null,
    finishedAt: null,
    reason: null,
  }]));

  return {
    _kind: 'task_graph',
    _tasks: tasks,
    _runtime: {
      taskStates,
      producedTags: [],
      replanRequested: false,
      replanTaskId: null,
    },
  };
}

export function choosePrimaryActiveRun(activeRuns = []) {
  if (!Array.isArray(activeRuns) || activeRuns.length === 0) return null;
  const sorted = [...activeRuns].sort((left, right) => {
    if (!!left.isPrimary !== !!right.isPrimary) return left.isPrimary ? -1 : 1;
    if (!!left.isManager !== !!right.isManager) return left.isManager ? -1 : 1;
    return (left.startTime || 0) - (right.startTime || 0);
  });
  return sorted[0] || null;
}

export function getStepGpuDemand(step, defaultWorkerResources = DEFAULT_WORKER_RESOURCES) {
  if (!step || typeof step !== 'object') return 0;
  if (step.type === 'parallel' || step.type === 'delay') return 0;
  const normalized = normalizeWorkerResources(step.resources, defaultWorkerResources);
  return normalized.gpus;
}

export function classifyParallelStepGpuFit(step, availability = {}, defaultWorkerResources = DEFAULT_WORKER_RESOURCES) {
  const requestedGpus = getStepGpuDemand(step, defaultWorkerResources);
  const totalGpuTokens = Number.isInteger(availability.totalGpuTokens) ? availability.totalGpuTokens : 0;
  const availableGpuTokens = Number.isInteger(availability.availableGpuTokens) ? availability.availableGpuTokens : 0;
  const hasAllocationLease = availability.hasAllocationLease !== false;

  if (requestedGpus <= 0) {
    return { status: 'runnable', requestedGpus, reason: null };
  }
  if (!hasAllocationLease) {
    return { status: 'unschedulable', requestedGpus, reason: 'no active allocation lease' };
  }
  if (requestedGpus > totalGpuTokens) {
    return {
      status: 'unschedulable',
      requestedGpus,
      reason: `needs ${requestedGpus} GPU token(s), only ${totalGpuTokens} configured`,
    };
  }
  if (requestedGpus <= availableGpuTokens) {
    return { status: 'runnable', requestedGpus, reason: null };
  }
  return {
    status: 'waiting',
    requestedGpus,
    reason: `needs ${requestedGpus} GPU token(s), ${availableGpuTokens} currently available`,
  };
}

export function selectBackfillParallelStepIndex(steps = [], availability = {}, defaultWorkerResources = DEFAULT_WORKER_RESOURCES) {
  if (!Array.isArray(steps) || steps.length === 0) return -1;
  let bestIndex = -1;
  let bestGpuDemand = -1;

  for (let index = 0; index < steps.length; index += 1) {
    const step = steps[index];
    if (!step || step.type !== 'agent') continue;
    const fit = classifyParallelStepGpuFit(step, availability, defaultWorkerResources);
    if (fit.status !== 'runnable') continue;

    if (fit.requestedGpus > bestGpuDemand) {
      bestIndex = index;
      bestGpuDemand = fit.requestedGpus;
    }
  }

  return bestIndex;
}

export function computeTaskGraphCriticalPathScores(tasks = []) {
  const taskList = Array.isArray(tasks) ? tasks : [];
  const taskById = new Map();
  for (const task of taskList) {
    if (task?.id) taskById.set(task.id, task);
  }

  const memo = new Map();
  const visiting = new Set();
  const scoreFor = (task) => {
    if (!task?.id) return 0;
    if (memo.has(task.id)) return memo.get(task.id);
    if (visiting.has(task.id)) return 0;
    visiting.add(task.id);
    const producedTags = new Set(task.producesTags || []);
    const dependents = [...taskById.values()].filter((candidate) => {
      if ((candidate.dependsOn || []).includes(task.id)) return true;
      return (candidate.dependsOnTags || []).some((tag) => producedTags.has(tag));
    });
    const maxDependent = dependents.reduce((best, dependent) => Math.max(best, scoreFor(dependent)), 0);
    visiting.delete(task.id);
    const score = 1 + maxDependent;
    memo.set(task.id, score);
    return score;
  };

  const result = {};
  for (const task of taskById.values()) {
    result[task.id] = scoreFor(task);
  }
  return result;
}

export function parseGpuTokenOrdinals(tokenNames = []) {
  const seen = new Set();
  const result = [];
  for (const tokenName of tokenNames) {
    const s = String(tokenName || '').trim();
    const match = s.match(/^gpu(\d+)$/i);
    const ordinal = match ? Number.parseInt(match[1], 10)
      : /^\d+$/.test(s) ? Number.parseInt(s, 10)
      : null;
    if (ordinal == null) throw new Error(`invalid GPU token/ordinal: ${tokenName}`);
    if (!seen.has(ordinal)) { seen.add(ordinal); result.push(ordinal); }
  }
  return result;
}

export function buildManagedWorkerTask(task, resourceGrant = null, resources = null, workerName = null) {
  const body = String(task || '').trim();
  if (!resourceGrant) return body;
  const gpuOrdinals = parseGpuTokenOrdinals(Array.isArray(resourceGrant.tokenNames) ? resourceGrant.tokenNames : []);
  const lines = [
    'Resource grant for this run:',
    `- grant_id: ${resourceGrant.id}`,
    resourceGrant.leaseJobId ? `- lease_job_id: ${resourceGrant.leaseJobId}` : null,
    resourceGrant.nodeName ? `- node_name: ${resourceGrant.nodeName}` : null,
    Array.isArray(resourceGrant.tokenNames) && resourceGrant.tokenNames.length
      ? `- gpu_tokens: ${resourceGrant.tokenNames.join(', ')}`
      : null,
    gpuOrdinals.length ? `- gpu_ordinals: ${gpuOrdinals.join(', ')}` : null,
    resources?.cpus ? `- recommended_cpus_per_task: ${resources.cpus}` : null,
    '',
    gpuOrdinals.length
      ? 'The runner is already inside the GPU allocation. Launch the worker directly in the current shell — do not use srun, HpcSubmit, or any cross-node attach.'
      : null,
    gpuOrdinals.length
      ? `Launcher pattern: CUDA_VISIBLE_DEVICES=${gpuOrdinals.join(',')} OMP_NUM_THREADS=${resources?.cpus || 1} bash <worker-script> <task-output-dir>`
      : null,
    'If your task specifies an output directory, use that exact directory. Do not substitute a default worker-runs path.',
    '',
    body,
  ].filter(Boolean);
  return lines.join('\n');
}
