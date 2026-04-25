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
    return {
      id,
      type: 'agent',
      agent,
      workerClass,
      task: work,
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
