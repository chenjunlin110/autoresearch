import {
  classifyParallelStepGpuFit,
  computeTaskGraphCriticalPathScores,
  normalizeWorkerClass,
} from './orchestration-utils.js';

function normalizeBusyWorkerNames(value) {
  const source = value instanceof Set ? [...value] : (Array.isArray(value) ? value : []);
  return new Set(source.map((item) => String(item || '').toLowerCase()).filter(Boolean));
}

export function inferWorkerClassFromMetadata({ content = '', role = '', name = '' } = {}) {
  const explicit = (String(content).match(/^worker_class:\s*(.+)$/m) || [])[1]?.trim()
    || (String(content).match(/^class:\s*(.+)$/m) || [])[1]?.trim();
  if (explicit) return normalizeWorkerClass(explicit);

  const normalizedRole = normalizeWorkerClass(role || '');
  if (normalizedRole?.includes('experiment') || normalizedRole?.includes('runner')) return 'experiment_runner';
  if (normalizedRole?.includes('analyst') || normalizedRole?.includes('analysis')) return 'analyst';
  if (normalizedRole?.includes('patch') || normalizedRole?.includes('implement')) return 'patcher';
  return normalizeWorkerClass(name);
}

export function findWorkerByName(workers = [], name = '') {
  return workers.find((worker) => worker.name.toLowerCase() === String(name || '').toLowerCase()) || null;
}

export function findWorkerForStep(step, workers = [], busyWorkerNames = new Set()) {
  const busy = normalizeBusyWorkerNames(busyWorkerNames);
  if (step?.agent) {
    const worker = findWorkerByName(workers, step.agent);
    if (!worker) return { worker: null, reason: `worker "${step.agent}" not found`, wait: false };
    if (busy.has(worker.name.toLowerCase())) {
      return { worker: null, reason: `worker "${worker.name}" is busy`, wait: true };
    }
    return { worker, reason: null, wait: false };
  }

  const workerClass = normalizeWorkerClass(step?.workerClass);
  if (!workerClass) {
    return { worker: null, reason: 'task has no agent or worker_class', wait: false };
  }

  const candidates = workers.filter((worker) => normalizeWorkerClass(worker.workerClass) === workerClass);
  if (candidates.length === 0) {
    return { worker: null, reason: `no worker with class "${workerClass}"`, wait: false };
  }

  const idle = candidates.find((worker) => !busy.has(worker.name.toLowerCase()));
  if (!idle) {
    return { worker: null, reason: `all workers with class "${workerClass}" are busy`, wait: true };
  }
  return { worker: idle, reason: null, wait: false };
}

function taskRank(task, fit, criticalPathScores) {
  return {
    priority: Number.isInteger(task.priority) ? task.priority : 0,
    criticalPath: criticalPathScores[task.id] || 1,
    utility: typeof task.utility === 'number' ? task.utility : 1,
    gpuDemand: fit.requestedGpus,
    estimatedRuntime: Number.isInteger(task.estimatedRuntimeSeconds) && task.estimatedRuntimeSeconds > 0
      ? task.estimatedRuntimeSeconds
      : Number.MAX_SAFE_INTEGER,
  };
}

function isBetterRank(candidate, best) {
  if (!best) return true;
  if (candidate.priority !== best.priority) return candidate.priority > best.priority;
  if (candidate.criticalPath !== best.criticalPath) return candidate.criticalPath > best.criticalPath;
  if (candidate.utility !== best.utility) return candidate.utility > best.utility;
  if (candidate.gpuDemand !== best.gpuDemand) return candidate.gpuDemand > best.gpuDemand;
  return candidate.estimatedRuntime < best.estimatedRuntime;
}

export function selectTaskGraphRunnableTask({
  tasks = [],
  availability = {},
  defaultWorkerResources,
  workers = [],
  running = [],
  plan = null,
} = {}) {
  let bestIndex = -1;
  let bestWorker = null;
  let bestRank = null;
  const busyWorkerNames = new Set(running.map((entry) => entry.workerName?.toLowerCase()).filter(Boolean));
  const criticalPathScores = plan ? computeTaskGraphCriticalPathScores(plan._tasks) : {};

  for (let index = 0; index < tasks.length; index += 1) {
    const task = tasks[index];
    const fit = classifyParallelStepGpuFit(task, availability, defaultWorkerResources);
    if (fit.status !== 'runnable') continue;

    const workerSelection = findWorkerForStep(task, workers, busyWorkerNames);
    if (!workerSelection.worker) continue;

    const rank = taskRank(task, fit, criticalPathScores);
    if (isBetterRank(rank, bestRank)) {
      bestIndex = index;
      bestWorker = workerSelection.worker;
      bestRank = rank;
    }
  }

  return { index: bestIndex, worker: bestWorker };
}
