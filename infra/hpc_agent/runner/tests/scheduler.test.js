import test from 'node:test';
import assert from 'node:assert/strict';

import { parseTaskGraphDocument } from '../src/orchestration-utils.js';
import {
  findWorkerForStep,
  inferWorkerClassFromMetadata,
  selectTaskGraphRunnableTask,
} from '../src/scheduler.js';

test('inferWorkerClassFromMetadata prefers frontmatter and falls back to role/name', () => {
  assert.equal(
    inferWorkerClassFromMetadata({
      content: 'worker_class: experiment-runner\n',
      role: 'Analyst',
      name: 'maya',
    }),
    'experiment_runner',
  );
  assert.equal(
    inferWorkerClassFromMetadata({ role: 'Patch Implementer', name: 'dev-1' }),
    'patcher',
  );
  assert.equal(
    inferWorkerClassFromMetadata({ role: '', name: 'GPU Worker 7' }),
    'gpu_worker_7',
  );
});

test('findWorkerForStep binds worker_class tasks to idle workers', () => {
  const workers = [
    { name: 'exp_runner_0', workerClass: 'experiment_runner' },
    { name: 'exp_runner_1', workerClass: 'experiment_runner' },
    { name: 'maya', workerClass: 'analyst' },
  ];

  assert.equal(
    findWorkerForStep({ workerClass: 'experiment-runner' }, workers, new Set(['exp_runner_0'])).worker.name,
    'exp_runner_1',
  );
  assert.deepEqual(
    findWorkerForStep({ workerClass: 'analyst' }, workers, new Set(['maya'])),
    { worker: null, reason: 'all workers with class "analyst" are busy', wait: true },
  );
  assert.deepEqual(
    findWorkerForStep({ agent: 'missing' }, workers),
    { worker: null, reason: 'worker "missing" not found', wait: false },
  );
});

test('selectTaskGraphRunnableTask ranks by priority, critical path, utility, GPU demand, and runtime', () => {
  const graph = parseTaskGraphDocument(`
<!-- TASK_GRAPH -->
{"tasks":[
  {"id":"exp_chain","worker_class":"experiment_runner","task":"chain head","resources":{"gpus":1},"priority":1,"utility":1,"estimated_runtime_seconds":60},
  {"id":"analyze_chain","worker_class":"analyst","task":"analyze","depends_on":["exp_chain"]},
  {"id":"exp_wide","worker_class":"experiment_runner","task":"wide","resources":{"gpus":2},"priority":1,"utility":1,"estimated_runtime_seconds":30},
  {"id":"exp_low","worker_class":"experiment_runner","task":"low","resources":{"gpus":1},"priority":0,"utility":99}
]}
<!-- /TASK_GRAPH -->
  `);
  const workers = [
    { name: 'exp_runner_0', workerClass: 'experiment_runner' },
    { name: 'exp_runner_1', workerClass: 'experiment_runner' },
    { name: 'maya', workerClass: 'analyst' },
  ];

  const runnable = [graph._tasks[2], graph._tasks[0], graph._tasks[3]];
  const selected = selectTaskGraphRunnableTask({
    tasks: runnable,
    availability: { hasAllocationLease: true, totalGpuTokens: 8, availableGpuTokens: 8 },
    defaultWorkerResources: { gpus: 0, cpus: 1 },
    workers,
    running: [],
    plan: graph,
  });

  assert.equal(selected.index, 1);
  assert.equal(selected.worker.name, 'exp_runner_0');

  const selectedWithBusyWorker = selectTaskGraphRunnableTask({
    tasks: [graph._tasks[0]],
    availability: { hasAllocationLease: true, totalGpuTokens: 8, availableGpuTokens: 8 },
    defaultWorkerResources: { gpus: 0, cpus: 1 },
    workers,
    running: [{ workerName: 'exp_runner_0' }],
    plan: graph,
  });
  assert.equal(selectedWithBusyWorker.worker.name, 'exp_runner_1');
});
