import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildManagedWorkerTask,
  classifyParallelStepGpuFit,
  choosePrimaryActiveRun,
  computeTaskGraphCriticalPathScores,
  normalizeOrchestrationConfig,
  normalizeScheduleStep,
  parseKillTasksDocument,
  parseScheduleDocument,
  parseTaskGraphDocument,
  selectBackfillParallelStepIndex,
} from '../src/orchestration-utils.js';

test('normalizeOrchestrationConfig supports single-manager aliases', () => {
  const config = normalizeOrchestrationConfig({
    mode: 'shared_allocation_manager_workers',
    manager: 'athena',
    maxConcurrentWorkers: '3',
    maxManagerPasses: '12',
    refillOnGraphDrain: 'false',
    liveReplanOnTaskComplete: 'true',
    liveReplanMinIntervalSeconds: '5',
    targetGpuUtilization: '0.75',
    defaultWorkerResources: { gpus: '1', cpus: '2' },
  });

  assert.equal(config.mode, 'single_manager');
  assert.equal(config.manager, 'athena');
  assert.equal(config.maxConcurrentWorkers, 3);
  assert.equal(config.maxManagerPasses, 12);
  assert.equal(config.refillOnGraphDrain, false);
  assert.equal(config.liveReplanOnTaskComplete, true);
  assert.equal(config.liveReplanMinIntervalSeconds, 5);
  assert.equal(config.targetGpuUtilization, 0.75);
  assert.deepEqual(config.defaultWorkerResources, { gpus: 1, cpus: 2 });
});

test('normalizeScheduleStep accepts sequential agent steps and legacy keyed form', () => {
  assert.deepEqual(
    normalizeScheduleStep({ agent: 'maya', task: 'Run tests', visibility: 'blind', resources: { gpus: 1 } }),
    {
      type: 'agent',
      agent: 'maya',
      task: 'Run tests',
      visibility: 'blind',
      resources: { gpus: 1, cpus: 1 },
      retries: 2,
    },
  );

  assert.deepEqual(
    normalizeScheduleStep({ leo: { task: 'Patch bug', visibility: 'focused', resources: { cpus: 2 } } }),
    {
      type: 'agent',
      agent: 'leo',
      task: 'Patch bug',
      visibility: 'focused',
      resources: { gpus: 0, cpus: 2 },
      retries: 2,
    },
  );
});

test('parseScheduleDocument accepts parallel groups', () => {
  const schedule = parseScheduleDocument(`
before
<!-- SCHEDULE -->
[
  {"parallel": [
    {"agent": "alice", "task": "train A", "resources": {"gpus": 1, "cpus": 1}},
    {"agent": "bob", "task": "train B", "resources": {"gpus": 1, "cpus": 1}}
  ]},
  {"delay": 5},
  {"agent": "carol", "task": "summarize"}
]
<!-- /SCHEDULE -->
after
  `);

  assert.ok(schedule);
  assert.equal(schedule._steps.length, 3);
  assert.equal(schedule._steps[0].type, 'parallel');
  assert.equal(schedule._steps[0].steps.length, 2);
  assert.equal(schedule._steps[0].steps[0].agent, 'alice');
  assert.deepEqual(schedule._steps[0].steps[1].resources, { gpus: 1, cpus: 1 });
  assert.equal(schedule._steps[2].agent, 'carol');
});

test('parseTaskGraphDocument accepts dependencies, tags, and replan barriers', () => {
  const graph = parseTaskGraphDocument(`
<!-- TASK_GRAPH -->
{
  "tasks": [
    {
      "id": "exp_a",
      "agent": "alice",
      "task": "run A",
      "resources": {"gpus": 1, "cpus": 1},
      "produces_tags": ["metrics:a"]
    },
    {
      "id": "analyze_a",
      "agent": "maya",
      "task": "analyze A",
      "depends_on": ["exp_a"],
      "depends_on_tags": ["metrics:a"],
      "replan_after": true
    }
  ]
}
<!-- /TASK_GRAPH -->
  `);

  assert.ok(graph);
  assert.equal(graph._kind, 'task_graph');
  assert.equal(graph._tasks.length, 2);
  assert.deepEqual(graph._tasks[0].producesTags, ['metrics:a']);
  assert.deepEqual(graph._tasks[1].dependsOn, ['exp_a']);
  assert.deepEqual(graph._tasks[1].dependsOnTags, ['metrics:a']);
  assert.equal(graph._tasks[1].replanAfter, true);
  assert.equal(graph._runtime.taskStates.exp_a.status, 'pending');
});

test('parseTaskGraphDocument accepts worker_class tasks without fixed agents', () => {
  const graph = parseTaskGraphDocument(`
<!-- TASK_GRAPH -->
{
  "tasks": [
    {
      "id": "exp_a",
      "worker_class": "experiment-runner",
      "task": "run A",
      "resources": {"gpus": 1},
      "priority": 2,
      "utility": 3.5,
      "estimated_runtime_seconds": 30,
      "produces_tags": ["metrics:a"]
    }
  ]
}
<!-- /TASK_GRAPH -->
  `);

  assert.ok(graph);
  assert.equal(graph._tasks[0].agent, null);
  assert.equal(graph._tasks[0].workerClass, 'experiment_runner');
  assert.equal(graph._tasks[0].priority, 2);
  assert.equal(graph._tasks[0].utility, 3.5);
  assert.equal(graph._tasks[0].estimatedRuntimeSeconds, 30);
});

test('parseTaskGraphDocument: param_patch carries execution_mode, edits, base_ref, rationale', () => {
  const graph = parseTaskGraphDocument(`
<!-- TASK_GRAPH -->
{"tasks":[
  {"id":"exp_42","worker_class":"experiment-runner","task":"betas sweep",
   "execution_mode":"param_patch","base_ref":"HEAD",
   "rationale":"betas=(0.9, 0.97) likely improves convergence",
   "edits":[
     {"file":"train.py","kind":"constant_replace","name":"ADAM_BETAS",
      "expected_old_repr":"(0.9, 0.95)","new_repr":"(0.9, 0.97)"}
   ]}
]}
<!-- /TASK_GRAPH -->
  `);
  assert.ok(graph);
  const task = graph._tasks[0];
  assert.equal(task.executionMode, 'param_patch');
  assert.equal(task.baseRef, 'HEAD');
  assert.equal(task.rationale, 'betas=(0.9, 0.97) likely improves convergence');
  assert.equal(task.edits.length, 1);
  assert.equal(task.edits[0].kind, 'constant_replace');
});

test('parseTaskGraphDocument: defaults execution_mode to code_edit and leaves edits null', () => {
  const graph = parseTaskGraphDocument(`
<!-- TASK_GRAPH -->
{"tasks":[{"id":"exp_a","agent":"alice","task":"do thing"}]}
<!-- /TASK_GRAPH -->
  `);
  assert.ok(graph);
  assert.equal(graph._tasks[0].executionMode, 'code_edit');
  assert.equal(graph._tasks[0].edits, null);
});

test('parseTaskGraphDocument: rejects param_patch without edits', () => {
  const errors = [];
  const graph = parseTaskGraphDocument(`
<!-- TASK_GRAPH -->
{"tasks":[{"id":"exp_a","agent":"alice","task":"x","execution_mode":"param_patch"}]}
<!-- /TASK_GRAPH -->
  `, undefined, { errors });
  assert.equal(graph, null);
  assert.ok(errors.some((e) => /requires non-empty "edits"/.test(e)),
    `expected edits-required error, got: ${errors.join(' | ')}`);
});

test('parseTaskGraphDocument: rejects malformed edit kinds', () => {
  const errors = [];
  const graph = parseTaskGraphDocument(`
<!-- TASK_GRAPH -->
{"tasks":[{"id":"exp_a","agent":"alice","task":"x","execution_mode":"param_patch",
  "edits":[{"file":"train.py","kind":"regex_replace","pattern":"foo","replacement":"bar"}]}]}
<!-- /TASK_GRAPH -->
  `, undefined, { errors });
  assert.equal(graph, null);
  assert.ok(errors.some((e) => /expected_count/.test(e)),
    `expected count error, got: ${errors.join(' | ')}`);
});

test('parseTaskGraphDocument: accepts well-formed early_stop and surfaces it on the task', () => {
  const graph = parseTaskGraphDocument(`
<!-- TASK_GRAPH -->
{"tasks":[{"id":"x","agent":"a","task":"y",
  "early_stop":{"check_at_seconds":90,"abort_if_loss_above":4.0}}]}
<!-- /TASK_GRAPH -->
  `);
  assert.ok(graph);
  assert.deepEqual(graph._tasks[0].earlyStop, {
    checkAtSeconds: 90,
    abortIfLossAbove: 4.0,
  });
});

test('parseTaskGraphDocument: rejects malformed early_stop', () => {
  const errors = [];
  const graph = parseTaskGraphDocument(`
<!-- TASK_GRAPH -->
{"tasks":[{"id":"x","agent":"a","task":"y",
  "early_stop":{"check_at_seconds":90}}]}
<!-- /TASK_GRAPH -->
  `, undefined, { errors });
  assert.equal(graph, null);
  assert.ok(errors.some((e) => /early_stop must be/.test(e)));
});

test('parseTaskGraphDocument: rejects unknown execution_mode', () => {
  const errors = [];
  const graph = parseTaskGraphDocument(`
<!-- TASK_GRAPH -->
{"tasks":[{"id":"x","agent":"a","task":"y","execution_mode":"yolo"}]}
<!-- /TASK_GRAPH -->
  `, undefined, { errors });
  assert.equal(graph, null);
  assert.ok(errors.some((e) => /invalid execution_mode/.test(e)));
});

test('parseTaskGraphDocument can depend on existing live graph task ids', () => {
  const graph = parseTaskGraphDocument(`
<!-- TASK_GRAPH -->
{"tasks":[
  {"id":"analyze_existing","agent":"maya","task":"analyze","depends_on":["exp_existing"]}
]}
<!-- /TASK_GRAPH -->
  `, undefined, { additionalKnownTaskIds: ['exp_existing'] });

  assert.ok(graph);
  assert.deepEqual(graph._tasks[0].dependsOn, ['exp_existing']);
});

test('computeTaskGraphCriticalPathScores ranks tasks on longer dependency chains', () => {
  const graph = parseTaskGraphDocument(`
<!-- TASK_GRAPH -->
{"tasks":[
  {"id":"a","agent":"alice","task":"A"},
  {"id":"b","agent":"bob","task":"B","depends_on":["a"]},
  {"id":"c","agent":"carol","task":"C","depends_on":["b"]},
  {"id":"d","agent":"dana","task":"D"}
]}
<!-- /TASK_GRAPH -->
  `);

  assert.deepEqual(computeTaskGraphCriticalPathScores(graph._tasks), {
    a: 3,
    b: 2,
    c: 1,
    d: 1,
  });
});

test('choosePrimaryActiveRun prefers primary manager then oldest run', () => {
  const selected = choosePrimaryActiveRun([
    { id: 'worker-1', isManager: false, isPrimary: false, startTime: 20 },
    { id: 'manager-1', isManager: true, isPrimary: true, startTime: 30 },
    { id: 'worker-0', isManager: false, isPrimary: false, startTime: 10 },
  ]);
  assert.equal(selected.id, 'manager-1');

  const fallback = choosePrimaryActiveRun([
    { id: 'worker-1', isManager: false, isPrimary: false, startTime: 20 },
    { id: 'worker-0', isManager: false, isPrimary: false, startTime: 10 },
  ]);
  assert.equal(fallback.id, 'worker-0');
});

test('classifyParallelStepGpuFit distinguishes runnable, waiting, and unschedulable steps', () => {
  const step = normalizeScheduleStep({ agent: 'maya', task: 'Run experiment', resources: { gpus: 2, cpus: 1 } });

  assert.deepEqual(
    classifyParallelStepGpuFit(step, { hasAllocationLease: true, totalGpuTokens: 4, availableGpuTokens: 2 }),
    { status: 'runnable', requestedGpus: 2, reason: null },
  );

  assert.deepEqual(
    classifyParallelStepGpuFit(step, { hasAllocationLease: true, totalGpuTokens: 4, availableGpuTokens: 1 }),
    { status: 'waiting', requestedGpus: 2, reason: 'needs 2 GPU token(s), 1 currently available' },
  );

  assert.deepEqual(
    classifyParallelStepGpuFit(step, { hasAllocationLease: false, totalGpuTokens: 4, availableGpuTokens: 4 }),
    { status: 'unschedulable', requestedGpus: 2, reason: 'no active allocation lease' },
  );
});

test('selectBackfillParallelStepIndex chooses the largest runnable task, not just queue head', () => {
  const steps = [
    normalizeScheduleStep({ agent: 'large', task: '4-gpu run', resources: { gpus: 4, cpus: 1 } }),
    normalizeScheduleStep({ agent: 'small-a', task: '1-gpu run', resources: { gpus: 1, cpus: 1 } }),
    normalizeScheduleStep({ agent: 'medium', task: '2-gpu run', resources: { gpus: 2, cpus: 1 } }),
    normalizeScheduleStep({ agent: 'small-b', task: 'cpu-only summary', resources: { gpus: 0, cpus: 1 } }),
  ];

  assert.equal(
    selectBackfillParallelStepIndex(steps, { hasAllocationLease: true, totalGpuTokens: 8, availableGpuTokens: 2 }),
    2,
  );

  assert.equal(
    selectBackfillParallelStepIndex(steps, { hasAllocationLease: true, totalGpuTokens: 8, availableGpuTokens: 1 }),
    1,
  );

  assert.equal(
    selectBackfillParallelStepIndex(steps, { hasAllocationLease: true, totalGpuTokens: 8, availableGpuTokens: 0 }),
    3,
  );
});

test('buildManagedWorkerTask prepends grant instructions', () => {
  const task = buildManagedWorkerTask(
    'Run the training job.',
    {
      id: 7,
      leaseJobId: '1575902',
      nodeName: 'fs-mbz-gpu-875',
      tokenNames: ['gpu2'],
    },
    { gpus: 1, cpus: 2 },
    'maya',
  );

  assert.match(task, /grant_id: 7/);
  assert.match(task, /gpu_tokens: gpu2/);
  assert.match(task, /gpu_ordinals: 2/);
  assert.match(task, /recommended_cpus_per_task: 2/);
  assert.match(task, /Launcher pattern: CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 bash <worker-script> <task-output-dir>/);
  assert.match(task, /do not use srun, HpcSubmit, or any cross-node attach/);
  assert.match(task, /use that exact directory/);
  assert.match(task, /Run the training job/);
});

test('parseKillTasksDocument extracts ids and is forgiving', () => {
  assert.deepEqual(
    parseKillTasksDocument('chatter <!-- KILL_TASKS --> ["exp_0042", "exp_0099"] <!-- /KILL_TASKS --> more chatter'),
    ['exp_0042', 'exp_0099'],
  );
  // dedup + trim
  assert.deepEqual(
    parseKillTasksDocument('<!-- KILL_TASKS -->[" exp_a ","exp_a"]<!-- /KILL_TASKS -->'),
    ['exp_a'],
  );
  // missing block, malformed JSON, non-string entries → []
  assert.deepEqual(parseKillTasksDocument('no block here'), []);
  assert.deepEqual(parseKillTasksDocument('<!-- KILL_TASKS -->[oops<!-- /KILL_TASKS -->'), []);
  assert.deepEqual(parseKillTasksDocument('<!-- KILL_TASKS -->[1,2,3]<!-- /KILL_TASKS -->'), []);
});
