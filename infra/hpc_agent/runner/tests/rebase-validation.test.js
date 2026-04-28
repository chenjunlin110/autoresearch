import test from 'node:test';
import assert from 'node:assert/strict';

// Phase 5 logic is methods on ProjectRunner; we exercise it via a
// minimal stub that fakes only what _maybeEnqueueRebaseValidation
// touches: currentSchedule (with _tasks + _runtime.taskStates),
// pendingRebaseValidations Set, _isTaskGraphPlan, _ensureTaskGraphRuntime,
// saveState, log helper. Importing ProjectRunner requires more setup
// than the test gives back, so we duplicate the helper here and verify
// its behavior — drift risk if the real method changes, but the surface
// is small enough to catch in code review.

function makeRunner() {
  const runner = {
    id: 'test',
    pendingRebaseValidations: new Set(),
    currentSchedule: {
      _kind: 'task_graph',
      _tasks: [],
      _runtime: { taskStates: {}, producedTags: [] },
    },
    _isTaskGraphPlan(plan) { return plan && plan._kind === 'task_graph' && Array.isArray(plan._tasks); },
    _ensureTaskGraphRuntime(plan) {
      if (!plan._runtime) plan._runtime = { taskStates: {}, producedTags: [] };
      if (!plan._runtime.taskStates) plan._runtime.taskStates = {};
    },
    saveState() {},
  };
  // Copy of the production method body. Keep in sync with server.js.
  runner._maybeEnqueueRebaseValidation = function (originId, originPicked) {
    if (!originPicked || !Array.isArray(originPicked.edits) || originPicked.edits.length === 0) return;
    if (originId.includes('__rebase')) return;
    if (this.pendingRebaseValidations.has(originId)) return;
    if (!this.currentSchedule || !this._isTaskGraphPlan(this.currentSchedule)) return;
    const rebaseId = `${originId}__rebase`;
    if (this.currentSchedule._tasks.some((t) => t.id === rebaseId)) return;
    const synthetic = {
      id: rebaseId,
      type: 'agent',
      executionMode: 'param_patch',
      edits: originPicked.edits,
      baseRef: 'HEAD',
      priority: 8,
    };
    this._ensureTaskGraphRuntime(this.currentSchedule);
    this.currentSchedule._tasks.push(synthetic);
    this.currentSchedule._runtime.taskStates[rebaseId] = { status: 'pending', attempts: 0 };
    this.pendingRebaseValidations.add(originId);
    this.saveState();
  };
  return runner;
}

test('enqueues rebase task when origin beat its base', () => {
  const r = makeRunner();
  const picked = { edits: [{ kind: 'constant_replace', name: 'X', new_repr: '42' }] };
  r._maybeEnqueueRebaseValidation('exp_001', picked);
  assert.equal(r.currentSchedule._tasks.length, 1);
  assert.equal(r.currentSchedule._tasks[0].id, 'exp_001__rebase');
  assert.equal(r.currentSchedule._tasks[0].executionMode, 'param_patch');
  assert.equal(r.currentSchedule._tasks[0].baseRef, 'HEAD');
  assert.deepEqual(r.currentSchedule._tasks[0].edits, picked.edits);
  assert.ok(r.pendingRebaseValidations.has('exp_001'));
});

test('refuses to re-rebase a rebase task (no infinite chain)', () => {
  const r = makeRunner();
  const picked = { edits: [{ kind: 'constant_replace', name: 'X', new_repr: '1' }] };
  r._maybeEnqueueRebaseValidation('exp_001__rebase', picked);
  assert.equal(r.currentSchedule._tasks.length, 0);
});

test('idempotent: same origin id only enqueues once', () => {
  const r = makeRunner();
  const picked = { edits: [{ kind: 'constant_replace', name: 'X', new_repr: '1' }] };
  r._maybeEnqueueRebaseValidation('exp_001', picked);
  r._maybeEnqueueRebaseValidation('exp_001', picked);
  assert.equal(r.currentSchedule._tasks.length, 1);
});

test('skips if origin has no edits', () => {
  const r = makeRunner();
  r._maybeEnqueueRebaseValidation('exp_001', { edits: [] });
  r._maybeEnqueueRebaseValidation('exp_002', null);
  r._maybeEnqueueRebaseValidation('exp_003', { edits: null });
  assert.equal(r.currentSchedule._tasks.length, 0);
});

test('skips if rebase task id already in plan (cross-cycle dedup)', () => {
  const r = makeRunner();
  r.currentSchedule._tasks.push({ id: 'exp_001__rebase', executionMode: 'param_patch' });
  const picked = { edits: [{ kind: 'constant_replace', name: 'X', new_repr: '1' }] };
  r._maybeEnqueueRebaseValidation('exp_001', picked);
  assert.equal(r.currentSchedule._tasks.length, 1); // still just the pre-existing one
  assert.ok(!r.pendingRebaseValidations.has('exp_001'));
});

test('writes runtime taskState for the new task', () => {
  const r = makeRunner();
  const picked = { edits: [{ kind: 'constant_replace', name: 'X', new_repr: '1' }] };
  r._maybeEnqueueRebaseValidation('exp_001', picked);
  assert.equal(r.currentSchedule._runtime.taskStates['exp_001__rebase']?.status, 'pending');
});
