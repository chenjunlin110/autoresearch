import test from 'node:test';
import assert from 'node:assert/strict';
import { applyDiversityQuota } from '../src/quota-diversity.js';

const cfgOn = {
  quotaDiversityEnabled: true,
  maxPerOperatorPerWake: 2,
  maxSameOperatorValuePerWake: 1,
  alwaysAllowValidation: true,
};

function task(id, edits, priority = 0, executionMode = 'param_patch') {
  return { id, edits, priority, executionMode };
}

test('flag off ⇒ no filtering', () => {
  const tasks = [
    task('a', [{ kind: 'constant_replace', name: 'X', new_repr: '1' }]),
    task('b', [{ kind: 'constant_replace', name: 'X', new_repr: '2' }]),
    task('c', [{ kind: 'constant_replace', name: 'X', new_repr: '3' }]),
  ];
  const r = applyDiversityQuota(tasks, { ...cfgOn, quotaDiversityEnabled: false });
  assert.equal(r.accepted.length, 3);
  assert.equal(r.dropped.length, 0);
});

test('drops same-axis overflow beyond maxPerOperatorPerWake', () => {
  const tasks = [
    task('a', [{ kind: 'constant_replace', name: 'X', new_repr: '1' }], 1),
    task('b', [{ kind: 'constant_replace', name: 'X', new_repr: '2' }], 1),
    task('c', [{ kind: 'constant_replace', name: 'X', new_repr: '3' }], 1),
    task('d', [{ kind: 'constant_replace', name: 'Y', new_repr: '1' }], 1),
  ];
  const r = applyDiversityQuota(tasks, cfgOn);
  // X axis quota 2; one X dropped, Y kept.
  assert.equal(r.accepted.length, 3);
  assert.equal(r.dropped.length, 1);
  assert.match(r.dropped[0].reason, /^axis_quota:X$/);
});

test('drops exact-duplicate (operator, value) within wave', () => {
  const tasks = [
    task('a', [{ kind: 'constant_replace', name: 'X', new_repr: '1' }]),
    task('b', [{ kind: 'constant_replace', name: 'X', new_repr: '1' }]),
  ];
  const r = applyDiversityQuota(tasks, cfgOn);
  assert.equal(r.accepted.length, 1);
  assert.equal(r.dropped.length, 1);
  assert.match(r.dropped[0].reason, /value_repeat/);
});

test('validation tasks are exempt', () => {
  const tasks = [
    task('v1', [], 100, 'baseline_repeat'),
    task('a', [{ kind: 'constant_replace', name: 'X', new_repr: '1' }]),
    task('b', [{ kind: 'constant_replace', name: 'X', new_repr: '2' }]),
    task('c', [{ kind: 'constant_replace', name: 'X', new_repr: '3' }]),
  ];
  const r = applyDiversityQuota(tasks, cfgOn);
  assert.ok(r.accepted.find((t) => t.id === 'v1'), 'baseline_repeat must pass');
  assert.equal(r.accepted.filter((t) => t.id !== 'v1').length, 2); // X quota=2
});

test('manager priority preserved within axis (high-priority kept; low-priority dropped)', () => {
  const tasks = [
    task('lo1', [{ kind: 'constant_replace', name: 'X', new_repr: '1' }], 1),
    task('lo2', [{ kind: 'constant_replace', name: 'X', new_repr: '2' }], 1),
    task('hi', [{ kind: 'constant_replace', name: 'X', new_repr: '3' }], 10),
  ];
  const r = applyDiversityQuota(tasks, cfgOn);
  assert.ok(r.accepted.find((t) => t.id === 'hi'));
  assert.equal(r.accepted.length, 2);
});

test('non-edit tasks pass through (no operators to count)', () => {
  const tasks = [
    task('code1', [], 0, 'code_edit'),
    task('a', [{ kind: 'constant_replace', name: 'X', new_repr: '1' }]),
  ];
  const r = applyDiversityQuota(tasks, cfgOn);
  assert.equal(r.accepted.length, 2);
});

test('preserves original task order for accepted tasks', () => {
  const tasks = [
    task('a', [{ kind: 'constant_replace', name: 'X', new_repr: '1' }], 1),
    task('b', [{ kind: 'constant_replace', name: 'Y', new_repr: '1' }], 5),
    task('c', [{ kind: 'constant_replace', name: 'Z', new_repr: '1' }], 1),
  ];
  const r = applyDiversityQuota(tasks, cfgOn);
  assert.deepEqual(r.accepted.map((t) => t.id), ['a', 'b', 'c']);
});
