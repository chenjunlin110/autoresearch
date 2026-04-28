import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { readLineage, persistHeadState, editsEqual, writeLineage } from '../src/lineage.js';
import { improvementFor, improvedBy } from '../src/metric-conventions.js';

test('persistHeadState round-trips through readLineage', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'head-state-'));
  const lineagePath = path.join(dir, 'lineage.json');
  writeLineage(lineagePath, { task: 'unit', entries: [], current_head_commit: 'a'.repeat(40) });
  persistHeadState(lineagePath, {
    commit: 'b'.repeat(40),
    metricKnown: true,
    metric: 0.9931,
    sourceExperiment: 'exp_seed',
  });
  const back = readLineage(lineagePath);
  assert.equal(back.current_head_commit, 'b'.repeat(40));
  assert.equal(back.current_head_metric_known, true);
  assert.equal(back.current_head_metric, 0.9931);
  assert.equal(back.current_head_metric_source_experiment, 'exp_seed');
  fs.rmSync(dir, { recursive: true, force: true });
});

test('persistHeadState with metricKnown=false clears metric and source', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'head-state-'));
  const lineagePath = path.join(dir, 'lineage.json');
  persistHeadState(lineagePath, { commit: 'a'.repeat(40), metricKnown: true, metric: 0.99, sourceExperiment: 'x' });
  persistHeadState(lineagePath, { metricKnown: false, metric: 0.5 /* should be ignored */, sourceExperiment: 'y' });
  const back = readLineage(lineagePath);
  assert.equal(back.current_head_metric_known, false);
  assert.equal(back.current_head_metric, null);
  assert.equal(back.current_head_metric_source_experiment, null);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('editsEqual: identical edits are equal', () => {
  const a = [{ kind: 'constant_replace', file: 'train.py', name: 'X', expected_old_repr: '1', new_repr: '2' }];
  const b = [{ kind: 'constant_replace', file: 'train.py', name: 'X', expected_old_repr: '1', new_repr: '2' }];
  assert.equal(editsEqual(a, b), true);
});

test('editsEqual: different new_repr ⇒ not equal', () => {
  const a = [{ kind: 'constant_replace', file: 'train.py', name: 'X', expected_old_repr: '1', new_repr: '2' }];
  const b = [{ kind: 'constant_replace', file: 'train.py', name: 'X', expected_old_repr: '1', new_repr: '3' }];
  assert.equal(editsEqual(a, b), false);
});

test('editsEqual: different lengths ⇒ not equal', () => {
  const a = [{ kind: 'constant_replace', file: 'train.py', name: 'X', expected_old_repr: '1', new_repr: '2' }];
  const b = [];
  assert.equal(editsEqual(a, b), false);
});

test('editsEqual: handles unified_diff and block_replace kinds', () => {
  const a = [{ kind: 'unified_diff', file: 't.py', diff: 'D' }];
  const b = [{ kind: 'unified_diff', file: 't.py', diff: 'D' }];
  assert.equal(editsEqual(a, b), true);
  const c = [{ kind: 'block_replace', file: 't.py', anchor_regex: 'A', end_regex: 'E', new_text: 'N' }];
  const d = [{ kind: 'block_replace', file: 't.py', anchor_regex: 'A', end_regex: 'E', new_text: 'N' }];
  assert.equal(editsEqual(c, d), true);
});

test('improvementFor: positive when candidate beats baseline (lower-is-better)', () => {
  assert.ok(Math.abs(improvementFor({ metric: 0.98 }, { metric: 0.99 }) - 0.01) < 1e-12);
  // Bare numbers also accepted
  assert.ok(Math.abs(improvementFor(0.98, 0.99) - 0.01) < 1e-12);
});

test('improvementFor: NaN inputs ⇒ NaN return', () => {
  assert.ok(Number.isNaN(improvementFor({ metric: NaN }, { metric: 0.99 })));
  assert.ok(Number.isNaN(improvementFor({ metric: 0.99 }, { metric: NaN })));
});

test('improvedBy: only true when delta > threshold', () => {
  assert.equal(improvedBy(0.98, 0.99, 0.005), true);
  assert.equal(improvedBy(0.98, 0.99, 0.05), false);
});
