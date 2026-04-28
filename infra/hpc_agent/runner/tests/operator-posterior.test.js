import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import {
  buildOperatorPosterior,
  formatCoverageMarkdown,
  extractOperators,
  operatorKeyForEdit,
  welfordSigma,
} from '../src/operator-posterior.js';

function makeEventsDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'op-post-'));
}

function writeDoc(dir, taskId, events) {
  const doc = { task_id: taskId, events };
  fs.writeFileSync(path.join(dir, `${taskId}.json`), JSON.stringify(doc));
}

test('operatorKeyForEdit handles all four kinds', () => {
  assert.equal(operatorKeyForEdit({ kind: 'constant_replace', name: 'ASPECT_RATIO' }), 'ASPECT_RATIO');
  const r = operatorKeyForEdit({ kind: 'regex_replace', pattern: 'foo', replacement: 'bar' });
  assert.match(r, /^_REGEX_/);
  const b = operatorKeyForEdit({ kind: 'block_replace', anchor_regex: 'def train', new_text: 'pass' });
  assert.match(b, /^_BLOCK_/);
  const u = operatorKeyForEdit({ kind: 'unified_diff', diff: '@@ -1 +1 @@\n-a\n+b\n' });
  assert.match(u, /^_DIFF_/);
});

test('Welford updates μ and σ correctly on synthetic single-edit events', () => {
  const dir = makeEventsDir();
  const baseCommit = 'a'.repeat(40);
  // Three single-edit experiments on ASPECT_RATIO. Improvements: +0.001,
  // +0.002, +0.003. yBase fixed at 0.992; candidate metrics 0.991, 0.990,
  // 0.989. Mean improvement = 0.002, sample stddev = 0.001.
  const cases = [
    { id: 'e1', value: '64', yCand: 0.991 },
    { id: 'e2', value: '80', yCand: 0.990 },
    { id: 'e3', value: '96', yCand: 0.989 },
  ];
  for (const c of cases) {
    writeDoc(dir, c.id, [
      { name: 'picked', execution_mode: 'param_patch', edits: [{ kind: 'constant_replace', name: 'ASPECT_RATIO', new_repr: c.value }], picked_base_commit: baseCommit, picked_base_metric_at_pick_time: 0.992 },
      { name: 'validated', canonical_success: true, val_bpb: c.yCand },
    ]);
  }
  const p = buildOperatorPosterior({ eventsDir: dir });
  const op = p.operators.get('ASPECT_RATIO');
  assert.equal(op.singleEditCount, 3);
  assert.ok(Math.abs(op.mean - 0.002) < 1e-9, `mean ${op.mean}`);
  const sigma = welfordSigma(op);
  assert.ok(sigma != null && Math.abs(sigma - 0.001) < 1e-9, `sigma ${sigma}`);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('multi-edit events touch coverage but not μ; COMPOSITION bucket gets the credit', () => {
  const dir = makeEventsDir();
  const base = 'a'.repeat(40);
  writeDoc(dir, 'm1', [
    {
      name: 'picked', execution_mode: 'param_patch',
      edits: [
        { kind: 'constant_replace', name: 'ASPECT_RATIO', new_repr: '96' },
        { kind: 'constant_replace', name: 'DEPTH', new_repr: '12' },
      ],
      picked_base_commit: base, picked_base_metric_at_pick_time: 0.992,
    },
    { name: 'validated', canonical_success: true, val_bpb: 0.990 },
  ]);
  const p = buildOperatorPosterior({ eventsDir: dir });
  assert.equal(p.totalSingleEdit, 0);
  assert.equal(p.totalMultiEdit, 1);
  const ar = p.operators.get('ASPECT_RATIO');
  assert.equal(ar.count, 1);
  assert.equal(ar.singleEditCount, 0);
  assert.equal(ar.mean, 0); // never updated by multi-edit
  assert.ok(ar.valuesSeen.has('96'));
  assert.equal(p.compositions.size, 1);
  const compKey = [...p.compositions.keys()][0];
  assert.match(compKey, /ASPECT_RATIO\+DEPTH/);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('skips μ/σ update when picked_base_metric_at_pick_time is missing', () => {
  const dir = makeEventsDir();
  writeDoc(dir, 'e1', [
    { name: 'picked', execution_mode: 'param_patch', edits: [{ kind: 'constant_replace', name: 'X', new_repr: '1' }], picked_base_commit: 'a'.repeat(40) /* no baseline metric */ },
    { name: 'validated', canonical_success: true, val_bpb: 0.99 },
  ]);
  const p = buildOperatorPosterior({ eventsDir: dir });
  assert.equal(p.totalSkippedNoBaseline, 1);
  const op = p.operators.get('X');
  assert.equal(op.count, 1);            // coverage updated
  assert.equal(op.singleEditCount, 0);  // μ/σ NOT updated
  fs.rmSync(dir, { recursive: true, force: true });
});

test('coverage markdown lists never-tested axes from registry', () => {
  const dir = makeEventsDir();
  // No events at all
  const p = buildOperatorPosterior({ eventsDir: dir });
  const md = formatCoverageMarkdown(p, [
    { name: 'ASPECT_RATIO', role: 'primary' },
    { name: 'DEPTH', role: 'primary' },
  ]);
  assert.match(md, /ASPECT_RATIO\s*: never tested/);
  assert.match(md, /DEPTH\s*: never tested/);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('extractOperators returns empty for non-param_patch picked events', () => {
  // edits unset
  assert.deepEqual(extractOperators({ name: 'picked' }), []);
  assert.deepEqual(extractOperators({ name: 'picked', edits: [] }), []);
});

test('positive μ means candidate beat baseline (lower-is-better convention)', () => {
  const dir = makeEventsDir();
  writeDoc(dir, 'e1', [
    { name: 'picked', execution_mode: 'param_patch', edits: [{ kind: 'constant_replace', name: 'X', new_repr: '1' }], picked_base_commit: 'a'.repeat(40), picked_base_metric_at_pick_time: 1.0 },
    { name: 'validated', canonical_success: true, val_bpb: 0.5 }, // huge improvement
  ]);
  const p = buildOperatorPosterior({ eventsDir: dir });
  const op = p.operators.get('X');
  assert.ok(op.mean > 0, `expected positive mean, got ${op.mean}`);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('regression produces negative μ', () => {
  const dir = makeEventsDir();
  writeDoc(dir, 'e1', [
    { name: 'picked', execution_mode: 'param_patch', edits: [{ kind: 'constant_replace', name: 'X', new_repr: '1' }], picked_base_commit: 'a'.repeat(40), picked_base_metric_at_pick_time: 0.99 },
    { name: 'validated', canonical_success: true, val_bpb: 1.05 },
  ]);
  const p = buildOperatorPosterior({ eventsDir: dir });
  const op = p.operators.get('X');
  assert.ok(op.mean < 0, `expected negative mean, got ${op.mean}`);
  fs.rmSync(dir, { recursive: true, force: true });
});
