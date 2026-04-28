import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { buildExperimentLedger, computeEditConfidence, formatLedgerMarkdown } from '../src/experiment-ledger.js';

function mkEvents(entries) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ledger-events-'));
  for (const [taskId, events] of entries) {
    fs.writeFileSync(path.join(dir, `${taskId}.json`), JSON.stringify({ task_id: taskId, events }));
  }
  return dir;
}

test('buildExperimentLedger: ranks completed by metric and surfaces running', () => {
  const dir = mkEvents([
    ['exp_a', [
      { name: 'picked', t: 100 },
      { name: 'validated', canonical_success: true, val_bpb: 1.05, training_seconds: 200 },
      { name: 'released', t: 200 },
    ]],
    ['exp_b', [
      { name: 'picked', t: 150 },
      { name: 'validated', canonical_success: true, val_bpb: 0.98, training_seconds: 220 },
      { name: 'released', t: 250 },
    ]],
    ['exp_running', [{ name: 'picked', t: 300 }]],
  ]);
  const ledger = buildExperimentLedger({ eventsDir: dir });
  assert.equal(ledger.topK.length, 2);
  assert.equal(ledger.topK[0].id, 'exp_b'); // 0.98 < 1.05
  assert.equal(ledger.topK[1].id, 'exp_a');
  assert.equal(ledger.running.length, 1);
  assert.equal(ledger.running[0].id, 'exp_running');
  assert.equal(ledger.totalCompleted, 2);
  assert.equal(ledger.totalFailed, 0);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('buildExperimentLedger: clusters failures by reason prefix', () => {
  const dir = mkEvents([
    ['exp_oom_1', [
      { name: 'picked', t: 100 },
      { name: 'validated', canonical_success: false, reason: 'OOM at compile time, batch=2**20' },
      { name: 'released', t: 110 },
    ]],
    ['exp_oom_2', [
      { name: 'picked', t: 200 },
      { name: 'validated', canonical_success: false, reason: 'OOM at compile time, batch=2**20' },
      { name: 'released', t: 210 },
    ]],
    ['exp_compile', [
      { name: 'picked', t: 300 },
      { name: 'validated', canonical_success: false, reason: 'compile error in attention path' },
      { name: 'released', t: 310 },
    ]],
  ]);
  const ledger = buildExperimentLedger({ eventsDir: dir });
  assert.equal(ledger.totalFailed, 3);
  assert.equal(ledger.failedClusters.length, 2);
  assert.equal(ledger.failedClusters[0].count, 2); // OOM cluster wins
  assert.match(ledger.failedClusters[0].reason, /OOM/);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('formatLedgerMarkdown: surfaces edit_summary next to hypothesis', () => {
  const dir = mkEvents([
    ['exp_aspect', [
      { name: 'picked', t: 100,
        rationale: 'wider aspect ratio',
        edit_summary: 'ASPECT_RATIO 64→96',
        base_ref: 'HEAD' },
      { name: 'validated', canonical_success: true, val_bpb: 0.99, training_seconds: 300 },
      { name: 'released', t: 400 },
    ]],
  ]);
  const ledger = buildExperimentLedger({ eventsDir: dir });
  const text = formatLedgerMarkdown(ledger);
  assert.match(text, /edit: ASPECT_RATIO 64→96/);
  assert.match(text, /hypothesis: wider aspect ratio/);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('formatLedgerMarkdown: surfaces rationale next to outcome', () => {
  const dir = mkEvents([
    ['exp_aspect_96', [
      { name: 'picked', t: 100,
        rationale: 'wider aspect ratio matches FFN at this depth',
        task_summary: 'sweep aspect=96',
        base_ref: 'HEAD' },
      { name: 'validated', canonical_success: true, val_bpb: 0.99, training_seconds: 300 },
      { name: 'released', t: 400 },
    ]],
  ]);
  const ledger = buildExperimentLedger({ eventsDir: dir });
  const text = formatLedgerMarkdown(ledger);
  assert.match(text, /exp_aspect_96/);
  assert.match(text, /hypothesis: wider aspect ratio matches FFN/);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('formatLedgerMarkdown: emits headers and entries when ledger non-empty', () => {
  const dir = mkEvents([
    ['exp_top', [
      { name: 'picked', t: 100 },
      { name: 'validated', canonical_success: true, val_bpb: 0.9, training_seconds: 300 },
      { name: 'released', t: 400 },
    ]],
  ]);
  const ledger = buildExperimentLedger({ eventsDir: dir });
  const text = formatLedgerMarkdown(ledger);
  assert.match(text, /Top-1 by val_bpb/);
  assert.match(text, /exp_top: val_bpb=0\.9000/);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('formatLedgerMarkdown: empty ledger returns empty string', () => {
  const dir = mkEvents([]);
  const ledger = buildExperimentLedger({ eventsDir: dir });
  assert.equal(formatLedgerMarkdown(ledger), '');
  fs.rmSync(dir, { recursive: true, force: true });
});

test('computeEditConfidence: warns when direct-executor recent success drops below 70%', () => {
  const dir = mkEvents([
    ['exp_p1', [
      { name: 'picked', t: 100 },
      { name: 'worker_spawned', worker: 'direct-executor' },
      { name: 'validated', canonical_success: false, reason: 'no module-level assignment' },
    ]],
    ['exp_p2', [
      { name: 'picked', t: 200 },
      { name: 'worker_spawned', worker: 'direct-executor' },
      { name: 'validated', canonical_success: false, reason: 'expected_old_repr mismatch' },
    ]],
    ['exp_p3', [
      { name: 'picked', t: 300 },
      { name: 'worker_spawned', worker: 'direct-executor' },
      { name: 'validated', canonical_success: true, val_bpb: 1.0 },
    ]],
  ]);
  const conf = computeEditConfidence({ eventsDir: dir });
  assert.ok(conf, 'expected confidence record');
  assert.equal(conf.total, 3);
  assert.equal(conf.succeeded, 1);
  assert.ok(Math.abs(conf.rate - 1 / 3) < 1e-9);
  assert.equal(conf.shouldWarn, true);
  assert.equal(conf.recentFailureIds.length, 2);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('computeEditConfidence: skips when no direct-executor tasks ran', () => {
  const dir = mkEvents([
    ['exp_a', [
      { name: 'picked', t: 100 },
      { name: 'worker_spawned', worker: 'alice' },
      { name: 'validated', canonical_success: true, val_bpb: 0.95 },
    ]],
  ]);
  const conf = computeEditConfidence({ eventsDir: dir });
  assert.equal(conf, null);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('buildExperimentLedger: missing eventsDir does not throw', () => {
  const ledger = buildExperimentLedger({ eventsDir: '/no/such/dir' });
  assert.deepEqual(ledger.topK, []);
  assert.equal(ledger.totalCompleted, 0);
});
