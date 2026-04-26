import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { buildCycleReport, formatCycleReportLine } from '../src/cycle-report.js';

function makeEventsDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'autoresearch-cycle-report-'));
}

function writeDoc(dir, taskId, events) {
  fs.writeFileSync(
    path.join(dir, `${taskId}.json`),
    JSON.stringify({ task_id: taskId, events }, null, 2),
  );
}

test('buildCycleReport: counts, time buckets, val_bpb best/mean', () => {
  const dir = makeEventsDir();
  const t0 = 1000;
  // task 1: full happy path
  writeDoc(dir, 'exp_a', [
    { name: 'picked', t: t0 + 1 },
    { name: 'granted', t: t0 + 2 },
    { name: 'worker_spawned', t: t0 + 3, attempt: 1 },
    { name: 'worker_returned', t: t0 + 50, attempt: 1, success: true, duration_ms: 47000 },
    { name: 'validated', t: t0 + 51, canonical_success: true, val_bpb: 0.9919, training_seconds: 300, mismatch: false },
    { name: 'released', t: t0 + 52 },
  ]);
  // task 2: LLM said success, canonical disagrees
  writeDoc(dir, 'exp_b', [
    { name: 'picked', t: t0 + 5 },
    { name: 'granted', t: t0 + 6 },
    { name: 'worker_spawned', t: t0 + 7, attempt: 1 },
    { name: 'worker_returned', t: t0 + 30, attempt: 1, success: true },
    { name: 'validated', t: t0 + 31, canonical_success: false, canonical_reason: 'metrics.val_bpb is not finite', mismatch: true, llm_claim_success: true },
    { name: 'released', t: t0 + 32 },
  ]);
  // task 3: walltime blocked, never ran
  writeDoc(dir, 'exp_c', [
    { name: 'picked', t: t0 + 4 },
    { name: 'walltime_blocked', t: t0 + 4, remaining_seconds: 60, required_seconds: 420 },
  ]);

  const report = buildCycleReport({
    eventsDir: dir,
    windowStart: t0,
    windowEnd: t0 + 100,
    gpuCount: 8,
    cycleId: '1',
  });
  assert.equal(report.counts.picked, 3);
  assert.equal(report.counts.granted, 2);
  assert.equal(report.counts.validated_success, 1);
  assert.equal(report.counts.validated_failure, 1);
  assert.equal(report.counts.truth_mismatch, 1);
  assert.equal(report.counts.walltime_blocked, 1);
  assert.equal(report.time_buckets_seconds.training_total, 300);
  assert.equal(report.val_bpb.n, 1);
  assert.equal(report.val_bpb.best, 0.9919);
  assert.equal(report.val_bpb.best_task_id, 'exp_a');
  // worker_llm_total = (50-3)+(30-7) = 47+23 = 70
  assert.equal(report.time_buckets_seconds.worker_llm_total, 70);
  // duty cycle = 300 / (100s × 8 GPU) = 0.375
  assert.equal(report.gpu_duty_cycle, 0.375);

  fs.rmSync(dir, { recursive: true, force: true });
});

test('buildCycleReport: ignores tasks picked outside the window', () => {
  const dir = makeEventsDir();
  writeDoc(dir, 'old', [{ name: 'picked', t: 100 }]);
  writeDoc(dir, 'new', [{ name: 'picked', t: 5000 }]);
  const report = buildCycleReport({
    eventsDir: dir,
    windowStart: 4000,
    windowEnd: 6000,
    gpuCount: 8,
  });
  assert.equal(report.counts.picked, 1);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('buildCycleReport: handles empty events dir', () => {
  const dir = makeEventsDir();
  const report = buildCycleReport({
    eventsDir: dir,
    windowStart: 0,
    windowEnd: 100,
    gpuCount: 8,
  });
  assert.equal(report.counts.picked, 0);
  assert.equal(report.gpu_duty_cycle, 0);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('buildCycleReport: handles missing/corrupt event files gracefully', () => {
  const dir = makeEventsDir();
  fs.writeFileSync(path.join(dir, 'broken.json'), 'not json');
  fs.writeFileSync(path.join(dir, 'half.json'), '{}');
  writeDoc(dir, 'good', [{ name: 'picked', t: 50 }]);
  const report = buildCycleReport({
    eventsDir: dir,
    windowStart: 0,
    windowEnd: 100,
    gpuCount: 8,
  });
  assert.equal(report.counts.picked, 1);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('formatCycleReportLine has compact one-line summary', () => {
  const r = buildCycleReport({
    eventsDir: '/dev/null/nonexistent',
    windowStart: 0,
    windowEnd: 100,
    gpuCount: 8,
  });
  const line = formatCycleReportLine(r);
  assert.match(line, /Cycle report/);
  assert.match(line, /gpu_duty/);
});
