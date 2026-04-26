import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { appendTaskEvent, readTaskEvents, taskEventsPath } from '../src/task-events.js';

function makeTmpDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'autoresearch-events-'));
}

test('appendTaskEvent creates the events file on first write', () => {
  const dir = makeTmpDir();
  appendTaskEvent(dir, 'exp_0001_baseline', 'picked', { worker: 'exp_runner_3' });
  const doc = readTaskEvents(dir, 'exp_0001_baseline');
  assert.equal(doc.task_id, 'exp_0001_baseline');
  assert.equal(doc.events.length, 1);
  assert.equal(doc.events[0].name, 'picked');
  assert.equal(doc.events[0].worker, 'exp_runner_3');
  assert.equal(typeof doc.events[0].t, 'number');
  fs.rmSync(dir, { recursive: true, force: true });
});

test('appendTaskEvent appends without rewriting earlier events', () => {
  const dir = makeTmpDir();
  appendTaskEvent(dir, 'exp_0002', 'picked');
  appendTaskEvent(dir, 'exp_0002', 'granted', { grant_id: 17 });
  appendTaskEvent(dir, 'exp_0002', 'released', { grant_id: 17 });
  const doc = readTaskEvents(dir, 'exp_0002');
  assert.deepEqual(doc.events.map((e) => e.name), ['picked', 'granted', 'released']);
  assert.equal(doc.events[1].grant_id, 17);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('appendTaskEvent sanitizes filesystem-hostile task ids', () => {
  const dir = makeTmpDir();
  const dirty = '../../../etc/passwd';
  appendTaskEvent(dir, dirty, 'picked');
  const expected = taskEventsPath(dir, dirty);
  assert.ok(fs.existsSync(expected));
  // and the resolved path stays inside `dir`
  assert.ok(path.resolve(expected).startsWith(path.resolve(dir)));
  fs.rmSync(dir, { recursive: true, force: true });
});

test('appendTaskEvent silently no-ops when given invalid args', () => {
  // Should not throw — tracing must never break the orchestrator.
  appendTaskEvent('', 'x', 'y');
  appendTaskEvent('/dev/null/cant-write-here', 'x', 'y');
  appendTaskEvent('/tmp', '', 'y');
  appendTaskEvent('/tmp', 'x', '');
});

test('appendTaskEvent recovers from a corrupt prior file', () => {
  const dir = makeTmpDir();
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(taskEventsPath(dir, 'exp_3'), 'not json {{');
  appendTaskEvent(dir, 'exp_3', 'picked');
  const doc = readTaskEvents(dir, 'exp_3');
  assert.equal(doc.events.length, 1);
  assert.equal(doc.events[0].name, 'picked');
  fs.rmSync(dir, { recursive: true, force: true });
});
