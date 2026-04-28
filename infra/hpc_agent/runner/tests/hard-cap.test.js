/**
 * Phase 2 verification: a deliberately hung wrapper must be killed by the
 * direct executor's hard-cap and its process group must come down with it.
 * We launch a wrapper that backgrounds a child sleeping forever, then
 * confirm both bash and the child are gone after the cap fires.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { runDirectWorker } from '../src/direct-executor.js';

function pidAlive(pid) {
  try { process.kill(pid, 0); return true; } catch { return false; }
}

test('runDirectWorker: hard cap takes down a hung wrapper and its descendant', async () => {
  const sandbox = fs.mkdtempSync(path.join(os.tmpdir(), 'hardcap-sb-'));
  const outputDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hardcap-out-'));
  const wrapper = path.join(sandbox, 'wrapper.sh');
  const childPidFile = path.join(outputDir, 'child.pid');
  // Wrapper backgrounds a long sleep, writes its pid, then waits on it
  // so SIGTERM to bash needs to flow through the process group to the
  // descendant. If the cap doesn't kill the pgroup, the descendant
  // outlives this test.
  fs.writeFileSync(wrapper, [
    '#!/usr/bin/env bash',
    'sleep 600 &',
    'child=$!',
    `printf '%s' "$child" > "${childPidFile}"`,
    'wait $child',
    '',
  ].join('\n'));
  fs.chmodSync(wrapper, 0o755);

  const start = Date.now();
  const verdict = await runDirectWorker({
    sandboxDir: sandbox,
    wrapperScript: wrapper,
    outputDir,
    hardCapMs: 1500,
  });
  const elapsed = Date.now() - start;

  assert.equal(verdict.ok, false);
  assert.ok(elapsed >= 1500 && elapsed < 8000,
    `expected hard cap ~1.5s, got ${elapsed}ms`);

  // Confirm the child process is no longer alive.
  const childPid = Number.parseInt(fs.readFileSync(childPidFile, 'utf-8').trim(), 10);
  assert.ok(Number.isInteger(childPid));
  // SIGKILL should have escalated within 30s; give the kernel a beat.
  await new Promise((resolve) => setTimeout(resolve, 200));
  assert.equal(pidAlive(childPid), false,
    `child pid ${childPid} should be dead after hard cap`);

  fs.rmSync(sandbox, { recursive: true, force: true });
  fs.rmSync(outputDir, { recursive: true, force: true });
});

test('runDirectWorker: abortSignal terminates the wrapper before hard cap', async () => {
  const sandbox = fs.mkdtempSync(path.join(os.tmpdir(), 'abort-sb-'));
  const outputDir = fs.mkdtempSync(path.join(os.tmpdir(), 'abort-out-'));
  const wrapper = path.join(sandbox, 'wrapper.sh');
  fs.writeFileSync(wrapper, '#!/usr/bin/env bash\nsleep 600\n');
  fs.chmodSync(wrapper, 0o755);

  const ac = new AbortController();
  setTimeout(() => ac.abort(), 500);
  const start = Date.now();
  const verdict = await runDirectWorker({
    sandboxDir: sandbox,
    wrapperScript: wrapper,
    outputDir,
    abortSignal: ac.signal,
    hardCapMs: 60000,
  });
  const elapsed = Date.now() - start;
  assert.equal(verdict.ok, false);
  assert.ok(elapsed < 8000, `expected fast abort, got ${elapsed}ms`);
  fs.rmSync(sandbox, { recursive: true, force: true });
  fs.rmSync(outputDir, { recursive: true, force: true });
});
