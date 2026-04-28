import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import Database from 'better-sqlite3';
import {
  createTokenRequest,
  ensureResourceOrchestrationTables,
  grantTokenRequest,
  upsertAllocationLease,
} from '../src/resource-orchestration.js';
import {
  buildSharedSlurmProbeCommand,
  buildSharedSlurmStepCommand,
  resolveSharedSlurmStepInput,
} from '../src/hpc-tool.js';
import { parseGpuTokenOrdinals } from '../src/orchestration-utils.js';

function createTempProjectDb() {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hpc-agent-hpc-tool-test-'));
  const dbPath = path.join(tempDir, 'project.db');
  const db = new Database(dbPath);
  db.pragma('journal_mode = WAL');
  db.pragma('foreign_keys = ON');
  return {
    tempDir,
    dbPath,
    db,
    cleanup() {
      try { db.close(); } catch {}
      fs.rmSync(tempDir, { recursive: true, force: true });
    },
  };
}

test('parseGpuTokenOrdinals accepts gpu token names and raw ordinals', () => {
  assert.deepEqual(parseGpuTokenOrdinals(['gpu0', 'gpu3', '3']), [0, 3]);
  assert.throws(() => parseGpuTokenOrdinals(['abc']), /invalid GPU token/);
});

test('resolveSharedSlurmStepInput derives job and GPUs from an active grant', () => {
  const fixture = createTempProjectDb();
  try {
    ensureResourceOrchestrationTables(fixture.db, { gpuCount: 8 });
    upsertAllocationLease(fixture.db, {
      owner: 'manager',
      jobId: '1575847',
      nodeName: 'fs-mbz-gpu-275',
    });
    const request = createTokenRequest(fixture.db, {
      workerName: 'worker-a',
      requestedCount: 2,
      rationale: 'train autoresearch',
    });
    const grant = grantTokenRequest(fixture.db, {
      requestId: request.id,
      managerName: 'manager',
      tokenNames: ['gpu2', 'gpu5'],
    });

    const spec = resolveSharedSlurmStepInput({
      mode: 'shared_step',
      grant_id: grant.id,
      command: 'uv run train.py',
      env: {
        AUTORESEARCH_DISABLE_COMPILE: '1',
      },
    }, {
      project_db_path: fixture.dbPath,
      agent_name: 'worker-a',
      shared_working_directory: '/workspace/autoresearch',
    });

    assert.equal(spec.jobId, '1575847');
    assert.equal(spec.command, 'uv run train.py');
    assert.deepEqual(spec.gpuTokens, ['gpu2', 'gpu5']);
    assert.deepEqual(spec.gpuOrdinals, [2, 5]);
    assert.equal(spec.env.CUDA_VISIBLE_DEVICES, '2,5');
    assert.equal(spec.env.OMP_NUM_THREADS, '1');
    assert.equal(spec.env.AUTORESEARCH_DISABLE_COMPILE, '1');
    assert.equal(spec.workingDirectory, '/workspace/autoresearch');
    assert.equal(spec.nodeName, 'fs-mbz-gpu-275');
  } finally {
    fixture.cleanup();
  }
});

test('resolveSharedSlurmStepInput rejects grants owned by a different worker', () => {
  const fixture = createTempProjectDb();
  try {
    ensureResourceOrchestrationTables(fixture.db, { gpuCount: 2 });
    upsertAllocationLease(fixture.db, {
      owner: 'manager',
      jobId: '1575847',
      nodeName: 'node-a',
    });
    const request = createTokenRequest(fixture.db, {
      workerName: 'worker-a',
      requestedCount: 1,
      rationale: 'exclusive worker run',
    });
    const grant = grantTokenRequest(fixture.db, {
      requestId: request.id,
      managerName: 'manager',
      tokenNames: ['gpu1'],
    });

    assert.throws(() => resolveSharedSlurmStepInput({
      mode: 'shared_step',
      grant_id: grant.id,
      command: 'python train.py',
    }, {
      project_db_path: fixture.dbPath,
      agent_name: 'worker-b',
    }), /belongs to worker-a, not worker-b/);
  } finally {
    fixture.cleanup();
  }
});

test('buildSharedSlurmStepCommand uses overlap and manual CUDA_VISIBLE_DEVICES', () => {
  const invocation = buildSharedSlurmStepCommand({
    jobId: '1575847',
    cpusPerTask: 1,
    env: {
      AUTORESEARCH_DISABLE_COMPILE: '1',
      CUDA_VISIBLE_DEVICES: '0,7',
      OMP_NUM_THREADS: '1',
    },
    workingDirectory: '/workspace/autoresearch',
    command: 'uv run train.py',
  });

  assert.equal(invocation.command, 'srun');
  assert.deepEqual(invocation.args.slice(0, 6), [
    '--jobid', '1575847',
    '--overlap',
    '--ntasks=1',
    '--cpus-per-task', '1',
  ]);
  assert.match(invocation.script, /cd '\/workspace\/autoresearch'/);
  assert.match(invocation.script, /export CUDA_VISIBLE_DEVICES='0,7'/);
  assert.match(invocation.script, /export AUTORESEARCH_DISABLE_COMPILE='1'/);
  assert.match(invocation.script, /uv run train\.py$/);
});

test('buildSharedSlurmProbeCommand creates a lightweight overlap probe', () => {
  const invocation = buildSharedSlurmProbeCommand('1575847');
  assert.equal(invocation.command, 'srun');
  assert.deepEqual(invocation.args.slice(0, 6), [
    '--jobid', '1575847',
    '--overlap',
    '--ntasks=1',
    '--cpus-per-task', '1',
  ]);
  assert.match(invocation.script, /torch_cuda_device_count/);
  assert.match(invocation.script, /SLURM_STEP_GPUS/);
});
