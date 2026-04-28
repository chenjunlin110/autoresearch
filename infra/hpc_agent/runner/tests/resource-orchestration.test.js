import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import Database from 'better-sqlite3';
import {
  createTokenRequest,
  ensureResourceOrchestrationTables,
  getResourceSummary,
  grantTokenRequest,
  listResourceTokens,
  releaseAllocationLease,
  releaseTokenGrant,
  upsertAllocationLease,
} from '../src/resource-orchestration.js';

function createTempProjectDb() {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hpc-agent-resource-test-'));
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

test('resource schema bootstrap seeds GPU tokens idempotently', () => {
  const fixture = createTempProjectDb();
  try {
    ensureResourceOrchestrationTables(fixture.db, { gpuCount: 4 });
    ensureResourceOrchestrationTables(fixture.db, { gpuCount: 4 });

    const tokens = listResourceTokens(fixture.db);
    assert.equal(tokens.length, 4);
    assert.deepEqual(tokens.map(token => token.tokenName), ['gpu0', 'gpu1', 'gpu2', 'gpu3']);
    assert.ok(tokens.every(token => token.available));
  } finally {
    fixture.cleanup();
  }
});

test('allocation lease and token lifecycle updates availability correctly', () => {
  const fixture = createTempProjectDb();
  try {
    ensureResourceOrchestrationTables(fixture.db, { gpuCount: 4 });
    const lease = upsertAllocationLease(fixture.db, {
      owner: 'manager',
      jobId: 'job-123',
      nodeName: 'node-a',
      metadata: { partition: 'debug' },
    });
    assert.equal(lease.jobId, 'job-123');
    assert.equal(lease.nodeName, 'node-a');

    const request = createTokenRequest(fixture.db, {
      workerName: 'worker-a',
      requestedCount: 2,
      rationale: 'train candidate',
      metadata: { trial: 7 },
    });
    assert.equal(request.status, 'pending');

    const grant = grantTokenRequest(fixture.db, {
      requestId: request.id,
      managerName: 'manager',
      tokenNames: ['gpu0', 'gpu1'],
    }, { gpuCount: 4 });
    assert.equal(grant.status, 'active');
    assert.deepEqual(grant.tokenNames, ['gpu0', 'gpu1']);
    assert.equal(grant.leaseJobId, 'job-123');

    const afterGrant = listResourceTokens(fixture.db);
    assert.equal(afterGrant.filter(token => token.available).length, 2);
    assert.equal(afterGrant.find(token => token.tokenName === 'gpu0').available, false);

    const releasedGrant = releaseTokenGrant(fixture.db, {
      grantId: grant.id,
      actor: 'manager',
      note: 'task finished',
    });
    assert.equal(releasedGrant.status, 'released');

    const afterRelease = listResourceTokens(fixture.db);
    assert.ok(afterRelease.every(token => token.available));

    assert.throws(() => releaseTokenGrant(fixture.db, {
      grantId: grant.id,
      actor: 'manager',
    }), /is not active/);
  } finally {
    fixture.cleanup();
  }
});

test('resource summary reflects active lease, grants, and released lease state', () => {
  const fixture = createTempProjectDb();
  try {
    ensureResourceOrchestrationTables(fixture.db, { gpuCount: 3, tokenPrefix: 'gpu' });
    upsertAllocationLease(fixture.db, {
      owner: 'manager',
      jobId: 'job-321',
      nodeName: 'node-b',
    });
    const request = createTokenRequest(fixture.db, {
      workerName: 'worker-b',
      requestedCount: 1,
      rationale: 'eval run',
    });
    grantTokenRequest(fixture.db, {
      requestId: request.id,
      managerName: 'manager',
      tokenNames: ['gpu2'],
    }, { gpuCount: 3 });

    const activeSummary = getResourceSummary(fixture.db, { gpuCount: 3 });
    assert.equal(activeSummary.allocation.jobId, 'job-321');
    assert.equal(activeSummary.tokens.total, 3);
    assert.equal(activeSummary.tokens.granted, 1);
    assert.equal(activeSummary.tokens.available, 2);
    assert.equal(activeSummary.requests.pending, 0);
    assert.equal(activeSummary.requests.granted, 1);
    assert.equal(activeSummary.grants.active, 1);

    releaseAllocationLease(fixture.db, { actor: 'manager', note: 'allocation ended' });
    const releasedSummary = getResourceSummary(fixture.db, { gpuCount: 3 });
    assert.equal(releasedSummary.allocation, null);
    assert.equal(releasedSummary.grants.active, 1);
  } finally {
    fixture.cleanup();
  }
});
