import test from 'node:test';
import assert from 'node:assert/strict';
import {
  _resetWalltimeCacheForTests,
  getSlurmDeadlineMs,
  remainingWalltimeMs,
} from '../src/slurm-walltime.js';

function withEnv(overrides, fn) {
  const restore = {};
  for (const k of Object.keys(overrides)) {
    restore[k] = process.env[k];
    if (overrides[k] === undefined) delete process.env[k];
    else process.env[k] = overrides[k];
  }
  try { return fn(); }
  finally {
    for (const k of Object.keys(restore)) {
      if (restore[k] === undefined) delete process.env[k];
      else process.env[k] = restore[k];
    }
  }
}

test('AUTORESEARCH_SLURM_END_TIME accepts ISO 8601', () => {
  _resetWalltimeCacheForTests();
  const future = new Date(Date.now() + 7200_000).toISOString();
  withEnv({ AUTORESEARCH_SLURM_END_TIME: future, SLURM_JOB_ID: undefined }, () => {
    const ms = getSlurmDeadlineMs();
    assert.equal(ms, Date.parse(future));
    const remaining = remainingWalltimeMs();
    assert.ok(remaining > 7100_000 && remaining <= 7200_000);
  });
});

test('AUTORESEARCH_SLURM_END_TIME accepts epoch seconds', () => {
  _resetWalltimeCacheForTests();
  const epochS = Math.floor((Date.now() + 1800_000) / 1000);
  withEnv({ AUTORESEARCH_SLURM_END_TIME: String(epochS), SLURM_JOB_ID: undefined }, () => {
    const ms = getSlurmDeadlineMs();
    assert.equal(ms, epochS * 1000);
  });
});

test('no SLURM env yields null remaining (no gate)', () => {
  _resetWalltimeCacheForTests();
  withEnv({ AUTORESEARCH_SLURM_END_TIME: undefined, SLURM_JOB_ID: undefined }, () => {
    assert.equal(getSlurmDeadlineMs(), 0);
    assert.equal(remainingWalltimeMs(), null);
  });
});

test('"Unknown" or junk falls back to 0 (no gate)', () => {
  _resetWalltimeCacheForTests();
  withEnv({ AUTORESEARCH_SLURM_END_TIME: 'Unknown', SLURM_JOB_ID: undefined }, () => {
    assert.equal(getSlurmDeadlineMs(), 0);
    assert.equal(remainingWalltimeMs(), null);
  });
  _resetWalltimeCacheForTests();
  withEnv({ AUTORESEARCH_SLURM_END_TIME: 'not a date', SLURM_JOB_ID: undefined }, () => {
    assert.equal(getSlurmDeadlineMs(), 0);
  });
});

test('expired allocation reports zero remaining, not negative', () => {
  _resetWalltimeCacheForTests();
  const past = new Date(Date.now() - 1000).toISOString();
  withEnv({ AUTORESEARCH_SLURM_END_TIME: past, SLURM_JOB_ID: undefined }, () => {
    assert.equal(remainingWalltimeMs(), 0);
  });
});
