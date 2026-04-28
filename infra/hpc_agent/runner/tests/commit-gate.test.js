import test from 'node:test';
import assert from 'node:assert/strict';
import { shouldCommit, formatGateDecision, DECISION } from '../src/commit-gate.js';

test('AUTO_KEEP when improvement clears threshold and base==head', () => {
  const d = shouldCommit({
    yCandidate: 0.9890,
    yHead: 0.9920,
    sigmaHat: 0.0005,
    sigmaHatSource: 'calibration',
    tRemainingMs: 0,
    tTotalMs: 3600_000,
    candidateBaseCommit: 'a'.repeat(40),
    currentHeadCommit: 'a'.repeat(40),
  });
  assert.equal(d.decision, DECISION.AUTO_KEEP);
  assert.equal(d.reason, 'passes_gate');
  assert.ok(d.improvement > d.threshold);
});

test('WAIT when improvement is below threshold', () => {
  const d = shouldCommit({
    yCandidate: 0.9919,
    yHead: 0.9920,
    sigmaHat: 0.001,
    tRemainingMs: 1800_000,
    tTotalMs: 3600_000,
    candidateBaseCommit: 'a'.repeat(40),
    currentHeadCommit: 'a'.repeat(40),
  });
  assert.equal(d.decision, DECISION.WAIT);
  assert.equal(d.reason, 'below_threshold');
});

test('AUTO_KEEP_REQUIRES_REBASE when candidate base != current head', () => {
  const d = shouldCommit({
    yCandidate: 0.98,
    yHead: 0.99,
    sigmaHat: 0.0005,
    candidateBaseCommit: 'a'.repeat(40),
    currentHeadCommit: 'b'.repeat(40),
  });
  assert.equal(d.decision, DECISION.AUTO_KEEP_REQUIRES_REBASE);
  assert.equal(d.reason, 'stale_baseline');
});

test('AUTO_KEEP_REQUIRES_REBASE when patch conflict simulated', () => {
  const d = shouldCommit({
    yCandidate: 0.98,
    yHead: 0.99,
    sigmaHat: 0.0005,
    candidateBaseCommit: 'a'.repeat(40),
    currentHeadCommit: 'a'.repeat(40),
    patchAppliesCleanly: false,
  });
  assert.equal(d.decision, DECISION.AUTO_KEEP_REQUIRES_REBASE);
  assert.equal(d.reason, 'patch_conflict_on_head');
});

test('DISABLED when yHead is null (HEAD metric unknown)', () => {
  const d = shouldCommit({
    yCandidate: 0.98,
    yHead: null,
    sigmaHat: 0.0005,
  });
  assert.equal(d.decision, DECISION.DISABLED);
  assert.equal(d.reason, 'head_metric_unknown');
});

test('threshold decays over remaining time (tau monotone in t)', () => {
  const common = {
    yCandidate: 0.991,
    yHead: 0.992,
    sigmaHat: 0.0005,
    candidateBaseCommit: 'a'.repeat(40),
    currentHeadCommit: 'a'.repeat(40),
  };
  const early = shouldCommit({ ...common, tRemainingMs: 3600_000, tTotalMs: 3600_000 });
  const late = shouldCommit({ ...common, tRemainingMs: 60_000, tTotalMs: 3600_000 });
  assert.ok(early.threshold > late.threshold, `expected early threshold > late, got ${early.threshold} vs ${late.threshold}`);
  assert.ok(early.tau > late.tau);
});

test('stale-worker term increases threshold', () => {
  const a = shouldCommit({ yCandidate: 0, yHead: 0.01, sigmaHat: 0.0005, numStaleWorkers: 0, candidateBaseCommit: 'a'.repeat(40), currentHeadCommit: 'a'.repeat(40) });
  const b = shouldCommit({ yCandidate: 0, yHead: 0.01, sigmaHat: 0.0005, numStaleWorkers: 5, candidateBaseCommit: 'a'.repeat(40), currentHeadCommit: 'a'.repeat(40) });
  assert.ok(b.threshold > a.threshold);
});

test('sigmaHat null falls back to fallbackSigmaHat with source=fallback', () => {
  const d = shouldCommit({
    yCandidate: 0.98,
    yHead: 0.99,
    sigmaHat: null,
    sigmaHatSource: 'unknown',
    fallbackSigmaHat: 0.001,
    candidateBaseCommit: 'a'.repeat(40),
    currentHeadCommit: 'a'.repeat(40),
  });
  assert.equal(d.sigmaHatSource, 'fallback');
  assert.equal(d.sigmaHatUsed, 0.001);
});

test('formatGateDecision produces a single line with key fields', () => {
  const d = shouldCommit({
    yCandidate: 0.989,
    yHead: 0.992,
    sigmaHat: 0.0005,
    sigmaHatSource: 'calibration',
    candidateBaseCommit: 'a'.repeat(40),
    currentHeadCommit: 'a'.repeat(40),
  });
  const line = formatGateDecision('exp_0042', d);
  assert.match(line, /\[gate\] exp_0042: WOULD_AUTO_KEEP/);
  assert.match(line, /reason=passes_gate/);
  assert.match(line, /source=calibration/);
});

test('candidate metric NaN returns DISABLED', () => {
  const d = shouldCommit({ yCandidate: NaN, yHead: 0.99, sigmaHat: 0.0005 });
  assert.equal(d.decision, DECISION.DISABLED);
  assert.equal(d.reason, 'candidate_metric_invalid');
});
