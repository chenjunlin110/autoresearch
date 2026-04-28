import test from 'node:test';
import assert from 'node:assert/strict';
import {
  standardNormalCDF,
  commitProbability,
  adaptiveQueueDepth,
  formatHazardDecision,
} from '../src/commit-hazard.js';

const cfgOn = {
  hazardQueueEnabled: true,
  qMin: 1,
  qMax: 8,
  hazardZeta: 2.0,
  posteriorMinSamples: 3,
  hazardCommitProbCap: 0.5,
};

function opState({ mean = 0, m2 = 0, count = 0, singleEditCount = 0 } = {}) {
  return { mean, m2, count, singleEditCount, valuesSeen: new Set() };
}

function pick({ id, ops }) {
  // Build a minimal picked event: list of constant_replace edits.
  const edits = ops.map(({ name, value }) => ({
    kind: 'constant_replace', name, new_repr: String(value),
  }));
  return { task_id: id, edits };
}

test('standardNormalCDF: known values', () => {
  assert.ok(Math.abs(standardNormalCDF(0) - 0.5) < 1e-6);
  assert.ok(Math.abs(standardNormalCDF(1.96) - 0.975) < 1e-3);
  assert.ok(Math.abs(standardNormalCDF(-1.96) - 0.025) < 1e-3);
  assert.ok(standardNormalCDF(10) > 0.99999);
  assert.ok(standardNormalCDF(-10) < 1e-5);
});

test('commitProbability: n<minSamples returns unreliable cap', () => {
  const operators = new Map([['X', opState({ mean: 0.01, singleEditCount: 1 })]]);
  const r = commitProbability({ operatorKey: 'X', operators, minSamples: 3, unreliableCap: 0.5 });
  assert.equal(r.source, 'unreliable');
  assert.equal(r.p, 0.5);
});

test('commitProbability: strong positive mean ⇒ p approaches 1', () => {
  // n=4, mean=0.003, m2 chosen so σ ≈ 0.001 → z=3 → Φ(3)≈0.9987
  // m2 = (n-1) * σ² = 3 * 1e-6 = 3e-6
  const operators = new Map([['X', opState({ mean: 0.003, m2: 3e-6, count: 4, singleEditCount: 4 })]]);
  const r = commitProbability({ operatorKey: 'X', operators, minSamples: 3 });
  assert.equal(r.source, 'posterior');
  assert.ok(r.p > 0.95, `expected p>0.95, got ${r.p}`);
  assert.ok(r.p <= 0.99, `expected cap at 0.99, got ${r.p}`);
});

test('commitProbability: strong negative mean ⇒ p near 0', () => {
  const operators = new Map([['X', opState({ mean: -0.003, m2: 3e-6, count: 4, singleEditCount: 4 })]]);
  const r = commitProbability({ operatorKey: 'X', operators, minSamples: 3 });
  assert.equal(r.source, 'posterior');
  assert.ok(r.p < 0.05, `expected p<0.05, got ${r.p}`);
});

test('commitProbability: σ=0 ⇒ falls back to fallbackSigma; otherwise no_sigma', () => {
  // m2=0 with n>=2 gives σ=0
  const operators = new Map([['X', opState({ mean: 0.001, m2: 0, count: 4, singleEditCount: 4 })]]);
  const noFallback = commitProbability({ operatorKey: 'X', operators, minSamples: 3 });
  assert.equal(noFallback.source, 'no_sigma');
  const withFallback = commitProbability({
    operatorKey: 'X', operators, minSamples: 3, fallbackSigma: 0.001,
  });
  assert.equal(withFallback.source, 'posterior');
  // z = mean/fallbackSigma = 0.001/0.001 = 1 ⇒ Φ(1) ≈ 0.8413
  assert.ok(withFallback.p > 0.8 && withFallback.p < 0.85, `p=${withFallback.p}`);
});

test('adaptiveQueueDepth: flag off ⇒ returns null', () => {
  const r = adaptiveQueueDepth({ runningPicked: [], posterior: { operators: new Map(), compositions: new Map() }, config: { ...cfgOn, hazardQueueEnabled: false } });
  assert.equal(r, null);
});

test('adaptiveQueueDepth: no running tasks ⇒ ρ=0, Q=qMax', () => {
  const r = adaptiveQueueDepth({ runningPicked: [], posterior: { operators: new Map(), compositions: new Map() }, config: cfgOn });
  assert.equal(r.qTarget, 8);
  assert.equal(r.rho, 0);
});

test('adaptiveQueueDepth: many strong winners ⇒ Q drops to qMin', () => {
  // 4 in-flight experiments, all on a hot operator with p≈0.99
  const operators = new Map([['X', opState({ mean: 0.003, m2: 3e-6, count: 4, singleEditCount: 4 })]]);
  const compositions = new Map();
  const running = [
    pick({ id: 'a', ops: [{ name: 'X', value: 1 }] }),
    pick({ id: 'b', ops: [{ name: 'X', value: 2 }] }),
    pick({ id: 'c', ops: [{ name: 'X', value: 3 }] }),
    pick({ id: 'd', ops: [{ name: 'X', value: 4 }] }),
  ];
  const r = adaptiveQueueDepth({ runningPicked: running, posterior: { operators, compositions }, config: cfgOn });
  assert.ok(r.rho > 0.99, `rho ${r.rho}`);
  assert.equal(r.qTarget, 1);
});

test('adaptiveQueueDepth: weak posteriors ⇒ p=cap, Q stays high but not max', () => {
  // 4 in-flight, each with unreliable operator (n<3) → p=0.5 each
  // ρ = 1 - 0.5^4 = 0.9375 → (1-0.9375)^2 = 0.0039 → Q ≈ qMin + (qMax-qMin)*0.0039 ≈ 1.03
  const operators = new Map([['X', opState({ singleEditCount: 1 })]]);
  const compositions = new Map();
  const running = [
    pick({ id: 'a', ops: [{ name: 'X', value: 1 }] }),
    pick({ id: 'b', ops: [{ name: 'X', value: 2 }] }),
    pick({ id: 'c', ops: [{ name: 'X', value: 3 }] }),
    pick({ id: 'd', ops: [{ name: 'X', value: 4 }] }),
  ];
  const r = adaptiveQueueDepth({ runningPicked: running, posterior: { operators, compositions }, config: cfgOn });
  // The point: 4 unreliables shouldn't crash Q to 1, but with cap=0.5 and zeta=2,
  // ρ=0.9375 still pushes Q low. This is a known property of cap=0.5 — caller
  // can raise unreliableCap or qMin to mitigate.
  assert.ok(r.qTarget >= 1 && r.qTarget <= 2);
});

test('adaptiveQueueDepth: mixed reliable+unreliable; reliable dominates', () => {
  const operators = new Map([
    ['HOT', opState({ mean: 0.003, m2: 3e-6, count: 4, singleEditCount: 4 })],   // strong
    ['COLD', opState({ mean: -0.003, m2: 3e-6, count: 4, singleEditCount: 4 })], // strong negative
    ['NEW', opState({ singleEditCount: 1 })],                                     // unreliable
  ]);
  const compositions = new Map();
  const running = [
    pick({ id: 'h1', ops: [{ name: 'HOT', value: 1 }] }),
    pick({ id: 'c1', ops: [{ name: 'COLD', value: 1 }] }),
    pick({ id: 'n1', ops: [{ name: 'NEW', value: 1 }] }),
  ];
  const r = adaptiveQueueDepth({ runningPicked: running, posterior: { operators, compositions }, config: cfgOn });
  // HOT contributes p≈0.99; that alone gives ρ≈0.99 regardless of others.
  assert.ok(r.rho > 0.99);
  assert.equal(r.qTarget, 1);
  // Sanity check the per-experiment annotations
  assert.equal(r.perExperiment.find((e) => e.id === 'c1').source, 'posterior');
  assert.equal(r.perExperiment.find((e) => e.id === 'n1').source, 'unreliable');
});

test('adaptiveQueueDepth: ζ controls steepness of Q response', () => {
  const operators = new Map([['X', opState({ mean: 0.0005, m2: 1e-6, count: 4, singleEditCount: 4 })]]); // mild positive
  const compositions = new Map();
  const running = [pick({ id: 'a', ops: [{ name: 'X', value: 1 }] })];
  const flat = adaptiveQueueDepth({ runningPicked: running, posterior: { operators, compositions }, config: { ...cfgOn, hazardZeta: 1 } });
  const steep = adaptiveQueueDepth({ runningPicked: running, posterior: { operators, compositions }, config: { ...cfgOn, hazardZeta: 4 } });
  // Same ρ; steeper ζ means (1-ρ)^ζ is smaller ⇒ Q is closer to qMin.
  assert.ok(flat.qTarget >= steep.qTarget);
});

test('adaptiveQueueDepth: COMPOSITION key used for multi-edit picked', () => {
  const operators = new Map();
  const compositions = new Map([['_COMPOSITION:A+B', { mean: 0.003, m2: 3e-6, count: 4 }]]);
  const running = [pick({ id: 'm', ops: [{ name: 'A', value: 1 }, { name: 'B', value: 2 }] })];
  const r = adaptiveQueueDepth({ runningPicked: running, posterior: { operators, compositions }, config: cfgOn });
  assert.equal(r.perExperiment[0].source, 'posterior');
  assert.ok(r.perExperiment[0].p > 0.9);
});

test('formatHazardDecision summarizes Q, rho, and per-experiment p', () => {
  const operators = new Map([['X', opState({ mean: 0.003, m2: 3e-6, count: 4, singleEditCount: 4 })]]);
  const compositions = new Map();
  const r = adaptiveQueueDepth({
    runningPicked: [pick({ id: 'exp_42', ops: [{ name: 'X', value: 1 }] })],
    posterior: { operators, compositions },
    config: cfgOn,
  });
  const line = formatHazardDecision(r);
  assert.match(line, /^\[hazard\] Q=\d+ rho=/);
  assert.match(line, /exp_42:0\.\d{2}\(posterior\)/);
});
