/**
 * @fileoverview ALPS Phase 2 — σ̂ estimator from task events.
 *
 * Two real sources of variance, one fallback:
 *
 *   1. **Calibration** — `baseline_repeat` tasks at run start. Same source
 *      commit, same harness, different PRNG seed. Sample variance over
 *      their metrics is an honest estimate of run-to-run noise.
 *
 *   2. **Online same-(operator,value) repeats** — when the manager
 *      dispatches the same `(operator, new_repr)` pair twice (or via a
 *      `baseline_repeat` later in the run), their variance refines σ̂.
 *
 *   3. **Fallback** — a per-task constant configured in
 *      `directExecutor.fallbackSigma`. Used when (1) and (2) are absent.
 *      Logged distinctly so we can audit which estimate the gate used.
 *
 * Combined estimate: inverse-variance-weighted average of (1) and (2)
 * when both are available with n ≥ 2 each. If only one source has n ≥ 2
 * we use that one. Otherwise we fall back.
 *
 * The estimator returns
 *   `{ sigmaHat, source, samples }`
 * where `source ∈ {"calibration", "online_repeats", "combined", "fallback"}`.
 */

import fs from 'fs';
import path from 'path';
import { extractOperators } from './operator-posterior.js';

const DEFAULT_METRIC_KEY = 'val_bpb';

function readDocs(eventsDir) {
  let entries = [];
  try { entries = fs.readdirSync(eventsDir); } catch { return []; }
  const docs = [];
  for (const name of entries) {
    if (!name.endsWith('.json') || name.endsWith('.tmp')) continue;
    try {
      const raw = fs.readFileSync(path.join(eventsDir, name), 'utf-8');
      const doc = JSON.parse(raw);
      if (doc && typeof doc === 'object' && Array.isArray(doc.events) && doc.task_id) {
        docs.push(doc);
      }
    } catch { /* tolerant */ }
  }
  return docs;
}

function sampleStddev(values) {
  if (!Array.isArray(values) || values.length < 2) return null;
  const n = values.length;
  const mean = values.reduce((s, x) => s + x, 0) / n;
  let m2 = 0;
  for (const x of values) m2 += (x - mean) * (x - mean);
  const variance = m2 / (n - 1);
  if (!Number.isFinite(variance) || variance < 0) return null;
  return Math.sqrt(variance);
}

/**
 * @param {Object=} args
 * @param {string=} args.eventsDir
 * @param {string=} args.metricKey
 * @param {number=} args.fallbackSigma  per-task fallback constant
 * @return {{sigmaHat: number, source: string, samples: number, sources: Object}}
 */
export function estimateNoise({
  eventsDir,
  metricKey = DEFAULT_METRIC_KEY,
  fallbackSigma = 0.0005,
} = {}) {
  const docs = readDocs(eventsDir);

  const calibrationMetrics = [];
  const repeatBuckets = new Map();

  for (const doc of docs) {
    const events = doc.events || [];
    const picked = events.find((e) => e.name === 'picked');
    const validated = events.find((e) => e.name === 'validated');
    if (!picked || !validated) continue;
    if (!validated.canonical_success) continue;
    const m = validated[metricKey];
    if (typeof m !== 'number' || !Number.isFinite(m)) continue;

    if (picked.execution_mode === 'baseline_repeat') {
      // Group calibrations by base commit so we don't mix runs from
      // different HEAD points; we only care about within-baseline
      // variance.
      const baseCommit = picked.picked_base_commit || picked.base_ref || 'unknown';
      let arr = repeatBuckets.get(`baseline:${baseCommit}`);
      if (!arr) { arr = []; repeatBuckets.set(`baseline:${baseCommit}`, arr); }
      arr.push(m);
      calibrationMetrics.push(m);
      continue;
    }

    // Online (operator, value) repeats. Single-edit experiments only.
    const ops = extractOperators(picked);
    if (ops.length !== 1) continue;
    const { operator, value } = ops[0];
    if (value == null) continue;
    const key = `op:${operator}:${value}:${picked.picked_base_commit || 'unknown'}`;
    let arr = repeatBuckets.get(key);
    if (!arr) { arr = []; repeatBuckets.set(key, arr); }
    arr.push(m);
  }

  // Calibration σ̂: pooled stddev across baseline_repeat buckets that
  // share a base commit. We use the largest bucket as the primary
  // sample.
  let calibSigma = null;
  let calibN = 0;
  for (const [key, arr] of repeatBuckets) {
    if (!key.startsWith('baseline:')) continue;
    if (arr.length < 2) continue;
    const s = sampleStddev(arr);
    if (s != null && (calibSigma == null || arr.length > calibN)) {
      calibSigma = s;
      calibN = arr.length;
    }
  }

  // Online σ̂: pool stddev across (op, value) buckets with ≥ 2 samples.
  // Pooled = sqrt(sum((n_i - 1) * s_i^2) / sum(n_i - 1)).
  let pooledNumer = 0;
  let pooledDenom = 0;
  let onlineN = 0;
  for (const [key, arr] of repeatBuckets) {
    if (key.startsWith('baseline:')) continue;
    if (arr.length < 2) continue;
    const s = sampleStddev(arr);
    if (s == null) continue;
    pooledNumer += (arr.length - 1) * s * s;
    pooledDenom += (arr.length - 1);
    onlineN += arr.length;
  }
  let onlineSigma = pooledDenom > 0 ? Math.sqrt(pooledNumer / pooledDenom) : null;

  if (calibSigma != null && onlineSigma != null && onlineN >= 2) {
    // Inverse-variance combination.
    const wCalib = (calibN - 1) / Math.max(calibSigma * calibSigma, 1e-12);
    const wOnline = pooledDenom / Math.max(onlineSigma * onlineSigma, 1e-12);
    const totalWeight = wCalib + wOnline;
    const combinedVar = (wCalib * calibSigma * calibSigma + wOnline * onlineSigma * onlineSigma) / totalWeight;
    return {
      sigmaHat: Math.sqrt(combinedVar),
      source: 'combined',
      samples: calibN + onlineN,
      sources: { calibSigma, calibN, onlineSigma, onlineN },
    };
  }
  if (calibSigma != null) {
    return { sigmaHat: calibSigma, source: 'calibration', samples: calibN, sources: { calibSigma, calibN } };
  }
  if (onlineSigma != null && onlineN >= 2) {
    return { sigmaHat: onlineSigma, source: 'online_repeats', samples: onlineN, sources: { onlineSigma, onlineN } };
  }
  return {
    sigmaHat: typeof fallbackSigma === 'number' && Number.isFinite(fallbackSigma) ? fallbackSigma : 0.0005,
    source: 'fallback',
    samples: 0,
    sources: {},
  };
}
