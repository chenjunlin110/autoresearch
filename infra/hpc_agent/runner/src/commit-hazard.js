/**
 * @fileoverview ALPS Phase 3 — adaptive queue depth via commit hazard.
 *
 * Replaces the fixed `liveReplanWatermarkRatio` with a queue depth that
 * shrinks when in-flight experiments look likely to win and grows
 * otherwise. Intuition: if 4 of 8 running candidates have a high
 * P[beat HEAD], dispatching more tasks against the same HEAD is
 * speculative work — those new tasks will probably be stale by the
 * time they validate. Hold the dispatch slot.
 *
 *     ρ_t = 1 - ∏_e (1 - p_e)              # commit hazard
 *     Q_t = Q_min + (Q_max - Q_min)(1-ρ)^ζ
 *
 * `p_e ≈ Φ(μ_o / σ_o)` from the operator posterior — the probability
 * that the operator's true improvement exceeds 0 under the current
 * Welford estimate. Multi-edit experiments use the COMPOSITION bucket.
 *
 * **Reliability gate** is critical: with n<minSamples the posterior is
 * uninformative; using a flat 0.2 prior on 8 workers gives
 * ρ≈1-0.8^8≈0.83 and Q collapses to Q_min in seconds. We instead cap
 * unreliable operators at `unreliableCap` (default 0.5) so they don't
 * dominate ρ until real data accumulates.
 *
 * Sign convention (lower-is-better metric): positive μ in the posterior
 * means the operator improves on average. P[improvement > 0] = Φ(μ/σ).
 */

import { extractOperators, welfordSigma } from './operator-posterior.js';

const SQRT_2 = Math.SQRT2;

/**
 * Standard-normal CDF Φ(z) via the Abramowitz-Stegun erf approximation.
 * Max error <1.5e-7, plenty for hazard math.
 */
export function standardNormalCDF(z) {
  if (!Number.isFinite(z)) return 0.5;
  const sign = z < 0 ? -1 : 1;
  const x = Math.abs(z) / SQRT_2;
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((
    1.061405429 * t - 1.453152027) * t)
    + 1.421413741) * t
    - 0.284496736) * t
    + 0.254829592) * t * Math.exp(-x * x);
  return 0.5 * (1 + sign * y);
}

/**
 * Estimate P[a single experiment with this operator beats current HEAD]
 * from the Welford posterior. Returns:
 *   - { p, source: "posterior" }   when n ≥ minSamples and σ > 0
 *   - { p: cap, source: "unreliable" } otherwise
 *
 * @param {Object} args
 * @param {string} args.operatorKey
 * @param {Map<string, Object>} args.operators  posterior.operators map
 * @param {Map<string, Object>=} args.compositions  posterior.compositions map (used for COMPOSITION keys)
 * @param {number=} args.minSamples
 * @param {number=} args.unreliableCap
 * @param {number=} args.fallbackSigma         used when posterior σ is null/0
 * @return {{p: number, source: string, n: number}}
 */
export function commitProbability({
  operatorKey,
  operators,
  compositions = null,
  minSamples = 3,
  unreliableCap = 0.5,
  fallbackSigma = null,
}) {
  if (!operatorKey || (!operators && !compositions)) {
    return { p: unreliableCap, source: 'no_posterior', n: 0 };
  }
  const isComposition = operatorKey.startsWith('_COMPOSITION:');
  const state = isComposition
    ? (compositions ? compositions.get(operatorKey) : null)
    : (operators ? operators.get(operatorKey) : null);
  if (!state) {
    return { p: unreliableCap, source: 'unknown_operator', n: 0 };
  }
  const n = isComposition ? state.count : state.singleEditCount;
  if (n < minSamples) {
    return { p: unreliableCap, source: 'unreliable', n };
  }
  let sigma = welfordSigma(state);
  if (!sigma || sigma <= 0) {
    if (fallbackSigma && fallbackSigma > 0) {
      sigma = fallbackSigma;
    } else {
      return { p: unreliableCap, source: 'no_sigma', n };
    }
  }
  const z = state.mean / sigma;
  const raw = standardNormalCDF(z);
  // Cap at 0.99 — a single experiment never has certainty 1.
  return { p: Math.min(raw, 0.99), source: 'posterior', n };
}

/**
 * Compute the adaptive target queue depth from the set of in-flight
 * experiments. Returns null when the feature is disabled — caller
 * falls back to its existing watermark.
 *
 * @param {Object} args
 * @param {Array<Object>} args.runningPicked  picked events for in-flight experiments
 * @param {Object} args.posterior              from buildOperatorPosterior
 * @param {Object} args.config                 hazard config
 * @param {boolean} args.config.hazardQueueEnabled
 * @param {number} args.config.qMin
 * @param {number} args.config.qMax
 * @param {number=} args.config.hazardZeta             (default 2.0)
 * @param {number=} args.config.posteriorMinSamples    (default 3)
 * @param {number=} args.config.hazardCommitProbCap    (default 0.5)
 * @param {number=} args.config.fallbackSigma
 * @return {{qTarget: number, rho: number, perExperiment: Array<{id: string, op: string, p: number, source: string}>}|null}
 */
export function adaptiveQueueDepth({ runningPicked, posterior, config }) {
  if (!config?.hazardQueueEnabled) return null;
  const qMin = Number.isInteger(config.qMin) ? config.qMin : 1;
  const qMax = Number.isInteger(config.qMax) ? config.qMax : 8;
  const zeta = Number.isFinite(config.hazardZeta) && config.hazardZeta > 0 ? config.hazardZeta : 2.0;
  const minSamples = Number.isInteger(config.posteriorMinSamples) ? config.posteriorMinSamples : 3;
  const unreliableCap = Number.isFinite(config.hazardCommitProbCap) ? config.hazardCommitProbCap : 0.5;
  const fallbackSigma = Number.isFinite(config.fallbackSigma) ? config.fallbackSigma : null;

  const perExperiment = [];
  for (const picked of runningPicked || []) {
    if (!picked) continue;
    const ops = extractOperators(picked);
    if (ops.length === 0) continue;
    let opKey;
    if (ops.length === 1) {
      opKey = ops[0].operator;
    } else {
      opKey = `_COMPOSITION:${ops.map((o) => o.operator).sort().join('+')}`;
    }
    const probe = commitProbability({
      operatorKey: opKey,
      operators: posterior?.operators,
      compositions: posterior?.compositions,
      minSamples,
      unreliableCap,
      fallbackSigma,
    });
    perExperiment.push({
      id: picked.task_id || picked.id || '?',
      op: opKey,
      p: probe.p,
      source: probe.source,
      n: probe.n,
    });
  }

  // ρ = 1 - ∏(1 - p). When no in-flight experiments contribute, ρ=0 and
  // Q stays at Q_max — explore freely until something running.
  let prod = 1;
  for (const e of perExperiment) prod *= (1 - e.p);
  const rho = 1 - prod;
  const qFloat = qMin + (qMax - qMin) * Math.pow(1 - rho, zeta);
  const qTarget = Math.max(qMin, Math.min(qMax, Math.round(qFloat)));
  return { qTarget, rho, perExperiment };
}

/**
 * One-line `[hazard]` log of the queue-depth decision so it can be
 * audited alongside `[gate]` lines.
 *
 * @param {ReturnType<typeof adaptiveQueueDepth>} result
 * @return {string}
 */
export function formatHazardDecision(result) {
  if (!result) return '';
  const opsSummary = result.perExperiment
    .map((e) => `${e.id}:${e.p.toFixed(2)}(${e.source})`)
    .slice(0, 6)
    .join(' ');
  return `[hazard] Q=${result.qTarget} rho=${result.rho.toFixed(3)} ${opsSummary}${result.perExperiment.length > 6 ? ' …' : ''}`;
}
