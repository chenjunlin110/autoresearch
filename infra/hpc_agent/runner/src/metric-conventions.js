/**
 * @fileoverview Single source of truth for the sign convention used by
 * everything that compares two metrics — the operator posterior, the
 * commit gate, the coverage manifest, the ledger renderer.
 *
 * **Convention (frozen):** every task in this codebase reports a
 * lower-is-better metric. `improvementFor(candidate, baseline)` returns
 * `baseline.metric - candidate.metric`, so a positive return value means
 * the candidate is better than the baseline. The operator posterior and
 * the commit gate consume that signed scalar directly: positive μ means
 * the operator on average improves on its baseline; positive
 * (yHead − yCandidate) is what the gate checks against τ·σ̂.
 *
 * If a future task plugin needs higher-is-better, override this helper
 * (e.g., return `candidate.metric - baseline.metric`) — every consumer
 * stays correct without touching their own arithmetic.
 *
 * Inputs are objects of the shape `{metric: number}` so callers don't
 * have to remember whether to pass the raw number or the experiment
 * record. We accept either: a bare number is treated as
 * `{metric: number}`.
 */

/**
 * @param {{metric: number} | number} candidate
 * @param {{metric: number} | number} baseline
 * @return {number} positive when candidate is better than baseline
 *   (lower-is-better convention)
 */
export function improvementFor(candidate, baseline) {
  const c = typeof candidate === 'number' ? candidate : candidate?.metric;
  const b = typeof baseline === 'number' ? baseline : baseline?.metric;
  if (typeof c !== 'number' || !Number.isFinite(c)) return NaN;
  if (typeof b !== 'number' || !Number.isFinite(b)) return NaN;
  return b - c;
}

/**
 * Convenience predicate: did the candidate beat the baseline by at
 * least `threshold`? threshold is in the same units as the metric.
 *
 * @param {{metric: number} | number} candidate
 * @param {{metric: number} | number} baseline
 * @param {number} threshold positive scalar
 * @return {boolean}
 */
export function improvedBy(candidate, baseline, threshold) {
  const delta = improvementFor(candidate, baseline);
  return Number.isFinite(delta) && delta > threshold;
}
