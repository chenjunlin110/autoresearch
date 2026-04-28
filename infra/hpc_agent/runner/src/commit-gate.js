/**
 * @fileoverview ALPS Phase 2 — statistical commit gate.
 *
 * Pure decision function. Given a candidate's metric, the current
 * HEAD's metric, an estimate of run-to-run noise σ̂, and how much
 * wall-time remains, decide whether the candidate "wins" by enough
 * margin to advance source/HEAD.
 *
 * Threshold is a time-decreasing multiple of σ̂:
 *
 *     τ_t = τ_min + (τ_max - τ_min) * (t_remaining / t_total)
 *     threshold = τ_t * σ̂ + c_stale * num_stale_workers
 *
 * Early in the run we are conservative (high τ); near the deadline we
 * relax the bar so a marginal but consistent improvement still gets
 * committed.
 *
 * The gate is **disabled** when HEAD's metric is unknown — without a
 * baseline we can't compute (yHead − yCandidate). The caller (the
 * runtime) is responsible for enqueuing a `baseline_repeat` to
 * re-establish HEAD.
 *
 * Stale candidates (baseCommit ≠ currentHeadCommit) are routed to
 * `AUTO_KEEP_REQUIRES_REBASE` rather than directly auto-applied — we
 * never apply a candidate measured on a different baseline; that's
 * Phase 5's territory.
 *
 * Sign convention: improvement = yHead − yCandidate (lower-is-better);
 * positive improvement means the candidate is better than HEAD.
 */

import { improvementFor } from './metric-conventions.js';

export const DECISION = Object.freeze({
  AUTO_KEEP: 'AUTO_KEEP',
  AUTO_KEEP_REQUIRES_REBASE: 'AUTO_KEEP_REQUIRES_REBASE',
  WAIT: 'WAIT',
  DISABLED: 'DISABLED',
});

/**
 * @param {Object} args
 * @param {number|null} args.yCandidate
 * @param {number|null} args.yHead
 * @param {number|null} args.sigmaHat
 * @param {string=} args.sigmaHatSource     "calibration" | "online_repeats" | "fallback" | "unknown"
 * @param {number=} args.sigmaHatSamples
 * @param {number=} args.tRemainingMs
 * @param {number=} args.tTotalMs
 * @param {number=} args.numStaleWorkers
 * @param {string|null} args.candidateBaseCommit
 * @param {string|null} args.currentHeadCommit
 * @param {boolean=} args.patchAppliesCleanly  default true (caller may set false to short-circuit)
 * @param {number=} args.tauMin               default 0.5
 * @param {number=} args.tauMax               default 2.0
 * @param {number=} args.cStale               default 0.0001 (per stale worker)
 * @param {number=} args.fallbackSigmaHat     default 0.0005 — used if sigmaHat null
 * @return {{
 *   decision: string,
 *   threshold: number|null,
 *   reason: string,
 *   improvement: number|null,
 *   tau: number|null,
 *   sigmaHatUsed: number|null,
 *   sigmaHatSource: string,
 *   inputs: Object,
 * }}
 */
export function shouldCommit({
  yCandidate,
  yHead,
  sigmaHat,
  sigmaHatSource = 'unknown',
  sigmaHatSamples = 0,
  tRemainingMs = 0,
  tTotalMs = 0,
  numStaleWorkers = 0,
  candidateBaseCommit = null,
  currentHeadCommit = null,
  patchAppliesCleanly = true,
  tauMin = 0.5,
  tauMax = 2.0,
  cStale = 0.0001,
  fallbackSigmaHat = 0.0005,
} = {}) {
  const inputs = {
    yCandidate, yHead, sigmaHat, sigmaHatSource, sigmaHatSamples,
    tRemainingMs, tTotalMs, numStaleWorkers,
    candidateBaseCommit, currentHeadCommit,
    patchAppliesCleanly, tauMin, tauMax, cStale, fallbackSigmaHat,
  };

  if (typeof yCandidate !== 'number' || !Number.isFinite(yCandidate)) {
    return { decision: DECISION.DISABLED, threshold: null, reason: 'candidate_metric_invalid', improvement: null, tau: null, sigmaHatUsed: null, sigmaHatSource, inputs };
  }
  if (typeof yHead !== 'number' || !Number.isFinite(yHead)) {
    return { decision: DECISION.DISABLED, threshold: null, reason: 'head_metric_unknown', improvement: null, tau: null, sigmaHatUsed: null, sigmaHatSource, inputs };
  }

  const improvement = improvementFor(yCandidate, yHead);

  // Stale baseline: candidate ran on a different commit than HEAD now.
  // Block auto-apply; surface for Phase 5's rebase-validation lifecycle.
  if (candidateBaseCommit && currentHeadCommit && candidateBaseCommit !== currentHeadCommit) {
    return {
      decision: DECISION.AUTO_KEEP_REQUIRES_REBASE,
      threshold: null,
      reason: 'stale_baseline',
      improvement,
      tau: null,
      sigmaHatUsed: null,
      sigmaHatSource,
      inputs,
    };
  }

  if (patchAppliesCleanly === false) {
    return {
      decision: DECISION.AUTO_KEEP_REQUIRES_REBASE,
      threshold: null,
      reason: 'patch_conflict_on_head',
      improvement,
      tau: null,
      sigmaHatUsed: null,
      sigmaHatSource,
      inputs,
    };
  }

  const usedSigma = (typeof sigmaHat === 'number' && Number.isFinite(sigmaHat) && sigmaHat > 0)
    ? sigmaHat
    : fallbackSigmaHat;
  const usedSource = (typeof sigmaHat === 'number' && Number.isFinite(sigmaHat) && sigmaHat > 0)
    ? sigmaHatSource
    : 'fallback';

  let frac = 0;
  if (Number.isFinite(tTotalMs) && tTotalMs > 0) {
    frac = Math.max(0, Math.min(1, (tRemainingMs || 0) / tTotalMs));
  }
  const tau = tauMin + (tauMax - tauMin) * frac;
  const stalePenalty = Math.max(0, numStaleWorkers || 0) * cStale;
  const threshold = tau * usedSigma + stalePenalty;

  if (improvement > threshold) {
    return {
      decision: DECISION.AUTO_KEEP,
      threshold,
      reason: 'passes_gate',
      improvement,
      tau,
      sigmaHatUsed: usedSigma,
      sigmaHatSource: usedSource,
      inputs,
    };
  }
  return {
    decision: DECISION.WAIT,
    threshold,
    reason: 'below_threshold',
    improvement,
    tau,
    sigmaHatUsed: usedSigma,
    sigmaHatSource: usedSource,
    inputs,
  };
}

/**
 * Format a decision as a one-line `[gate]` log entry.
 *
 * @param {string} experimentId
 * @param {ReturnType<typeof shouldCommit>} d
 * @return {string}
 */
export function formatGateDecision(experimentId, d) {
  const tag = d.decision === DECISION.AUTO_KEEP ? 'WOULD_AUTO_KEEP'
    : d.decision === DECISION.AUTO_KEEP_REQUIRES_REBASE ? 'WOULD_AUTO_KEEP_REQUIRES_REBASE'
    : d.decision === DECISION.WAIT ? 'WOULD_NOT_KEEP'
    : 'DISABLED';
  const parts = [`[gate] ${experimentId}: ${tag}`];
  if (d.threshold != null) parts.push(`threshold=${d.threshold.toExponential(3)}`);
  if (d.tau != null) parts.push(`tau=${d.tau.toFixed(2)}`);
  if (d.sigmaHatUsed != null) parts.push(`sigmaHat=${d.sigmaHatUsed.toExponential(3)}`);
  if (d.sigmaHatSource) parts.push(`source=${d.sigmaHatSource}`);
  if (d.improvement != null && Number.isFinite(d.improvement)) parts.push(`improvement=${d.improvement.toExponential(3)}`);
  parts.push(`reason=${d.reason}`);
  if (d.inputs.candidateBaseCommit && d.inputs.currentHeadCommit) {
    const a = d.inputs.candidateBaseCommit.slice(0, 7);
    const b = d.inputs.currentHeadCommit.slice(0, 7);
    if (a !== b) parts.push(`base=${a} head=${b}`);
  }
  return parts.join(' ');
}
