/**
 * @fileoverview ALPS Phase 1 — online operator posterior.
 *
 * For each "operator" (axis being searched, e.g. ASPECT_RATIO,
 * MATRIX_LR), maintain Welford's online estimate of the *improvement*
 * the operator delivers over its baseline, plus coverage telemetry
 * (which values were tried, how often, average runtime, invalid rate).
 *
 * Key design decisions:
 * - Single-edit experiments only contribute to per-operator μ/σ.
 *   Multi-edit experiments touch the joint COMPOSITION bucket, so an
 *   operator's posterior never picks up credit for a peer's contribution.
 * - The baseline metric is the metric of the experiment's
 *   `picked_base_commit` *at the time it was picked*. Stored on the
 *   picked event as `picked_base_metric_at_pick_time` (Phase 0). When
 *   absent, we update coverage only — no μ/σ fabrication from a
 *   baseline we never actually measured.
 * - Sign convention: `improvement = baseline.metric - candidate.metric`
 *   (lower-is-better). Positive μ ⇒ operator improves on average.
 * - Online only. No cross-run priors in this PR; we will not fabricate
 *   "learned-from-prior-search" effects we did not produce.
 */

import fs from 'fs';
import path from 'path';
import { improvementFor } from './metric-conventions.js';

const DEFAULT_METRIC_KEY = 'val_bpb';
const RUNTIME_EMA_ALPHA = 0.4;
const COMPOSITION_PREFIX = '_COMPOSITION:';
const BLOCK_PREFIX = '_BLOCK_';
const DIFF_PREFIX = '_DIFF_';

/**
 * @typedef {Object} OperatorState
 * @property {string} key
 * @property {number} count          total experiments touching this operator
 * @property {number} singleEditCount experiments where this op was sole edit
 * @property {number} mean           Welford running mean of improvement
 * @property {number} m2             Welford sum-of-squares
 * @property {Array<string>} valuesSeen sorted distinct new_repr values tried
 * @property {number} invalidCount   how many touching experiments were not canonical_success
 * @property {number|null} runtimeAvg EMA over training_seconds for canonical_success runs
 * @property {string|null} bestValue  new_repr that produced max improvement so far
 * @property {number|null} bestImprovement
 *
 * @typedef {Object} CompositionState
 * @property {string} key                    `_COMPOSITION:<sorted op list>`
 * @property {number} count
 * @property {number} mean
 * @property {number} m2
 * @property {number|null} bestImprovement
 *
 * @typedef {Object} PosteriorReport
 * @property {Map<string, OperatorState>} operators
 * @property {Map<string, CompositionState>} compositions
 * @property {number} totalSingleEdit
 * @property {number} totalMultiEdit
 * @property {number} totalSkippedNoBaseline
 */

function newOperatorState(key) {
  return {
    key,
    count: 0,
    singleEditCount: 0,
    mean: 0,
    m2: 0,
    valuesSeen: new Set(),
    invalidCount: 0,
    runtimeAvg: null,
    bestValue: null,
    bestImprovement: null,
  };
}

function newCompositionState(key) {
  return { key, count: 0, mean: 0, m2: 0, bestImprovement: null };
}

/**
 * Welford online update. The caller passes the running n explicitly
 * (which it has bumped immediately before calling). We avoid using
 * `state.count` because for OperatorState, count tracks every touch
 * (including multi-edit experiments) and is therefore *not* the
 * correct Welford divisor; the divisor is `singleEditCount`.
 */
function welfordUpdate(state, x, n) {
  if (n < 1) return;
  const delta = x - state.mean;
  state.mean += delta / n;
  const delta2 = x - state.mean;
  state.m2 += delta * delta2;
}

/**
 * Operator key extraction for one structured edit. Mirrors the keying
 * the plan describes: name for constant_replace / regex_replace, anchor
 * digest for block_replace, content digest for unified_diff.
 *
 * @param {Object} edit
 * @return {string|null}
 */
export function operatorKeyForEdit(edit) {
  if (!edit || typeof edit !== 'object') return null;
  if (edit.kind === 'constant_replace') {
    return typeof edit.name === 'string' && edit.name ? edit.name : null;
  }
  if (edit.kind === 'regex_replace') {
    if (typeof edit.name === 'string' && edit.name) return edit.name;
    const pattern = typeof edit.pattern === 'string' ? edit.pattern.slice(0, 64) : '';
    return `_REGEX_${pattern}`;
  }
  if (edit.kind === 'block_replace') {
    const anchor = typeof edit.anchor_regex === 'string' ? edit.anchor_regex.slice(0, 32) : '';
    return `${BLOCK_PREFIX}${anchor}`;
  }
  if (edit.kind === 'unified_diff') {
    const diff = typeof edit.diff === 'string' ? edit.diff.slice(0, 256) : '';
    let h = 0;
    for (let i = 0; i < diff.length; i += 1) h = ((h << 5) - h + diff.charCodeAt(i)) | 0;
    return `${DIFF_PREFIX}${(h >>> 0).toString(16)}`;
  }
  return null;
}

/**
 * Extract the (operator, value) pairs from a picked event's edit list.
 * Returns `[]` for non-param_patch experiments (no structured edits).
 *
 * @param {Object} pickedEvent
 * @return {Array<{operator: string, value: string|null}>}
 */
export function extractOperators(pickedEvent) {
  if (!pickedEvent || !Array.isArray(pickedEvent.edits)) return [];
  const out = [];
  for (const edit of pickedEvent.edits) {
    const key = operatorKeyForEdit(edit);
    if (!key) continue;
    let value = null;
    if (edit.kind === 'constant_replace') value = edit.new_repr ?? null;
    else if (edit.kind === 'regex_replace') value = edit.replacement ?? null;
    else if (edit.kind === 'block_replace') value = edit.new_text ? edit.new_text.slice(0, 32) : null;
    else if (edit.kind === 'unified_diff') value = null;
    out.push({ operator: key, value });
  }
  return out;
}

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
    } catch {
      // tolerate corrupt entry; same convention as experiment-ledger
    }
  }
  return docs;
}

/**
 * Compute the operator posterior from the task-events directory.
 *
 * @param {Object=} args
 * @param {string=} args.eventsDir
 * @param {string=} args.metricKey
 * @return {PosteriorReport}
 */
export function buildOperatorPosterior({
  eventsDir,
  metricKey = DEFAULT_METRIC_KEY,
} = {}) {
  const operators = new Map();
  const compositions = new Map();
  let totalSingleEdit = 0;
  let totalMultiEdit = 0;
  let totalSkippedNoBaseline = 0;

  const docs = readDocs(eventsDir);
  for (const doc of docs) {
    const events = doc.events || [];
    const picked = events.find((e) => e.name === 'picked');
    const validated = events.find((e) => e.name === 'validated');
    if (!picked) continue;

    const ops = extractOperators(picked);
    if (ops.length === 0) continue;

    // Coverage tracking — applies whether or not we update μ/σ.
    const opKeys = ops.map((o) => o.operator);
    for (const { operator, value } of ops) {
      if (!operators.has(operator)) operators.set(operator, newOperatorState(operator));
      const state = operators.get(operator);
      state.count += 1;
      if (value != null) state.valuesSeen.add(value);
      if (validated && !validated.canonical_success) state.invalidCount += 1;
      if (validated && validated.canonical_success && typeof validated.training_seconds === 'number') {
        state.runtimeAvg = state.runtimeAvg == null
          ? validated.training_seconds
          : RUNTIME_EMA_ALPHA * validated.training_seconds + (1 - RUNTIME_EMA_ALPHA) * state.runtimeAvg;
      }
    }

    // μ/σ update needs a successful validation AND a known baseline
    // metric at pick time. Multi-edit experiments touch the COMPOSITION
    // bucket only.
    if (!validated || !validated.canonical_success) continue;
    const candidateMetric = validated[metricKey];
    if (typeof candidateMetric !== 'number' || !Number.isFinite(candidateMetric)) continue;

    const baseMetric = picked.picked_base_metric_at_pick_time;
    if (typeof baseMetric !== 'number' || !Number.isFinite(baseMetric)) {
      totalSkippedNoBaseline += 1;
      continue;
    }
    const delta = improvementFor(candidateMetric, baseMetric);
    if (!Number.isFinite(delta)) continue;

    if (ops.length === 1) {
      const { operator, value } = ops[0];
      const state = operators.get(operator);
      state.singleEditCount += 1;
      welfordUpdate(state, delta, state.singleEditCount);
      if (state.bestImprovement == null || delta > state.bestImprovement) {
        state.bestImprovement = delta;
        state.bestValue = value ?? state.bestValue;
      }
      totalSingleEdit += 1;
    } else {
      const compKey = `${COMPOSITION_PREFIX}${[...opKeys].sort().join('+')}`;
      if (!compositions.has(compKey)) compositions.set(compKey, newCompositionState(compKey));
      const comp = compositions.get(compKey);
      comp.count += 1;
      welfordUpdate(comp, delta, comp.count);
      if (comp.bestImprovement == null || delta > comp.bestImprovement) {
        comp.bestImprovement = delta;
      }
      totalMultiEdit += 1;
    }
  }

  return {
    operators,
    compositions,
    totalSingleEdit,
    totalMultiEdit,
    totalSkippedNoBaseline,
  };
}

/**
 * Welford variance → unbiased σ estimate. Returns null when n < 2.
 * For OperatorState, n is `singleEditCount` (NOT `count` — that one
 * tracks every touch including multi-edit experiments). For
 * CompositionState, n is `count`. We pick whichever exists.
 */
export function welfordSigma(state) {
  if (!state) return null;
  const n = typeof state.singleEditCount === 'number' ? state.singleEditCount : state.count;
  if (typeof n !== 'number' || n < 2) return null;
  const variance = state.m2 / (n - 1);
  if (!Number.isFinite(variance) || variance < 0) return null;
  return Math.sqrt(variance);
}

function fmtSigned(x, digits = 4) {
  if (x == null || !Number.isFinite(x)) return 'n/a';
  const sign = x >= 0 ? '+' : '−';
  return `${sign}${Math.abs(x).toFixed(digits)}`;
}

/**
 * Render the coverage manifest as a markdown block for the manager prompt.
 *
 * @param {PosteriorReport} posterior
 * @param {Array<{name: string, role?: string}>=} searchAxes  registry
 * @param {Object=} opts
 * @param {boolean=} opts.lowerIsBetter   default true (used in the legend)
 * @return {string}
 */
export function formatCoverageMarkdown(posterior, searchAxes = [], opts = {}) {
  const lowerIsBetter = opts.lowerIsBetter !== false;
  if (!posterior) return '';
  const { operators, compositions } = posterior;
  if (operators.size === 0 && compositions.size === 0 && (!searchAxes || searchAxes.length === 0)) {
    return '';
  }

  const lines = [];
  lines.push('> **Axis coverage** (this run; online posteriors, no cross-run priors).');
  lines.push(`> Sign: improvement = ${lowerIsBetter ? 'baseline − candidate' : 'candidate − baseline'}; positive μ means the operator beats its baseline on average.`);

  const knownAxes = new Map();
  for (const axis of (searchAxes || [])) {
    if (!axis || typeof axis.name !== 'string') continue;
    knownAxes.set(axis.name, axis);
  }

  // Primary axes first (registry order), then any operators encountered
  // outside the registry.
  const primaryRows = [];
  const otherRows = [];

  for (const [name, axis] of knownAxes) {
    primaryRows.push(renderOperatorRow(name, operators.get(name) || null, axis));
  }
  for (const [name, state] of operators) {
    if (knownAxes.has(name)) continue;
    if (name.startsWith(COMPOSITION_PREFIX)) continue;
    otherRows.push(renderOperatorRow(name, state, { role: 'unregistered' }));
  }

  if (primaryRows.length > 0) {
    lines.push('>');
    lines.push('> primary axes:');
    for (const r of primaryRows) lines.push(`>   ${r}`);
  }
  if (otherRows.length > 0) {
    lines.push('>');
    lines.push('> other operators (not in searchAxes registry):');
    for (const r of otherRows) lines.push(`>   ${r}`);
  }

  if (compositions.size > 0) {
    lines.push('>');
    lines.push('> compositions (multi-edit; credited jointly, not to individual axes):');
    const comps = [...compositions.values()].sort((a, b) => b.count - a.count).slice(0, 6);
    for (const c of comps) {
      const ops = c.key.slice(COMPOSITION_PREFIX.length);
      const sigma = welfordSigma(c);
      const sigmaStr = sigma != null ? `, σ=${sigma.toFixed(4)}` : '';
      const best = c.bestImprovement != null ? `; best ${fmtSigned(c.bestImprovement)}` : '';
      lines.push(`>   {${ops}}: n=${c.count}; μ=${fmtSigned(c.mean)}${sigmaStr}${best}`);
    }
  }

  if (posterior.totalSkippedNoBaseline > 0) {
    lines.push('>');
    lines.push(`> note: skipped μ/σ update on ${posterior.totalSkippedNoBaseline} experiment(s) — baseline metric at pick time was unknown (gate may have been disabled).`);
  }

  return lines.join('\n');
}

function renderOperatorRow(name, state, axis) {
  if (!state || state.count === 0) {
    return `${name.padEnd(16)}: never tested`;
  }
  const valuesArr = [...state.valuesSeen].sort();
  const valuesStr = valuesArr.length > 0
    ? `tried {${valuesArr.slice(0, 6).join(', ')}${valuesArr.length > 6 ? ', …' : ''}}`
    : 'tried (no values recorded)';
  const sigma = welfordSigma(state);
  const stats = state.singleEditCount > 0
    ? `n_single=${state.singleEditCount}; μ=${fmtSigned(state.mean)}${sigma != null ? `, σ=${sigma.toFixed(4)}` : ''}`
    : `n_single=0 (only co-edits seen)`;
  const best = state.bestValue != null && state.bestImprovement != null
    ? `; best ${state.bestValue} (${fmtSigned(state.bestImprovement)})`
    : '';
  const invalid = state.invalidCount > 0 ? ` [${state.invalidCount} invalid]` : '';
  return `${name.padEnd(16)}: ${valuesStr}; ${stats}${best}${invalid}`;
}
