/**
 * @fileoverview Aggregates per-task event traces from
 * `<projectDir>/task-events/*.json` into a single "what happened during
 * this cycle" view: time buckets, GPU duty cycle, mismatch count, and
 * the best metric observed.
 *
 * Only events whose `picked` timestamp falls within `[windowStart,
 * windowEnd]` are included, so back-to-back manager cycles produce
 * disjoint reports.
 *
 * The aggregator is forgiving: missing events fall through, malformed
 * JSON is skipped, and no exception ever propagates to the orchestrator.
 */

import fs from 'fs';
import path from 'path';

/** Default metric key — matches the autoresearch task; see result-validator.js. */
const DEFAULT_METRIC_KEY = 'val_bpb';

/**
 * @typedef {Object} CycleReport
 * @property {string|null} cycle_id
 * @property {number|undefined} window_start  epoch seconds
 * @property {number|undefined} window_end    epoch seconds
 * @property {number} cycle_seconds
 * @property {number} gpu_count
 * @property {number} available_gpu_seconds
 * @property {Record<string, number>} counts
 * @property {{worker_llm_total: number, training_total: number}} time_buckets_seconds
 * @property {number} gpu_duty_cycle           training_seconds / available_gpu_seconds
 * @property {number} worker_llm_gpu_share     worker_llm_seconds / available_gpu_seconds
 * @property {{best: number|null, best_task_id: string|null,
 *            mean: number|null, n: number}} val_bpb metric summary (legacy
 *   field name kept for backward compatibility with autoresearch dashboards)
 */

/**
 * Read every `<taskId>.json` under `eventsDir`. Skips `.tmp` half-writes
 * and unparseable entries silently.
 * @param {string} eventsDir
 * @return {Array<Record<string, unknown>>}
 */
function readEventsDir(eventsDir) {
  let entries = [];
  try {
    entries = fs.readdirSync(eventsDir);
  } catch {
    return [];
  }
  const docs = [];
  for (const name of entries) {
    if (!name.endsWith('.json') || name.endsWith('.tmp')) continue;
    try {
      const doc = JSON.parse(fs.readFileSync(path.join(eventsDir, name), 'utf-8'));
      if (doc && Array.isArray(doc.events)) docs.push(doc);
    } catch {
      // Skip corrupt entry.
    }
  }
  return docs;
}

/**
 * @param {Record<string, unknown>|null|undefined} doc
 * @param {string} name
 * @return {Record<string, unknown>|null}
 */
function eventOf(doc, name) {
  if (!doc?.events) return null;
  return doc.events.find((event) => event.name === name) || null;
}

/**
 * @param {Record<string, unknown>|null|undefined} doc
 * @param {string} name
 * @return {Array<Record<string, unknown>>}
 */
function eventsOf(doc, name) {
  if (!doc?.events) return [];
  return doc.events.filter((event) => event.name === name);
}

/**
 * @param {number} t epoch seconds
 * @param {number|undefined} windowStart
 * @param {number|undefined} windowEnd
 * @return {boolean}
 */
function inWindow(t, windowStart, windowEnd) {
  if (typeof t !== 'number') return false;
  return t >= windowStart && t <= windowEnd;
}

function round2(x) {
  return Math.round(x * 100) / 100;
}
function round3(x) {
  return Math.round(x * 1000) / 1000;
}
function round4(x) {
  return Math.round(x * 10000) / 10000;
}

/**
 * Build a one-shot aggregate report for the time window
 * `[windowStart, windowEnd]` (epoch seconds).
 *
 * @param {Object=} args
 * @param {string=} args.eventsDir
 * @param {number=} args.windowStart
 * @param {number=} args.windowEnd
 * @param {number=} args.gpuCount    pool size; default 8
 * @param {string|null=} args.cycleId optional label preserved in output
 * @param {string=} args.metricKey   key on `validated.metric_value`/`val_bpb`
 *   to track best/mean. Default matches the autoresearch convention.
 * @return {CycleReport}
 */
export function buildCycleReport({
  eventsDir,
  windowStart,
  windowEnd,
  gpuCount = 8,
  cycleId = null,
  metricKey = DEFAULT_METRIC_KEY,
} = {}) {
  const docs = readEventsDir(eventsDir);
  const cycleSeconds = Math.max(0, (windowEnd ?? 0) - (windowStart ?? 0));
  const availableGpuSeconds = cycleSeconds * gpuCount;

  let nPicked = 0;
  let nGranted = 0;
  let nWorkerSpawned = 0;
  let nWorkerReturnedSuccess = 0;
  let nWorkerReturnedFail = 0;
  let nCancelled = 0;
  let nKilledByTimeout = 0;
  let nValidatedTrue = 0;
  let nValidatedFalse = 0;
  let nMismatch = 0;
  let nWalltimeBlocked = 0;
  let nGrantBlocked = 0;
  let workerLlmSecondsSum = 0;
  let trainingSecondsSum = 0;
  let metricSum = 0;
  let metricCount = 0;
  let bestMetric = Infinity;
  let bestTaskId = null;

  for (const doc of docs) {
    if (!doc) continue;
    const picked = eventOf(doc, 'picked');
    if (!picked || !inWindow(picked.t, windowStart, windowEnd)) continue;
    nPicked += 1;
    if (eventOf(doc, 'granted')) nGranted += 1;
    if (eventOf(doc, 'walltime_blocked')) nWalltimeBlocked += 1;
    if (eventOf(doc, 'grant_blocked')) nGrantBlocked += 1;

    const spawns = eventsOf(doc, 'worker_spawned');
    const returns = eventsOf(doc, 'worker_returned');
    nWorkerSpawned += spawns.length;
    for (let i = 0; i < returns.length; i += 1) {
      const ret = returns[i];
      const spawn = spawns[i];
      if (spawn && typeof spawn.t === 'number' && typeof ret.t === 'number') {
        workerLlmSecondsSum += Math.max(0, ret.t - spawn.t);
      }
      if (ret.success) nWorkerReturnedSuccess += 1;
      else nWorkerReturnedFail += 1;
      if (ret.cancelled) nCancelled += 1;
      if (ret.killed_by_timeout) nKilledByTimeout += 1;
    }

    const validations = eventsOf(doc, 'validated');
    for (const v of validations) {
      if (v.canonical_success) nValidatedTrue += 1;
      else nValidatedFalse += 1;
      if (v.mismatch) nMismatch += 1;
      if (typeof v.training_seconds === 'number') {
        trainingSecondsSum += v.training_seconds;
      }
      // The validator records the metric under its canonical key (default
      // `val_bpb`). Keep the legacy field name in the report output to
      // preserve any downstream dashboards that key off `val_bpb`.
      const metricValue = v[metricKey];
      if (typeof metricValue === 'number' && Number.isFinite(metricValue)) {
        metricSum += metricValue;
        metricCount += 1;
        if (metricValue < bestMetric) {
          bestMetric = metricValue;
          bestTaskId = doc.task_id;
        }
      }
    }
  }

  const dutyCycle = availableGpuSeconds > 0 ? trainingSecondsSum / availableGpuSeconds : 0;
  const llmShare = cycleSeconds > 0 ? workerLlmSecondsSum / (cycleSeconds * gpuCount) : 0;

  return {
    cycle_id: cycleId,
    window_start: windowStart,
    window_end: windowEnd,
    cycle_seconds: cycleSeconds,
    gpu_count: gpuCount,
    available_gpu_seconds: availableGpuSeconds,
    counts: {
      picked: nPicked,
      granted: nGranted,
      worker_spawned: nWorkerSpawned,
      worker_returned_success: nWorkerReturnedSuccess,
      worker_returned_failure: nWorkerReturnedFail,
      cancelled: nCancelled,
      killed_by_timeout: nKilledByTimeout,
      walltime_blocked: nWalltimeBlocked,
      grant_blocked: nGrantBlocked,
      validated_success: nValidatedTrue,
      validated_failure: nValidatedFalse,
      truth_mismatch: nMismatch,
    },
    time_buckets_seconds: {
      worker_llm_total: round2(workerLlmSecondsSum),
      training_total: round2(trainingSecondsSum),
    },
    gpu_duty_cycle: round3(dutyCycle),
    worker_llm_gpu_share: round3(llmShare),
    val_bpb: {
      best: Number.isFinite(bestMetric) ? bestMetric : null,
      best_task_id: bestTaskId,
      mean: metricCount > 0 ? round4(metricSum / metricCount) : null,
      n: metricCount,
    },
  };
}

/**
 * Persist a report to `<reportsDir>/cycle-<isoTimestamp>.json`.
 * @param {Object=} args
 * @param {string=} args.reportsDir
 * @param {CycleReport=} args.report
 * @return {string|null} the file path, or null on error
 */
export function writeCycleReport({ reportsDir, report } = {}) {
  if (!reportsDir || !report) return null;
  try {
    fs.mkdirSync(reportsDir, { recursive: true });
    const ts = new Date(report.window_end * 1000).toISOString().replace(/[:.]/g, '-');
    const file = path.join(reportsDir, `cycle-${ts}.json`);
    fs.writeFileSync(file, JSON.stringify(report, null, 2));
    return file;
  } catch {
    return null;
  }
}

/**
 * Compact one-line summary for the orchestrator log.
 * @param {CycleReport|null|undefined} report
 * @return {string}
 */
export function formatCycleReportLine(report) {
  if (!report) return 'cycle report unavailable';
  const c = report.counts || {};
  const tb = report.time_buckets_seconds || {};
  const v = report.val_bpb || {};
  return (
    `Cycle report: ${c.picked || 0} picked, ${c.validated_success || 0} ok, `
    + `${c.validated_failure || 0} failed, ${c.cancelled || 0} cancelled, `
    + `${c.killed_by_timeout || 0} timeout, ${c.walltime_blocked || 0} wt-blocked, `
    + `${c.truth_mismatch || 0} mismatch | `
    + `train ${tb.training_total || 0}s, worker_llm ${tb.worker_llm_total || 0}s | `
    + `gpu_duty ${(report.gpu_duty_cycle * 100).toFixed(1)}% | `
    + `best metric ${v.best != null ? v.best.toFixed(4) : 'n/a'}`
    + (v.best_task_id ? ` (${v.best_task_id})` : '')
  );
}
