import fs from 'fs';
import path from 'path';

// Aggregates per-task event traces from <projectDir>/task-events/*.json into
// a single "what happened during this cycle" view: time buckets, GPU duty
// cycle, mismatch count, top stalls.
//
// Only events with timestamps within [windowStart, windowEnd] are included
// for the per-cycle view. The aggregator is forgiving: missing events fall
// through, never throws.

function readEventsDir(eventsDir) {
  let entries = [];
  try { entries = fs.readdirSync(eventsDir); } catch { return []; }
  const docs = [];
  for (const name of entries) {
    if (!name.endsWith('.json') || name.endsWith('.tmp')) continue;
    try {
      const doc = JSON.parse(fs.readFileSync(path.join(eventsDir, name), 'utf-8'));
      if (doc && Array.isArray(doc.events)) docs.push(doc);
    } catch { /* skip corrupt entry */ }
  }
  return docs;
}

function eventOf(doc, name) {
  if (!doc?.events) return null;
  return doc.events.find((e) => e.name === name) || null;
}

function eventsOf(doc, name) {
  if (!doc?.events) return [];
  return doc.events.filter((e) => e.name === name);
}

function inWindow(t, windowStart, windowEnd) {
  if (typeof t !== 'number') return false;
  return t >= windowStart && t <= windowEnd;
}

// Build a one-shot report. windowStart/windowEnd are epoch seconds.
export function buildCycleReport({
  eventsDir,
  windowStart,
  windowEnd,
  gpuCount = 8,
  cycleId = null,
} = {}) {
  const docs = readEventsDir(eventsDir);
  const cycleSeconds = Math.max(0, (windowEnd ?? 0) - (windowStart ?? 0));
  const availableGpuSeconds = cycleSeconds * gpuCount;

  let nPicked = 0, nGranted = 0, nWorkerSpawned = 0, nWorkerReturnedSuccess = 0,
      nWorkerReturnedFail = 0, nCancelled = 0, nKilledByTimeout = 0,
      nValidatedTrue = 0, nValidatedFalse = 0, nMismatch = 0,
      nWalltimeBlocked = 0, nGrantBlocked = 0;
  let workerLlmSecondsSum = 0;
  let trainingSecondsSum = 0;
  let trainStartToEndSum = 0;
  let valBpbSum = 0, valBpbCount = 0, bestVal = Infinity, bestId = null;

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
      if (ret.success) nWorkerReturnedSuccess += 1; else nWorkerReturnedFail += 1;
      if (ret.cancelled) nCancelled += 1;
      if (ret.killed_by_timeout) nKilledByTimeout += 1;
    }

    const validations = eventsOf(doc, 'validated');
    for (const v of validations) {
      if (v.canonical_success) nValidatedTrue += 1; else nValidatedFalse += 1;
      if (v.mismatch) nMismatch += 1;
      if (typeof v.training_seconds === 'number') trainingSecondsSum += v.training_seconds;
      if (typeof v.val_bpb === 'number' && Number.isFinite(v.val_bpb)) {
        valBpbSum += v.val_bpb;
        valBpbCount += 1;
        if (v.val_bpb < bestVal) { bestVal = v.val_bpb; bestId = doc.task_id; }
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
      best: Number.isFinite(bestVal) ? bestVal : null,
      best_task_id: bestId,
      mean: valBpbCount > 0 ? round4(valBpbSum / valBpbCount) : null,
      n: valBpbCount,
    },
  };
}

function round2(x) { return Math.round(x * 100) / 100; }
function round3(x) { return Math.round(x * 1000) / 1000; }
function round4(x) { return Math.round(x * 10000) / 10000; }

export function writeCycleReport({ reportsDir, report }) {
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

export function formatCycleReportLine(report) {
  if (!report) return 'cycle report unavailable';
  const c = report.counts || {};
  const tb = report.time_buckets_seconds || {};
  const v = report.val_bpb || {};
  return (
    `Cycle report: ${c.picked || 0} picked, ${c.validated_success || 0} ok, ` +
    `${c.validated_failure || 0} failed, ${c.cancelled || 0} cancelled, ` +
    `${c.killed_by_timeout || 0} timeout, ${c.walltime_blocked || 0} wt-blocked, ` +
    `${c.truth_mismatch || 0} mismatch | ` +
    `train ${tb.training_total || 0}s, worker_llm ${tb.worker_llm_total || 0}s | ` +
    `gpu_duty ${(report.gpu_duty_cycle * 100).toFixed(1)}% | ` +
    `best val_bpb ${v.best != null ? v.best.toFixed(4) : 'n/a'}` +
    (v.best_task_id ? ` (${v.best_task_id})` : '')
  );
}
