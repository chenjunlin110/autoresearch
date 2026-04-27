/**
 * @fileoverview Compact summary of past + in-flight experiments, injected
 * into the manager's per-cycle context so it doesn't need to `ls
 * experiments/` and read 50+ `metrics.json` files on every replan.
 *
 * Built directly from the `task-events/` directory (Phase 1.1) so the
 * ledger is always consistent with the canonical event trace — no
 * separate state, no schema drift. Each completed experiment becomes a
 * one-line record (~120 chars). With 100+ experiments the ledger stays
 * roughly constant size.
 *
 * Output sections:
 *   - top_k:           best `metricKey` so far (default `val_bpb`, lower=better)
 *   - recent:          most recent terminal experiments
 *   - running:         in-flight tasks
 *   - failed_clusters: groups of failures by short reason prefix
 */

import fs from 'fs';
import path from 'path';

const DEFAULT_METRIC_KEY = 'val_bpb';
const TOP_K = 10;
const RECENT_K = 10;
const FAILURE_REASON_PREFIX = 60;

/**
 * @typedef {Object} ExperimentRecord
 * @property {string} id
 * @property {string} status        completed | failed | running
 * @property {number=} metric        the optimized metric value (e.g. val_bpb)
 * @property {number=} trainingSeconds
 * @property {string=} reason        for failed/running tasks
 * @property {number=} startedAt     epoch seconds (picked event)
 * @property {number=} finishedAt    epoch seconds (released event)
 *
 * @typedef {Object} CompactLedger
 * @property {Array<ExperimentRecord>} topK
 * @property {Array<ExperimentRecord>} recent
 * @property {Array<ExperimentRecord>} running
 * @property {Array<{reason: string, count: number, ids: Array<string>}>} failedClusters
 * @property {number} totalCompleted
 * @property {number} totalFailed
 */

function readEvents(eventsDir) {
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
      // Skip corrupt entry; the event-trace writer is forgiving by design.
    }
  }
  return docs;
}

function reduceDoc(doc, metricKey) {
  const events = doc.events || [];
  const findFirst = (name) => events.find((e) => e.name === name) || null;
  const findLast = (name) => {
    for (let i = events.length - 1; i >= 0; i -= 1) {
      if (events[i].name === name) return events[i];
    }
    return null;
  };
  const picked = findFirst('picked');
  const released = findLast('released');
  const validated = findLast('validated');
  const workerReturned = findLast('worker_returned');
  const grantBlocked = findLast('grant_blocked');
  const walltimeBlocked = findLast('walltime_blocked');

  const startedAt = typeof picked?.t === 'number' ? picked.t : null;
  const finishedAt = typeof released?.t === 'number' ? released.t : null;
  // Manager-provided framing captured at pick time (Phase 5.4 epistemic
  // memory loop). Either field may be absent for tasks emitted before
  // the field was wired.
  const rationale = typeof picked?.rationale === 'string' ? picked.rationale : null;
  const taskSummary = typeof picked?.task_summary === 'string' ? picked.task_summary : null;
  const baseRef = typeof picked?.base_ref === 'string' ? picked.base_ref : null;
  const editSummary = typeof picked?.edit_summary === 'string' ? picked.edit_summary : null;

  const baseRecord = { id: doc.task_id, rationale, taskSummary, baseRef, editSummary };

  if (validated) {
    const metric = validated[metricKey];
    if (validated.canonical_success && typeof metric === 'number' && Number.isFinite(metric)) {
      return {
        ...baseRecord,
        status: 'completed',
        metric,
        trainingSeconds: typeof validated.training_seconds === 'number' ? validated.training_seconds : null,
        startedAt,
        finishedAt,
      };
    }
    return {
      ...baseRecord,
      status: 'failed',
      reason: validated.reason || 'validated_failure',
      startedAt,
      finishedAt,
    };
  }
  if (workerReturned && !workerReturned.success) {
    return {
      ...baseRecord,
      status: 'failed',
      reason: workerReturned.killed_by_timeout ? 'killed_by_timeout' : (workerReturned.cancelled ? 'cancelled' : 'worker_returned_failure'),
      startedAt,
      finishedAt,
    };
  }
  if (grantBlocked) {
    return {
      ...baseRecord,
      status: 'failed',
      reason: `grant_blocked: ${grantBlocked.reason || 'unknown'}`,
      startedAt,
    };
  }
  if (walltimeBlocked) {
    return {
      ...baseRecord,
      status: 'failed',
      reason: 'walltime_blocked',
      startedAt,
    };
  }
  if (picked && !released) {
    return {
      ...baseRecord,
      status: 'running',
      startedAt,
    };
  }
  return null;
}

/**
 * Build the compact ledger from the task-events directory.
 *
 * @param {Object=} args
 * @param {string=} args.eventsDir
 * @param {string=} args.metricKey  defaults to val_bpb
 * @param {boolean=} args.lowerIsBetter  defaults true
 * @return {CompactLedger}
 */
export function buildExperimentLedger({
  eventsDir,
  metricKey = DEFAULT_METRIC_KEY,
  lowerIsBetter = true,
} = {}) {
  const docs = readEvents(eventsDir);
  const records = docs.map((doc) => reduceDoc(doc, metricKey)).filter(Boolean);

  const completed = records.filter((r) => r.status === 'completed');
  const failed = records.filter((r) => r.status === 'failed');
  const running = records.filter((r) => r.status === 'running');

  completed.sort((a, b) => (lowerIsBetter ? a.metric - b.metric : b.metric - a.metric));
  const topK = completed.slice(0, TOP_K);

  const terminals = [...completed, ...failed].sort((a, b) => (b.finishedAt || b.startedAt || 0) - (a.finishedAt || a.startedAt || 0));
  const recent = terminals.slice(0, RECENT_K);

  // Cluster failures by leading prefix of `reason` so the manager sees
  // "all 7 OOMs" rather than seven near-identical lines.
  const clusters = new Map();
  for (const f of failed) {
    const key = (f.reason || 'unknown').slice(0, FAILURE_REASON_PREFIX);
    if (!clusters.has(key)) clusters.set(key, { reason: key, count: 0, ids: [] });
    const entry = clusters.get(key);
    entry.count += 1;
    if (entry.ids.length < 5) entry.ids.push(f.id);
  }
  const failedClusters = [...clusters.values()].sort((a, b) => b.count - a.count);

  return {
    topK,
    recent,
    running,
    failedClusters,
    totalCompleted: completed.length,
    totalFailed: failed.length,
  };
}

/**
 * Compute the recent `param_patch` success rate from task-events.
 * Tasks dispatched via the direct executor write `worker_spawned` with
 * `worker: 'direct-executor'`; we count those + their final
 * `validated.canonical_success` to decide if the manager has been
 * emitting good edits. Returns `null` if there's nothing to report yet.
 *
 * @param {Object=} args
 * @param {string=} args.eventsDir
 * @param {number=} args.windowSize  how many most-recent direct tasks to inspect
 * @param {number=} args.successThreshold  trigger threshold (default 0.7)
 * @return {{rate: number, succeeded: number, total: number, recentFailureIds: Array<string>, shouldWarn: boolean}|null}
 */
export function computeEditConfidence({
  eventsDir,
  windowSize = 20,
  successThreshold = 0.7,
} = {}) {
  const docs = readEvents(eventsDir);
  // Pair each direct-executor doc with its picked timestamp so we can
  // sort newest-first and take a recency window.
  const direct = [];
  for (const doc of docs) {
    const events = doc.events || [];
    const spawn = events.find((e) => e.name === 'worker_spawned' && e.worker === 'direct-executor');
    if (!spawn) continue;
    const picked = events.find((e) => e.name === 'picked');
    const validated = events.find((e) => e.name === 'validated');
    direct.push({
      id: doc.task_id,
      pickedAt: typeof picked?.t === 'number' ? picked.t : 0,
      success: !!validated?.canonical_success,
      validated: !!validated,
    });
  }
  if (direct.length === 0) return null;
  direct.sort((a, b) => b.pickedAt - a.pickedAt);
  const window = direct.slice(0, windowSize);
  const terminal = window.filter((d) => d.validated);
  if (terminal.length === 0) return null;
  const succeeded = terminal.filter((d) => d.success).length;
  const rate = succeeded / terminal.length;
  return {
    rate,
    succeeded,
    total: terminal.length,
    recentFailureIds: terminal.filter((d) => !d.success).slice(0, 5).map((d) => d.id),
    shouldWarn: rate < successThreshold,
  };
}

/**
 * Render the ledger as a markdown block for injection into the manager
 * prompt. Returns an empty string when no experiments have run yet.
 *
 * @param {CompactLedger} ledger
 * @param {string=} metricKey  shown in headers/cells; default val_bpb
 * @return {string}
 */
/**
 * Pull the most informative one-liner the manager wrote about this
 * experiment when it dispatched: the explicit `rationale` field, or
 * else the first line of the task body. Keeps the ledger row readable.
 *
 * @param {ExperimentRecord} record
 * @return {string|null}
 */
function frameOf(record) {
  const text = record.rationale || record.taskSummary || null;
  if (!text) return null;
  return text.length > 140 ? `${text.slice(0, 137)}…` : text;
}

export function formatLedgerMarkdown(ledger, metricKey = DEFAULT_METRIC_KEY) {
  if (!ledger) return '';
  const sections = [];
  if (ledger.topK.length > 0) {
    // Top-K is what we want the manager to *exploit* — show its own
    // hypothesis next to each so it can recognize "wider aspect helps"
    // as a real direction instead of a coincidence on a single id.
    sections.push(`> **Top-${ledger.topK.length} by ${metricKey} (lower is better):**`);
    for (const record of ledger.topK) {
      const train = record.trainingSeconds != null
        ? ` (${Math.round(record.trainingSeconds)}s train)`
        : '';
      const frame = frameOf(record);
      const lineage = record.baseRef && record.baseRef !== 'HEAD' ? ` ← ${record.baseRef}` : '';
      const head = `> - ${record.id}: ${metricKey}=${record.metric.toFixed(4)}${train}${lineage}`;
      sections.push(head);
      if (record.editSummary) sections.push(`>     edit: ${record.editSummary}`);
      if (frame) sections.push(`>     hypothesis: ${frame}`);
    }
  }
  if (ledger.running.length > 0) {
    sections.push('', `> **Running (${ledger.running.length}):** ${ledger.running.map((r) => r.id).join(', ')}`);
  }
  if (ledger.recent.length > 0) {
    // Recent is for *learning* — what just happened to inform what
    // to try next. Surface the rationale here too so the manager
    // sees its own framing turn into a result.
    sections.push('', `> **Recent ${Math.min(ledger.recent.length, 8)} (newest first):**`);
    for (const record of ledger.recent.slice(0, 8)) {
      const tag = record.status === 'completed'
        ? `${metricKey}=${record.metric.toFixed(4)}`
        : `failed: ${(record.reason || '').slice(0, 60)}`;
      sections.push(`> - ${record.id} — ${tag}`);
      if (record.editSummary) sections.push(`>     edit: ${record.editSummary}`);
      const frame = frameOf(record);
      if (frame) sections.push(`>     hypothesis: ${frame}`);
    }
  }
  if (ledger.failedClusters.length > 0) {
    sections.push('', `> **Failure clusters (${ledger.totalFailed} total):**`);
    for (const cluster of ledger.failedClusters.slice(0, 5)) {
      const sample = cluster.ids.slice(0, 3).join(', ');
      sections.push(`> - ${cluster.count}× ${cluster.reason}${cluster.ids.length > 3 ? '…' : ''} (e.g. ${sample})`);
    }
  }
  return sections.join('\n');
}
