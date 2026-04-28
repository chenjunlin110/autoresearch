/**
 * @fileoverview Per-task event log written to
 * `<projectDir>/task-events/<taskId>.json`. Each lifecycle transition
 * appends one event via {@link appendTaskEvent}. The downstream
 * `cycle-report.js` aggregator reads these files at the end of a manager
 * cycle to compute per-cycle telemetry (GPU duty cycle, time buckets,
 * mismatch counts).
 *
 * Event vocabulary (Phase 1, expand in later phases):
 *   - `picked` — task was selected by the scheduler from the ready queue
 *   - `granted` — GPU token grant created
 *   - `worker_spawned` — runAgent (or direct executor) about to start an attempt
 *   - `worker_returned` — attempt finished; success/failure flags follow
 *   - `validated` — orchestrator parsed result.txt + metrics.json
 *   - `released` — GPU token released; task lifecycle done
 *
 * Concurrency: each call performs a read-mutate-write on the per-task JSON
 * via an atomic tmp+rename. The orchestrator is single-threaded so two
 * writes to the same taskId can only interleave at await points; this
 * function performs no awaits, so the sequence is safe without a mutex.
 *
 * Errors: swallow silently. Tracing must never break the orchestrator.
 */

import fs from 'fs';
import path from 'path';

const FILENAME_FORBIDDEN = /[^a-zA-Z0-9_.-]/g;
const MAX_TASK_ID_LENGTH = 200;

/**
 * Reduce an arbitrary taskId to a safe filesystem-safe segment.
 * @param {string|number|null|undefined} taskId
 * @return {string} sanitized name; never empty, never contains path separators
 */
function sanitizeTaskId(taskId) {
  const raw = String(taskId || 'unknown');
  return raw.replace(FILENAME_FORBIDDEN, '_').slice(0, MAX_TASK_ID_LENGTH) || 'unknown';
}

/**
 * Resolve the JSON file path for a given task's event log.
 * @param {string} eventsDir
 * @param {string|number} taskId
 * @return {string} absolute or relative path joining `eventsDir` and `<taskId>.json`
 */
export function taskEventsPath(eventsDir, taskId) {
  return path.join(eventsDir, `${sanitizeTaskId(taskId)}.json`);
}

/**
 * Append one event to the task's JSON log. No-op if any required argument
 * is missing or the underlying filesystem operation fails — tracing is
 * advisory and must never break the orchestrator.
 *
 * @param {string} eventsDir directory holding `<taskId>.json` files
 * @param {string|number} taskId identifier; sanitized into a filename
 * @param {string} eventName event vocabulary name (see file header)
 * @param {Record<string, unknown>=} extra additional fields merged into the event
 */
export function appendTaskEvent(eventsDir, taskId, eventName, extra = {}) {
  if (!eventsDir || !taskId || !eventName) return;
  try {
    fs.mkdirSync(eventsDir, { recursive: true });
    const file = taskEventsPath(eventsDir, taskId);
    let doc;
    try {
      doc = JSON.parse(fs.readFileSync(file, 'utf-8'));
      if (!doc || typeof doc !== 'object' || !Array.isArray(doc.events)) {
        doc = { task_id: String(taskId), events: [] };
      }
    } catch {
      doc = { task_id: String(taskId), events: [] };
    }
    doc.events.push({ name: eventName, t: Date.now() / 1000, ...extra });
    const tmp = `${file}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify(doc, null, 2));
    fs.renameSync(tmp, file);
  } catch {
    // Swallow: tracing must never break the orchestrator.
  }
}

/**
 * Load the full event document for a task.
 * @param {string} eventsDir
 * @param {string|number} taskId
 * @return {{task_id: string, events: Array<Record<string, unknown>>}|null}
 */
export function readTaskEvents(eventsDir, taskId) {
  try {
    return JSON.parse(fs.readFileSync(taskEventsPath(eventsDir, taskId), 'utf-8'));
  } catch {
    return null;
  }
}
