import fs from 'fs';
import path from 'path';

// Per-task event log written to <projectDir>/task-events/<taskId>.json.
// Each call appends one event; we atomic-rename via a tmp file so concurrent
// writes from different async paths in the same Node process don't tear.
//
// The orchestrator is single-threaded so two writes to the same taskId can
// only interleave at await points, which never happen inside this function.
// That makes the read-mutate-write sequence safe without a mutex.
//
// Event vocabulary (Phase 1, expand in later phases):
//   picked         — task was selected by the scheduler from the ready queue
//   granted        — GPU token grant created
//   worker_spawned — runAgent (or direct executor) about to start an attempt
//   worker_returned— attempt finished; success/failure flags follow
//   validated      — orchestrator parsed result.txt + metrics.json
//   released       — GPU token released; task lifecycle done

const FILENAME_FORBIDDEN = /[^a-zA-Z0-9_.-]/g;

function sanitizeTaskId(taskId) {
  const s = String(taskId || 'unknown');
  return s.replace(FILENAME_FORBIDDEN, '_').slice(0, 200) || 'unknown';
}

export function taskEventsPath(eventsDir, taskId) {
  return path.join(eventsDir, `${sanitizeTaskId(taskId)}.json`);
}

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
    // Tracing must never break the orchestrator; swallow and move on.
  }
}

export function readTaskEvents(eventsDir, taskId) {
  try {
    return JSON.parse(fs.readFileSync(taskEventsPath(eventsDir, taskId), 'utf-8'));
  } catch {
    return null;
  }
}
