/**
 * @fileoverview ALPS Phase 4 — quota-based diversity filter.
 *
 * Cap how many same-axis tasks one manager wake-up can dispatch, and
 * collapse exact same-(operator, value) duplicates within a wave.
 * Behind a disabled-by-default flag (`quotaDiversityEnabled`) so the
 * filter is opt-in.
 *
 * Manager priority is preserved within buckets: tasks are sorted
 * priority-first, and overflow is dropped from the lowest-priority
 * same-op tail. Validation tasks (baseline_repeat / head_validation)
 * are exempt by default.
 */

import { extractOperators } from './operator-posterior.js';

/**
 * @typedef {Object} QuotaConfig
 * @property {boolean} quotaDiversityEnabled
 * @property {number} maxPerOperatorPerWake
 * @property {number} maxSameOperatorValuePerWake
 * @property {boolean} alwaysAllowValidation
 *
 * @typedef {Object} QuotaResult
 * @property {Array<Object>} accepted
 * @property {Array<{task: Object, reason: string}>} dropped
 */

function isValidationTask(task) {
  if (!task || typeof task !== 'object') return false;
  if (task.executionMode === 'baseline_repeat') return true;
  if (Array.isArray(task.producesTags) && task.producesTags.some((t) => typeof t === 'string' && t.startsWith('head_validation:'))) {
    return true;
  }
  return false;
}

function pickedFromTask(task) {
  // The quota filter runs before the picked event is written, but the
  // operator-extraction logic only needs the edit array. We pass a
  // shaped object that extractOperators can read.
  return { edits: Array.isArray(task.edits) ? task.edits : [] };
}

/**
 * Apply the quota-diversity filter to a candidate task batch.
 *
 * @param {Array<Object>} tasks  parsed TASK_GRAPH candidates
 * @param {QuotaConfig} config
 * @return {QuotaResult}
 */
export function applyDiversityQuota(tasks, config) {
  if (!config || !config.quotaDiversityEnabled || !Array.isArray(tasks) || tasks.length === 0) {
    return { accepted: tasks || [], dropped: [] };
  }
  const maxPerOp = config.maxPerOperatorPerWake > 0 ? config.maxPerOperatorPerWake : 4;
  const maxValueRepeats = config.maxSameOperatorValuePerWake > 0 ? config.maxSameOperatorValuePerWake : 1;
  const alwaysAllowValidation = config.alwaysAllowValidation !== false;

  // Stable priority sort: highest priority first. Tasks with no priority
  // sort to position-only (use original index as tiebreaker).
  const indexed = tasks.map((t, i) => ({ task: t, index: i }));
  indexed.sort((a, b) => {
    const pa = typeof a.task.priority === 'number' ? a.task.priority : 0;
    const pb = typeof b.task.priority === 'number' ? b.task.priority : 0;
    if (pa !== pb) return pb - pa;
    return a.index - b.index;
  });

  const opCount = new Map();
  const valueCount = new Map();
  const accepted = [];
  const dropped = [];

  for (const { task } of indexed) {
    if (alwaysAllowValidation && isValidationTask(task)) {
      accepted.push(task);
      continue;
    }
    const ops = extractOperators(pickedFromTask(task));
    if (ops.length === 0) {
      // Non-edit tasks (or tasks whose edits we can't parse) pass
      // through. The quota only restricts tasks with structured edits.
      accepted.push(task);
      continue;
    }
    let dropReason = null;
    for (const { operator, value } of ops) {
      const c = opCount.get(operator) || 0;
      if (c >= maxPerOp) { dropReason = `axis_quota:${operator}`; break; }
      if (value != null) {
        const vKey = `${operator}:${value}`;
        const vc = valueCount.get(vKey) || 0;
        if (vc >= maxValueRepeats) { dropReason = `value_repeat:${vKey}`; break; }
      }
    }
    if (dropReason) {
      dropped.push({ task, reason: dropReason });
      continue;
    }
    for (const { operator, value } of ops) {
      opCount.set(operator, (opCount.get(operator) || 0) + 1);
      if (value != null) {
        const vKey = `${operator}:${value}`;
        valueCount.set(vKey, (valueCount.get(vKey) || 0) + 1);
      }
    }
    accepted.push(task);
  }

  // Restore original order among accepted so downstream scheduling
  // (which may rely on emission order for dependency resolution within
  // a wave) sees the same sequence the manager intended, minus drops.
  const acceptedSet = new Set(accepted);
  const inOrder = tasks.filter((t) => acceptedSet.has(t));
  return { accepted: inOrder, dropped };
}
