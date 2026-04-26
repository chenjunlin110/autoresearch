/**
 * @fileoverview One-shot lookup of the Slurm allocation's end time, cached
 * for the lifetime of the orchestrator process. Slurm doesn't move the
 * deadline mid-job, so caching is safe; callers can use
 * {@link _resetWalltimeCacheForTests} from unit tests.
 *
 * Resolution order:
 *   1. `AUTORESEARCH_SLURM_END_TIME` env override (ISO 8601 or epoch
 *      seconds; intended for tests and headless re-runs).
 *   2. `scontrol show job $SLURM_JOB_ID -o` → `EndTime=` field.
 *   3. Give up — return `0`. Callers treat that as "no gate".
 *
 * Used by the walltime admission gate in `_executeTaskGraph` to reject
 * tasks whose estimated runtime would exceed the remaining allocation.
 */

import { execSync } from 'child_process';

const SCONTROL_TIMEOUT_MS = 5000;

let cachedDeadlineMs = null;

/**
 * Coerce a raw deadline string into epoch milliseconds. Accepts ISO 8601,
 * epoch seconds (10 digits), and epoch milliseconds (13 digits). Returns
 * 0 on any unrecognized form so callers degrade to "no gate".
 *
 * @param {string|null|undefined} raw
 * @return {number} epoch millis, or 0 when unparseable
 */
function parseDeadline(raw) {
  if (!raw || typeof raw !== 'string') return 0;
  const trimmed = raw.trim();
  if (!trimmed || trimmed === 'Unknown') return 0;
  if (/^\d{10}$/.test(trimmed)) return Number.parseInt(trimmed, 10) * 1000;
  if (/^\d{13}$/.test(trimmed)) return Number.parseInt(trimmed, 10);
  const ms = Date.parse(trimmed);
  return Number.isFinite(ms) ? ms : 0;
}

/**
 * Return the Slurm allocation deadline as epoch millis, cached after the
 * first call. Returns `0` when no source is available — callers should
 * treat that as "no walltime gate active".
 *
 * @return {number}
 */
export function getSlurmDeadlineMs() {
  if (cachedDeadlineMs !== null) return cachedDeadlineMs;

  const override = process.env.AUTORESEARCH_SLURM_END_TIME;
  if (override) {
    cachedDeadlineMs = parseDeadline(override);
    return cachedDeadlineMs;
  }

  const jobId = process.env.SLURM_JOB_ID;
  if (!jobId) {
    cachedDeadlineMs = 0;
    return 0;
  }
  try {
    const out = execSync(`scontrol show job ${jobId} -o`, {
      encoding: 'utf-8',
      timeout: SCONTROL_TIMEOUT_MS,
      stdio: ['ignore', 'pipe', 'ignore'],
    });
    const match = out.match(/EndTime=(\S+)/);
    cachedDeadlineMs = parseDeadline(match ? match[1] : '');
  } catch {
    cachedDeadlineMs = 0;
  }
  return cachedDeadlineMs;
}

/**
 * Milliseconds remaining before the allocation deadline. Returns `null`
 * when no deadline is known so callers can short-circuit the gate. Clamps
 * to 0 once the deadline has passed.
 *
 * @param {number=} nowMs override for the current time, useful in tests
 * @return {number|null}
 */
export function remainingWalltimeMs(nowMs = Date.now()) {
  const deadline = getSlurmDeadlineMs();
  if (!deadline) return null;
  return Math.max(0, deadline - nowMs);
}

/**
 * Reset the deadline cache. Test-only; production callers should not
 * need this. Exported with a leading underscore so the convention reads
 * "internal-but-importable" at call sites.
 */
export function _resetWalltimeCacheForTests() {
  cachedDeadlineMs = null;
}
