import { execSync } from 'child_process';

// One-shot lookup of the Slurm allocation's end time. Cached for the lifetime
// of the orchestrator process — Slurm doesn't move the deadline mid-job.
//
// Three sources, in order:
//   1. AUTORESEARCH_SLURM_END_TIME (test override; ISO 8601 or epoch seconds)
//   2. scontrol show job $SLURM_JOB_ID -o → EndTime field
//   3. give up — return 0 (caller treats as "no gate")
//
// The walltime admission gate uses this to refuse tasks that wouldn't finish
// before the allocation expires.

let cachedDeadlineMs = null;

function parseDeadline(raw) {
  if (!raw || typeof raw !== 'string') return 0;
  const trimmed = raw.trim();
  if (!trimmed || trimmed === 'Unknown') return 0;
  // Pure epoch seconds.
  if (/^\d{10}$/.test(trimmed)) return Number.parseInt(trimmed, 10) * 1000;
  // Pure epoch milliseconds.
  if (/^\d{13}$/.test(trimmed)) return Number.parseInt(trimmed, 10);
  // ISO 8601 (Slurm prints e.g. "2026-04-25T22:30:00").
  const ms = Date.parse(trimmed);
  return Number.isFinite(ms) ? ms : 0;
}

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
      timeout: 5000,
      stdio: ['ignore', 'pipe', 'ignore'],
    });
    const m = out.match(/EndTime=(\S+)/);
    cachedDeadlineMs = parseDeadline(m ? m[1] : '');
  } catch {
    cachedDeadlineMs = 0;
  }
  return cachedDeadlineMs;
}

export function remainingWalltimeMs(nowMs = Date.now()) {
  const deadline = getSlurmDeadlineMs();
  if (!deadline) return null;
  return Math.max(0, deadline - nowMs);
}

// Reset the cache. Tests only — production callers should not need this.
export function _resetWalltimeCacheForTests() {
  cachedDeadlineMs = null;
}
