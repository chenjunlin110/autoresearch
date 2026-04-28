/**
 * @fileoverview Defensive nvidia-smi probe used after a task is force-killed
 * to detect a process that survived SIGKILL and still owns its CUDA context.
 * If anything is still on the GPU we wait a short window and re-check before
 * the orchestrator reissues the freed token; otherwise the next task lands
 * on a non-empty GPU and OOMs at compile time.
 *
 * The probe is best-effort: nvidia-smi missing or slow returns "no leak"
 * rather than blocking the scheduler.
 */

import { execFileSync } from 'child_process';

const NVIDIA_SMI_TIMEOUT_MS = 3000;

/**
 * Return the list of (gpu_index, pid) pairs currently holding compute
 * contexts. Returns an empty array on any error so callers degrade safely.
 *
 * @return {Array<{gpu: number, pid: number}>}
 */
export function listComputeProcesses() {
  let out;
  try {
    out = execFileSync(
      'nvidia-smi',
      ['--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
      { encoding: 'utf-8', timeout: NVIDIA_SMI_TIMEOUT_MS, stdio: ['ignore', 'pipe', 'ignore'] },
    );
  } catch {
    return [];
  }
  const rows = [];
  for (const raw of out.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    const parts = line.split(',').map((s) => s.trim());
    if (parts.length < 2) continue;
    const pid = Number.parseInt(parts[1], 10);
    if (!Number.isInteger(pid)) continue;
    rows.push({ gpu_uuid: parts[0], pid });
  }
  return rows;
}

/**
 * Wait briefly for `pid` to leave the GPU. Used by the orchestrator after
 * SIGKILL to make sure the process actually released its CUDA context.
 *
 * @param {number} pid
 * @param {Object=} opts
 * @param {number=} opts.maxWaitMs    total wait budget (default 8s)
 * @param {number=} opts.pollEveryMs  poll interval (default 500ms)
 * @return {Promise<{released: boolean, waitedMs: number}>}
 */
export async function waitForGpuRelease(pid, { maxWaitMs = 8000, pollEveryMs = 500 } = {}) {
  if (!Number.isInteger(pid)) return { released: true, waitedMs: 0 };
  const start = Date.now();
  while (Date.now() - start < maxWaitMs) {
    const procs = listComputeProcesses();
    if (!procs.some((p) => p.pid === pid)) {
      return { released: true, waitedMs: Date.now() - start };
    }
    await new Promise((resolve) => setTimeout(resolve, pollEveryMs));
  }
  return { released: false, waitedMs: Date.now() - start };
}
