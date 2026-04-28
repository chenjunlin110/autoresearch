/**
 * @fileoverview Canonical truth source for one task run's outcome.
 *
 * Every task plugin's wrapper writes two artifacts the framework can read:
 *   - `result.txt`   — at minimum `exit_code=<int>`. Optional: `output_dir=`,
 *     `metrics_path=`.
 *   - `metrics.json` — at minimum the metric the manager optimizes (a finite
 *     number). The autoresearch task uses `val_bpb`; the
 *     {@link DEFAULT_METRIC_KEY} default mirrors that. Other tasks pass their
 *     own key in `metricKey`.
 *
 * This module decides "did the run actually succeed" from disk, NOT from the
 * worker LLM's natural-language claim. Worker stdout is debug only.
 *
 * Phase 1 uses this only to LOG mismatch with the LLM's claim
 * (instrumentation, not enforcement). Phase 3+ swaps the orchestrator's
 * source of truth to here.
 */

import fs from 'fs';
import path from 'path';

/** Default metric key — preserves the autoresearch convention. */
const DEFAULT_METRIC_KEY = 'val_bpb';

/**
 * @typedef {Object} ResultTxt
 * @property {number} exit_code
 * @property {string=} output_dir
 * @property {string=} metrics_path
 */

/**
 * @typedef {Object} ValidationVerdict
 * @property {boolean} success
 * @property {string=} reason
 * @property {number=} exitCode
 * @property {Record<string, unknown>=} metrics
 */

/**
 * Parse `result.txt` body (key=value lines). Rejects content that lacks an
 * integer `exit_code=` line.
 * @param {string} text raw file contents
 * @return {ResultTxt|null}
 */
function parseResultTxt(text) {
  if (typeof text !== 'string' || !text.trim()) return null;
  const out = {};
  for (const line of text.split(/\r?\n/)) {
    const match = line.match(/^([a-zA-Z_][a-zA-Z0-9_]*)=(.*)$/);
    if (!match) continue;
    out[match[1]] = match[2];
  }
  if (!('exit_code' in out)) return null;
  const exitCode = Number.parseInt(out.exit_code, 10);
  if (!Number.isInteger(exitCode)) return null;
  return {
    exit_code: exitCode,
    output_dir: out.output_dir,
    metrics_path: out.metrics_path,
  };
}

/**
 * Read + parse a JSON file; returns `null` on any error so callers can
 * decide what failure means without try/catch.
 * @param {string} filePath
 * @return {Record<string, unknown>|null}
 */
function readJsonFile(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  } catch {
    return null;
  }
}

/**
 * Read `<outputDir>/result.txt`.
 * @param {string} outputDir
 * @return {ResultTxt|null}
 */
export function readResultTxt(outputDir) {
  if (!outputDir) return null;
  const filePath = path.join(outputDir, 'result.txt');
  let text;
  try {
    text = fs.readFileSync(filePath, 'utf-8');
  } catch {
    return null;
  }
  return parseResultTxt(text);
}

/**
 * Read `metrics.json` either from `metricsPathOverride` (when provided) or
 * `<outputDir>/metrics.json`.
 * @param {string} outputDir
 * @param {string|null=} metricsPathOverride absolute path; takes precedence
 * @return {Record<string, unknown>|null}
 */
export function readMetricsJson(outputDir, metricsPathOverride = null) {
  const filePath = metricsPathOverride
    || (outputDir ? path.join(outputDir, 'metrics.json') : null);
  if (!filePath) return null;
  return readJsonFile(filePath);
}

/**
 * Decide whether the run at `outputDir` succeeded by inspecting the wrapper
 * artifacts on disk. The contract is:
 *   1. `result.txt` must exist and parse, with `exit_code = 0`.
 *   2. `metrics.json` must exist and contain a finite numeric value at
 *      `metricKey` (default `val_bpb` to preserve autoresearch behavior).
 *
 * @param {Object=} args
 * @param {string=} args.outputDir directory the wrapper wrote to
 * @param {string|null=} args.metricsPath absolute override for metrics.json
 * @param {string=} args.metricKey JSON key on metrics.json holding the optimized number
 * @return {ValidationVerdict}
 */
export function validateExperimentResult({
  outputDir,
  metricsPath = null,
  metricKey = DEFAULT_METRIC_KEY,
} = {}) {
  if (!outputDir) {
    return { success: false, reason: 'no output_dir provided' };
  }
  const result = readResultTxt(outputDir);
  if (!result) {
    return { success: false, reason: 'missing or unparseable result.txt' };
  }
  if (result.exit_code !== 0) {
    return {
      success: false,
      reason: `train wrapper exited non-zero: exit_code=${result.exit_code}`,
      exitCode: result.exit_code,
    };
  }
  const metrics = readMetricsJson(outputDir, metricsPath || result.metrics_path || null);
  if (!metrics || typeof metrics !== 'object') {
    return { success: false, reason: 'missing or unparseable metrics.json', exitCode: 0 };
  }
  const value = metrics[metricKey];
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return {
      success: false,
      reason: `metrics.${metricKey} is not a finite number (got ${JSON.stringify(value)})`,
      exitCode: 0,
      metrics,
    };
  }
  return { success: true, exitCode: 0, metrics };
}

/**
 * Best-effort extraction of the experiment output directory from a worker
 * task body. Manager prompts conventionally say "Output directory:
 * experiments/<id>" but format varies. Returns the path (relative or
 * absolute), or `null` if we cannot find one.
 *
 * The orchestrator passes the result to {@link validateExperimentResult}
 * after the worker LLM returns; on `null` it records a
 * `validate_skipped: no_output_dir` event rather than a false success.
 *
 * Phase 3 replaces this with an explicit `outputDir` in the TASK_GRAPH
 * schema; Phase 1 has to extract from natural language.
 *
 * @param {string|null|undefined} taskBody
 * @return {string|null}
 */
export function extractOutputDirFromTaskBody(taskBody) {
  if (typeof taskBody !== 'string' || !taskBody) return null;
  const labeled = taskBody.match(/Output\s+directory[:\s]+([^\s'"`,]+)/i)
    || taskBody.match(/output_dir[:\s=]+([^\s'"`,]+)/i)
    || taskBody.match(/outputs?\s+under[:\s]+([^\s'"`,]+)/i);
  if (labeled) return labeled[1].replace(/[.,;]+$/, '');
  const inline = taskBody.match(/\b(experiments\/[A-Za-z0-9_./-]+)/);
  if (inline) return inline[1].replace(/[.,;]+$/, '');
  return null;
}
