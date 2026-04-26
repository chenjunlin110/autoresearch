import fs from 'fs';
import path from 'path';

// Canonical truth source for an autoresearch experiment outcome.
//
// The bash wrapper writes:
//   result.txt   : exit_code=<int>\noutput_dir=<path>\nmetrics_path=<path>
//   metrics.json : { val_bpb, training_seconds, num_steps, peak_vram_mb, ... }
//
// This module decides "did the experiment actually succeed" from disk, NOT from
// the worker LLM's natural-language claim. Worker stdout is debug only.
//
// Contract:
//   validateExperimentResult({ outputDir, metricsPath? })
//     -> { success: boolean, reason?: string, metrics?: object, exitCode?: number }
//
// Phase 1 uses this only to LOG mismatch with the LLM's claim (instrumentation,
// not enforcement). Phase 3+ swaps the orchestrator's source of truth to here.

function parseResultTxt(text) {
  if (typeof text !== 'string' || !text.trim()) return null;
  const out = {};
  for (const line of text.split(/\r?\n/)) {
    const m = line.match(/^([a-zA-Z_][a-zA-Z0-9_]*)=(.*)$/);
    if (!m) continue;
    out[m[1]] = m[2];
  }
  if (!('exit_code' in out)) return null;
  const exitCode = Number.parseInt(out.exit_code, 10);
  if (!Number.isInteger(exitCode)) return null;
  return { exit_code: exitCode, output_dir: out.output_dir, metrics_path: out.metrics_path };
}

function readJsonFile(p) {
  try {
    return JSON.parse(fs.readFileSync(p, 'utf-8'));
  } catch {
    return null;
  }
}

export function readResultTxt(outputDir) {
  if (!outputDir) return null;
  const p = path.join(outputDir, 'result.txt');
  let text;
  try { text = fs.readFileSync(p, 'utf-8'); } catch { return null; }
  return parseResultTxt(text);
}

export function readMetricsJson(outputDir, metricsPathOverride = null) {
  const p = metricsPathOverride || (outputDir ? path.join(outputDir, 'metrics.json') : null);
  if (!p) return null;
  return readJsonFile(p);
}

export function validateExperimentResult({ outputDir, metricsPath = null } = {}) {
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
  if (typeof metrics.val_bpb !== 'number' || !Number.isFinite(metrics.val_bpb)) {
    return {
      success: false,
      reason: `metrics.val_bpb is not a finite number (got ${JSON.stringify(metrics.val_bpb)})`,
      exitCode: 0,
      metrics,
    };
  }
  return { success: true, exitCode: 0, metrics };
}

// Best-effort extraction of the experiment output directory from a worker
// task body. Manager prompts say "Output directory: experiments/<id>" but
// formats vary. Returns the experiment dir path (relative or absolute), or
// null if we can't find it. The orchestrator will pass this to the
// validator after the worker LLM returns; if it's null we record a
// `validate_skipped: no_output_dir` event rather than a false-success.
//
// Phase 3 replaces this with an explicit `outputDir` in the TASK_GRAPH
// schema, but Phase 1 has to extract from natural language.
export function extractOutputDirFromTaskBody(taskBody) {
  if (typeof taskBody !== 'string' || !taskBody) return null;
  // Most explicit form first.
  const labeled = taskBody.match(/Output\s+directory[:\s]+([^\s'"`,]+)/i)
    || taskBody.match(/output_dir[:\s=]+([^\s'"`,]+)/i)
    || taskBody.match(/outputs?\s+under[:\s]+([^\s'"`,]+)/i);
  if (labeled) return labeled[1].replace(/[.,;]+$/, '');
  // Fallback: any "experiments/<token>" path that doesn't include a glob.
  const inline = taskBody.match(/\b(experiments\/[A-Za-z0-9_./-]+)/);
  if (inline) return inline[1].replace(/[.,;]+$/, '');
  return null;
}
