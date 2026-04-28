/**
 * @fileoverview KEEP_EXPERIMENT mechanic: when the manager anoints a
 * winner, fast-forward `tasks/<task>/source/`'s HEAD by re-applying the
 * winner's structured edits and committing them. Subsequent
 * `param_patch` tasks with `base_ref: "HEAD"` then start from this new
 * baseline — so wins compound across the parallel search instead of
 * each new experiment forking from the same untouched root.
 *
 * This is what closes the gap to Karpathy's serial agent: the
 * cumulative branching primitive ("git advance on win") expressed as a
 * one-shot manager directive that the framework executes.
 *
 * Race-condition note: experiments already in flight when a KEEP fires
 * have stale sandboxes (cloned from the pre-KEEP HEAD). They still
 * complete and report metrics; the manager decides whether to KEEP
 * those next, KILL_TASKS them mid-flight, or ignore them. Lineage is
 * a manager decision, not a timing artifact.
 */

import fs from 'fs';
import path from 'path';
import { execFileSync } from 'child_process';
import { applyEdits } from './edits.js';

// 120s rather than 30s because `git add -A && git commit` on the task's
// source/ runs against a network filesystem (weka) where directory walks
// can be slow even on a small repo. We saw 30s timeouts on KEEP_EXPERIMENT
// for the autoresearch source/ during the 1580312 verification run.
const GIT_TIMEOUT_MS = 120000;

/**
 * @typedef {Object} LineageEntry
 * @property {string} experiment_id
 * @property {string} commit          SHA of source/ HEAD after this keep
 * @property {string} reason          manager-supplied rationale
 * @property {string} kept_at         ISO timestamp
 * @property {Array<Object>} edits    raw edit specs that were applied
 *
 * @typedef {Object} Lineage
 * @property {string} task
 * @property {Array<LineageEntry>} entries
 * @property {string|null} current_head_commit
 */

function gitText(args, cwd) {
  return execFileSync('git', args, {
    cwd,
    encoding: 'utf-8',
    timeout: GIT_TIMEOUT_MS,
    stdio: ['ignore', 'pipe', 'pipe'],
  }).trim();
}

function gitSilent(args, cwd) {
  execFileSync('git', args, {
    cwd,
    timeout: GIT_TIMEOUT_MS,
    stdio: ['ignore', 'ignore', 'pipe'],
  });
}

/**
 * Read the lineage file; returns the empty shape when missing.
 * @param {string} lineagePath
 * @return {Lineage}
 */
export function readLineage(lineagePath) {
  const empty = {
    task: null,
    entries: [],
    current_head_commit: null,
    // ALPS HEAD-metric tracking. metric_known=false disables the
    // commit gate until a baseline_repeat (or no-edit experiment)
    // re-establishes the metric.
    current_head_metric_known: false,
    current_head_metric: null,
    current_head_metric_source_experiment: null,
  };
  if (!lineagePath) return empty;
  try {
    const raw = fs.readFileSync(lineagePath, 'utf-8');
    const doc = JSON.parse(raw);
    if (!doc || typeof doc !== 'object') return empty;
    return {
      task: doc.task ?? null,
      entries: Array.isArray(doc.entries) ? doc.entries : [],
      current_head_commit: doc.current_head_commit ?? null,
      current_head_metric_known: doc.current_head_metric_known === true,
      current_head_metric: typeof doc.current_head_metric === 'number'
        && Number.isFinite(doc.current_head_metric) ? doc.current_head_metric : null,
      current_head_metric_source_experiment: typeof doc.current_head_metric_source_experiment === 'string'
        ? doc.current_head_metric_source_experiment : null,
    };
  } catch {
    return empty;
  }
}

export function writeLineage(lineagePath, doc) {
  fs.mkdirSync(path.dirname(lineagePath), { recursive: true });
  const tmp = `${lineagePath}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(doc, null, 2) + '\n');
  fs.renameSync(tmp, lineagePath);
}

/**
 * Update HEAD-metric tracking on the lineage file without touching
 * entries. Used by the framework's head-state setter so
 * `currentHeadMetricKnown` survives a runner restart.
 *
 * @param {string} lineagePath
 * @param {{
 *   commit?: string|null,
 *   metricKnown: boolean,
 *   metric?: number|null,
 *   sourceExperiment?: string|null,
 * }} headState
 */
export function persistHeadState(lineagePath, headState) {
  if (!lineagePath) return;
  const lineage = readLineage(lineagePath);
  if (headState.commit !== undefined) {
    lineage.current_head_commit = headState.commit;
  }
  lineage.current_head_metric_known = headState.metricKnown === true;
  lineage.current_head_metric = lineage.current_head_metric_known
    ? (typeof headState.metric === 'number' ? headState.metric : null)
    : null;
  lineage.current_head_metric_source_experiment = lineage.current_head_metric_known
    ? (typeof headState.sourceExperiment === 'string' ? headState.sourceExperiment : null)
    : null;
  writeLineage(lineagePath, lineage);
}

/**
 * Compare two structured edit arrays for exact equality. Used by the
 * `evaluateKeepMetricValidity` helper in server.js to confirm the edits
 * we just applied match what the candidate ran with — necessary
 * condition for transferring the candidate's metric to the new HEAD.
 *
 * Compares by edit kind and kind-relevant fields; ignores extra fields
 * the parser may have added.
 *
 * @param {Array<Object>|null|undefined} a
 * @param {Array<Object>|null|undefined} b
 * @return {boolean}
 */
export function editsEqual(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b)) return false;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    const ea = a[i];
    const eb = b[i];
    if (!ea || !eb || typeof ea !== 'object' || typeof eb !== 'object') return false;
    if (ea.file !== eb.file) return false;
    if (ea.kind !== eb.kind) return false;
    if (ea.kind === 'constant_replace') {
      if (ea.name !== eb.name) return false;
      if (ea.expected_old_repr !== eb.expected_old_repr) return false;
      if (ea.new_repr !== eb.new_repr) return false;
    } else if (ea.kind === 'regex_replace') {
      if (ea.pattern !== eb.pattern) return false;
      if (ea.replacement !== eb.replacement) return false;
      if ((ea.expected_count ?? null) !== (eb.expected_count ?? null)) return false;
    } else if (ea.kind === 'block_replace') {
      if (ea.anchor_regex !== eb.anchor_regex) return false;
      if (ea.end_regex !== eb.end_regex) return false;
      if (ea.new_text !== eb.new_text) return false;
    } else if (ea.kind === 'unified_diff') {
      if (ea.diff !== eb.diff) return false;
    }
  }
  return true;
}

/**
 * Apply an experiment's edits to source/, commit, and append a lineage
 * entry. Returns `{ok, commit?, reason?}`. Idempotent on
 * experiment_id — a repeat keep is a no-op (already in lineage).
 *
 * @param {Object} args
 * @param {string} args.sourceRepoPath  task plugin's source/ git repo
 * @param {string} args.lineagePath     where lineage.json lives
 * @param {string} args.taskName        for the lineage doc header
 * @param {string} args.experimentId
 * @param {string=} args.reason
 * @param {Array<Object>} args.edits    raw edit specs from the picked event
 * @return {{ok: boolean, commit?: string, reason?: string, alreadyKept?: boolean}}
 */
export function applyKeepExperiment({
  sourceRepoPath,
  lineagePath,
  taskName,
  experimentId,
  reason = '',
  edits,
}) {
  if (!sourceRepoPath || !lineagePath || !experimentId) {
    return { ok: false, reason: 'missing required argument' };
  }
  if (!Array.isArray(edits) || edits.length === 0) {
    return { ok: false, reason: 'no edits to apply (was the experiment a param_patch?)' };
  }

  const lineage = readLineage(lineagePath);
  if (lineage.entries.some((e) => e.experiment_id === experimentId)) {
    return { ok: true, alreadyKept: true, commit: lineage.current_head_commit };
  }

  // Apply the edits in-place on source/. applyEdits is per-file atomic.
  const editResult = applyEdits(sourceRepoPath, edits);
  if (!editResult.ok) {
    return { ok: false, reason: `edits failed against source/: ${editResult.reason}` };
  }

  let commit;
  try {
    gitSilent(['add', '-A'], sourceRepoPath);
    const status = gitText(['status', '--porcelain'], sourceRepoPath);
    if (!status) {
      return { ok: false, reason: 'edits applied but produced no diff against source/' };
    }
    const message = `KEEP ${experimentId}` + (reason ? `: ${reason}` : '');
    gitSilent([
      '-c', 'user.name=lineage-keeper',
      '-c', 'user.email=lineage@example.com',
      'commit', '-q', '-m', message,
    ], sourceRepoPath);
    commit = gitText(['rev-parse', 'HEAD'], sourceRepoPath);
  } catch (e) {
    return { ok: false, reason: `git commit on source/ failed: ${e.message}` };
  }

  const entry = {
    experiment_id: experimentId,
    commit,
    reason,
    kept_at: new Date().toISOString(),
    edits,
  };
  lineage.entries.push(entry);
  lineage.current_head_commit = commit;
  if (!lineage.task) lineage.task = taskName || null;
  writeLineage(lineagePath, lineage);

  return { ok: true, commit };
}

/**
 * Render the lineage as a compact markdown block for injection into the
 * manager's per-cycle context. Returns empty string when nothing has
 * been kept yet.
 *
 * @param {Lineage} lineage
 * @return {string}
 */
export function formatLineageMarkdown(lineage) {
  if (!lineage || !Array.isArray(lineage.entries) || lineage.entries.length === 0) {
    return '';
  }
  const lines = ['> **Kept lineage** (each row is the cumulative state of `source/`; new `param_patch` tasks build on the latest):'];
  for (const entry of lineage.entries) {
    const reason = entry.reason ? ` — ${entry.reason}` : '';
    const sha = (entry.commit || '').slice(0, 7);
    lines.push(`> - ${entry.experiment_id} (${sha})${reason}`);
  }
  if (lineage.current_head_commit) {
    lines.push(`> current source/HEAD = ${lineage.current_head_commit.slice(0, 7)}; \`base_ref: "HEAD"\` starts here.`);
  }
  return lines.join('\n');
}
