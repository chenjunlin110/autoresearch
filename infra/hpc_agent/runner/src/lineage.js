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

const GIT_TIMEOUT_MS = 30000;

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
  if (!lineagePath) return { task: null, entries: [], current_head_commit: null };
  try {
    const raw = fs.readFileSync(lineagePath, 'utf-8');
    const doc = JSON.parse(raw);
    if (!doc || typeof doc !== 'object') return { task: null, entries: [], current_head_commit: null };
    return {
      task: doc.task ?? null,
      entries: Array.isArray(doc.entries) ? doc.entries : [],
      current_head_commit: doc.current_head_commit ?? null,
    };
  } catch {
    return { task: null, entries: [], current_head_commit: null };
  }
}

function writeLineage(lineagePath, doc) {
  fs.mkdirSync(path.dirname(lineagePath), { recursive: true });
  const tmp = `${lineagePath}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(doc, null, 2) + '\n');
  fs.renameSync(tmp, lineagePath);
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
