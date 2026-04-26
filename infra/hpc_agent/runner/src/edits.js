/**
 * @fileoverview Deterministic source-edit primitives for the direct executor.
 *
 * The manager LLM emits an `edits[]` array whose items are dispatched here,
 * one of four kinds:
 *
 *   1. constant_replace — module-level Python assignment, AST-normalized
 *      comparison so `(0.9, 0.95)` matches both `( 0.9 , 0.95 )` and any
 *      expression that evaluates to the same Python tuple. Implemented by
 *      shelling out to `edits-ast.py`.
 *   2. regex_replace    — pattern + replacement with a mandatory
 *      `expected_count` to fail loud on drift in the matched text.
 *   3. block_replace    — anchor regex + end regex, replace everything
 *      between (inclusive of the matched lines).
 *   4. unified_diff     — escape hatch; applied via `git apply` against the
 *      working tree.
 *
 * Edits are atomic per file: {@link applyEditsToFile} either writes the
 * fully-edited contents back or leaves the file untouched. Cross-file
 * orchestration (rolling back already-written files) is the executor's job.
 */

import fs from 'fs';
import path from 'path';
import { execFileSync } from 'child_process';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const AST_HELPER_PATH = path.join(HERE, 'edits-ast.py');
const PYTHON_TIMEOUT_MS = 10000;

const EDIT_KINDS = new Set([
  'constant_replace',
  'regex_replace',
  'block_replace',
  'unified_diff',
]);

/**
 * @typedef {Object} EditResult
 * @property {boolean} ok
 * @property {string=} reason          short human-readable failure cause
 * @property {string=} kind            edit kind that ran
 * @property {number=} matchCount      regex_replace / block_replace
 */

/**
 * Apply a list of edits scoped to a single file. Returns once the new
 * contents have been written, or with `ok=false` and the original file
 * unchanged on the first failure.
 *
 * @param {string} filePath           absolute path to the source file
 * @param {Array<Object>} edits       same-file edits in apply order
 * @return {EditResult}
 */
export function applyEditsToFile(filePath, edits) {
  if (!Array.isArray(edits) || edits.length === 0) {
    return { ok: false, reason: 'no edits provided' };
  }
  let src;
  try {
    src = fs.readFileSync(filePath, 'utf-8');
  } catch (e) {
    return { ok: false, reason: `cannot read ${filePath}: ${e.message}` };
  }

  // constant_replace edits to the same file are passed in one batch to the
  // Python helper so it can splice them all atomically. Other kinds run
  // sequentially against the in-memory buffer.
  let cursor = 0;
  let current = src;
  while (cursor < edits.length) {
    const edit = edits[cursor];
    if (!edit || typeof edit !== 'object' || !EDIT_KINDS.has(edit.kind)) {
      return { ok: false, reason: `unknown edit kind: ${edit?.kind}` };
    }
    if (edit.kind === 'constant_replace') {
      const batch = [];
      while (cursor < edits.length && edits[cursor]?.kind === 'constant_replace') {
        batch.push(edits[cursor]);
        cursor += 1;
      }
      const result = applyConstantReplaceBatch(current, batch);
      if (!result.ok) return result;
      current = result.newSrc;
      continue;
    }
    let result;
    if (edit.kind === 'regex_replace') {
      result = applyRegexReplace(current, edit);
    } else if (edit.kind === 'block_replace') {
      result = applyBlockReplace(current, edit);
    } else if (edit.kind === 'unified_diff') {
      // Unified diffs touch the working tree directly via `git apply`, so
      // they bypass the in-memory buffer. We flush whatever's in `current`
      // first, then apply the patch from disk.
      try {
        fs.writeFileSync(filePath, current);
      } catch (e) {
        return { ok: false, reason: `failed flushing edits: ${e.message}` };
      }
      result = applyUnifiedDiff(filePath, edit);
      if (result.ok) {
        try { current = fs.readFileSync(filePath, 'utf-8'); }
        catch (e) { return { ok: false, reason: `cannot re-read after diff: ${e.message}` }; }
      }
    }
    if (!result.ok) return result;
    if (result.newSrc != null) current = result.newSrc;
    cursor += 1;
  }

  try {
    fs.writeFileSync(filePath, current);
  } catch (e) {
    return { ok: false, reason: `cannot write ${filePath}: ${e.message}` };
  }
  return { ok: true };
}

/**
 * Run all `constant_replace` edits for a file as a single Python pass. The
 * helper either returns the fully-rewritten text or refuses and the file
 * is left untouched.
 *
 * @param {string} src
 * @param {Array<Object>} edits     all known to be constant_replace
 * @return {{ok: boolean, newSrc?: string, reason?: string}}
 */
function applyConstantReplaceBatch(src, edits) {
  const request = {
    src,
    edits: edits.map((edit) => ({
      name: edit.name,
      expected_old_repr: edit.expected_old_repr,
      new_repr: edit.new_repr,
    })),
  };
  let stdout;
  try {
    stdout = execFileSync('python3', [AST_HELPER_PATH], {
      input: JSON.stringify(request),
      encoding: 'utf-8',
      timeout: PYTHON_TIMEOUT_MS,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
  } catch (e) {
    return { ok: false, reason: `constant_replace helper failed: ${e.message}` };
  }
  let parsed;
  try {
    parsed = JSON.parse(stdout);
  } catch (e) {
    return { ok: false, reason: `constant_replace helper produced bad JSON: ${e.message}` };
  }
  if (!parsed.ok) {
    return { ok: false, reason: `constant_replace: ${parsed.reason}` };
  }
  return { ok: true, newSrc: parsed.new_src };
}

/**
 * `regex_replace` requires an explicit `expected_count` so a typo in the
 * pattern fails loudly instead of silently leaving the file unchanged.
 *
 * @param {string} src
 * @param {Object} edit
 * @return {{ok: boolean, newSrc?: string, reason?: string, matchCount?: number}}
 */
function applyRegexReplace(src, edit) {
  const { pattern, replacement, expected_count: expectedCount } = edit;
  if (typeof pattern !== 'string' || typeof replacement !== 'string') {
    return { ok: false, reason: 'regex_replace requires string pattern and replacement' };
  }
  if (!Number.isInteger(expectedCount) || expectedCount < 1) {
    return { ok: false, reason: 'regex_replace requires expected_count >= 1' };
  }
  let regex;
  try {
    regex = new RegExp(pattern, edit.flags || 'g');
  } catch (e) {
    return { ok: false, reason: `regex_replace pattern is invalid: ${e.message}` };
  }
  if (!regex.global) {
    return { ok: false, reason: 'regex_replace pattern must be global (use flags: "g")' };
  }
  const matches = src.match(regex) || [];
  if (matches.length !== expectedCount) {
    return {
      ok: false,
      reason: `regex_replace expected ${expectedCount} match(es), found ${matches.length}`,
      matchCount: matches.length,
    };
  }
  return { ok: true, newSrc: src.replace(regex, replacement), matchCount: matches.length };
}

/**
 * `block_replace` finds the first `anchor_regex` match, then the first
 * `end_regex` match starting after it, and replaces everything between
 * (inclusive of the matched lines) with `new_text`.
 *
 * @param {string} src
 * @param {Object} edit
 * @return {{ok: boolean, newSrc?: string, reason?: string}}
 */
function applyBlockReplace(src, edit) {
  const { anchor_regex: anchor, end_regex: end, new_text: newText } = edit;
  if (typeof anchor !== 'string' || typeof end !== 'string' || typeof newText !== 'string') {
    return { ok: false, reason: 'block_replace requires anchor_regex, end_regex, new_text' };
  }
  let anchorRe;
  let endRe;
  try {
    anchorRe = new RegExp(anchor, 'm');
    endRe = new RegExp(end, 'm');
  } catch (e) {
    return { ok: false, reason: `block_replace regex invalid: ${e.message}` };
  }
  const startMatch = src.match(anchorRe);
  if (!startMatch) return { ok: false, reason: 'block_replace anchor not found' };
  const startIdx = startMatch.index;
  const tail = src.slice(startIdx + startMatch[0].length);
  const endMatch = tail.match(endRe);
  if (!endMatch) return { ok: false, reason: 'block_replace end_regex not found after anchor' };
  const endIdx = startIdx + startMatch[0].length + endMatch.index + endMatch[0].length;
  return { ok: true, newSrc: src.slice(0, startIdx) + newText + src.slice(endIdx) };
}

/**
 * Apply a unified-diff blob to the working tree via `git apply`. The diff
 * must be valid against the current file contents.
 *
 * @param {string} filePath          file the diff touches; used as cwd hint
 * @param {Object} edit
 * @return {{ok: boolean, reason?: string}}
 */
function applyUnifiedDiff(filePath, edit) {
  const diff = edit.diff;
  if (typeof diff !== 'string' || !diff.trim()) {
    return { ok: false, reason: 'unified_diff requires non-empty diff text' };
  }
  const cwd = path.dirname(filePath);
  try {
    execFileSync('git', ['apply', '--unidiff-zero', '--whitespace=nowarn', '-'], {
      input: diff,
      cwd,
      timeout: PYTHON_TIMEOUT_MS,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
  } catch (e) {
    return { ok: false, reason: `git apply failed: ${e.message}` };
  }
  return { ok: true };
}

/**
 * Group edits by file (preserving order within a file) and apply each
 * group atomically. On any failure, files already written remain edited;
 * the caller is expected to roll the working tree back via git.
 *
 * @param {string} workTreeRoot   absolute path; edits resolve relative to here
 * @param {Array<Object>} edits   each must have `file` (relative path)
 * @return {{ok: boolean, applied: number, reason?: string, failedAt?: number}}
 */
export function applyEdits(workTreeRoot, edits) {
  if (!Array.isArray(edits) || edits.length === 0) {
    return { ok: false, applied: 0, reason: 'no edits provided' };
  }
  const groups = new Map();
  edits.forEach((edit, index) => {
    if (!edit || typeof edit.file !== 'string' || !edit.file.trim()) {
      throw new Error(`edit #${index} missing 'file'`);
    }
    if (!groups.has(edit.file)) groups.set(edit.file, []);
    groups.get(edit.file).push({ edit, index });
  });
  let applied = 0;
  for (const [file, items] of groups) {
    const target = path.resolve(workTreeRoot, file);
    if (!target.startsWith(path.resolve(workTreeRoot) + path.sep)
        && target !== path.resolve(workTreeRoot)) {
      return {
        ok: false, applied, failedAt: items[0].index,
        reason: `edit target ${file} escapes the work tree`,
      };
    }
    const result = applyEditsToFile(target, items.map((item) => item.edit));
    if (!result.ok) {
      return { ok: false, applied, failedAt: items[0].index, reason: result.reason };
    }
    applied += items.length;
  }
  return { ok: true, applied };
}
