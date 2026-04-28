/**
 * @fileoverview The non-LLM execution path for tasks the manager has marked
 * `execution_mode: "param_patch"`. The flow is:
 *
 *   1. {@link prepareDirectSandbox}: clone the source repo to a fresh
 *      per-task sandbox, hard-checkout the resolved parent SHA, apply the
 *      manager-provided `edits[]` atomically, commit, and confirm the
 *      working tree still parses cleanly.
 *   2. {@link runDirectWorker}: spawn the task plugin's bash wrapper inside
 *      that sandbox under its own process group (`detached: true`), pipe
 *      output to disk, enforce a hard wall cap, and read the canonical
 *      verdict from `result.txt` + `metrics.json`.
 *
 * Both steps are idempotent only at the granularity of a unique branch
 * name (the experiment id). Re-running with the same id refuses the
 * sandbox prep so a confused manager doesn't quietly overwrite a prior
 * experiment's commit.
 *
 * The executor is task-agnostic: it never imports anything from
 * `tasks/<name>/`. Plugins pass their wrapper path, environment, and
 * metric key through `runDirectTask`.
 *
 * Failure handling is intentionally manager-driven, not orchestrator-
 * driven. On any failure we write a structured `failure.json` next to
 * the (would-be) output dir and surface its path in the orchestrator
 * log; the manager reads it on its next replan and decides whether to
 * emit a corrected `param_patch`, an `llm_repair` task, or move on. We
 * deliberately do NOT auto-emit `llm_repair` siblings — letting the
 * manager react keeps it from looping on the same broken edit and
 * matches the project-wide "trust the LLM" philosophy.
 */

import fs from 'fs';
import path from 'path';
import { spawn, execFileSync } from 'child_process';
import { applyEdits } from './edits.js';
import { validateExperimentResult } from './result-validator.js';

const GIT_TIMEOUT_MS = 30000;
const PYTHON_PARSE_TIMEOUT_MS = 15000;

/**
 * @typedef {Object} EditSpec
 * @property {string} file
 * @property {string} kind
 *
 * @typedef {Object} PreparedSandbox
 * @property {boolean} ok
 * @property {string=} reason
 * @property {string=} sandboxDir   absolute path to the prepared work tree
 * @property {string=} baseCommit   SHA the parent ref resolved to
 * @property {string=} finalCommit  SHA after edits + commit
 *
 * @typedef {Object} DirectRunResult
 * @property {boolean} ok
 * @property {string=} reason
 * @property {number=} exitCode
 * @property {boolean=} timedOut
 * @property {Record<string, unknown>=} metrics
 * @property {string=} outputDir
 * @property {string=} sandboxDir
 * @property {string=} baseCommit
 * @property {string=} finalCommit
 * @property {string=} failurePath  where failure.json was written, if any
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
 * Resolve a ref (branch, tag, or SHA) to a full commit SHA in the source
 * repo. Refuses to return anything that doesn't look like a 40-hex SHA
 * because callers rely on the SHA being immutable.
 *
 * @param {string} sourceRepoPath
 * @param {string} ref
 * @return {string|null}
 */
export function resolveCommitSha(sourceRepoPath, ref) {
  if (!ref || typeof ref !== 'string') return null;
  let sha;
  try {
    sha = gitText(['rev-parse', '--verify', `${ref}^{commit}`], sourceRepoPath);
  } catch {
    return null;
  }
  return /^[0-9a-f]{40}$/.test(sha) ? sha : null;
}

/**
 * Clone, checkout parent SHA, apply edits, commit. The sandbox dir is
 * `<sandboxParentDir>/<branchName>` and must not already exist. On any
 * failure the dir is removed before returning so the caller can retry
 * with a corrected edit set.
 *
 * @param {Object} args
 * @param {string} args.sourceRepoPath  absolute path to the canonical repo
 * @param {string} args.sandboxParentDir directory that will hold the sandbox
 * @param {string} args.branchName       unique experiment id
 * @param {string} args.parentRef        ref into sourceRepo; we resolve to SHA
 * @param {Array<EditSpec>} args.edits   manager-provided edit list
 * @param {string=} args.commitMessage   commit subject; default uses branchName
 * @param {string=} args.authorName
 * @param {string=} args.authorEmail
 * @param {Array<string>=} args.parsePythonFiles  files to ast.parse-check
 *   after editing; default infers `.py` files from the edits[] list
 * @return {PreparedSandbox}
 */
export function prepareDirectSandbox({
  sourceRepoPath,
  sandboxParentDir,
  branchName,
  parentRef,
  edits,
  commitMessage = null,
  authorName = 'direct-executor',
  authorEmail = 'direct-executor@example.com',
  parsePythonFiles = null,
} = {}) {
  if (!sourceRepoPath || !sandboxParentDir || !branchName || !parentRef) {
    return { ok: false, reason: 'missing required argument' };
  }
  if (!Array.isArray(edits) || edits.length === 0) {
    return { ok: false, reason: 'no edits provided' };
  }
  const baseCommit = resolveCommitSha(sourceRepoPath, parentRef);
  if (!baseCommit) {
    return { ok: false, reason: `cannot resolve parent ref "${parentRef}" in ${sourceRepoPath}` };
  }
  const sandboxDir = path.join(sandboxParentDir, branchName);
  if (fs.existsSync(sandboxDir)) {
    return {
      ok: false,
      reason: `experiment_id_already_used: ${sandboxDir} already exists`,
    };
  }
  fs.mkdirSync(sandboxParentDir, { recursive: true });

  const cleanup = () => {
    try { fs.rmSync(sandboxDir, { recursive: true, force: true }); } catch {}
  };

  try {
    gitSilent(['clone', '--no-hardlinks', '-q', sourceRepoPath, sandboxDir], sandboxParentDir);
    gitSilent(['config', 'user.name', authorName], sandboxDir);
    gitSilent(['config', 'user.email', authorEmail], sandboxDir);
    gitSilent(['checkout', '-q', baseCommit], sandboxDir);
    gitSilent(['checkout', '-q', '-B', branchName], sandboxDir);
  } catch (e) {
    cleanup();
    return { ok: false, reason: `git setup failed: ${e.message}`, baseCommit };
  }

  const editResult = applyEdits(sandboxDir, edits);
  if (!editResult.ok) {
    cleanup();
    return {
      ok: false,
      reason: `edits failed: ${editResult.reason}`,
      baseCommit,
    };
  }

  const filesToParse = parsePythonFiles
    || [...new Set(edits.map((edit) => edit.file).filter((f) => typeof f === 'string' && f.endsWith('.py')))];
  for (const relativePath of filesToParse) {
    const target = path.resolve(sandboxDir, relativePath);
    try {
      execFileSync(
        'python3',
        ['-c', 'import ast, sys; ast.parse(open(sys.argv[1]).read())', target],
        { timeout: PYTHON_PARSE_TIMEOUT_MS, stdio: ['ignore', 'ignore', 'pipe'] },
      );
    } catch (e) {
      cleanup();
      return {
        ok: false,
        reason: `post-edit ast.parse failed for ${relativePath}: ${e.message}`,
        baseCommit,
      };
    }
  }

  let finalCommit;
  try {
    gitSilent(['add', '-A'], sandboxDir);
    const status = gitText(['status', '--porcelain'], sandboxDir);
    if (!status) {
      cleanup();
      return {
        ok: false,
        reason: 'edits produced no diff against parent (likely matched current contents)',
        baseCommit,
      };
    }
    gitSilent([
      'commit', '-q', '-m', commitMessage || `direct-executor: ${branchName}`,
    ], sandboxDir);
    finalCommit = gitText(['rev-parse', 'HEAD'], sandboxDir);
  } catch (e) {
    cleanup();
    return { ok: false, reason: `git commit failed: ${e.message}`, baseCommit };
  }

  return { ok: true, sandboxDir, baseCommit, finalCommit };
}

/**
 * Spawn the task plugin's bash wrapper inside a prepared sandbox. The
 * wrapper is launched with `detached: true` so its bash + python tree
 * gets a fresh process group; on abort or hard-cap we send SIGTERM (then
 * SIGKILL) to `-pgid`, which flushes the whole subtree.
 *
 * @param {Object} args
 * @param {string} args.sandboxDir
 * @param {string} args.wrapperScript  absolute path to the bash wrapper
 * @param {Array<string>} args.wrapperArgs
 * @param {string} args.outputDir      where the wrapper writes result.txt etc.
 * @param {Record<string, string>=} args.env extra env merged onto process.env
 * @param {AbortSignal=} args.abortSignal
 * @param {number=} args.hardCapMs     fallback wall cap (default 900s)
 * @param {string=} args.metricKey     forwarded to validator (default val_bpb)
 * @param {string=} args.stdoutPath    capture wrapper stdout/stderr (default
 *                                     `<outputDir>/worker.log`)
 * @return {Promise<DirectRunResult>}
 */
export function runDirectWorker({
  sandboxDir,
  wrapperScript,
  wrapperArgs = [],
  outputDir,
  env = {},
  abortSignal = null,
  hardCapMs = 900000,
  metricKey = 'val_bpb',
  stdoutPath = null,
} = {}) {
  if (!sandboxDir || !wrapperScript || !outputDir) {
    return Promise.resolve({ ok: false, reason: 'missing required argument' });
  }
  fs.mkdirSync(outputDir, { recursive: true });
  const logPath = stdoutPath || path.join(outputDir, 'worker.log');
  const logFd = fs.openSync(logPath, 'a');

  return new Promise((resolve) => {
    // CRITICAL: tell the wrapper to run training from THIS experiment's
    // sandbox, not the shared source/ dir. Without this the wrapper
    // defaults to `$script_dir/source` and ignores cwd entirely, so
    // every concurrent experiment fights over the same source/train.py
    // and the per-experiment edits applied above are never seen by
    // python. We pass a generic env name (DIRECT_EXECUTOR_REPO_ROOT)
    // that all task wrappers respect as highest-priority repo_root.
    const childEnv = {
      ...process.env,
      ...env,
      DIRECT_EXECUTOR_REPO_ROOT: sandboxDir,
    };
    const child = spawn('bash', [wrapperScript, outputDir, ...wrapperArgs], {
      cwd: sandboxDir,
      detached: true,
      stdio: ['ignore', logFd, logFd],
      env: childEnv,
    });
    child.unref();

    let settled = false;
    const finalize = (verdict) => {
      if (settled) return;
      settled = true;
      try { fs.closeSync(logFd); } catch {}
      resolve(verdict);
    };

    const killGroup = (signal) => {
      try { process.kill(-child.pid, signal); } catch {}
    };

    const hardCapTimer = setTimeout(() => {
      killGroup('SIGTERM');
      setTimeout(() => killGroup('SIGKILL'), 30000);
    }, hardCapMs);

    let aborted = false;
    const onAbort = () => {
      aborted = true;
      killGroup('SIGTERM');
      setTimeout(() => killGroup('SIGKILL'), 5000);
    };
    if (abortSignal) {
      if (abortSignal.aborted) onAbort();
      else abortSignal.addEventListener('abort', onAbort, { once: true });
    }

    child.once('error', (err) => {
      clearTimeout(hardCapTimer);
      finalize({ ok: false, reason: `wrapper spawn failed: ${err.message}` });
    });
    child.once('exit', (code, signal) => {
      clearTimeout(hardCapTimer);
      if (abortSignal) {
        try { abortSignal.removeEventListener('abort', onAbort); } catch {}
      }
      const timedOut = signal === 'SIGTERM' || signal === 'SIGKILL' || aborted;
      const verdict = validateExperimentResult({ outputDir, metricKey });
      const exitCode = typeof code === 'number' ? code : null;
      finalize({
        ok: verdict.success,
        reason: verdict.success ? undefined : (verdict.reason || `exit_code=${exitCode}, signal=${signal}`),
        exitCode,
        timedOut,
        metrics: verdict.metrics,
        outputDir,
      });
    });
  });
}

/**
 * Compose {@link prepareDirectSandbox} and {@link runDirectWorker}. On
 * sandbox-prep failure, writes a `failure.json` next to the (would-be)
 * output dir so the orchestrator can hand it to an `llm_repair`
 * follow-up. On wrapper failure, the worker's `train.log` is the
 * post-mortem; we still surface `failure.json` so the schema is
 * uniform.
 *
 * @param {Object} args   union of prepareDirectSandbox + runDirectWorker args
 * @return {Promise<DirectRunResult>}
 */
export async function runDirectTask(args) {
  const prep = prepareDirectSandbox(args);
  if (!prep.ok) {
    const failurePath = writeFailureRecord(args.outputDir, {
      stage: 'prepare_sandbox',
      reason: prep.reason,
      base_commit: prep.baseCommit ?? null,
      attempted_edits: args.edits || [],
    });
    return { ok: false, reason: prep.reason, failurePath, baseCommit: prep.baseCommit };
  }
  const run = await runDirectWorker({
    ...args,
    sandboxDir: prep.sandboxDir,
  });
  const result = {
    ...run,
    sandboxDir: prep.sandboxDir,
    baseCommit: prep.baseCommit,
    finalCommit: prep.finalCommit,
  };
  if (!run.ok) {
    result.failurePath = writeFailureRecord(args.outputDir, {
      stage: 'run_worker',
      reason: run.reason,
      base_commit: prep.baseCommit,
      final_commit: prep.finalCommit,
      sandbox_dir: prep.sandboxDir,
      attempted_edits: args.edits || [],
      exit_code: run.exitCode ?? null,
      timed_out: !!run.timedOut,
    });
  }
  return result;
}

/**
 * Persist a structured `failure.json` next to the worker output. Returns
 * the absolute path on success, or `null` if the directory is missing.
 *
 * @param {string|null|undefined} outputDir
 * @param {Record<string, unknown>} body
 * @return {string|null}
 */
export function writeFailureRecord(outputDir, body) {
  if (!outputDir) return null;
  try {
    fs.mkdirSync(outputDir, { recursive: true });
    const file = path.join(outputDir, 'failure.json');
    fs.writeFileSync(file, JSON.stringify(body, null, 2));
    return file;
  } catch {
    return null;
  }
}
