import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { execFileSync } from 'child_process';
import {
  prepareDirectSandbox,
  resolveCommitSha,
  runDirectWorker,
  runDirectTask,
} from '../src/direct-executor.js';

function mkSourceRepo(trainPyContents) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'direct-exec-src-'));
  execFileSync('git', ['init', '-q'], { cwd: root });
  execFileSync('git', ['config', 'user.email', 'src@example.com'], { cwd: root });
  execFileSync('git', ['config', 'user.name', 'src'], { cwd: root });
  fs.writeFileSync(path.join(root, 'train.py'), trainPyContents);
  execFileSync('git', ['add', '.'], { cwd: root });
  execFileSync('git', ['commit', '-q', '-m', 'init'], { cwd: root });
  return root;
}

test('resolveCommitSha returns a 40-hex SHA for HEAD', () => {
  const src = mkSourceRepo('X = 1\n');
  const sha = resolveCommitSha(src, 'HEAD');
  assert.match(sha, /^[0-9a-f]{40}$/);
  fs.rmSync(src, { recursive: true, force: true });
});

test('resolveCommitSha returns null for an unknown ref', () => {
  const src = mkSourceRepo('X = 1\n');
  assert.equal(resolveCommitSha(src, 'no-such-branch'), null);
  fs.rmSync(src, { recursive: true, force: true });
});

test('prepareDirectSandbox: clones, edits, commits, surfaces final SHA', () => {
  const src = mkSourceRepo('DEPTH = 8\nADAM_BETAS = (0.9, 0.95)\n');
  const parentDir = fs.mkdtempSync(path.join(os.tmpdir(), 'direct-exec-parent-'));
  const result = prepareDirectSandbox({
    sourceRepoPath: src,
    sandboxParentDir: parentDir,
    branchName: 'exp_0001',
    parentRef: 'HEAD',
    edits: [
      { file: 'train.py', kind: 'constant_replace', name: 'DEPTH',
        expected_old_repr: '8', new_repr: '12' },
    ],
  });
  assert.equal(result.ok, true, result.reason);
  assert.match(result.baseCommit, /^[0-9a-f]{40}$/);
  assert.match(result.finalCommit, /^[0-9a-f]{40}$/);
  assert.notEqual(result.baseCommit, result.finalCommit);
  const newSrc = fs.readFileSync(path.join(result.sandboxDir, 'train.py'), 'utf-8');
  assert.equal(newSrc, 'DEPTH = 12\nADAM_BETAS = (0.9, 0.95)\n');
  fs.rmSync(src, { recursive: true, force: true });
  fs.rmSync(parentDir, { recursive: true, force: true });
});

test('prepareDirectSandbox: refuses when sandbox dir already exists (idempotence)', () => {
  const src = mkSourceRepo('X = 1\n');
  const parentDir = fs.mkdtempSync(path.join(os.tmpdir(), 'direct-exec-parent-'));
  fs.mkdirSync(path.join(parentDir, 'exp_0001'));
  const result = prepareDirectSandbox({
    sourceRepoPath: src,
    sandboxParentDir: parentDir,
    branchName: 'exp_0001',
    parentRef: 'HEAD',
    edits: [{ file: 'train.py', kind: 'constant_replace', name: 'X',
      expected_old_repr: '1', new_repr: '2' }],
  });
  assert.equal(result.ok, false);
  assert.match(result.reason, /experiment_id_already_used/);
  fs.rmSync(src, { recursive: true, force: true });
  fs.rmSync(parentDir, { recursive: true, force: true });
});

test('prepareDirectSandbox: rolls back when an edit fails', () => {
  const src = mkSourceRepo('DEPTH = 8\n');
  const parentDir = fs.mkdtempSync(path.join(os.tmpdir(), 'direct-exec-parent-'));
  const result = prepareDirectSandbox({
    sourceRepoPath: src,
    sandboxParentDir: parentDir,
    branchName: 'exp_0002',
    parentRef: 'HEAD',
    edits: [
      { file: 'train.py', kind: 'constant_replace', name: 'DEPTH',
        expected_old_repr: '7', new_repr: '9' },
    ],
  });
  assert.equal(result.ok, false);
  assert.match(result.reason, /edits failed/);
  assert.equal(fs.existsSync(path.join(parentDir, 'exp_0002')), false,
    'sandbox dir should have been removed on edit failure');
  fs.rmSync(src, { recursive: true, force: true });
  fs.rmSync(parentDir, { recursive: true, force: true });
});

test('prepareDirectSandbox: ast.parse-checks edited Python files', () => {
  const src = mkSourceRepo('DEPTH = 8\n');
  const parentDir = fs.mkdtempSync(path.join(os.tmpdir(), 'direct-exec-parent-'));
  // unified_diff that produces a syntax error in train.py
  const badDiff = [
    'diff --git a/train.py b/train.py',
    '--- a/train.py',
    '+++ b/train.py',
    '@@ -1 +1 @@',
    '-DEPTH = 8',
    '+DEPTH = (1,',
    '',
  ].join('\n');
  const result = prepareDirectSandbox({
    sourceRepoPath: src,
    sandboxParentDir: parentDir,
    branchName: 'exp_0003',
    parentRef: 'HEAD',
    edits: [{ file: 'train.py', kind: 'unified_diff', diff: badDiff }],
  });
  assert.equal(result.ok, false);
  assert.match(result.reason, /post-edit ast.parse failed/);
  fs.rmSync(src, { recursive: true, force: true });
  fs.rmSync(parentDir, { recursive: true, force: true });
});

test('runDirectWorker: success path reads metrics.json and returns ok=true', async () => {
  const sandbox = fs.mkdtempSync(path.join(os.tmpdir(), 'direct-exec-sb-'));
  const outputDir = fs.mkdtempSync(path.join(os.tmpdir(), 'direct-exec-out-'));
  // A trivial wrapper that writes a valid result.txt + metrics.json.
  const wrapper = path.join(sandbox, 'wrapper.sh');
  fs.writeFileSync(wrapper, [
    '#!/usr/bin/env bash',
    'set -e',
    'out="$1"',
    'mkdir -p "$out"',
    'echo \'{"val_bpb": 1.234, "training_seconds": 12.5}\' > "$out/metrics.json"',
    'printf "exit_code=0\\noutput_dir=%s\\nmetrics_path=%s/metrics.json\\n" "$out" "$out" > "$out/result.txt"',
    '',
  ].join('\n'));
  fs.chmodSync(wrapper, 0o755);
  const verdict = await runDirectWorker({
    sandboxDir: sandbox,
    wrapperScript: wrapper,
    outputDir,
  });
  assert.equal(verdict.ok, true, verdict.reason);
  assert.equal(verdict.exitCode, 0);
  assert.equal(verdict.metrics.val_bpb, 1.234);
  fs.rmSync(sandbox, { recursive: true, force: true });
  fs.rmSync(outputDir, { recursive: true, force: true });
});

test('runDirectTask: writes failure.json on prepare failure', async () => {
  const src = mkSourceRepo('X = 1\n');
  const parentDir = fs.mkdtempSync(path.join(os.tmpdir(), 'direct-exec-parent-'));
  const outputDir = fs.mkdtempSync(path.join(os.tmpdir(), 'direct-exec-out-'));
  const result = await runDirectTask({
    sourceRepoPath: src,
    sandboxParentDir: parentDir,
    branchName: 'exp_0004',
    parentRef: 'HEAD',
    edits: [{ file: 'train.py', kind: 'constant_replace', name: 'MISSING',
      expected_old_repr: '0', new_repr: '1' }],
    wrapperScript: '/bin/true', // never reached
    outputDir,
  });
  assert.equal(result.ok, false);
  assert.match(result.reason, /no module-level assignment/);
  assert.ok(result.failurePath, 'failure.json should be written');
  const body = JSON.parse(fs.readFileSync(result.failurePath, 'utf-8'));
  assert.equal(body.stage, 'prepare_sandbox');
  assert.match(body.reason, /no module-level assignment/);
  assert.equal(body.attempted_edits.length, 1);
  fs.rmSync(src, { recursive: true, force: true });
  fs.rmSync(parentDir, { recursive: true, force: true });
  fs.rmSync(outputDir, { recursive: true, force: true });
});
