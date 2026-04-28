import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { execFileSync } from 'child_process';
import { applyKeepExperiment, formatLineageMarkdown, readLineage } from '../src/lineage.js';

function makeSourceRepo(initialContents) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'lineage-src-'));
  execFileSync('git', ['init', '-q'], { cwd: root });
  execFileSync('git', ['config', 'user.email', 't@example.com'], { cwd: root });
  execFileSync('git', ['config', 'user.name', 't'], { cwd: root });
  fs.writeFileSync(path.join(root, 'train.py'), initialContents);
  execFileSync('git', ['add', '.'], { cwd: root });
  execFileSync('git', ['commit', '-q', '-m', 'init'], { cwd: root });
  return root;
}

function readSrc(root, file) {
  return fs.readFileSync(path.join(root, file), 'utf-8');
}

test('applyKeepExperiment: applies edits, commits, advances HEAD, writes lineage entry', () => {
  const root = makeSourceRepo('DEPTH = 8\nADAM_BETAS = (0.9, 0.95)\n');
  const lineagePath = path.join(root, '.lineage.json');
  const result = applyKeepExperiment({
    sourceRepoPath: root,
    lineagePath,
    taskName: 'autoresearch-dag-full',
    experimentId: 'exp_0010_aspect',
    reason: 'aspect win',
    edits: [
      { file: 'train.py', kind: 'constant_replace', name: 'DEPTH',
        expected_old_repr: '8', new_repr: '12' },
    ],
  });
  assert.equal(result.ok, true);
  assert.match(result.commit, /^[0-9a-f]{40}$/);

  // Source/HEAD has the new value
  assert.equal(readSrc(root, 'train.py'), 'DEPTH = 12\nADAM_BETAS = (0.9, 0.95)\n');

  // Lineage doc shape
  const lineage = readLineage(lineagePath);
  assert.equal(lineage.entries.length, 1);
  assert.equal(lineage.entries[0].experiment_id, 'exp_0010_aspect');
  assert.equal(lineage.entries[0].reason, 'aspect win');
  assert.equal(lineage.entries[0].commit, result.commit);
  assert.equal(lineage.current_head_commit, result.commit);
  assert.equal(lineage.task, 'autoresearch-dag-full');

  fs.rmSync(root, { recursive: true, force: true });
});

test('applyKeepExperiment: cumulative — second keep stacks on first', () => {
  const root = makeSourceRepo('DEPTH = 8\nADAM_BETAS = (0.9, 0.95)\n');
  const lineagePath = path.join(root, '.lineage.json');

  const r1 = applyKeepExperiment({
    sourceRepoPath: root, lineagePath, taskName: 't', experimentId: 'exp_a',
    reason: 'depth up', edits: [{ file: 'train.py', kind: 'constant_replace',
      name: 'DEPTH', expected_old_repr: '8', new_repr: '12' }],
  });
  assert.equal(r1.ok, true);

  const r2 = applyKeepExperiment({
    sourceRepoPath: root, lineagePath, taskName: 't', experimentId: 'exp_b',
    reason: 'beta tweak', edits: [{ file: 'train.py', kind: 'constant_replace',
      name: 'ADAM_BETAS', expected_old_repr: '(0.9, 0.95)', new_repr: '(0.9, 0.97)' }],
  });
  assert.equal(r2.ok, true);
  assert.notEqual(r2.commit, r1.commit);

  // Both edits are in HEAD now
  assert.equal(readSrc(root, 'train.py'), 'DEPTH = 12\nADAM_BETAS = (0.9, 0.97)\n');

  const lineage = readLineage(lineagePath);
  assert.equal(lineage.entries.length, 2);
  assert.equal(lineage.entries.map((e) => e.experiment_id).join(','), 'exp_a,exp_b');
  assert.equal(lineage.current_head_commit, r2.commit);

  fs.rmSync(root, { recursive: true, force: true });
});

test('applyKeepExperiment: idempotent on duplicate id', () => {
  const root = makeSourceRepo('X = 1\n');
  const lineagePath = path.join(root, '.lineage.json');
  const edits = [{ file: 'train.py', kind: 'constant_replace',
    name: 'X', expected_old_repr: '1', new_repr: '2' }];

  const r1 = applyKeepExperiment({
    sourceRepoPath: root, lineagePath, taskName: 't',
    experimentId: 'exp_dup', reason: 'first', edits,
  });
  const r2 = applyKeepExperiment({
    sourceRepoPath: root, lineagePath, taskName: 't',
    experimentId: 'exp_dup', reason: 'second (would mismatch expected_old_repr)', edits,
  });
  assert.equal(r1.ok, true);
  assert.equal(r2.ok, true);
  assert.equal(r2.alreadyKept, true);
  assert.equal(r2.commit, r1.commit);

  const lineage = readLineage(lineagePath);
  assert.equal(lineage.entries.length, 1);

  fs.rmSync(root, { recursive: true, force: true });
});

test('applyKeepExperiment: bad edits do not advance HEAD or write lineage', () => {
  const root = makeSourceRepo('Y = 5\n');
  const lineagePath = path.join(root, '.lineage.json');
  const before = execFileSync('git', ['rev-parse', 'HEAD'],
    { cwd: root, encoding: 'utf-8' }).trim();

  const result = applyKeepExperiment({
    sourceRepoPath: root, lineagePath, taskName: 't',
    experimentId: 'exp_bad', reason: '', edits: [
      { file: 'train.py', kind: 'constant_replace', name: 'MISSING',
        expected_old_repr: '0', new_repr: '1' },
    ],
  });
  assert.equal(result.ok, false);
  assert.match(result.reason, /no module-level assignment/);

  const after = execFileSync('git', ['rev-parse', 'HEAD'],
    { cwd: root, encoding: 'utf-8' }).trim();
  assert.equal(after, before, 'HEAD should not move on edit failure');
  assert.equal(fs.existsSync(lineagePath), false, 'lineage.json should not be written on failure');

  fs.rmSync(root, { recursive: true, force: true });
});

test('formatLineageMarkdown: renders kept entries with short SHAs', () => {
  const lineage = {
    task: 't',
    entries: [
      { experiment_id: 'exp_a', commit: 'abcdef0123456789abcdef0123456789abcdef01',
        reason: 'aspect win', kept_at: '2026-04-27T00:00:00Z', edits: [] },
      { experiment_id: 'exp_b', commit: 'fedcba9876543210fedcba9876543210fedcba98',
        reason: '', kept_at: '2026-04-27T00:01:00Z', edits: [] },
    ],
    current_head_commit: 'fedcba9876543210fedcba9876543210fedcba98',
  };
  const md = formatLineageMarkdown(lineage);
  assert.match(md, /Kept lineage/);
  assert.match(md, /exp_a \(abcdef0\) — aspect win/);
  assert.match(md, /exp_b \(fedcba9\)/);
  assert.match(md, /current source\/HEAD = fedcba9/);
});

test('formatLineageMarkdown: empty lineage → empty string', () => {
  assert.equal(formatLineageMarkdown(null), '');
  assert.equal(formatLineageMarkdown({ entries: [] }), '');
});
