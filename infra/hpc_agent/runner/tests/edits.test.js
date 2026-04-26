import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { execFileSync } from 'child_process';
import { applyEditsToFile, applyEdits } from '../src/edits.js';

function tmpFile(contents, suffix = '.py') {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'edits-test-'));
  const file = path.join(dir, `f${suffix}`);
  fs.writeFileSync(file, contents);
  return { dir, file };
}

test('constant_replace: rewrites a tuple literal in place', () => {
  const { dir, file } = tmpFile('A = 1\nADAM_BETAS = (0.9, 0.95)\nB = 2\n');
  const result = applyEditsToFile(file, [
    { kind: 'constant_replace', name: 'ADAM_BETAS',
      expected_old_repr: '(0.9, 0.95)', new_repr: '(0.9, 0.97)' },
  ]);
  assert.equal(result.ok, true);
  assert.equal(fs.readFileSync(file, 'utf-8'), 'A = 1\nADAM_BETAS = (0.9, 0.97)\nB = 2\n');
  fs.rmSync(dir, { recursive: true, force: true });
});

test('constant_replace: AST-normalizes so 524288 matches 2 ** 19', () => {
  const { dir, file } = tmpFile('WIDTH = 2 ** 19\n');
  const result = applyEditsToFile(file, [
    { kind: 'constant_replace', name: 'WIDTH', expected_old_repr: '524288', new_repr: '1048576' },
  ]);
  assert.equal(result.ok, true);
  assert.equal(fs.readFileSync(file, 'utf-8'), 'WIDTH = 1048576\n');
  fs.rmSync(dir, { recursive: true, force: true });
});

test('constant_replace: refuses when current value disagrees with expected_old_repr', () => {
  const { dir, file } = tmpFile('DEPTH = 8\n');
  const result = applyEditsToFile(file, [
    { kind: 'constant_replace', name: 'DEPTH', expected_old_repr: '7', new_repr: '9' },
  ]);
  assert.equal(result.ok, false);
  assert.match(result.reason, /does not match/);
  // file untouched
  assert.equal(fs.readFileSync(file, 'utf-8'), 'DEPTH = 8\n');
  fs.rmSync(dir, { recursive: true, force: true });
});

test('constant_replace: refuses when symbol is not at module scope', () => {
  const { dir, file } = tmpFile('def f():\n    DEPTH = 8\n    return DEPTH\n');
  const result = applyEditsToFile(file, [
    { kind: 'constant_replace', name: 'DEPTH', expected_old_repr: '8', new_repr: '9' },
  ]);
  assert.equal(result.ok, false);
  assert.match(result.reason, /no module-level assignment/);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('constant_replace: refuses when new_repr would not parse', () => {
  const { dir, file } = tmpFile('X = 1\n');
  const result = applyEditsToFile(file, [
    { kind: 'constant_replace', name: 'X', expected_old_repr: '1', new_repr: 'def(' },
  ]);
  assert.equal(result.ok, false);
  assert.match(result.reason, /not a parseable expression/);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('constant_replace: handles AnnAssign (typed assignment)', () => {
  const { dir, file } = tmpFile('DEPTH: int = 8\n');
  const result = applyEditsToFile(file, [
    { kind: 'constant_replace', name: 'DEPTH', expected_old_repr: '8', new_repr: '9' },
  ]);
  assert.equal(result.ok, true);
  assert.equal(fs.readFileSync(file, 'utf-8'), 'DEPTH: int = 9\n');
  fs.rmSync(dir, { recursive: true, force: true });
});

test('regex_replace: substitutes when match count equals expected_count', () => {
  const { dir, file } = tmpFile('foo()\nbar()\nfoo()\n', '.txt');
  const result = applyEditsToFile(file, [
    { kind: 'regex_replace', pattern: 'foo\\(\\)', replacement: 'baz()', expected_count: 2 },
  ]);
  assert.equal(result.ok, true);
  assert.equal(fs.readFileSync(file, 'utf-8'), 'baz()\nbar()\nbaz()\n');
  fs.rmSync(dir, { recursive: true, force: true });
});

test('regex_replace: refuses when expected_count is wrong', () => {
  const { dir, file } = tmpFile('foo()\nbar()\nfoo()\n', '.txt');
  const result = applyEditsToFile(file, [
    { kind: 'regex_replace', pattern: 'foo\\(\\)', replacement: 'baz()', expected_count: 1 },
  ]);
  assert.equal(result.ok, false);
  assert.match(result.reason, /expected 1 match.*found 2/);
  // file untouched
  assert.equal(fs.readFileSync(file, 'utf-8'), 'foo()\nbar()\nfoo()\n');
  fs.rmSync(dir, { recursive: true, force: true });
});

test('regex_replace: requires expected_count >= 1', () => {
  const { dir, file } = tmpFile('foo\n', '.txt');
  const result = applyEditsToFile(file, [
    { kind: 'regex_replace', pattern: 'foo', replacement: 'bar' },
  ]);
  assert.equal(result.ok, false);
  assert.match(result.reason, /expected_count/);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('block_replace: replaces region between anchor and end markers', () => {
  const src = ['head', 'BEGIN', 'old1', 'old2', 'END', 'tail'].join('\n');
  const { dir, file } = tmpFile(src, '.txt');
  const result = applyEditsToFile(file, [
    { kind: 'block_replace', anchor_regex: '^BEGIN$', end_regex: '^END$', new_text: 'NEW' },
  ]);
  assert.equal(result.ok, true);
  assert.equal(fs.readFileSync(file, 'utf-8'), 'head\nNEW\ntail');
  fs.rmSync(dir, { recursive: true, force: true });
});

test('block_replace: fails when anchor not found', () => {
  const { dir, file } = tmpFile('head\nbody\ntail\n', '.txt');
  const result = applyEditsToFile(file, [
    { kind: 'block_replace', anchor_regex: '^MISSING$', end_regex: 'tail', new_text: 'X' },
  ]);
  assert.equal(result.ok, false);
  assert.match(result.reason, /anchor not found/);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('unified_diff: applies a small patch via git apply', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'edits-diff-'));
  execFileSync('git', ['init', '-q'], { cwd: dir });
  execFileSync('git', ['config', 'user.email', 'test@example.com'], { cwd: dir });
  execFileSync('git', ['config', 'user.name', 'Test'], { cwd: dir });
  const file = path.join(dir, 'a.txt');
  fs.writeFileSync(file, 'one\ntwo\nthree\n');
  execFileSync('git', ['add', '.'], { cwd: dir });
  execFileSync('git', ['commit', '-q', '-m', 'init'], { cwd: dir });
  const diff = [
    'diff --git a/a.txt b/a.txt',
    '--- a/a.txt',
    '+++ b/a.txt',
    '@@ -1,3 +1,3 @@',
    ' one',
    '-two',
    '+TWO',
    ' three',
    '',
  ].join('\n');
  const result = applyEditsToFile(file, [{ kind: 'unified_diff', diff }]);
  assert.equal(result.ok, true, `expected ok, got ${result.reason}`);
  assert.equal(fs.readFileSync(file, 'utf-8'), 'one\nTWO\nthree\n');
  fs.rmSync(dir, { recursive: true, force: true });
});

test('applyEdits: groups by file and rejects edits that escape the work tree', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'edits-root-'));
  fs.writeFileSync(path.join(root, 'a.py'), 'X = 1\n');
  const result = applyEdits(root, [
    { file: '../escape.py', kind: 'constant_replace', name: 'X',
      expected_old_repr: '1', new_repr: '2' },
  ]);
  assert.equal(result.ok, false);
  assert.match(result.reason, /escapes the work tree/);
  fs.rmSync(root, { recursive: true, force: true });
});

test('applyEdits: applies edits across two files in one batch', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'edits-batch-'));
  fs.writeFileSync(path.join(root, 'a.py'), 'X = 1\n');
  fs.writeFileSync(path.join(root, 'b.py'), 'Y = 2\n');
  const result = applyEdits(root, [
    { file: 'a.py', kind: 'constant_replace', name: 'X', expected_old_repr: '1', new_repr: '11' },
    { file: 'b.py', kind: 'constant_replace', name: 'Y', expected_old_repr: '2', new_repr: '22' },
  ]);
  assert.equal(result.ok, true);
  assert.equal(result.applied, 2);
  assert.equal(fs.readFileSync(path.join(root, 'a.py'), 'utf-8'), 'X = 11\n');
  assert.equal(fs.readFileSync(path.join(root, 'b.py'), 'utf-8'), 'Y = 22\n');
  fs.rmSync(root, { recursive: true, force: true });
});
