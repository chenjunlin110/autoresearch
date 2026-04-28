import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildClaudeExecInvocation,
  resolveClaudeCliModel,
} from '../src/agent-runner.js';

test('resolveClaudeCliModel strips supported provider prefixes', () => {
  assert.equal(resolveClaudeCliModel('anthropic/claude-opus-4-7'), 'claude-opus-4-7');
  assert.equal(resolveClaudeCliModel('claude/claude-sonnet-4-6'), 'claude-sonnet-4-6');
  assert.equal(resolveClaudeCliModel('claude-haiku-4-5'), 'claude-haiku-4-5');
  assert.equal(resolveClaudeCliModel('mid'), null);
  assert.equal(resolveClaudeCliModel('gpt-5.4'), null);
});

test('buildClaudeExecInvocation includes workspace, writable roots, and model', () => {
  const inv = buildClaudeExecInvocation({
    cwd: '/workspace/repo',
    writableRoots: ['/workspace/repo', '/workspace/project-data'],
    model: 'anthropic/claude-opus-4-7',
    outputFile: '/tmp/final.txt',
  });
  assert.equal(inv.command, 'claude');
  assert.ok(inv.args.includes('--print'));
  assert.ok(inv.args.includes('--dangerously-skip-permissions'));
  const joined = inv.args.join(' ');
  assert.match(joined, /--add-dir \/workspace\/repo/);
  assert.match(joined, /--add-dir \/workspace\/project-data/);
  assert.match(joined, /--model claude-opus-4-7/);
  assert.equal(inv.resolvedModel, 'claude-opus-4-7');
});

test('buildClaudeExecInvocation skips --model for abstract tier names', () => {
  const inv = buildClaudeExecInvocation({
    cwd: '/workspace/repo',
    model: 'mid',
    outputFile: '/tmp/final.txt',
  });
  assert.equal(inv.resolvedModel, null);
  assert.ok(!inv.args.includes('--model'));
});
