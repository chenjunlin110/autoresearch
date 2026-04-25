import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildCodexExecInvocation,
  resolveCodexCliModel,
} from '../src/agent-runner.js';

test('resolveCodexCliModel strips supported provider prefixes', () => {
  assert.equal(resolveCodexCliModel('openai-codex/gpt-5.3-codex'), 'gpt-5.3-codex');
  assert.equal(resolveCodexCliModel('openai/gpt-5.4'), 'gpt-5.4');
  assert.equal(resolveCodexCliModel('gpt-5.4'), 'gpt-5.4');
  assert.equal(resolveCodexCliModel('mid'), null);
  assert.equal(resolveCodexCliModel('claude-sonnet-4-6'), null);
});

test('buildCodexExecInvocation includes workspace and extra writable roots', () => {
  const invocation = buildCodexExecInvocation({
    cwd: '/workspace/repo',
    writableRoots: ['/workspace/repo', '/workspace/project-data'],
    model: 'openai/gpt-5.4',
    outputFile: '/tmp/final.txt',
  });

  assert.equal(invocation.command, 'codex');
  assert.deepEqual(invocation.args.slice(0, 10), [
    'exec',
    '--color', 'never',
    '--ignore-user-config',
    '--ignore-rules',
    '--dangerously-bypass-approvals-and-sandbox',
    '--output-last-message', '/tmp/final.txt',
    '-C', '/workspace/repo',
  ]);
  assert.match(invocation.args.join(' '), /-C \/workspace\/repo/);
  assert.match(invocation.args.join(' '), /--add-dir \/workspace\/project-data/);
  assert.match(invocation.args.join(' '), /-m gpt-5.4/);
  assert.equal(invocation.args.at(-1), '-');
  assert.equal(invocation.resolvedModel, 'gpt-5.4');
});
