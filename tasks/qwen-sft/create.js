#!/usr/bin/env node
import { createQwenSftWorkspace } from './manager.js';

function parseArgs(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) {
      options[key] = true;
      continue;
    }
    options[key] = value;
    index += 1;
  }
  return options;
}

const args = parseArgs(process.argv.slice(2));

if (!args['artifact-root']) {
  console.error('Usage: create.js --artifact-root <dir> [--project-id <name>] [--experiment-name <name>] [--gpu-count <n>] [--time-budget-seconds <s>]');
  process.exit(1);
}

const result = createQwenSftWorkspace({
  artifactRoot: args['artifact-root'],
  projectId: args['project-id'] || 'qwen-sft',
  experimentName: args['experiment-name'] || 'experiments',
  workerScriptPath: args['worker-script'],
  timeBudgetSeconds: args['time-budget-seconds'] || 600,
  gpuCount: args['gpu-count'] || 8,
  experimentWorkerCount: args['experiment-worker-count'] || args['gpu-count'] || 8,
  agentRuntime: args['agent-runtime'] || process.env.AUTORESEARCH_AGENT_RUNTIME || 'codex_cli',
});

process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
