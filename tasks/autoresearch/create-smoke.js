#!/usr/bin/env node
import { createAutoresearchDagSmokeWorkspace } from './manager-smoke.js';

function parseArgs(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith('--')) {
      continue;
    }
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
  console.error('Usage: create-autoresearch-dag-smoke.js --artifact-root <dir> [--project-id <name>] [--experiment-name <name>]');
  process.exit(1);
}

const result = createAutoresearchDagSmokeWorkspace({
  artifactRoot: args['artifact-root'],
  projectId: args['project-id'] || 'autoresearch-dag-smoke',
  experimentName: args['experiment-name'] || 'exp1',
  workerScriptPath: args['worker-script'],
  timeBudgetSeconds: args['time-budget-seconds'] || 10,
  minimumIterations: args['minimum-iterations'] || 2,
  maximumIterations: args['maximum-iterations'] || 3,
  gpuCount: args['gpu-count'] || 8,
  experimentWorkerCount: args['experiment-worker-count'] || args['gpu-count'] || 8,
});

process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
