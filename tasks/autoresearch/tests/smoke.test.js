import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { createAutoresearchDagSmokeWorkspace } from '../manager-smoke.js';

test('createAutoresearchDagSmokeWorkspace uses repo-relative experiment output dirs', () => {
  const artifactRoot = mkdtempSync(path.join(tmpdir(), 'autoresearch-dag-live-'));

  try {
    const workspace = createAutoresearchDagSmokeWorkspace({
      artifactRoot,
      workerScriptPath: '/mnt/weka/home/junlin.chen/workspace/autoresearch/infra/hpc_agent/runner/scripts/run-autoresearch-worker.sh',
      timeBudgetSeconds: 10,
      gpuCount: 8,
    });

    const readme = readFileSync(workspace.readmePath, 'utf8');
    const managerProgram = readFileSync(workspace.managerProgramPath, 'utf8');
    const workerProgram = readFileSync(workspace.workerProgramPath, 'utf8');
    const projectsYaml = readFileSync(workspace.projectsYamlPath, 'utf8');
    const config = readFileSync(workspace.configPath, 'utf8');
    const workerSkill = readFileSync(workspace.experimentWorkerSkillPaths[0], 'utf8');

    assert.equal(workspace.expOutputDir, path.join(workspace.repoRoot, 'exp1'));
    assert.equal(workspace.experimentWorkerSkillPaths.length, 8);
    assert.equal(workspace.managerProgramPath, path.join(workspace.repoRoot, 'program.md'));
    assert.equal(workspace.workerProgramPath, path.join(workspace.repoRoot, 'worker_program.md'));
    assert.match(readme, /Manager program: `program\.md`/);
    assert.match(readme, /Worker program: `worker_program\.md`/);
    assert.match(readme, /research portfolio, not identical replicates/);
    assert.match(readme, /triggers live manager replanning/);
    assert.match(readme, /do not depend on currently running task ids/);
    assert.match(readme, /tags that currently running tasks have not produced yet/);
    assert.match(readme, /exp1\/<experiment_id>/);
    assert.match(readme, /Keep `AUTORESEARCH_TIME_BUDGET_SECONDS=10` fixed/);
    assert.match(readme, /at most 3 waves/);
    assert.match(readme, /Never use decimals like 0\.5/);
    assert.match(readme, /wave1\/aspect56/);
    assert.match(managerProgram, /# autoresearch manager program/);
    assert.match(managerProgram, /supports live replanning while a task graph is still running/);
    assert.match(managerProgram, /Every time a worker finishes and returns its resource token/);
    assert.match(managerProgram, /containing only additional tasks to append/);
    assert.match(managerProgram, /must not depend on currently-running task ids/);
    assert.match(managerProgram, /not-yet-produced tags/);
    assert.match(managerProgram, /ready backlog of about 16 experiments/);
    assert.match(managerProgram, /Every experiment task you emit must tell the worker to:/);
    assert.match(managerProgram, /Read `worker_program\.md` before running/);
    assert.match(workerProgram, /# autoresearch worker program/);
    assert.match(workerProgram, /Experiment runner protocol/);
    assert.match(workerProgram, /Analysis worker protocol/);
    assert.match(config, /liveReplanOnTaskComplete: true/);
    assert.doesNotMatch(readme, new RegExp(`${path.basename(artifactRoot)}-${path.basename(artifactRoot)}`));
    assert.match(readme, /exp1\/<experiment_id>\/metrics\.json/);
    assert.match(projectsYaml, new RegExp(workspace.repoRoot.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
    assert.match(workerSkill, /worker_class: experiment_runner/);
    assert.match(workerSkill, /read `worker_program\.md`/);
    assert.match(workerSkill, /except do not change the fixed time budget/);
  } finally {
    rmSync(artifactRoot, { recursive: true, force: true });
  }
});
