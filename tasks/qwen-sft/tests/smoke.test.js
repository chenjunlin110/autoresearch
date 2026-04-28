import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { createQwenSftWorkspace } from '../manager.js';

test('createQwenSftWorkspace generates a workspace with val_loss metric and DATA_MIX search surface', () => {
  const artifactRoot = mkdtempSync(path.join(tmpdir(), 'qwen-sft-smoke-'));

  try {
    const workspace = createQwenSftWorkspace({
      artifactRoot,
      timeBudgetSeconds: 600,
      gpuCount: 8,
    });

    assert.equal(workspace.experimentWorkerSkillPaths.length, 8);
    assert.equal(workspace.expOutputDir, path.join(workspace.repoRoot, 'experiments'));
    assert.ok(existsSync(workspace.configPath), 'config.yaml must exist');
    assert.ok(existsSync(workspace.projectsYamlPath), 'projects.yaml must exist');
    assert.ok(existsSync(workspace.statePath), 'state.json must exist');
    assert.ok(existsSync(workspace.managerProgramPath), 'program.md must exist');
    assert.ok(existsSync(workspace.workerProgramPath), 'worker_program.md must exist');
    assert.ok(existsSync(workspace.mayaSkillPath), 'maya skill must exist');

    const config = readFileSync(workspace.configPath, 'utf8');
    assert.match(config, /metricKey: val_loss/);
    assert.match(config, /timeBudgetSeconds: 600/);
    assert.match(config, /liveReplanOnTaskComplete: true/);
    assert.match(config, /liveReplanMinIntervalSeconds: 30/);
    assert.match(config, /directExecutor:/);
    assert.match(config, /enabled: true/);
    assert.match(config, /HF_HOME:/);

    const managerProgram = readFileSync(workspace.managerProgramPath, 'utf8');
    assert.match(managerProgram, /qwen-sft manager program/);
    assert.match(managerProgram, /DATA_MIX/);
    assert.match(managerProgram, /constant_replace/);
    assert.match(managerProgram, /execution_mode/);
    assert.match(managerProgram, /val_loss/);
    assert.match(managerProgram, /600s SFT budget/);
    assert.match(managerProgram, /Tulu-3/i);
    assert.match(managerProgram, /five buckets/i);
    assert.match(managerProgram, /Rationale is mandatory/);

    const workerProgram = readFileSync(workspace.workerProgramPath, 'utf8');
    assert.match(workerProgram, /qwen-sft worker program/);
    assert.match(workerProgram, /SFT_TIME_BUDGET_SECONDS=600/);
    assert.match(workerProgram, /val_loss/);

    const initialState = JSON.parse(readFileSync(workspace.statePath, 'utf8'));
    assert.equal(initialState.isPaused, false);

    const projectsYaml = readFileSync(workspace.projectsYamlPath, 'utf8');
    assert.match(projectsYaml, /qwen-sft:/);
    assert.match(projectsYaml, new RegExp(workspace.repoRoot.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));

    const workerSkill = readFileSync(workspace.experimentWorkerSkillPaths[0], 'utf8');
    assert.match(workerSkill, /worker_class: experiment_runner/);
    assert.match(workerSkill, /SFT_TIME_BUDGET_SECONDS=600/);
    assert.match(workerSkill, /Read `worker_program\.md`/);
  } finally {
    rmSync(artifactRoot, { recursive: true, force: true });
  }
});
