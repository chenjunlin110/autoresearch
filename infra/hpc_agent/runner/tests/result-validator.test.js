import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import {
  extractOutputDirFromTaskBody,
  readResultTxt,
  validateExperimentResult,
} from '../src/result-validator.js';

function makeExpDir({ resultTxt, metricsJson }) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'autoresearch-result-'));
  if (resultTxt !== undefined) fs.writeFileSync(path.join(dir, 'result.txt'), resultTxt);
  if (metricsJson !== undefined) {
    fs.writeFileSync(
      path.join(dir, 'metrics.json'),
      typeof metricsJson === 'string' ? metricsJson : JSON.stringify(metricsJson),
    );
  }
  return dir;
}

test('readResultTxt parses key=value lines and rejects blank or missing exit_code', () => {
  const dir = makeExpDir({ resultTxt: 'exit_code=0\noutput_dir=/x\nmetrics_path=/x/m.json\n' });
  const r = readResultTxt(dir);
  assert.equal(r.exit_code, 0);
  assert.equal(r.output_dir, '/x');
  fs.rmSync(dir, { recursive: true, force: true });

  const empty = makeExpDir({ resultTxt: '' });
  assert.equal(readResultTxt(empty), null);
  fs.rmSync(empty, { recursive: true, force: true });

  const bad = makeExpDir({ resultTxt: 'no exit code here\n' });
  assert.equal(readResultTxt(bad), null);
  fs.rmSync(bad, { recursive: true, force: true });
});

test('validateExperimentResult: clean success', () => {
  const dir = makeExpDir({
    resultTxt: 'exit_code=0\n',
    metricsJson: { val_bpb: 0.9919, training_seconds: 300.1, num_steps: 1000 },
  });
  const v = validateExperimentResult({ outputDir: dir });
  assert.equal(v.success, true);
  assert.equal(v.metrics.val_bpb, 0.9919);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('validateExperimentResult: honors metrics_path from result.txt when absolute', () => {
  const dir = makeExpDir({ resultTxt: 'exit_code=0\n' });
  const altPath = path.join(dir, 'alt-metrics.json');
  fs.writeFileSync(altPath, JSON.stringify({ val_bpb: 0.99 }));
  fs.writeFileSync(
    path.join(dir, 'result.txt'),
    `exit_code=0\nmetrics_path=${altPath}\n`,
  );
  const v = validateExperimentResult({ outputDir: dir });
  assert.equal(v.success, true);
  assert.equal(v.metrics.val_bpb, 0.99);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('validateExperimentResult: train exit non-zero is failure', () => {
  const dir = makeExpDir({
    resultTxt: 'exit_code=137\n',
    metricsJson: { val_bpb: 0.9919 },
  });
  const v = validateExperimentResult({ outputDir: dir });
  assert.equal(v.success, false);
  assert.match(v.reason, /exit_code=137/);
  assert.equal(v.exitCode, 137);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('validateExperimentResult: missing metrics.json is failure even if exit_code=0', () => {
  const dir = makeExpDir({ resultTxt: 'exit_code=0\n' });
  const v = validateExperimentResult({ outputDir: dir });
  assert.equal(v.success, false);
  assert.match(v.reason, /metrics\.json/);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('validateExperimentResult: NaN/missing val_bpb is failure', () => {
  const dir = makeExpDir({
    resultTxt: 'exit_code=0\n',
    metricsJson: { val_bpb: null, training_seconds: 300 },
  });
  const v = validateExperimentResult({ outputDir: dir });
  assert.equal(v.success, false);
  assert.match(v.reason, /val_bpb/);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('validateExperimentResult: missing result.txt is failure', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'autoresearch-result-'));
  const v = validateExperimentResult({ outputDir: dir });
  assert.equal(v.success, false);
  assert.match(v.reason, /result\.txt/);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('extractOutputDirFromTaskBody picks up labeled and inline paths', () => {
  assert.equal(
    extractOutputDirFromTaskBody('Output directory: experiments/exp_0001_baseline'),
    'experiments/exp_0001_baseline',
  );
  assert.equal(
    extractOutputDirFromTaskBody('Run wrapper with output_dir=experiments/exp_0042'),
    'experiments/exp_0042',
  );
  assert.equal(
    extractOutputDirFromTaskBody('Write outputs under experiments/exp_0099_depth9'),
    'experiments/exp_0099_depth9',
  );
  assert.equal(
    extractOutputDirFromTaskBody('please run experiments/exp_0007_aspect_72/sandbox/repo'),
    'experiments/exp_0007_aspect_72/sandbox/repo',
  );
  assert.equal(extractOutputDirFromTaskBody('no path here'), null);
  assert.equal(extractOutputDirFromTaskBody(''), null);
  assert.equal(extractOutputDirFromTaskBody(null), null);
});
