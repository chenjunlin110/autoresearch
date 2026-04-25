import { spawn } from 'child_process';
import Database from 'better-sqlite3';
import { getAllocationLease, getTokenGrant } from './resource-orchestration.js';
import { parseGpuTokenOrdinals } from './orchestration-utils.js';

const SHARED_STEP_TIMEOUT_MS = parseInt(process.env.HPC_SHARED_STEP_TIMEOUT_MS || '21600000', 10); // 6h default
const SHARED_STEP_OUTPUT_LIMIT = parseInt(process.env.HPC_SHARED_STEP_OUTPUT_LIMIT || '120000', 10);
const SHELL_SINGLE_QUOTE_ESCAPE = `'\"'\"'`;
export function getHpcToolDefinition() {
  return {
    name: 'HpcSubmit',
    description:
      'Attach a worker step to an existing shared SLURM allocation, or probe that allocation. ' +
      'In shared-allocation mode, the step uses srun --overlap --ntasks=1 and manual CUDA_VISIBLE_DEVICES assignment. ' +
      'Returns the step stdout/stderr when the run finishes. ' +
      'The agent is paused while waiting — no tokens are consumed during the wait.',
    input_schema: {
      type: 'object',
      properties: {
        mode: {
          type: 'string',
          description: 'Execution mode. Use "shared_probe" to inspect an existing shared allocation, or "shared_step" to attach a worker step to it.',
        },
        task_name: {
          type: 'string',
          description: 'Optional short name for the step, used only for logging.',
        },
        command: {
          type: 'string',
          description: 'Shell command(s) to execute inside the shared allocation (for example "uv run train.py").',
        },
        walltime: {
          type: 'string',
          description: 'Maximum local wait time in HH:MM:SS format for a shared step. Defaults to 6:00:00.',
        },
        shared_job_id: {
          type: 'string',
          description: 'Existing shared allocation job id for local shared-allocation mode.',
        },
        grant_id: {
          type: 'number',
          description: 'Active token grant id authorizing this worker to use a subset of GPUs from the shared allocation.',
        },
        gpu_tokens: {
          type: 'array',
          items: { type: 'string' },
          description: 'Granted GPU token names such as ["gpu0"] or ["gpu2","gpu5"]. In shared-step mode these are converted to CUDA_VISIBLE_DEVICES ordinals.',
        },
        gpu_ids: {
          type: 'array',
          items: { type: 'number' },
          description: 'Numeric GPU ordinals to expose via CUDA_VISIBLE_DEVICES. Prefer gpu_tokens when using resource orchestration grants.',
        },
        cpus_per_task: {
          type: 'number',
          description: 'CPU count for a shared-allocation worker step. Defaults to 1.',
        },
        working_directory: {
          type: 'string',
          description: 'Working directory to cd into before running the shared-allocation command.',
        },
        env: {
          type: 'object',
          description: 'Extra environment variables to export before running the shared-allocation command.',
        },
      },
      required: ['mode'],
    },
  };
}

function parseWalltimeSeconds(walltime) {
  const m = String(walltime || '').match(/^(\d+):(\d{2}):(\d{2})$/);
  if (!m) return 3600;
  return parseInt(m[1]) * 3600 + parseInt(m[2]) * 60 + parseInt(m[3]);
}

function ensureNonEmptyString(value, name) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`${name} is required`);
  }
  return value.trim();
}

function normalizePositiveInteger(value, fallback) {
  if (typeof value === 'number' && Number.isInteger(value) && value > 0) return value;
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number.parseInt(value, 10);
    if (Number.isInteger(parsed) && parsed > 0) return parsed;
  }
  return fallback;
}

function shellEscape(value) {
  return `'${String(value).replaceAll("'", SHELL_SINGLE_QUOTE_ESCAPE)}'`;
}

function normalizeStringList(value) {
  if (Array.isArray(value)) {
    return value.map(item => String(item).trim()).filter(Boolean);
  }
  if (typeof value === 'string' && value.trim()) {
    return value.split(',').map(item => item.trim()).filter(Boolean);
  }
  return [];
}


function normalizeSharedEnv(rawEnv = {}) {
  if (!rawEnv || typeof rawEnv !== 'object' || Array.isArray(rawEnv)) return {};
  const env = {};
  for (const [key, value] of Object.entries(rawEnv)) {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) {
      throw new Error(`invalid environment variable name: ${key}`);
    }
    env[key] = String(value);
  }
  return env;
}

function withProjectDb(defaults, fn) {
  const dbPath = defaults.project_db_path || defaults.projectDbPath || null;
  if (!dbPath) {
    throw new Error('shared allocation mode requires hpcDefaults.project_db_path');
  }
  const db = new Database(dbPath, { fileMustExist: true });
  try {
    return fn(db);
  } finally {
    try { db.close(); } catch {}
  }
}

function loadGrantContext(grantId, defaults = {}) {
  const normalizedGrantId = normalizePositiveInteger(grantId, 0);
  if (!normalizedGrantId) return null;
  return withProjectDb(defaults, (db) => {
    const grant = getTokenGrant(db, normalizedGrantId);
    if (!grant) {
      throw new Error(`token grant ${normalizedGrantId} not found`);
    }
    if (grant.status !== 'active') {
      throw new Error(`token grant ${normalizedGrantId} is not active`);
    }
    const activeLease = getAllocationLease(db, { includeInactive: true });
    return { grant, activeLease };
  });
}

function resolveRequestedGpuTokens(input = {}, grantContext = null) {
  const tokenNames = normalizeStringList(input.gpu_tokens);
  if (tokenNames.length > 0) return tokenNames;

  const gpuIds = normalizeStringList(input.gpu_ids);
  if (gpuIds.length > 0) return gpuIds.map(id => `gpu${Number.parseInt(id, 10)}`);

  if (grantContext?.grant?.tokenNames?.length) {
    return grantContext.grant.tokenNames;
  }
  return [];
}

export function resolveSharedSlurmStepInput(input = {}, defaults = {}) {
  const grantContext = loadGrantContext(input.grant_id, defaults);
  const requestedGpuTokens = resolveRequestedGpuTokens(input, grantContext);
  if (requestedGpuTokens.length === 0) {
    throw new Error('shared_step requires gpu_tokens/gpu_ids or an active grant_id');
  }

  const actor = typeof defaults.agent_name === 'string' ? defaults.agent_name.trim() : null;
  if (actor && grantContext?.grant?.workerName && grantContext.grant.workerName !== actor) {
    throw new Error(`token grant ${grantContext.grant.id} belongs to ${grantContext.grant.workerName}, not ${actor}`);
  }

  if (grantContext?.grant?.tokenNames?.length) {
    const expected = [...grantContext.grant.tokenNames].sort();
    const actual = [...requestedGpuTokens].sort();
    if (JSON.stringify(expected) !== JSON.stringify(actual)) {
      throw new Error(`token grant ${grantContext.grant.id} authorizes ${expected.join(', ')}, got ${actual.join(', ')}`);
    }
  }

  const jobId = (
    (typeof input.shared_job_id === 'string' && input.shared_job_id.trim())
    || grantContext?.grant?.leaseJobId
    || grantContext?.activeLease?.jobId
    || defaults.shared_job_id
    || defaults.sharedJobId
    || ''
  ).trim();
  if (!jobId) {
    throw new Error('shared_step requires shared_job_id or a lease-backed active grant_id');
  }

  if (grantContext?.grant?.leaseJobId && grantContext.grant.leaseJobId !== jobId) {
    throw new Error(`shared_step job ${jobId} does not match grant lease job ${grantContext.grant.leaseJobId}`);
  }

  const cpusPerTask = normalizePositiveInteger(input.cpus_per_task, 1);
  const gpuOrdinals = parseGpuTokenOrdinals(requestedGpuTokens);
  const env = {
    ...normalizeSharedEnv(defaults.shared_env || defaults.sharedEnv || {}),
    ...normalizeSharedEnv(input.env),
  };
  env.CUDA_VISIBLE_DEVICES = gpuOrdinals.join(',');
  if (!Object.prototype.hasOwnProperty.call(env, 'OMP_NUM_THREADS')) {
    env.OMP_NUM_THREADS = String(cpusPerTask);
  }

  const workingDirectory = (
    (typeof input.working_directory === 'string' && input.working_directory.trim())
    || defaults.shared_working_directory
    || defaults.sharedWorkingDirectory
    || ''
  ).trim() || null;

  return {
    taskName: typeof input.task_name === 'string' && input.task_name.trim() ? input.task_name.trim() : 'shared-step',
    command: ensureNonEmptyString(input.command, 'command'),
    jobId,
    grantId: normalizePositiveInteger(input.grant_id, 0) || null,
    gpuTokens: requestedGpuTokens,
    gpuOrdinals,
    cpusPerTask,
    env,
    workingDirectory,
    nodeName: grantContext?.grant?.nodeName || grantContext?.activeLease?.nodeName || null,
  };
}

export function buildSharedSlurmStepCommand(spec) {
  const scriptParts = [];
  if (spec.workingDirectory) {
    scriptParts.push(`cd ${shellEscape(spec.workingDirectory)}`);
  }
  for (const [key, value] of Object.entries(spec.env)) {
    scriptParts.push(`export ${key}=${shellEscape(value)}`);
  }
  scriptParts.push(spec.command);
  const script = scriptParts.join(' && ');

  // Manual CUDA_VISIBLE_DEVICES assignment is intentionally used here because it
  // matched the real cluster behavior during overlap tests more reliably than
  // srun GPU-binding flags.
  return {
    command: 'srun',
    args: [
      '--jobid', spec.jobId,
      '--overlap',
      '--ntasks=1',
      '--cpus-per-task', String(spec.cpusPerTask),
      'bash',
      '-lc',
      script,
    ],
    script,
  };
}

export function buildSharedSlurmProbeCommand(sharedJobId) {
  const jobId = ensureNonEmptyString(sharedJobId, 'shared_job_id');
  const probeScript = [
    'python3 - <<\'PY\'',
    'import json, os, socket',
    'data = {',
    '  "hostname": socket.gethostname(),',
    '  "slurm_job_id": os.environ.get("SLURM_JOB_ID"),',
    '  "slurm_step_gpus": os.environ.get("SLURM_STEP_GPUS"),',
    '  "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),',
    '  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),',
    '}',
    'try:',
    '  import torch',
    '  data["torch_cuda_device_count"] = torch.cuda.device_count()',
    'except Exception as exc:',
    '  data["torch_error"] = str(exc)',
    'print(json.dumps(data, sort_keys=True))',
    'PY',
  ].join('\n');

  return {
    command: 'srun',
    args: [
      '--jobid', jobId,
      '--overlap',
      '--ntasks=1',
      '--cpus-per-task', '1',
      'bash',
      '-lc',
      probeScript,
    ],
    script: probeScript,
  };
}

function executeLocalSpawn(command, args, { cwd = process.cwd(), timeoutMs = SHARED_STEP_TIMEOUT_MS } = {}) {
  return new Promise((resolve) => {
    const proc = spawn(command, args, {
      cwd,
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: true,
      env: process.env,
    });
    let stdout = '';
    let stderr = '';
    let outputTruncated = false;
    let settled = false;
    let timedOut = false;

    const appendOutput = (current, chunk) => {
      if (current.length >= SHARED_STEP_OUTPUT_LIMIT) {
        outputTruncated = true;
        return current;
      }
      const remaining = SHARED_STEP_OUTPUT_LIMIT - current.length;
      const next = chunk.length > remaining ? chunk.slice(0, remaining) : chunk;
      if (next.length < chunk.length) outputTruncated = true;
      return current + next;
    };

    const killProc = (signal) => {
      try { process.kill(-proc.pid, signal); } catch {}
    };

    proc.stdout.on('data', (chunk) => {
      stdout = appendOutput(stdout, chunk.toString('utf-8'));
    });
    proc.stderr.on('data', (chunk) => {
      stderr = appendOutput(stderr, chunk.toString('utf-8'));
    });

    const timer = setTimeout(() => {
      timedOut = true;
      killProc('SIGTERM');
      setTimeout(() => killProc('SIGKILL'), 5000);
    }, timeoutMs);

    const finish = (payload) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(payload);
    };

    proc.on('close', (code, signal) => {
      let output = '';
      if (stdout) output += stdout;
      if (stderr) output += `${output ? '\n' : ''}${stderr}`;
      if (outputTruncated) {
        output += '\n\n... (shared step output truncated) ...';
      }
      finish({
        ok: code === 0 && !timedOut,
        code,
        signal,
        timedOut,
        output: output || '(no output)',
      });
    });

    proc.on('error', (err) => {
      finish({
        ok: false,
        code: null,
        signal: null,
        timedOut,
        output: `Error executing ${command}: ${err.message}`,
      });
    });
  });
}

function formatSharedStepResult(kind, spec, result) {
  const status = result.timedOut ? 'TIMED_OUT' : (result.ok ? 'COMPLETED' : 'FAILED');
  const detailLines = [
    `[HpcSubmit] Shared ${kind} ${status}.`,
    `  job_id:        ${spec.jobId}`,
  ];
  if (spec.grantId) detailLines.push(`  grant_id:      ${spec.grantId}`);
  if (spec.gpuTokens?.length) detailLines.push(`  gpu_tokens:    ${spec.gpuTokens.join(', ')}`);
  if (spec.cpusPerTask) detailLines.push(`  cpus_per_task: ${spec.cpusPerTask}`);
  if (spec.nodeName) detailLines.push(`  node_name:     ${spec.nodeName}`);
  if (result.code !== null && result.code !== undefined) detailLines.push(`  exit_code:     ${result.code}`);
  if (result.signal) detailLines.push(`  signal:        ${result.signal}`);
  return `${detailLines.join('\n')}\n\n=== STEP OUTPUT ===\n${result.output}\n=== END OUTPUT ===`;
}

async function executeSharedSlurmStep(input, defaults = {}) {
  try {
    const spec = resolveSharedSlurmStepInput(input, defaults);
    const invocation = buildSharedSlurmStepCommand(spec);
    const result = await executeLocalSpawn(invocation.command, invocation.args, {
      timeoutMs: parseWalltimeSeconds(input.walltime || '6:00:00') * 1000,
    });
    return formatSharedStepResult('step', spec, result);
  } catch (err) {
    return `[HpcSubmit Error] Shared step failed: ${err.message}`;
  }
}

async function executeSharedSlurmProbe(input, defaults = {}) {
  try {
    const jobId = (
      (typeof input.shared_job_id === 'string' && input.shared_job_id.trim())
      || defaults.shared_job_id
      || defaults.sharedJobId
      || ''
    ).trim();
    if (!jobId) {
      throw new Error('shared_probe requires shared_job_id');
    }
    const invocation = buildSharedSlurmProbeCommand(jobId);
    const result = await executeLocalSpawn(invocation.command, invocation.args, { timeoutMs: 120000 });
    return formatSharedStepResult('probe', { jobId, cpusPerTask: 1 }, result);
  } catch (err) {
    return `[HpcSubmit Error] Shared probe failed: ${err.message}`;
  }
}

/**
 * @param {Object} input   - Tool input from the agent
 * @param {Object} defaults - Project-level HPC defaults from config.yaml hpc section
 */
export async function executeHpcSubmit(input, defaults = {}) {
  if (input?.mode === 'shared_probe') {
    return executeSharedSlurmProbe(input, defaults);
  }
  if (input?.mode === 'shared_step' || input?.shared_job_id || input?.grant_id) {
    return executeSharedSlurmStep(input, defaults);
  }
  return [
    '[HpcSubmit Error] Legacy remote submission mode has been removed.',
    'Use mode="shared_probe" to inspect an existing shared allocation, or mode="shared_step" to attach work to it.',
    'Configure shared_job_id and GPU grants in the project-local orchestration state.',
  ].join('\n');
}
