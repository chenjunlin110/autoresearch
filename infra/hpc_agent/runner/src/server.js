#!/usr/bin/env node
/**
 * HPC Agent - Multi-project AI Agent Orchestrator
 * 
 * Central service that manages multiple repo-based agent projects.
 * Includes the API server and orchestrator endpoints.
 */

import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn, execSync } from 'child_process';
import yaml from 'js-yaml';
import Database from 'better-sqlite3';
import { runAgentWithAPI, runAgentWithClaudeCLI, runAgentWithCodexCLI } from './agent-runner.js';
import {
  buildEmptyResourceSummary,
  createTokenRequest,
  ensureResourceOrchestrationTables,
  getAllocationLease,
  getResourceSummary,
  grantTokenRequest,
  listResourceTokens,
  normalizeResourceOrchestrationConfig,
  releaseTokenGrant,
} from './resource-orchestration.js';
import {
  buildManagedWorkerTask,
  classifyParallelStepGpuFit,
  choosePrimaryActiveRun,
  normalizeDirectExecutorConfig,
  normalizeOrchestrationConfig,
  normalizeScheduleStep,
  parseKillTasksDocument,
  parseScheduleDocument,
  parseTaskGraphDocument,
  selectBackfillParallelStepIndex,
} from './orchestration-utils.js';
import {
  findWorkerForStep,
  inferWorkerClassFromMetadata,
  selectTaskGraphRunnableTask,
} from './scheduler.js';
import { appendTaskEvent } from './task-events.js';
import { extractOutputDirFromTaskBody, validateExperimentResult } from './result-validator.js';
import { remainingWalltimeMs } from './slurm-walltime.js';
import { buildCycleReport, formatCycleReportLine, writeCycleReport } from './cycle-report.js';
import { listComputeProcesses } from './gpu-probe.js';
import { runDirectTask } from './direct-executor.js';
import {
  buildExperimentLedger,
  computeEditConfidence,
  formatLedgerMarkdown,
} from './experiment-ledger.js';
import { listSessions as chatListSessions, createSession as chatCreateSession, getSession as chatGetSession, deleteSession as chatDeleteSession, updateSessionPreferences as chatUpdateSessionPreferences, streamChatMessage, getActiveStream, isStreaming as isChatStreaming, saveMessage as chatSaveMessage } from './chat.js';
import { resolveModel, callModel, buildUserMessage, getModels as getPiModels } from './providers/index.js';
import { buildCustomTierMap, resolveProviderRuntime } from './providers/custom-config.js';
import { startOAuthLogin, submitManualCode, checkOAuthStatus, getAccessToken as getOAuthAccessToken, clearCredentials as clearOAuthCredentials, listOAuthProviders, loadCredentials as loadOAuthCredentials } from './oauth.js';
import {
  loadKeyPool, addKey, addOAuthKey, removeKey, updateKey, reorderKeys,
  getKeyPoolSafe, resolveKeyForProject, markRateLimited, markKeySucceeded, migrateFromEnv,
  detectTokenProvider as detectTokenProviderFromPool,
} from './key-pool.js';
import webpush from 'web-push';
import { config as loadDotenv } from 'dotenv';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');

function expandHomePath(value) {
  if (!value) return value;
  if (value === '~') return process.env.HOME;
  if (value.startsWith('~/')) return path.join(process.env.HOME, value.slice(2));
  return value;
}

// Load local runner .env first, then app-home .env for generated/runtime values.
loadDotenv({ path: path.join(ROOT, '.env') });
const TBC_HOME_EARLY = expandHomePath(process.env.TBC_HOME) || path.join(process.env.HOME, '.hpc_agent');
loadDotenv({ path: path.join(TBC_HOME_EARLY, '.env') });
const TBC_HOME = expandHomePath(process.env.TBC_HOME) || path.join(process.env.HOME, '.hpc_agent');
fs.mkdirSync(TBC_HOME, { recursive: true });

function maskToken(token) {
  if (!token || token.length < 8) return '****';
  return token.slice(0, 4) + '****' + token.slice(-4);
}

function detectTokenProvider(token) {
  if (!token) return null;
  if (token.startsWith('sk-ant-')) return 'anthropic';
  if (token.startsWith('sk-proj-') || token.startsWith('sk-')) return 'openai';
  if (token.startsWith('AIzaSy')) return 'google';
  // MiniMax keys cannot be reliably auto-detected by prefix.
  // Use explicit provider field in project token settings.
  return 'unknown';
}

// Parse retry cooldown from rate-limit error messages
function parseSummarizeCooldown(message) {
  if (!message) return 5 * 60_000;
  const minMatch = message.match(/~?(\d+)\s*min/i);
  if (minMatch) return parseInt(minMatch[1]) * 60_000;
  const hourMatch = message.match(/(\d+)\s*h(?:ours?)?/i);
  if (hourMatch) return parseInt(hourMatch[1]) * 3600_000;
  const secMatch = message.match(/(\d+)\s*s(?:ec(?:onds?)?)?/i);
  if (secMatch) return parseInt(secMatch[1]) * 1000;
  return 5 * 60_000;
}

function normalizeAgentRuntime(value) {
  const runtime = String(value || '').trim().toLowerCase();
  if (runtime === 'api') return 'api';
  if (runtime === 'codex-cli' || runtime === 'codex_cli' || runtime === 'codex') return 'codex_cli';
  if (runtime === 'claude-cli' || runtime === 'claude_cli' || runtime === 'claude' || runtime === 'claude-code' || runtime === 'claude_code') return 'claude_cli';
  return 'codex_cli';
}

// Model tier system — maps abstract tiers to provider-specific models
const MODEL_TIERS = {
  anthropic: {
    high:  { model: 'claude-opus-4-6', reasoningEffort: 'high' },
    mid:   { model: 'claude-sonnet-4-6', reasoningEffort: 'high' },
    low:   { model: 'claude-sonnet-4-6' },
    xlow:  { model: 'claude-haiku-4-5-20251001' },
  },
  openai: {
    high:  { model: 'gpt-5.3-codex', reasoningEffort: 'xhigh' },
    mid:   { model: 'gpt-5.3-codex', reasoningEffort: 'high' },
    low:   { model: 'gpt-5.3-codex', reasoningEffort: 'medium' },
    xlow:  { model: 'gpt-4.1-mini' },
  },
  google: {
    high:  { model: 'gemini-3.1-pro-preview', reasoningEffort: 'high' },
    mid:   { model: 'gemini-3.1-pro-preview', reasoningEffort: 'medium' },
    low:   { model: 'gemini-3-flash-preview' },
    xlow:  { model: 'gemini-3-flash-preview' },
  },
  minimax: {
    high:  { model: 'minimax/MiniMax-M2.5' },
    mid:   { model: 'minimax/MiniMax-M2.5' },
    low:   { model: 'minimax/MiniMax-M2.5' },
    xlow:  { model: 'minimax/MiniMax-M2.5' },
  },
  'openai-codex': {
    high:  { model: 'openai-codex/gpt-5.3-codex', reasoningEffort: 'xhigh' },
    mid:   { model: 'openai-codex/gpt-5.3-codex', reasoningEffort: 'high' },
    low:   { model: 'openai-codex/gpt-5.3-codex', reasoningEffort: 'medium' },
    xlow:  { model: 'openai-codex/gpt-5.3-codex', reasoningEffort: 'low' },
  },
};

function inferProviderFromModel(model) {
  const raw = String(model || '').trim().toLowerCase();
  if (!raw) return null;
  if (raw.startsWith('openai-codex/')) return 'openai-codex';
  if (raw.startsWith('openai/')) return 'openai';
  if (raw.startsWith('anthropic/')) return 'anthropic';
  if (raw.startsWith('google/') || raw.startsWith('gemini/')) return 'google';
  if (raw.startsWith('minimax/')) return 'minimax';
  if (raw.startsWith('claude-')) return 'anthropic';
  if (raw.startsWith('gpt-') || raw.startsWith('o1') || raw.startsWith('o3') || raw.startsWith('o4')) return 'openai';
  if (raw.startsWith('gemini-')) return 'google';
  return null;
}

function resolveModelTier(tierOrModel, provider, projectModels) {
  const tier = (tierOrModel || '').toLowerCase().trim();
  // Project-level model overrides take priority only when compatible with the
  // currently selected provider.
  if (projectModels && projectModels[tier]) {
    const override = projectModels[tier];
    const overrideModel = override.includes('@') ? override.split('@', 2)[0] : override;
    const overrideProvider = inferProviderFromModel(overrideModel);
    if (!overrideProvider || overrideProvider === provider) {
      // Support "model@effort" format (e.g. "gpt-5.3-codex@xhigh")
      if (override.includes('@')) {
        const [model, reasoningEffort] = override.split('@', 2);
        return { model, reasoningEffort };
      }
      return { model: override };
    }
  }
  const tiers = MODEL_TIERS[provider];
  if (tiers && tiers[tier]) {
    return tiers[tier];
  }
  // Not a tier — treat as explicit model name
  return { model: tierOrModel };
}

function getProviderRuntimeSelection({ provider, modelTier, keyResult, projectModels }) {
  return resolveProviderRuntime({
    provider,
    modelTier,
    keyResult,
    projectModels,
    resolveModelTier,
  });
}

function parseExplicitModelSelection(model) {
  const value = typeof model === 'string' ? model.trim() : '';
  if (!value) return { model: null, reasoningEffort: null };
  if (value.includes('@')) {
    const [selectedModel, reasoningEffort] = value.split('@', 2);
    return {
      model: selectedModel || null,
      reasoningEffort: reasoningEffort || null,
    };
  }
  return { model: value, reasoningEffort: null };
}

function formatStoredChatErrorMessage({ error, statusCode, source, cooldownMs }) {
  if (source === 'local_cooldown') {
    return `This key is currently rate limited by HPC Agent${cooldownMs ? ` for about ${Math.ceil(cooldownMs / 60_000)}m` : ''}.`;
  }
  if (source === 'provider_429' || statusCode === 429) {
    return `Provider returned a 429/rate-limit error.${error ? `\n\n${error}` : ''}`;
  }
  if ((statusCode || 0) >= 500) {
    return `Server error (${statusCode}).${error ? `\n\n${error}` : ''}`;
  }
  return error || 'Failed to send message.';
}

function detectProviderFromToken(token) {
  if (!token) return 'anthropic';
  const p = detectTokenProvider(token);
  return (p === 'unknown' || !p) ? 'anthropic' : p;
}

// Strip meta directive blocks from agent responses (keep human-readable text only)
function stripMetaBlocks(text) {
  if (!text) return text;
  return text
    .replace(/<!--\s*(SCHEDULE|MILESTONE|CLAIM_COMPLETE|VERIFY_PASS|VERIFY_FAIL|EXAM_PASS|EXAM_FAIL)\s*-->[\s\S]*?<!--\s*\/\1\s*-->/g, '')
    .replace(/<!--\s*(CLAIM_COMPLETE|VERIFY_PASS|VERIFY_FAIL|EXAM_PASS|EXAM_FAIL)\s*-->/g, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

// --- Web Push (VAPID) --- Auto-generate keys if missing
let VAPID_PUBLIC = process.env.VAPID_PUBLIC_KEY;
let VAPID_PRIVATE = process.env.VAPID_PRIVATE_KEY;
const VAPID_EMAIL = process.env.VAPID_EMAIL || 'mailto:admin@example.com';
if (!VAPID_PUBLIC || !VAPID_PRIVATE) {
  const envPath = path.join(TBC_HOME, '.env');
  const vapidKeys = webpush.generateVAPIDKeys();
  VAPID_PUBLIC = vapidKeys.publicKey;
  VAPID_PRIVATE = vapidKeys.privateKey;
  // Append to .env file
  let envContent = '';
  try { envContent = fs.readFileSync(envPath, 'utf-8'); } catch {}
  const lines = [];
  if (!envContent.includes('VAPID_PUBLIC_KEY=')) lines.push(`VAPID_PUBLIC_KEY=${VAPID_PUBLIC}`);
  if (!envContent.includes('VAPID_PRIVATE_KEY=')) lines.push(`VAPID_PRIVATE_KEY=${VAPID_PRIVATE}`);
  if (!envContent.includes('VAPID_EMAIL=')) lines.push(`VAPID_EMAIL=${VAPID_EMAIL}`);
  if (lines.length) {
    fs.appendFileSync(envPath, (envContent.endsWith('\n') ? '' : '\n') + lines.join('\n') + '\n');
    log('Auto-generated VAPID keys and saved to .env');
  }
}
if (VAPID_PUBLIC && VAPID_PRIVATE) {
  webpush.setVapidDetails(VAPID_EMAIL, VAPID_PUBLIC, VAPID_PRIVATE);
}
const pushSubscriptions = new Map(); // endpoint -> subscription

// --- Configuration ---
const PORT = process.env.TBC_PORT || 3100;
const ALLOW_CUSTOM_PROVIDER = process.env.TBC_ALLOW_CUSTOM_PROVIDER === 'true';

// Ensure TBC_HOME exists
if (!fs.existsSync(TBC_HOME)) {
  fs.mkdirSync(TBC_HOME, { recursive: true });
}
if (!fs.existsSync(path.join(TBC_HOME, 'logs'))) {
  fs.mkdirSync(path.join(TBC_HOME, 'logs'), { recursive: true });
}

// --- State ---
const projects = new Map(); // projectId -> ProjectRunner

// --- SSE notification system ---
const sseClients = new Set();
const notifications = []; // In-memory notification store
const MAX_NOTIFICATIONS = 200;

// Throttled status broadcasts — at most once per second per project
const _statusBroadcastTimers = new Map();
function broadcastStatusUpdate(projectId) {
  if (_statusBroadcastTimers.has(projectId)) return; // already scheduled
  _statusBroadcastTimers.set(projectId, setTimeout(() => {
    _statusBroadcastTimers.delete(projectId);
    const runner = projects.get(projectId);
    if (!runner) return;
    const data = JSON.stringify({ type: 'status-update', project: projectId, status: runner.getStatus() });
    for (const client of sseClients) {
      client.write(`data: ${data}\n\n`);
    }
  }, 500)); // 500ms debounce
}

function broadcastReportUpdate(projectId, reportId, agent, cycle) {
  const data = JSON.stringify({ type: 'report-new', project: projectId, reportId, agent, cycle });
  for (const client of sseClients) {
    client.write(`data: ${data}\n\n`);
  }
}

function broadcastLiveAgentEvent(projectId, event) {
  const data = JSON.stringify({ type: 'agent-log-event', project: projectId, event });
  for (const client of sseClients) {
    client.write(`data: ${data}\n\n`);
  }
}

function broadcastEvent(event) {
  const messages = {
    milestone: `📌 New milestone: ${event.title}`,
    verified: `✅ Milestone verified: ${event.title}`,
    'verify-fail': `❌ Verification failed: ${event.title}`,
    phase: `🔄 Phase → ${event.phase}`,
    error: `⚠️ ${event.message}`,
    'agent-done': `${event.success ? '✓' : '✗'} ${event.agent}: ${event.summary || 'no response'}`,
    'project-complete': `🏁 Project ${event.success ? 'completed' : 'ended'}: ${event.message}`,
  };
  const notification = {
    id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
    type: event.type,
    project: event.project,
    message: messages[event.type] || JSON.stringify(event),
    timestamp: new Date().toISOString(),
    read: false,
    detailed: event.type === 'agent-done',
  };
  notifications.unshift(notification);
  if (notifications.length > MAX_NOTIFICATIONS) notifications.length = MAX_NOTIFICATIONS;
  const data = JSON.stringify({ ...event, notification });
  for (const client of sseClients) {
    client.write(`data: ${data}\n\n`);
  }
  // Web Push
  if (VAPID_PUBLIC && pushSubscriptions.size > 0) {
    const pushPayload = JSON.stringify({
      title: `HPC Agent: ${event.project || ''}`,
      body: notification.message,
      tag: `hpc-agent-${event.type}-${event.project}`,
      detailed: notification.detailed || false,
    });
    for (const [endpoint, sub] of pushSubscriptions) {
      webpush.sendNotification(sub, pushPayload).catch(err => {
        if (err.statusCode === 404 || err.statusCode === 410) {
          pushSubscriptions.delete(endpoint);
        }
      });
    }
  }
}
const startTime = Date.now();

// --- Logging ---
function log(msg, projectId = null) {
  const ts = new Date().toLocaleString('sv-SE', { hour12: false }).replace(',', '');
  const prefix = projectId ? `[${projectId}]` : '[hpc-agent]';
  const line = `${ts} ${prefix} ${msg}`;
  console.log(line);
  if (projectId) {
    const runner = projects.get(projectId);
    if (runner) {
      const logPath = runner.orchestratorLogPath;
      try { fs.appendFileSync(logPath, line + '\n'); } catch {}
    }
  }
}

// --- GitHub URL Parser ---
function parseGithubUrl(url) {
  const match = url.match(/github\.com\/([^\/]+)\/([^\/\s.]+)/);
  if (!match) return null;
  const [, username, reponame] = match;
  const id = `${username}/${reponame}`;
  const projectDir = path.join(TBC_HOME, 'dev', 'src', 'github.com', username, reponame);
  const repoDir = path.join(projectDir, 'repo');
  const cloneUrl = `https://github.com/${username}/${reponame}.git`;
  return { id, username, reponame, projectDir, repoDir, cloneUrl };
}

function normalizeLocalProjectId(value) {
  const id = String(value || '').trim();
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(id)) {
    throw new Error('Project ID must start with a letter or number and contain only letters, numbers, dots, underscores, and dashes.');
  }
  return id;
}

function resolveProjectDataDir(id, projectPath) {
  try {
    const remoteUrl = execSync('git remote get-url origin', {
      cwd: projectPath,
      encoding: 'utf-8',
      timeout: 5000,
      stdio: 'pipe',
    }).trim();
    const match = remoteUrl.match(/github\.com[:/]([^/]+)\/([^/.]+)/);
    if (match) {
      return path.join(TBC_HOME, 'dev', 'src', 'github.com', match[1], match[2]);
    }
  } catch {}
  return path.join(TBC_HOME, 'local', id);
}

function readInitialSpec(projectPath, projectDataDir) {
  const knowledgeSpecPath = path.join(projectDataDir, 'knowledge', 'spec.md');
  const repoSpecPath = path.join(projectPath, 'spec.md');
  const specPath = fs.existsSync(knowledgeSpecPath) ? knowledgeSpecPath : repoSpecPath;
  if (!fs.existsSync(specPath)) return { hasSpec: false, specContent: null };
  return { hasSpec: true, specContent: fs.readFileSync(specPath, 'utf-8') };
}

function writeProjectSpec(projectDataDir, spec) {
  if (!spec || (!spec.whatToBuild && !spec.successCriteria)) return;
  const knowledgeDir = path.join(projectDataDir, 'knowledge');
  const specPath = path.join(knowledgeDir, 'spec.md');
  const specContent = `# Project Specification\n\n## What do you want to build?\n\n${spec.whatToBuild || ''}\n\n## How do you consider the project is success?\n\n${spec.successCriteria || ''}\n`;
  fs.mkdirSync(knowledgeDir, { recursive: true });
  fs.writeFileSync(specPath, specContent);
}

// --- Project Runner ---
class ProjectRunner {
  constructor(id, config) {
    this.id = id;
    this.path = config.path.replace(/^~/, process.env.HOME);
    // Optional explicit data dir from projects.yaml. Takes precedence over the
    // git-remote-derived path so workspace generators can pin the runner's
    // reads to the same directory they wrote (config.yaml, skills, state.json,
    // project.db all live there).
    this.explicitDataDir = typeof config.dataDir === 'string' && config.dataDir.trim()
      ? config.dataDir.replace(/^~/, process.env.HOME)
      : null;
    this.enabled = config.enabled !== false;
    this.archived = config.archived === true;
    this.cycleCount = 0;   // Cycles: manager + worker runs
    this.epochCount = 0;   // Epochs: full Athena → implementation → verification → Athena loops
    this.currentAgent = null;
    this.currentAgentProcess = null;
    this.currentAgentStartTime = null;
    this.isPaused = false;
    this.sleepUntil = null;
    this.wakeNow = false;
    this.running = false;
    this.lastComputedSleepMs = null; // Cached sleep interval
    this.currentSchedule = null;
    this.abortCurrentCycle = false;
    this.phase = 'idle';
    this.isComplete = false;
    this.completionSuccess = false;
    this.completionMessage = null;
    this.consecutiveFailures = 0; // Track consecutive agent failures for auto-pause
    this.currentAgentLog = [];
    this.currentAgentModel = null; this.currentAgentCost = 0; this.currentAgentUsage = null; this.currentAgentKeyId = null; this.currentAgentVisibility = null;
    this.resourceOrchestrationSnapshot = null;
    this.activeRuns = new Map();
    this.nextRunId = 1;
    this.currentRunId = null;
    this._repo = null;
  }

  /**
   * Centralized state mutation with invariant enforcement.
   * All state changes should go through this method.
   * Automatically saves state and enforces consistency rules.
   */
  setState(patch, { save = true } = {}) {
    // Apply patch
    for (const [key, value] of Object.entries(patch)) {
      this[key] = value;
    }

    // Enforce invariants
    // 1. Complete projects are always paused
    if (this.isComplete) {
      this.isPaused = true;
    }
    // 2. Paused projects have no sleep countdown
    if (this.isPaused) {
      this.sleepUntil = null;
      this.lastComputedSleepMs = null;
    }
    if (save) this.saveState();

    // Broadcast status update via SSE for instant dashboard refresh
    broadcastStatusUpdate(this.id);
  }

  get projectDir() {
    if (this.explicitDataDir) return this.explicitDataDir;
    const repo = this.repo;
    if (repo) {
      return path.join(TBC_HOME, 'dev', 'src', 'github.com', ...repo.split('/'));
    }
    return path.join(TBC_HOME, 'local', this.id);
  }

  get agentDir() {
    return this.projectDir;
  }

  get chatsDir() {
    return path.join(this.projectDir, 'chats');
  }

  get agentsDir() {
    return path.join(this.projectDir, 'agents');
  }

  get responsesDir() {
    return path.join(this.projectDir, 'responses');
  }

  get uploadsDir() {
    return path.join(this.projectDir, 'uploads');
  }

  get projectDbPath() {
    return path.join(this.projectDir, 'project.db');
  }

  get orchestratorLogPath() {
    return path.join(this.projectDir, 'orchestrator.log');
  }

  get statePath() {
    return path.join(this.projectDir, 'state.json');
  }

  get taskEventsDir() {
    return path.join(this.projectDir, 'task-events');
  }

  get stopPath() {
    return path.join(this.projectDir, 'STOP');
  }

  getAgentNotesDir(agentName) {
    return path.join(this.agentsDir, agentName);
  }

  getOperationalPaths() {
    return [
      this.projectDbPath,
      this.orchestratorLogPath,
      this.responsesDir,
      this.agentsDir,
      this.uploadsDir,
      this.statePath,
      this.stopPath,
    ];
  }

  get skillsDir() {
    return path.join(this.projectDir, 'skills');
  }

  get knowledgeDir() {
    return path.join(this.projectDir, 'knowledge');
  }

  get workerSkillsDir() {
    return path.join(this.skillsDir, 'workers');
  }

  get repo() {
    if (this._repo === null) {
      try {
        const remoteUrl = execSync('git remote get-url origin', {
          cwd: this.path,
          encoding: 'utf-8',
          stdio: 'pipe',
        }).trim();
        const match = remoteUrl.match(/github\.com[:/]([^/]+\/[^/.]+)/);
        this._repo = match ? match[1] : null;
      } catch {
        this._repo = null;
      }
    }
    return this._repo;
  }

  get configPath() {
    return path.join(this.projectDir, 'config.yaml');
  }

  loadConfig() {
    const defaults = {
      cycleIntervalMs: 0,
      agentTimeoutMs: 3600000,
      model: 'mid',
      agentRuntime: 'codex_cli',
      budgetPer24h: 0,
      orchestration: normalizeOrchestrationConfig(),
      resourceOrchestration: normalizeResourceOrchestrationConfig(),
    };
    try {
      const raw = fs.readFileSync(this.configPath, 'utf-8');
      const config = yaml.load(raw) || {};
      // Validate numeric fields
      for (const key of ['cycleIntervalMs', 'agentTimeoutMs', 'budgetPer24h']) {
        if (config[key] !== undefined && (typeof config[key] !== 'number' || config[key] < 0)) {
          log(`WARNING: Invalid ${key} in config.yaml (${config[key]}), using default`, this.id);
          config[key] = defaults[key];
        }
      }
      return {
        ...defaults,
        ...config,
        agentRuntime: normalizeAgentRuntime(config.agentRuntime || defaults.agentRuntime),
        orchestration: normalizeOrchestrationConfig(config.orchestration),
        resourceOrchestration: normalizeResourceOrchestrationConfig(config.resourceOrchestration),
        directExecutor: normalizeDirectExecutorConfig(config.directExecutor),
      };
    } catch (e) {
      return defaults;
    }
  }

  saveConfig(content) {
    fs.mkdirSync(this.projectDir, { recursive: true });
    yaml.load(content); // Validate YAML
    fs.writeFileSync(this.configPath, content);
  }

  getOrchestrationConfig(config = null) {
    const source = config || this.loadConfig();
    return normalizeOrchestrationConfig(source.orchestration);
  }

  getAgentRuntime(config = null) {
    const source = config || this.loadConfig();
    return normalizeAgentRuntime(source.agentRuntime);
  }

  isSingleManagerMode(config = null) {
    return this.getOrchestrationConfig(config).mode === 'single_manager';
  }

  _createRunState(agent, { mode = null, task = null, visibility = null, primary = false, managerName = null, resources = null, resourceGrant = null, taskId = null } = {}) {
    const runState = {
      id: `run-${this.nextRunId++}`,
      agentName: agent.name,
      isManager: !!agent.isManager,
      isPrimary: !!primary,
      mode,
      task,
      managerName,
      visibility: visibility || { mode: 'full', issues: [] },
      resources,
      resourceGrant,
      startTime: Date.now(),
      log: [],
      model: null,
      cost: 0,
      usage: null,
      keyId: null,
      abortController: null,
      taskId,
    };
    this.activeRuns.set(runState.id, runState);
    this._syncCurrentAgentSnapshot();
    return runState;
  }

  _clearCurrentAgentSnapshot({ broadcast = true } = {}) {
    this.currentRunId = null;
    this.currentAgent = null;
    this.currentAgentProcess = null;
    this.currentAgentStartTime = null;
    this.currentAgentLog = [];
    this.currentAgentModel = null;
    this.currentAgentCost = 0;
    this.currentAgentUsage = null;
    this.currentAgentKeyId = null;
    this.currentAgentVisibility = null;
    if (broadcast) broadcastStatusUpdate(this.id);
  }

  _syncCurrentAgentSnapshot() {
    const selected = choosePrimaryActiveRun([...this.activeRuns.values()]);
    if (!selected) {
      this._clearCurrentAgentSnapshot();
      return;
    }
    this.currentRunId = selected.id;
    this.currentAgent = selected.agentName;
    this.currentAgentProcess = selected.abortController ? {
      kill: () => selected.abortController.abort(),
    } : null;
    this.currentAgentStartTime = selected.startTime;
    this.currentAgentLog = selected.log;
    this.currentAgentModel = selected.model;
    this.currentAgentCost = selected.cost || 0;
    this.currentAgentUsage = selected.usage || null;
    this.currentAgentKeyId = selected.keyId || null;
    this.currentAgentVisibility = selected.visibility || { mode: 'full', issues: [] };
    broadcastStatusUpdate(this.id);
  }

  _removeRunState(runState) {
    if (!runState?.id) return;
    this.activeRuns.delete(runState.id);
    this._syncCurrentAgentSnapshot();
  }

  _abortAllRuns() {
    for (const run of this.activeRuns.values()) {
      try { run.abortController?.abort(); } catch {}
    }
  }

  loadAgents() {
    const managers = [];
    const workers = [];
    
    const managersDir = path.join(ROOT, 'agent', 'managers');
    const workersDir = this.workerSkillsDir;
    
    const parseRole = (content) => {
      // Prefer frontmatter role: field
      const fmRole = (content.match(/^role:\s*(.+)$/m) || [])[1]?.trim();
      if (fmRole) return fmRole;
      // Fallback: match "# Name (Role)" in markdown
      const match = content.match(/^#\s*\w+\s*\(([^)]+)\)/m);
      return match ? match[1] : null;
    };

    const shortenModel = (model) => {
      if (!model) return null;
      const versionMatch = model.match(/(opus|sonnet|haiku)-(\d+)(?:-(\d+))?/i);
      if (versionMatch) {
        const name = versionMatch[1].toLowerCase();
        const major = versionMatch[2];
        const minor = versionMatch[3];
        return minor && minor.length <= 2 ? `${name} ${major}.${minor}` : `${name} ${major}`;
      }
      if (model.includes('opus')) return 'opus';
      if (model.includes('sonnet')) return 'sonnet';
      if (model.includes('haiku')) return 'haiku';
      return model;
    };
    
    const parseModel = (content) => {
      const match = content.match(/^model:\s*(.+)$/m);
      return match ? shortenModel(match[1].trim()) : null;
    };
    
    const config = this.loadConfig();
    const managerOverrides = config.managers || {};
    
    if (fs.existsSync(managersDir)) {
      for (const file of fs.readdirSync(managersDir)) {
        if (file.endsWith('.md')) {
          const name = file.replace('.md', '');
          const content = fs.readFileSync(path.join(managersDir, file), 'utf-8');
          const overrides = managerOverrides[name] || {};
          // Disabled check: config override takes priority, then frontmatter
          const isDisabled = overrides.disabled !== undefined ? overrides.disabled : /^disabled:\s*true$/m.test(content);
          if (isDisabled) continue;
          // Model: config override takes priority, then frontmatter
          const frontmatterModel = (content.match(/^model:\s*(.+)$/m) || [])[1]?.trim() || null;
          const rawModel = overrides.model || frontmatterModel;
          managers.push({
            name,
            role: parseRole(content),
            displayName: overrides.displayName || null,
            model: shortenModel(rawModel),
            rawModel,
            isManager: true,
          });
        }
      }
    }
    
    const workerOverrides = config.workers || {};
    if (fs.existsSync(workersDir)) {
      for (const file of fs.readdirSync(workersDir)) {
        if (file.endsWith('.md')) {
          const name = file.replace('.md', '');
          const content = fs.readFileSync(path.join(workersDir, file), 'utf-8');
          if (/^disabled:\s*true$/m.test(content)) continue;
          const reportsTo = (content.match(/^reports_to:\s*(.+)$/m) || [])[1]?.trim() || null;
          const role = parseRole(content);
          workers.push({
            name,
            role,
            displayName: workerOverrides[name]?.displayName || null,
            model: parseModel(content),
            rawModel: (content.match(/^model:\s*(.+)$/m) || [])[1]?.trim() || null,
            isManager: false,
            reportsTo,
            workerClass: inferWorkerClassFromMetadata({ content, role, name }),
          });
        }
      }
    }
    
    const costSummary = this.getCostSummary();
    for (const agent of [...managers, ...workers]) {
      const agentCost = costSummary.agents[agent.name];
      agent.totalCost = agentCost ? agentCost.totalCost : 0;
      agent.last24hCost = agentCost ? agentCost.last24hCost : 0;
      agent.lastCallCost = agentCost ? agentCost.lastCallCost : 0;
      agent.avgCallCost = agentCost ? agentCost.avgCallCost : 0;
      agent.callCount = agentCost ? agentCost.callCount : 0;
    }

    return { managers, workers };
  }

  getAgentDetails(agentName) {
    const workersDir = this.workerSkillsDir;
    const managersDir = path.join(ROOT, 'agent', 'managers');
    const agentNotesDir = this.getAgentNotesDir(agentName);
    
    let skillPath = path.join(workersDir, `${agentName}.md`);
    let isManager = false;
    if (!fs.existsSync(skillPath)) {
      skillPath = path.join(managersDir, `${agentName}.md`);
      isManager = true;
    }
    
    if (!fs.existsSync(skillPath)) {
      return null;
    }
    
    const skill = fs.readFileSync(skillPath, 'utf-8');
    
    let agentFiles = [];
    if (fs.existsSync(agentNotesDir)) {
      agentFiles = fs.readdirSync(agentNotesDir).flatMap(f => {
        const filePath = path.join(agentNotesDir, f);
        const stat = fs.statSync(filePath);
        if (!stat.isFile()) return [];
        return [{
          name: f,
          size: stat.size,
          modified: stat.mtime,
          content: stat.size < 50000 ? fs.readFileSync(filePath, 'utf-8') : null
        }];
      });
    }
    
    // Get last response from response log
    let lastResponse = null;
    let lastRawOutput = null;
    const responseLogPath = path.join(this.responsesDir, `${agentName}.log`);
    const rawLogPath = path.join(this.responsesDir, `${agentName}.raw.log`);
    
    const getLastBlock = (filePath, maxChars = 15000) => {
      if (!fs.existsSync(filePath)) return null;
      try {
        const content = fs.readFileSync(filePath, 'utf-8');
        const blocks = content.split(/={60,}/);
        if (blocks.length >= 2) {
          const lastBlock = blocks.slice(-2).join('').trim();
          return lastBlock.length > maxChars ? lastBlock.slice(-maxChars) : lastBlock;
        }
      } catch {}
      return null;
    };
    
    lastResponse = getLastBlock(responseLogPath);
    lastRawOutput = getLastBlock(rawLogPath);
    
    // Extract model from frontmatter
    const modelMatch = skill.match(/^model:\s*(.+)$/m);
    const frontmatterModel = modelMatch ? modelMatch[1].trim() : null;
    
    // Check config override (config.managers.<name>.model or config.workers.<name>.model)
    const config = this.loadConfig();
    const overrides = (isManager ? config.managers : config.workers) || {};
    const configModel = overrides[agentName]?.model || null;
    const model = configModel || frontmatterModel || null;
    
    // Read shared rules: everyone.md + role-specific (worker.md or manager.md)
    let everyone = null;
    let roleRules = null;
    try { everyone = fs.readFileSync(path.join(ROOT, 'agent', 'everyone.md'), 'utf-8'); } catch {}
    try { roleRules = fs.readFileSync(path.join(ROOT, 'agent', isManager ? 'manager.md' : 'worker.md'), 'utf-8'); } catch {}

    // Resolve role (from frontmatter) and displayName (from config override)
    const role = (skill.match(/^role:\s*(.+)$/m) || [])[1]?.trim() || null;
    const cfg = this.loadConfig();
    const bucket = isManager ? (cfg.managers || {}) : (cfg.workers || {});
    const displayName = bucket[agentName]?.displayName || null;

    return { name: agentName, isManager, role, displayName, skill, agentFiles, lastResponse, lastRawOutput, model, everyone, roleRules };
  }

  getLogs(lines = 50) {
    const logPath = this.orchestratorLogPath;
    if (!fs.existsSync(logPath)) return [];
    const content = fs.readFileSync(logPath, 'utf-8');
    return content.split('\n').filter(l => l.trim()).slice(-lines);
  }

  getCostSummary() {
    const empty = { totalCost: 0, last24hCost: 0, lastCycleCost: 0, avgCycleCost: 0, lastCycleDuration: 0, avgCycleDuration: 0, agents: {} };
    try {
      const db = this.getDb();
      // Ensure cost columns exist
      try { db.exec('ALTER TABLE reports ADD COLUMN cost REAL'); } catch {}
      try { db.exec('ALTER TABLE reports ADD COLUMN duration_ms INTEGER'); } catch {}

      const cutoff = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();

      // Total cost
      const totalCost = db.prepare('SELECT COALESCE(SUM(cost), 0) as v FROM reports').get().v;
      const last24hCost = db.prepare('SELECT COALESCE(SUM(cost), 0) as v FROM reports WHERE created_at > ?').get(cutoff).v;

      // Per-cycle data
      const cycles = db.prepare('SELECT cycle, SUM(cost) as cost, SUM(duration_ms) as duration FROM reports WHERE cost IS NOT NULL GROUP BY cycle ORDER BY cycle ASC').all();
      let lastCycleCost = 0, avgCycleCost = 0, lastCycleDuration = 0, avgCycleDuration = 0;
      if (cycles.length > 0) {
        const last = cycles[cycles.length - 1];
        lastCycleCost = last.cost || 0;
        lastCycleDuration = last.duration || 0;
        const totalCycleCost = cycles.reduce((s, c) => s + (c.cost || 0), 0);
        const totalCycleDuration = cycles.reduce((s, c) => s + (c.duration || 0), 0);
        avgCycleCost = totalCycleCost / cycles.length;
        avgCycleDuration = totalCycleDuration / cycles.length;
      }

      // Per-agent data
      const agentRows = db.prepare(`SELECT agent,
        COALESCE(SUM(cost), 0) as totalCost,
        COALESCE(SUM(CASE WHEN created_at > ? THEN cost ELSE 0 END), 0) as last24hCost,
        COUNT(*) as callCount
        FROM reports WHERE cost IS NOT NULL GROUP BY agent`).all(cutoff);
      const agents = {};
      for (const row of agentRows) {
        const lastCall = db.prepare('SELECT cost FROM reports WHERE agent = ? AND cost IS NOT NULL ORDER BY id DESC LIMIT 1').get(row.agent);
        agents[row.agent] = {
          totalCost: row.totalCost,
          last24hCost: row.last24hCost,
          callCount: row.callCount,
          lastCallCost: lastCall?.cost || 0,
          avgCallCost: row.callCount > 0 ? row.totalCost / row.callCount : 0,
        };
      }

      db.close();

      return { totalCost, last24hCost, lastCycleCost, avgCycleCost, lastCycleDuration, avgCycleDuration, agents };
    } catch {
      return empty;
    }
  }

  computeSleepInterval() {
    const config = this.loadConfig();
    const budgetPer24h = config.budgetPer24h || 0;
    const MIN_SLEEP = 10000;       // 10s
    const MAX_SLEEP = 7200000;     // 2h

    // If no budget set, fall back to fixed interval
    if (budgetPer24h <= 0) {
      return Math.max(config.cycleIntervalMs || 0, MIN_SLEEP);
    }

    const minFloor = config.cycleIntervalMs > 0 ? config.cycleIntervalMs : MIN_SLEEP;

    // Query cost data from SQLite reports table
    const cutoff = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
    let cycleCosts = [];
    let spent24h = 0;
    let oldestTime24h = Infinity;

    try {
      const db = this.getDb();
      try { db.exec('ALTER TABLE reports ADD COLUMN cost REAL'); } catch {}
      try { db.exec('ALTER TABLE reports ADD COLUMN duration_ms INTEGER'); } catch {}

      spent24h = db.prepare('SELECT COALESCE(SUM(cost), 0) as v FROM reports WHERE created_at > ?').get(cutoff).v;
      const oldest = db.prepare('SELECT MIN(created_at) as v FROM reports WHERE created_at > ?').get(cutoff);
      if (oldest?.v) oldestTime24h = new Date(oldest.v).getTime();

      const cycles = db.prepare('SELECT cycle, SUM(cost) as cost, MAX(duration_ms) as duration FROM reports WHERE cost IS NOT NULL GROUP BY cycle ORDER BY cycle ASC').all();
      cycleCosts = cycles.map(c => ({ cost: c.cost || 0, duration: c.duration || 0 }));
      db.close();
    } catch {}

    const remaining = budgetPer24h - spent24h;

    // Budget exhaustion: sleep until oldest entry rolls off
    if (remaining <= 0) {
      if (oldestTime24h < Infinity) {
        const rolloffAt = oldestTime24h + 24 * 60 * 60 * 1000;
        const waitMs = Math.max(rolloffAt - Date.now(), MIN_SLEEP);
        log(`Budget exhausted ($${spent24h.toFixed(2)}/$${budgetPer24h}), sleeping until oldest entry rolls off`, this.id);
        return Math.min(waitMs, MAX_SLEEP);
      }
      log(`Budget exhausted, sleeping max`, this.id);
      return MAX_SLEEP;
    }

    const n = cycleCosts.length;

    // Cold start: no historical data
    if (n === 0) {
      const { managers, workers } = this.loadAgents();
      const agentCount = managers.length + workers.length || 3;
      const model = (config.model || '').toLowerCase();
      let perAgentCost;
      if (model.includes('opus')) perAgentCost = 2.50;
      else if (model.includes('haiku')) perAgentCost = 0.50;
      else perAgentCost = 1.50; // sonnet default

      const estimatedCycleCost = perAgentCost * agentCount;
      const agentTimeout = config.agentTimeoutMs > 0 ? config.agentTimeoutMs : 900000;
      const estimatedCycleDuration = (agentTimeout / 2) * agentCount;

      const nAffordable = Math.floor(remaining / (estimatedCycleCost * 1.5)); // k=1.5 for cold start
      if (nAffordable <= 0) return MAX_SLEEP;
      const sleepMs = (86400000 / nAffordable) - estimatedCycleDuration;
      log(`Cold start: est cycle cost $${estimatedCycleCost.toFixed(2)}, affordable=${nAffordable}, sleep=${Math.round(sleepMs / 1000)}s`, this.id);
      return Math.max(minFloor, Math.min(sleepMs, MAX_SLEEP));
    }

    // Compute EMA of cycle costs and durations (alpha=0.3) with outlier dampening
    const alpha = 0.3;
    let emaCost = cycleCosts[0].cost;
    let emaDuration = cycleCosts[0].duration;

    for (let i = 1; i < n; i++) {
      let cycleCost = cycleCosts[i].cost;

      // Outlier dampening: if cost > 3x EMA and we have >= 3 data points, clamp to 2x EMA
      if (i >= 3 && cycleCost > 3 * emaCost) {
        cycleCost = 2 * emaCost;
      }

      emaCost = alpha * cycleCost + (1 - alpha) * emaCost;
      emaDuration = alpha * cycleCosts[i].duration + (1 - alpha) * emaDuration;
    }

    // Conservatism factor: k = 1.0 + 0.5 / sqrt(n)
    const k = 1.0 + 0.5 / Math.sqrt(n);

    const nAffordable = Math.floor(remaining / (emaCost * k));
    if (nAffordable <= 0) {
      log(`Budget nearly exhausted (remaining=$${remaining.toFixed(2)}, est/cycle=$${emaCost.toFixed(2)}), sleeping max`, this.id);
      return MAX_SLEEP;
    }

    const sleepMs = (86400000 / nAffordable) - emaDuration;
    log(`Budget: $${spent24h.toFixed(2)}/$${budgetPer24h} spent, est/cycle=$${emaCost.toFixed(2)}, k=${k.toFixed(2)}, affordable=${nAffordable}, sleep=${Math.round(Math.max(minFloor, Math.min(sleepMs, MAX_SLEEP)) / 1000)}s`, this.id);
    return Math.max(minFloor, Math.min(sleepMs, MAX_SLEEP));
  }

  getBudgetStatus() {
    const config = this.loadConfig();
    const budgetPer24h = config.budgetPer24h || 0;
    if (budgetPer24h <= 0) return null;

    const costSummary = this.getCostSummary();
    const spent24h = costSummary.last24hCost;
    const remaining24h = budgetPer24h - spent24h;
    const percentUsed = budgetPer24h > 0 ? (spent24h / budgetPer24h) * 100 : 0;
    const exhausted = remaining24h <= 0;

    return {
      budgetPer24h,
      spent24h,
      remaining24h,
      percentUsed,
      computedSleepMs: this.lastComputedSleepMs, // Use cached value
      exhausted
    };
  }

  _syncAgentRegistry(db) {
    const upsert = db.prepare(`
      INSERT INTO agents (name, role, reports_to, model, disabled)
      VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(name) DO UPDATE SET
        role = excluded.role,
        reports_to = excluded.reports_to,
        model = excluded.model,
        disabled = excluded.disabled
    `);
    const managersDir = path.join(ROOT, 'agent', 'managers');
    const workersDir = this.workerSkillsDir;
    const parseRole = (content) => (content.match(/^role:\s*(.+)$/m) || [])[1]?.trim() || null;
    const parseModel = (content) => (content.match(/^model:\s*(.+)$/m) || [])[1]?.trim() || null;

    if (fs.existsSync(managersDir)) {
      for (const file of fs.readdirSync(managersDir)) {
        if (!file.endsWith('.md')) continue;
        const content = fs.readFileSync(path.join(managersDir, file), 'utf-8');
        const name = file.replace('.md', '');
        const disabled = /^disabled:\s*true$/m.test(content) ? 1 : 0;
        upsert.run(name, parseRole(content), null, parseModel(content), disabled);
      }
    }

    if (fs.existsSync(workersDir)) {
      for (const file of fs.readdirSync(workersDir)) {
        if (!file.endsWith('.md')) continue;
        const content = fs.readFileSync(path.join(workersDir, file), 'utf-8');
        const name = file.replace('.md', '');
        const disabled = /^disabled:\s*true$/m.test(content) ? 1 : 0;
        const reportsTo = (content.match(/^reports_to:\s*(.+)$/m) || [])[1]?.trim() || null;
        upsert.run(name, parseRole(content), reportsTo, parseModel(content), disabled);
      }
    }
  }

  _resolveAllowedIssueClosers(db, issueCreator) {
    if (issueCreator === 'human' || issueCreator === 'chat') {
      return { allowed: new Set(['human', 'chat']), special: 'chat-human' };
    }
    return { allowed: new Set([issueCreator]), special: 'agent' };
  }

  getDb() {
    const dbPath = this.projectDbPath;
    const db = new Database(dbPath);
    db.pragma('journal_mode = WAL');
    db.pragma('foreign_keys = ON');
    db.exec(`
      CREATE TABLE IF NOT EXISTS agents (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL, role TEXT, reports_to TEXT, model TEXT, disabled INTEGER DEFAULT 0, created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')));
      CREATE TABLE IF NOT EXISTS issues (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, body TEXT DEFAULT '', status TEXT DEFAULT 'open', creator TEXT NOT NULL, assignee TEXT, labels TEXT DEFAULT '', created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')), updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')), updated_by TEXT, closed_at TEXT, closed_by TEXT);
      CREATE TABLE IF NOT EXISTS comments (id INTEGER PRIMARY KEY AUTOINCREMENT, issue_id INTEGER NOT NULL REFERENCES issues(id), author TEXT NOT NULL, body TEXT NOT NULL, created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')));
      CREATE TABLE IF NOT EXISTS milestones (id INTEGER PRIMARY KEY AUTOINCREMENT, description TEXT NOT NULL, cycles_budget INTEGER DEFAULT 20, cycles_used INTEGER DEFAULT 0, phase TEXT DEFAULT 'implementation', status TEXT DEFAULT 'active', created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')), completed_at TEXT);
      CREATE TABLE IF NOT EXISTS tbc_prs (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, summary TEXT DEFAULT '', base_branch TEXT NOT NULL, head_branch TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open', 'merged', 'closed')), issue_ids TEXT DEFAULT '[]', test_status TEXT DEFAULT 'unknown', github_pr_number INTEGER, github_pr_url TEXT, created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')), updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')));
    `);
    try { db.exec('ALTER TABLE issues ADD COLUMN updated_by TEXT'); } catch {}
    try { db.exec('ALTER TABLE issues ADD COLUMN closed_by TEXT'); } catch {}
    db.exec(`
      UPDATE tbc_prs
      SET status = CASE
        WHEN status = 'merged' THEN 'merged'
        WHEN status IN ('closed', 'completed', 'superseded') THEN 'closed'
        ELSE 'open'
      END
      WHERE status NOT IN ('open', 'merged', 'closed');
      CREATE TRIGGER IF NOT EXISTS tbc_prs_status_insert_check
      BEFORE INSERT ON tbc_prs
      FOR EACH ROW
      WHEN NEW.status NOT IN ('open', 'merged', 'closed')
      BEGIN
        SELECT RAISE(ABORT, 'invalid tbc_prs.status');
      END;
      CREATE TRIGGER IF NOT EXISTS tbc_prs_status_update_check
      BEFORE UPDATE OF status ON tbc_prs
      FOR EACH ROW
      WHEN NEW.status NOT IN ('open', 'merged', 'closed')
      BEGIN
        SELECT RAISE(ABORT, 'invalid tbc_prs.status');
      END;
    `);
    ensureResourceOrchestrationTables(db, this.loadConfig().resourceOrchestration);
    this._syncAgentRegistry(db);
    return db;
  }

  getResourceStatusSnapshot() {
    const config = this.loadConfig().resourceOrchestration;
    let db = null;
    try {
      db = this.getDb();
      const summary = getResourceSummary(db, config);
      this.resourceOrchestrationSnapshot = summary;
      return summary;
    } catch (e) {
      const fallback = this.resourceOrchestrationSnapshot || buildEmptyResourceSummary(config);
      return {
        ...fallback,
        error: e.message,
      };
    } finally {
      try { db?.close(); } catch {}
    }
  }

  async getComments(author, page = 1, perPage = 20) {
    try {
      const db = this.getDb();
      let query, countQuery, params;
      if (author) {
        query = `SELECT c.id, c.issue_id, c.author, c.body, c.created_at FROM comments c WHERE c.author = ? ORDER BY c.created_at DESC LIMIT ? OFFSET ?`;
        countQuery = `SELECT COUNT(*) as total FROM comments WHERE author = ?`;
        params = [author, perPage, (page - 1) * perPage];
      } else {
        query = `SELECT c.id, c.issue_id, c.author, c.body, c.created_at FROM comments c ORDER BY c.created_at DESC LIMIT ? OFFSET ?`;
        countQuery = `SELECT COUNT(*) as total FROM comments`;
        params = [perPage, (page - 1) * perPage];
      }
      const comments = db.prepare(query).all(...params).map(c => ({ ...c, agent: c.author }));
      const { total } = author ? db.prepare(countQuery).get(author) : db.prepare(countQuery).get();
      db.close();
      return {
        comments,
        total,
        page,
        perPage,
        hasMore: page * perPage < total
      };
    } catch (e) {
      return { comments: [], total: 0, error: e.message };
    }
  }

  async getPRs(status = 'open') {
    try {
      const db = this.getDb();
      let query = `
        SELECT id, title, summary, base_branch, head_branch, status, issue_ids, test_status, github_pr_number, github_pr_url, created_at, updated_at
        FROM tbc_prs
      `;
      const params = [];
      if (status === 'open' || status === 'merged' || status === 'closed') {
        query += ` WHERE status = ?`;
        params.push(status);
      }
      query += `
        ORDER BY updated_at DESC, id DESC
        LIMIT 50
      `;
      const prs = db.prepare(query).all(...params);
      db.close();
      return prs.map(pr => ({
        ...pr,
        number: pr.id,
        headRefName: pr.head_branch,
        baseRefName: pr.base_branch,
        shortTitle: pr.title,
        issueIds: (() => { try { return JSON.parse(pr.issue_ids || '[]'); } catch { return []; } })(),
      }));
    } catch {
      return [];
    }
  }

  async getPR(prId) {
    try {
      const db = this.getDb();
      const pr = db.prepare(`
        SELECT id, title, summary, base_branch, head_branch, status, issue_ids, test_status, github_pr_number, github_pr_url, created_at, updated_at
        FROM tbc_prs
        WHERE id = ?
      `).get(prId);
      db.close();
      if (!pr) return null;
      return {
        ...pr,
        number: pr.id,
        headRefName: pr.head_branch,
        baseRefName: pr.base_branch,
        shortTitle: pr.title,
        issueIds: (() => { try { return JSON.parse(pr.issue_ids || '[]'); } catch { return []; } })(),
      };
    } catch {
      return null;
    }
  }

  async getIssues() {
    try {
      const db = this.getDb();
      const issues = db.prepare(`
        SELECT i.*, (SELECT COUNT(*) FROM comments c WHERE c.issue_id = i.id) as comment_count
        FROM issues i ORDER BY i.created_at DESC
      `).all();
      db.close();
      return issues;
    } catch {
      return [];
    }
  }

  async createIssue(title, body = '', creator = 'human', assignee = null) {
    if (!title?.trim()) throw new Error('Missing issue title');
    try {
      const db = this.getDb();
      const now = new Date().toISOString();
      const result = db.prepare(
        `INSERT INTO issues (title, body, creator, assignee, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)`
      ).run(title.trim(), body.trim(), creator, assignee || null, now, now);
      db.close();
      return { success: true, issueId: result.lastInsertRowid };
    } catch (e) {
      throw new Error(`Failed to create issue: ${e.message}`);
    }
  }

  getStatus() {
    return {
      id: this.id,
      path: this.path,
      repo: this.repo,
      enabled: this.enabled,
      archived: this.archived,
      running: this.running,
      paused: this.isPaused,
      pauseReason: this.pauseReason || null,
      cycleCount: this.cycleCount,
      epochCount: this.epochCount,
      currentAgent: this.currentAgent,
      currentAgentModel: this.currentAgentModel,
      currentAgentKeyId: this.currentAgentKeyId || null,
      currentAgentVisibility: this.currentAgentVisibility || { mode: 'full', issues: [] },
      currentAgentRuntime: this.currentAgentStartTime
        ? Math.floor((Date.now() - this.currentAgentStartTime) / 1000)
        : null,
      activeRuns: [...this.activeRuns.values()].map((run) => ({
        id: run.id,
        agent: run.agentName,
        isManager: run.isManager,
        isPrimary: run.isPrimary,
        managerName: run.managerName || null,
        startTime: run.startTime,
        runtimeSeconds: run.startTime ? Math.floor((Date.now() - run.startTime) / 1000) : null,
        visibility: run.visibility || { mode: 'full', issues: [] },
        model: run.model || null,
        keyId: run.keyId || null,
        cost: run.cost || 0,
        resources: run.resources || null,
        grantId: run.resourceGrant?.id || null,
        gpuTokens: run.resourceGrant?.tokenNames || null,
      })),
      sleeping: this.sleepUntil !== null && !this.isPaused,
      sleepUntil: this.isPaused ? null : this.sleepUntil,
      schedule: this.currentSchedule || null,
      phase: this.phase,
      isComplete: this.isComplete || false,
      completionSuccess: this.completionSuccess || false,
      completionMessage: this.completionMessage || null,
      config: this.loadConfig(),
      agents: this.loadAgents(),
      cost: this.getCostSummary(),
      budget: this.getBudgetStatus(),
      resourceOrchestration: this.getResourceStatusSnapshot(),
    };
  }

  bootstrapPreview() {
    const projectDataExists = fs.existsSync(this.projectDir);
    let projectDataContents = [];
    if (projectDataExists) {
      projectDataContents = fs.readdirSync(this.projectDir).filter(name => !['repo', 'knowledge', 'skills', 'config.yaml'].includes(name));
    }
    // Read spec.md and check roadmap.md from private knowledge base
    let specContent = null;
    const specPath = path.join(this.knowledgeDir, 'spec.md');
    try { specContent = fs.readFileSync(specPath, 'utf-8'); } catch {}
    const hasRoadmap = fs.existsSync(path.join(this.knowledgeDir, 'roadmap.md'));
    return { available: true, projectDataEmpty: projectDataContents.length === 0, repo: this.repo, specContent, hasRoadmap };
  }

  bootstrap(options = {}) {
    // 0. Kill any running agent and pause the project
    if (this.activeRuns.size > 0) {
      this._abortAllRuns();
      this.activeRuns.clear();
      this._clearCurrentAgentSnapshot({ broadcast: false });
      log(`Killed running agents for bootstrap`, this.id);
    }
    this.isPaused = true;
    this.pauseReason = 'Bootstrapping';
    this.completedAgents = [];
    this.currentCycleId = null;
    this.currentSchedule = null;

    // 1. Wipe project operational state only, keep repo/knowledge/skills intact
    for (const target of this.getOperationalPaths()) {
      if (!fs.existsSync(target)) continue;
      fs.rmSync(target, { recursive: true, force: true });
    }
    log(`Cleared project operational state`, this.id);
    fs.mkdirSync(this.projectDir, { recursive: true });

    // 2. Reset cycle count and save state
    this.setState({
      cycleCount: 0,
      epochCount: 0,
      phase: 'idle',
      isComplete: false,
      completionSuccess: false,
      completionMessage: null,
      isPaused: true,
      pauseReason: 'Bootstrapped — resume when ready',
    });
    log(`Reset cycle count, project paused`, this.id);

    // 3. Remove roadmap.md from private knowledge base if requested
    if (options.removeRoadmap) {
      const roadmapPath = path.join(this.knowledgeDir, 'roadmap.md');
      if (fs.existsSync(roadmapPath)) {
        try {
          fs.unlinkSync(roadmapPath);
          log(`Removed private roadmap.md`, this.id);
        } catch (e) {
          log(`Warning: failed to remove private roadmap.md: ${e.message}`, this.id);
        }
      }
    }

    // 4. Update private spec.md if requested
    if (options.spec && options.spec.mode !== 'keep') {
      const specPath = path.join(this.knowledgeDir, 'spec.md');
      let newContent = '';
      if (options.spec.mode === 'edit') {
        newContent = options.spec.content || '';
      } else if (options.spec.mode === 'new') {
        const what = (options.spec.whatToBuild || '').trim();
        const criteria = (options.spec.successCriteria || '').trim();
        newContent = `# Project Spec\n\n## What to Build\n\n${what}\n\n## Success Criteria\n\n${criteria}\n`;
      }
      if (newContent) {
        try {
          fs.writeFileSync(specPath, newContent);
          log(`Updated private knowledge/spec.md`, this.id);
        } catch (e) {
          log(`Warning: failed to update spec.md: ${e.message}`, this.id);
        }
      }
    }

    return { bootstrapped: true };
  }

  _writeReport(agentName, body, { success = true, durationMs = 0 } = {}) {
    const durationStr = `${Math.floor(durationMs / 60000)}m ${Math.floor((durationMs % 60000) / 1000)}s`;
    const startedAt = new Date();
    const endedAt = new Date();
    const reportBody = `> ⏱ Started: ${startedAt.toLocaleString('sv-SE')} | Ended: ${endedAt.toLocaleString('sv-SE')} | Duration: ${durationStr}\n\n${body.trim()}`;
    const db = this.getDb();
    db.exec(`CREATE TABLE IF NOT EXISTS reports (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      cycle INTEGER NOT NULL,
      agent TEXT NOT NULL,
      body TEXT NOT NULL,
      summary TEXT,
      created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    )`);
    try { db.exec('ALTER TABLE reports ADD COLUMN summary TEXT'); } catch {}
    try { db.exec('ALTER TABLE reports ADD COLUMN cost REAL'); } catch {}
    try { db.exec('ALTER TABLE reports ADD COLUMN duration_ms INTEGER'); } catch {}
    try { db.exec('ALTER TABLE reports ADD COLUMN input_tokens INTEGER'); } catch {}
    try { db.exec('ALTER TABLE reports ADD COLUMN output_tokens INTEGER'); } catch {}
    try { db.exec('ALTER TABLE reports ADD COLUMN cache_read_tokens INTEGER'); } catch {}
    try { db.exec('ALTER TABLE reports ADD COLUMN success INTEGER'); } catch {}
    try { db.exec('ALTER TABLE reports ADD COLUMN model TEXT'); } catch {}
    try { db.exec('ALTER TABLE reports ADD COLUMN timed_out INTEGER'); } catch {}
    try { db.exec('ALTER TABLE reports ADD COLUMN key_id TEXT'); } catch {}
    try { db.exec('ALTER TABLE reports ADD COLUMN visibility_mode TEXT'); } catch {}
    try { db.exec('ALTER TABLE reports ADD COLUMN visibility_issues TEXT'); } catch {}
    db.prepare(`INSERT INTO reports (cycle, agent, body, created_at, cost, duration_ms, input_tokens, output_tokens, cache_read_tokens, success, model, timed_out, key_id, visibility_mode, visibility_issues)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(
      this.cycleCount, agentName, reportBody, new Date().toISOString(),
      null, durationMs,
      null, null, null,
      success ? 1 : 0, null, 0,
      null,
      'full', JSON.stringify([])
    );
    const lastId = db.prepare('SELECT last_insert_rowid() as id').get().id;
    db.close();
    log(`Saved report for ${agentName}`, this.id);
    broadcastReportUpdate(this.id, lastId, agentName, this.cycleCount);
    return { reportId: lastId };
  }

  async runDoctor() {
    const config = this.loadConfig();
    const doctorAgent = { name: 'doctor', isManager: true, rawModel: 'high' };
    const task = [
      'Inspect this project and act only as an AI Doctor agent.',
      '',
      'Your job is to inspect and repair project layout drift. Do not rely on any built-in deterministic doctor behavior. You are the doctor.',
      '',
      'Canonical layout:',
      '- repo/',
      '- knowledge/',
      '- skills/',
      '- project.db',
      '- orchestrator.log',
      '- responses/',
      '- agents/',
      '',
      'Required behavior:',
      '- Inspect the actual filesystem.',
      '- Repair missing or misplaced project files when it is safe.',
      '- Ensure required directories and files exist after repair.',
      '- If known agent directories under agents/ are missing, create them.',
      '- Do not change product code in repo/ unless absolutely necessary for the repair itself.',
      '- Prefer move/rename over copy when safe.',
      '',
      'At the end, write a concise doctor report with these sections exactly:',
      '## Doctor Check',
      'Layout status: ...',
      '',
      '### Required paths',
      '- ...',
      '',
      '### Repair actions',
      '- ...',
      '',
      'If something could not be fixed, say why clearly.',
    ].join('\n');
    return await this.runAgent(doctorAgent, config, 'doctor', task, { mode: 'full', issues: [] });
  }

  async start() {
    if (this.running) return;
    // Validate project path exists
    if (!fs.existsSync(this.path)) {
      log(`ERROR: Project path does not exist: ${this.path}`, this.id);
      return;
    }

    // Ensure project directories exist
    fs.mkdirSync(this.projectDir, { recursive: true });
    fs.mkdirSync(this.chatsDir, { recursive: true });
    fs.mkdirSync(this.agentsDir, { recursive: true });
    fs.mkdirSync(this.responsesDir, { recursive: true });
    fs.mkdirSync(this.skillsDir, { recursive: true });
    fs.mkdirSync(this.knowledgeDir, { recursive: true });
    fs.mkdirSync(path.join(this.projectDir, 'knowledge', 'analysis'), { recursive: true });
    fs.mkdirSync(path.join(this.projectDir, 'knowledge', 'decisions'), { recursive: true });
    fs.mkdirSync(this.workerSkillsDir, { recursive: true });
    
    // Load persisted state
    this.loadState();
    
    this.running = true;
    log(`Starting project runner (data: ${this.projectDir}, cycle: ${this.cycleCount})`, this.id);
    this.runLoop();
  }

  loadState() {
    const statePath = this.statePath;
    try {
      if (fs.existsSync(statePath)) {
        const state = JSON.parse(fs.readFileSync(statePath, 'utf-8'));
        this.cycleCount = state.cycleCount || 0;
        this.epochCount = state.epochCount || 0;
        this.completedAgents = state.completedAgents || [];
        this.currentCycleId = state.currentCycleId || null;
        this.currentSchedule = state.currentSchedule || null;
        if (state.isPaused !== undefined) this.isPaused = state.isPaused;
        this.phase = state.phase || 'idle';
        this.isComplete = state.isComplete || false;
        this.completionSuccess = state.completionSuccess || false;
        this.completionMessage = state.completionMessage || null;
        this.resourceOrchestrationSnapshot = state.resourceOrchestration || null;
        this._resetTaskGraphRunningTasks(this.currentSchedule);
        log(`Loaded state: cycle ${this.cycleCount}, completed: [${this.completedAgents.join(', ')}]${this.isPaused ? ', paused' : ''}`, this.id);
      } else {
        // New project — start paused
        this.setState({ isPaused: true, pauseReason: 'New project (paused by default)' }, { save: false });
        this.completedAgents = [];
        this.currentCycleId = null;
        this.currentSchedule = null;
        this.resourceOrchestrationSnapshot = null;
      }
    } catch (e) {
      log(`Failed to load state: ${e.message}`, this.id);
      this.completedAgents = [];
      this.currentCycleId = null;
      this.currentSchedule = null;
      this.resourceOrchestrationSnapshot = null;
    }
  }

  saveState() {
    const statePath = this.statePath;
    try {
      const resourceOrchestration = this.getResourceStatusSnapshot();
      const state = {
        cycleCount: this.cycleCount,
        epochCount: this.epochCount || 0,
        completedAgents: this.completedAgents || [],
        currentCycleId: this.currentCycleId,
        currentSchedule: this.currentSchedule || null,
        isPaused: this.isPaused || false,
        phase: this.phase,
        isComplete: this.isComplete || false,
        completionSuccess: this.completionSuccess || false,
        completionMessage: this.completionMessage || null,
        resourceOrchestration,
        lastUpdated: new Date().toISOString()
      };
      fs.writeFileSync(statePath, JSON.stringify(state, null, 2));
    } catch (e) {
      log(`Failed to save state: ${e.message}`, this.id);
    }
  }

  stop() {
    this.running = false;
    this._abortAllRuns();
    log(`Stopped project runner`, this.id);
  }

  pause() {
    this.setState({ isPaused: true, pauseReason: null });
    log(`Paused`, this.id);
  }

  resume() {
    if (this.isComplete) {
      log(`Reopening completed project`, this.id);
      this.setState({
        isComplete: false,
        completionSuccess: false,
        completionMessage: null,
        isPaused: false,
        pauseReason: null,
        phase: 'idle',
        currentSchedule: null,
        completedAgents: [],
      });
    } else {
      this.setState({ isPaused: false, pauseReason: null });
    }
    this.wakeNow = true;
    log(`Resumed`, this.id);
  }

  skip() {
    if (this.currentAgentProcess) {
      log(`Skipping current agent`, this.id);
      this.currentAgentProcess.kill('SIGTERM');
    } else if (this.sleepUntil) {
      log(`Skipping sleep`, this.id);
      this.wakeNow = true;
    }
  }

  // Kill Run: terminate the current agent, move to next in schedule
  killRun() {
    if (this.currentAgentProcess) {
      log(`🔴 Kill Run: terminating current agent`, this.id);
      this.currentAgentProcess.kill('SIGTERM');
    }
  }

  // Kill Cycle: terminate current agent + skip remaining workers in schedule
  killCycle() {
    log(`🔴 Kill Cycle: terminating agent and clearing schedule`, this.id);
    this._abortAllRuns();
    this.currentSchedule = null;
    this.completedAgents = [];
    this.saveState();
  }

  // Kill Epoch: terminate everything + force back to Athena
  killEpoch() {
    log(`🔴 Kill Epoch: terminating agent, clearing schedule`, this.id);
    this.abortCurrentCycle = true;
    this._abortAllRuns();
    this.currentSchedule = null;
    this.completedAgents = [];
    this.saveState();
  }

  // Wait while paused, auto-resuming after intervalMs. Optional condition check to resume early.
  async _autoPauseWait(intervalMs, resumeCondition = null) {
    const retryAt = Date.now() + intervalMs;
    while (this.isPaused && this.running && !this.wakeNow) {
      await sleep(5000);
      // Check if it's time to auto-retry
      if (Date.now() >= retryAt) {
        if (resumeCondition && !resumeCondition()) {
          // Condition not met, keep waiting (check again in 2h)
          log(`Auto-retry check: condition not met, waiting another 2h`, this.id);
          return this._autoPauseWait(intervalMs, resumeCondition);
        }
        log(`Auto-resuming after ${Math.round(intervalMs / 60000)}m pause`, this.id);
        this.isPaused = false;
        this.pauseReason = null;
        return;
      }
    }
    // Manually resumed or stopped
    if (!this.isPaused) {
      this.pauseReason = null;
    }
  }

  async sleepDelay(minutes, label) {
    const ms = Math.min(Math.max(parseFloat(minutes) || 0, 0), 120) * 60000;
    if (ms <= 0) return;
    log(`⏳ Waiting ${Math.round(ms / 60000)}m after ${label}...`, this.id);
    this.sleepUntil = Date.now() + ms;
    let slept = 0;
    while (slept < ms && !this.wakeNow && this.running && !this.abortCurrentCycle) {
      await sleep(5000);
      slept += 5000;
      while (this.isPaused && !this.wakeNow && this.running && !this.abortCurrentCycle) { await sleep(1000); }
    }
    this.sleepUntil = null;
  }

  _parseVisibility(value, task) {
    const visMode = typeof value === 'string'
      ? value
      : (typeof value === 'object' ? value.visibility : undefined);
    if (!visMode || visMode === 'full') return null;
    if (visMode === 'blind') return { mode: 'blind', issues: [] };
    if (visMode === 'focused') {
      const issueIds = (task || '').match(/#(\d+)/g)?.map(m => m.slice(1)) || [];
      return { mode: 'focused', issues: issueIds };
    }
    return null;
  }

  parseSchedule(resultText, options = {}) {
    const defaultWorkerResources = this.getOrchestrationConfig().defaultWorkerResources;
    const taskGraphErrors = [];
    const taskGraph = parseTaskGraphDocument(resultText, defaultWorkerResources, {
      additionalKnownTaskIds: options.additionalKnownTaskIds || [],
      errors: taskGraphErrors,
    });
    if (taskGraph) return taskGraph;
    const source = String(resultText || '');
    if (source.includes('<!-- TASK_GRAPH -->') && taskGraphErrors.length > 0) {
      log(`TASK_GRAPH rejected: ${taskGraphErrors.join('; ')}`, this.id);
    }

    const schedule = parseScheduleDocument(resultText, defaultWorkerResources);
    if (!schedule && (source.includes('<!-- SCHEDULE -->') || source.includes('<!-- TASK_GRAPH -->'))) {
      log(`Failed to parse orchestration plan`, this.id);
    }
    return schedule;
  }

  _normalizeScheduleStepForExecution(step, config) {
    if (!step) return null;
    if (step.type) return step;
    return normalizeScheduleStep(step, this.getOrchestrationConfig(config).defaultWorkerResources);
  }

  _getManagedWorkers(managerName = null) {
    const ownerName = typeof managerName === 'string' ? managerName.toLowerCase() : null;
    return this.loadAgents().workers.filter((worker) => {
      if (!ownerName) return true;
      return (worker.reportsTo || '').toLowerCase() === ownerName;
    });
  }

  _buildSingleManagerContext(config, orchestration) {
    const resource = this.getResourceStatusSnapshot();
    const workers = this._getManagedWorkers(orchestration.manager);
    const workerLines = workers.length
      ? workers.map((worker) => {
          const parts = [worker.name];
          if (worker.role) parts.push(`role=${worker.role}`);
          if (worker.workerClass) parts.push(`class=${worker.workerClass}`);
          if (worker.model) parts.push(`model=${worker.model}`);
          return `> - ${parts.join(' | ')}`;
        })
      : ['> - none discovered yet'];
    const lines = [
      '> **Runtime mode:** single-manager resource-aware orchestration',
      `> **Cycle:** ${this.cycleCount}`,
      `> **Manager:** ${orchestration.manager}`,
      '',
      '> **Available workers:**',
      ...workerLines,
      '',
      '> **Current resource status:**',
      `> - allocation_job_id: ${resource?.allocation?.jobId || 'none'}`,
      `> - allocation_state: ${resource?.allocation?.state || 'inactive'}`,
      `> - available_gpu_tokens: ${resource?.tokens?.available ?? 0}`,
      `> - granted_gpu_tokens: ${resource?.tokens?.granted ?? 0}`,
      `> - pending_requests: ${resource?.requests?.pending ?? 0}`,
      `> - target_gpu_utilization: ${Math.round((orchestration.targetGpuUtilization ?? 1) * 100)}%`,
      `> - live_replan_on_task_complete: ${orchestration.liveReplanOnTaskComplete ? 'enabled' : 'disabled'}`,
      '',
      '> Use one manager cycle to plan and dispatch workers. Prefer enough independent READY GPU tasks to keep available GPUs busy.',
      '> When dependency structure matters, prefer TASK_GRAPH with id/depends_on/depends_on_tags/produces_tags/replan_after.',
      '> In TASK_GRAPH, prefer worker_class for pooled workers and use agent only when a specific worker is required.',
      '> If later work depends on prior outputs, use fan-out/fan-in: many independent runs, then one analyst step.',
      '> If a worker needs shared-allocation compute, include a resources object, for example {"gpus": 1, "cpus": 1}.',
      '> Treat completed runs as feedback for the next batch. Do not stop after one wave unless the task program explicitly says one wave is enough.',
      '> If live replanning is enabled, you may be called while workers are still running; in that case append only new independent tasks to the live TASK_GRAPH.',
      '> If the task program specifies a minimum iteration or wave count, do not emit PROJECT_COMPLETE before that count is reached.',
      '> When improvement is still plausible, emit another TASK_GRAPH instead of PROJECT_COMPLETE. Stop only when the configured budget/goal/convergence criteria are satisfied.',
      '> Task-specific worker invocation (which wrapper script, which env vars) is described by the task program loaded into your skill file; do not assume a fixed wrapper path.',
      '> If the task is actually complete, emit <!-- PROJECT_COMPLETE --> with success=true.',
    ];

    // Phase 5.2: compact experiment ledger from task-events. Saves the
    // manager from `ls experiments/` + reading 50+ metrics.json files on
    // every replan; payload stays roughly constant past N=20 experiments.
    try {
      const directConfig = config?.directExecutor || null;
      const metricKey = directConfig?.metricKey || 'val_bpb';
      const ledger = buildExperimentLedger({
        eventsDir: this.taskEventsDir,
        metricKey,
      });
      const markdown = formatLedgerMarkdown(ledger, metricKey);
      if (markdown) {
        lines.push('', markdown);
      }
      // Phase 5.3: edit-confidence addendum. When recent param_patch
      // success drops below 70%, telegraph it loudly so the manager re-
      // reads train.py instead of emitting the same broken edit shape.
      const confidence = computeEditConfidence({ eventsDir: this.taskEventsDir });
      if (confidence?.shouldWarn) {
        const failed = confidence.total - confidence.succeeded;
        lines.push(
          '',
          `> **⚠ param_patch confidence low:** ${failed}/${confidence.total} of your recent direct-executor tasks failed (rate ${(confidence.rate * 100).toFixed(0)}%). Read \`train.py\` (or the parent's train.py) to refresh \`expected_old_repr\` before emitting more edits — most failures are stale reads. Recent failures: ${confidence.recentFailureIds.join(', ')}.`,
        );
      }
    } catch (e) {
      // Ledger is a hint, not a contract; never fail the manager cycle on it.
      try { log(`Experiment ledger build failed: ${e.message}`, this.id); } catch {}
    }

    return lines.join('\n');
  }

  _tryGrantWorkerResources(step, worker, managerName, config) {
    const requestedGpus = step.resources?.gpus || 0;
    if (!requestedGpus) return { grant: null, request: null, blocked: false };

    let db = null;
    try {
      db = this.getDb();
      const lease = getAllocationLease(db);
      const leaseRequired = config?.resourceOrchestration?.grantRequiresLease !== false;
      if (leaseRequired && !lease) {
        return { blocked: true, reason: 'no active allocation lease' };
      }
      const available = listResourceTokens(db)
        .filter((token) => token.available)
        .slice(0, requestedGpus)
        .map((token) => token.tokenName);
      if (available.length < requestedGpus) {
        return {
          blocked: true,
          reason: `needs ${requestedGpus} GPU token(s), ${available.length} available`,
        };
      }
      const request = createTokenRequest(db, {
        workerName: worker.name,
        requestedCount: requestedGpus,
        rationale: step.task.slice(0, 240),
        metadata: {
          managerName,
          scheduleAgent: worker.name,
          resources: step.resources,
        },
      });
      const grant = grantTokenRequest(db, {
        requestId: request.id,
        managerName,
        tokenNames: available,
      }, config.resourceOrchestration);
      return { blocked: false, request, grant };
    } finally {
      try { db?.close(); } catch {}
    }
  }

  _releaseWorkerGrant(grantId, actor) {
    if (!grantId) return;
    let db = null;
    try {
      db = this.getDb();
      releaseTokenGrant(db, {
        grantId,
        actor: actor || 'manager',
        note: 'worker finished',
      });
    } catch (e) {
      log(`Failed to release token grant #${grantId}: ${e.message}`, this.id);
    } finally {
      try { db?.close(); } catch {}
    }
  }

  _getParallelGpuAvailability() {
    let db = null;
    try {
      db = this.getDb();
      const lease = getAllocationLease(db);
      const tokens = listResourceTokens(db);
      const availableGpuTokens = tokens.filter((token) => token.available).length;
      // grantRequiresLease=false means the runner is already inside the GPU
      // allocation (sbatch on a compute node) and no lease row gets written.
      // Treat that as having effective allocation so admission doesn't block.
      const resourceConfig = this.loadConfig().resourceOrchestration || {};
      const leaseRequired = resourceConfig.grantRequiresLease !== false;
      return {
        hasAllocationLease: leaseRequired ? !!lease : true,
        totalGpuTokens: tokens.length,
        availableGpuTokens,
      };
    } catch {
      return {
        hasAllocationLease: false,
        totalGpuTokens: 0,
        availableGpuTokens: 0,
      };
    } finally {
      try { db?.close(); } catch {}
    }
  }

  _isTaskGraphPlan(plan) {
    return !!plan && plan._kind === 'task_graph' && Array.isArray(plan._tasks);
  }

  // Mark a set of task ids as cancelled, abort any active worker LLM runs for
  // those tasks, and let the existing finally branch release their GPU grants.
  // Manager invokes this via a `<!-- KILL_TASKS -->` block.
  _killTaskGraphTasks(plan, ids) {
    if (!this._isTaskGraphPlan(plan) || !Array.isArray(ids) || ids.length === 0) return 0;
    this._ensureTaskGraphRuntime(plan);
    const runtime = plan._runtime;
    if (!(runtime.cancelledTaskIds instanceof Set)) {
      runtime.cancelledTaskIds = new Set(Array.isArray(runtime.cancelledTaskIds) ? runtime.cancelledTaskIds : []);
    }
    let killed = 0;
    const knownIds = new Set(plan._tasks.map((task) => task.id));
    for (const id of ids) {
      if (!knownIds.has(id)) {
        log(`Manager kill ignored: unknown task id "${id}"`, this.id);
        continue;
      }
      const state = runtime.taskStates[id];
      if (!state || ['completed', 'failed', 'blocked', 'cancelled'].includes(state.status)) continue;
      runtime.cancelledTaskIds.add(id);
      runtime.taskStates[id] = {
        ...state,
        status: 'cancelled',
        finishedAt: new Date().toISOString(),
        reason: 'killed by manager',
      };
      // Abort any active worker LLM run tagged with this task id. The LLM CLI
      // is spawned with a process group and SIGTERM'd via abortController, so
      // the worker's bash + train.py descendants are also taken down.
      for (const runState of this.activeRuns.values()) {
        if (runState.taskId === id && runState.abortController) {
          try { runState.abortController.abort(); } catch {}
        }
      }
      log(`Manager killed task ${id}`, this.id);
      killed += 1;
    }
    if (killed) this.saveState();
    return killed;
  }

  _ensureTaskGraphRuntime(plan) {
    if (!this._isTaskGraphPlan(plan)) return;
    if (!plan._runtime || typeof plan._runtime !== 'object' || Array.isArray(plan._runtime)) {
      plan._runtime = {};
    }
    if (!plan._runtime.taskStates || typeof plan._runtime.taskStates !== 'object' || Array.isArray(plan._runtime.taskStates)) {
      plan._runtime.taskStates = {};
    }
    if (!Array.isArray(plan._runtime.producedTags)) {
      plan._runtime.producedTags = [];
    }
    if (typeof plan._runtime.replanRequested !== 'boolean') {
      plan._runtime.replanRequested = false;
    }
    if (!Object.prototype.hasOwnProperty.call(plan._runtime, 'replanTaskId')) {
      plan._runtime.replanTaskId = null;
    }
    if (!(plan._runtime.cancelledTaskIds instanceof Set)) {
      plan._runtime.cancelledTaskIds = new Set(
        Array.isArray(plan._runtime.cancelledTaskIds) ? plan._runtime.cancelledTaskIds : [],
      );
    }

    for (const task of plan._tasks) {
      if (!plan._runtime.taskStates[task.id] || typeof plan._runtime.taskStates[task.id] !== 'object') {
        plan._runtime.taskStates[task.id] = {
          status: 'pending',
          attempts: 0,
          startedAt: null,
          finishedAt: null,
          reason: null,
        };
      }
    }
  }

  _resetTaskGraphRunningTasks(plan) {
    if (!this._isTaskGraphPlan(plan)) return;
    this._ensureTaskGraphRuntime(plan);
    let changed = false;
    for (const task of plan._tasks) {
      const state = plan._runtime.taskStates[task.id];
      if (state?.status === 'running') {
        plan._runtime.taskStates[task.id] = {
          ...state,
          status: 'pending',
          reason: 'reset after runner restart',
        };
        changed = true;
      }
    }
    if (changed) this.saveState();
  }

  _getTaskGraphDependencyState(plan, task) {
    const taskStates = plan?._runtime?.taskStates || {};
    const producedTags = new Set(plan?._runtime?.producedTags || []);
    const unmetDependencies = [];
    const failedDependencies = [];

    for (const dependencyId of task.dependsOn || []) {
      const dependencyState = taskStates[dependencyId]?.status || 'pending';
      if (dependencyState === 'failed' || dependencyState === 'blocked') {
        failedDependencies.push(dependencyId);
      } else if (dependencyState !== 'completed') {
        unmetDependencies.push(dependencyId);
      }
    }

    const unmetTags = (task.dependsOnTags || []).filter((tag) => !producedTags.has(tag));
    return { unmetDependencies, failedDependencies, unmetTags };
  }

  _finalizeBlockedTaskGraphTasks(plan) {
    if (!this._isTaskGraphPlan(plan)) return 0;
    this._ensureTaskGraphRuntime(plan);
    let blockedCount = 0;
    for (const task of plan._tasks) {
      const state = plan._runtime.taskStates[task.id];
      if (!state || state.status !== 'pending') continue;
      const deps = this._getTaskGraphDependencyState(plan, task);
      if (deps.failedDependencies.length === 0) continue;
      plan._runtime.taskStates[task.id] = {
        ...state,
        status: 'blocked',
        finishedAt: new Date().toISOString(),
        reason: `blocked by failed dependencies: ${deps.failedDependencies.join(', ')}`,
      };
      blockedCount += 1;
    }
    if (blockedCount > 0) this.saveState();
    return blockedCount;
  }

  _getTaskGraphSummary(plan) {
    if (!this._isTaskGraphPlan(plan)) return {};
    this._ensureTaskGraphRuntime(plan);
    const byStatus = {};
    for (const task of plan._tasks) {
      const status = plan._runtime.taskStates[task.id]?.status || 'pending';
      byStatus[status] = (byStatus[status] || 0) + 1;
    }
    return {
      total: plan._tasks.length,
      byStatus,
      producedTags: [...(plan._runtime.producedTags || [])],
    };
  }

  _getReadyTaskGraphTasks(plan) {
    if (!this._isTaskGraphPlan(plan)) return [];
    this._ensureTaskGraphRuntime(plan);
    const ready = [];
    for (const task of plan._tasks) {
      const state = plan._runtime.taskStates[task.id];
      if (!state || state.status !== 'pending') continue;
      const deps = this._getTaskGraphDependencyState(plan, task);
      if (deps.failedDependencies.length > 0 || deps.unmetDependencies.length > 0 || deps.unmetTags.length > 0) continue;
      ready.push(task);
    }
    return ready;
  }

  _mergeTaskGraphIntoCurrentPlan(plan, incoming) {
    if (!this._isTaskGraphPlan(plan) || !this._isTaskGraphPlan(incoming)) {
      return { added: 0, skipped: 0, skippedRunningDeps: 0, extendedBarriers: 0 };
    }
    this._ensureTaskGraphRuntime(plan);
    this._ensureTaskGraphRuntime(incoming);
    const existingIds = new Set(plan._tasks.map((task) => task.id));
    const runningIds = new Set(plan._tasks
      .filter((task) => plan._runtime.taskStates[task.id]?.status === 'running')
      .map((task) => task.id));
    const producedTags = new Set(plan._runtime.producedTags || []);
    const runningProducedTags = new Set(plan._tasks
      .filter((task) => plan._runtime.taskStates[task.id]?.status === 'running')
      .flatMap((task) => task.producesTags || [])
      .filter((tag) => !producedTags.has(tag)));
    const pendingReplanBarriers = plan._tasks.filter((task) => {
      if (!task.replanAfter) return false;
      const status = plan._runtime.taskStates[task.id]?.status || 'pending';
      return status === 'pending';
    });
    const rejectedIncomingIds = new Set();
    let changed = true;
    while (changed) {
      changed = false;
      const rejectedIncomingTags = new Set(incoming._tasks
        .filter((task) => rejectedIncomingIds.has(task.id))
        .flatMap((task) => task.producesTags || []));
      for (const task of incoming._tasks) {
        if (rejectedIncomingIds.has(task.id) || existingIds.has(task.id)) continue;
        const dependsOnRejectedIncoming = (task.dependsOn || []).some((dependencyId) => rejectedIncomingIds.has(dependencyId));
        const dependsOnRunningExisting = (task.dependsOn || []).some((dependencyId) => runningIds.has(dependencyId));
        const dependsOnRunningExistingTag = (task.dependsOnTags || []).some((tag) => runningProducedTags.has(tag));
        const dependsOnRejectedIncomingTag = (task.dependsOnTags || []).some((tag) => rejectedIncomingTags.has(tag));
        if (
          dependsOnRejectedIncoming
          || dependsOnRunningExisting
          || dependsOnRunningExistingTag
          || dependsOnRejectedIncomingTag
        ) {
          rejectedIncomingIds.add(task.id);
          changed = true;
        }
      }
    }
    let added = 0;
    let skipped = 0;
    let skippedRunningDeps = 0;
    let extendedBarriers = 0;
    for (const task of incoming._tasks) {
      if (existingIds.has(task.id)) {
        skipped += 1;
        continue;
      }
      if (rejectedIncomingIds.has(task.id)) {
        skippedRunningDeps += 1;
        continue;
      }
      plan._tasks.push(task);
      plan._runtime.taskStates[task.id] = {
        status: 'pending',
        attempts: 0,
        startedAt: null,
        finishedAt: null,
        reason: null,
      };
      existingIds.add(task.id);
      added += 1;
      if (!task.replanAfter) {
        for (const barrier of pendingReplanBarriers) {
          if (!barrier.dependsOn.includes(task.id)) {
            barrier.dependsOn.push(task.id);
            extendedBarriers += 1;
          }
          for (const tag of task.producesTags || []) {
            if (!barrier.dependsOnTags.includes(tag)) {
              barrier.dependsOnTags.push(tag);
            }
          }
        }
      }
    }
    if (added > 0) {
      this.currentSchedule = plan;
      this.saveState();
    }
    return { added, skipped, skippedRunningDeps, extendedBarriers };
  }

  _buildLiveTaskGraphManagerContext(plan, orchestration, event = {}) {
    const summary = this._getTaskGraphSummary(plan);
    const ready = this._getReadyTaskGraphTasks(plan);
    const runningTasks = plan._tasks
      .filter((task) => plan._runtime.taskStates[task.id]?.status === 'running')
      .map((task) => task.id);
    const completedTasks = plan._tasks
      .filter((task) => plan._runtime.taskStates[task.id]?.status === 'completed')
      .map((task) => task.id);
    const pendingTasks = plan._tasks
      .filter((task) => plan._runtime.taskStates[task.id]?.status === 'pending')
      .map((task) => task.id);
    const base = this._buildSingleManagerContext(this.loadConfig(), orchestration);
    return [
      base,
      '',
      '> **Live non-blocking replanning trigger:**',
      `> - completed_task: ${event.completedTaskId || 'unknown'}`,
      `> - available_gpu_tokens: ${event.availableGpuTokens ?? 'unknown'}`,
      `> - running_tasks: ${runningTasks.length ? runningTasks.join(', ') : 'none'}`,
      `> - ready_tasks: ${ready.length ? ready.map((task) => task.id).join(', ') : 'none'}`,
      `> - pending_tasks: ${pendingTasks.length ? pendingTasks.join(', ') : 'none'}`,
      `> - completed_tasks: ${completedTasks.length ? completedTasks.join(', ') : 'none'}`,
      `> - produced_tags: ${summary.producedTags.length ? summary.producedTags.join(', ') : 'none'}`,
      '',
      '> A worker just finished and returned its resource token. Inspect the new result, the still-running tasks, and previous artifacts before planning.',
      '> Some workers may still be running. Do not wait for the current task graph to drain if useful independent work can be queued now.',
      '> Emit `<!-- TASK_GRAPH -->` with only additional tasks to append to the live graph. Do not repeat existing task ids.',
      '> New experiment tasks should be independent of currently-running task ids and tags that currently-running tasks have not produced yet. They may depend on completed task ids or already-produced tags.',
      '> If no useful independent work can be queued, emit no schedule.',
      '> Do not emit `PROJECT_COMPLETE` while there are running tasks.',
    ].join('\n');
  }

  async _maybeLiveReplanTaskGraph(plan, config, managerName, orchestration, managerAgent, {
    completedTaskId = null,
    running = [],
  } = {}) {
    if (!orchestration.liveReplanOnTaskComplete) return { added: 0, skipped: 0, ran: false };
    if (!managerAgent) return { added: 0, skipped: 0, ran: false };
    if (!this._isTaskGraphPlan(plan)) return { added: 0, skipped: 0, ran: false };
    if (running.length === 0) return { added: 0, skipped: 0, ran: false };

    this._ensureTaskGraphRuntime(plan);
    const runtime = plan._runtime;
    const now = Date.now();
    const minIntervalMs = (orchestration.liveReplanMinIntervalSeconds || 0) * 1000;
    if (runtime.liveReplanInProgress) return { added: 0, skipped: 0, ran: false };
    if (minIntervalMs > 0 && runtime.lastLiveReplanAt && now - Date.parse(runtime.lastLiveReplanAt) < minIntervalMs) {
      return { added: 0, skipped: 0, ran: false };
    }

    const availability = this._getParallelGpuAvailability();
    if ((availability.availableGpuTokens || 0) <= 0) return { added: 0, skipped: 0, ran: false };
    runtime.liveReplanInProgress = true;
    runtime.lastLiveReplanAt = new Date(now).toISOString();
    runtime.lastLiveReplanTaskId = completedTaskId || null;
    this.saveState();

    try {
      log(`Live replanning after ${completedTaskId || 'task completion'}: ${availability.availableGpuTokens} GPU token(s) available while graph still has running tasks`, this.id);
      const context = this._buildLiveTaskGraphManagerContext(plan, orchestration, {
        completedTaskId,
        availableGpuTokens: availability.availableGpuTokens,
      });
      const result = await this.runAgent(managerAgent, config, 'single-manager-live-replan', context, { mode: 'full', issues: [] });
      if (!result?.success || !result?.resultText) return { added: 0, skipped: 0, killed: 0, ran: true };
      // Apply kill requests first so any newly-appended task that depends on
      // a killed task's tag will be rejected by the merge step.
      const killIds = parseKillTasksDocument(result.resultText);
      const killed = killIds.length ? this._killTaskGraphTasks(plan, killIds) : 0;
      const incoming = this.parseSchedule(result.resultText, {
        additionalKnownTaskIds: plan._tasks.map((task) => task.id),
      });
      if (!this._isTaskGraphPlan(incoming)) {
        if (String(result.resultText || '').includes('<!-- PROJECT_COMPLETE -->')) {
          log('Ignoring live PROJECT_COMPLETE while task graph still has running workers', this.id);
        }
        return { added: 0, skipped: 0, killed, ran: true };
      }
      const merged = this._mergeTaskGraphIntoCurrentPlan(plan, incoming);
      if (merged.added > 0) {
        log(`Live replanning appended ${merged.added} task(s) to current graph${merged.extendedBarriers ? ` and extended ${merged.extendedBarriers} barrier dependency/dependencies` : ''}${merged.skipped || merged.skippedRunningDeps ? ` (${merged.skipped} duplicate id(s), ${merged.skippedRunningDeps} running-dependent task(s) skipped)` : ''}`, this.id);
      } else {
        log(`Live replanning appended no tasks${merged.skipped || merged.skippedRunningDeps ? ` (${merged.skipped} duplicate id(s), ${merged.skippedRunningDeps} running-dependent task(s) skipped)` : ''}`, this.id);
      }
      return { ...merged, ran: true };
    } finally {
      runtime.liveReplanInProgress = false;
      this.saveState();
    }
  }

  // Phase 1 instrumentation: parse canonical result.txt + metrics.json from
  // the experiment output dir (best-effort extracted from the task body) and
  // record a `validated` event capturing the canonical truth alongside the
  // LLM's stdout-driven success claim. Until Phase 3 flips the source of
  // truth, mismatches are logged but not enforced — this lets us measure
  // the disagreement rate on real workloads first.
  _recordCanonicalValidation(taskId, step, lastResult) {
    try {
      if (!taskId) return;
      const outputDir = extractOutputDirFromTaskBody(step?.task);
      if (!outputDir) {
        appendTaskEvent(this.taskEventsDir, taskId, 'validate_skipped', {
          reason: 'no_output_dir_in_task_body',
        });
        return;
      }
      const resolvedDir = path.isAbsolute(outputDir) ? outputDir : path.join(this.path, outputDir);
      const verdict = validateExperimentResult({ outputDir: resolvedDir });
      const llmClaim = !!lastResult?.success;
      const mismatch = llmClaim !== verdict.success;
      appendTaskEvent(this.taskEventsDir, taskId, 'validated', {
        canonical_success: verdict.success,
        canonical_reason: verdict.reason || null,
        exit_code: verdict.exitCode ?? null,
        val_bpb: verdict.metrics?.val_bpb ?? null,
        training_seconds: verdict.metrics?.training_seconds ?? null,
        num_steps: verdict.metrics?.num_steps ?? null,
        llm_claim_success: llmClaim,
        mismatch,
        output_dir: resolvedDir,
      });
      if (mismatch) {
        log(`Truth mismatch on ${taskId}: LLM claimed ${llmClaim ? 'success' : 'failure'}, canonical says ${verdict.success ? 'success' : 'failure'} (${verdict.reason || 'ok'})`, this.id);
      }
    } catch (e) {
      // Validation must never break the orchestrator.
      try { log(`Canonical validation crashed for ${taskId}: ${e.message}`, this.id); } catch {}
    }
  }

  async _runWorkerStepWithRetries(step, worker, config, managerName, grantContext = null, { markAgentCompleted = true, taskId = null, isCancelled = () => false } = {}) {
    const visibility = this._parseVisibility(step.visibility, step.task);
    const task = buildManagedWorkerTask(step.task, grantContext?.grant || null, step.resources || null, worker.name);
    const maxRetries = step.retries ?? 2;
    let attempt = 0;
    let total = 0;
    let failures = 0;
    let succeeded = false;
    let lastResult = null;

    while (attempt <= maxRetries && !succeeded && this.running && !this.abortCurrentCycle) {
      if (isCancelled()) {
        log(`Task ${taskId || worker.name} cancelled by manager; skipping further attempts`, this.id);
        lastResult = { success: false, cancelled: true, killedByTimeout: false };
        break;
      }
      if (attempt > 0) {
        log(`Retrying ${worker.name} (attempt ${attempt + 1}/${maxRetries + 1})`, this.id);
      }
      appendTaskEvent(this.taskEventsDir, taskId, 'worker_spawned', {
        attempt: attempt + 1,
        worker: worker.name,
      });
      // Outer-ring hard cap: anything still running past hardCapMs is a hung
      // wrapper or a leaked subprocess that the inner `timeout` couldn't
      // reach. Fire abortController on the run with this taskId; runAgent
      // resolves with killedByTimeout, and the loop exits without retry.
      const hardCapMs = (step.hardCapSeconds || 900) * 1000;
      let hardCapFired = false;
      const hardCapTimer = setTimeout(() => {
        for (const runState of this.activeRuns.values()) {
          if (runState.taskId !== taskId || !runState.abortController) continue;
          hardCapFired = true;
          log(`Hard cap fired on ${taskId} after ${Math.round(hardCapMs / 1000)}s; aborting`, this.id);
          try { runState.abortController.abort(); } catch {}
        }
      }, hardCapMs);
      try {
        lastResult = await this.runAgent(worker, config, null, task, visibility, {
          primary: false,
          managerName,
          resources: step.resources || null,
          resourceGrant: grantContext?.grant || null,
          taskId,
        });
      } finally {
        clearTimeout(hardCapTimer);
      }
      if (hardCapFired) {
        lastResult = { ...(lastResult || {}), success: false, killedByTimeout: true };
      }
      total += 1;
      appendTaskEvent(this.taskEventsDir, taskId, 'worker_returned', {
        attempt: attempt + 1,
        success: !!lastResult?.success,
        killed_by_timeout: !!lastResult?.killedByTimeout,
        cancelled: !!lastResult?.cancelled,
        hard_cap_fired: hardCapFired,
        duration_ms: lastResult?.durationMs ?? null,
      });
      // Phase 1 instrumentation: read canonical result from disk and log
      // disagreement with the LLM's claim. Source of truth stays the LLM
      // for now; we measure mismatch rate before flipping in Phase 3.
      this._recordCanonicalValidation(taskId, step, lastResult);
      if (isCancelled()) {
        // Aborted mid-run; the LLM CLI was killed via runState.abortController.
        // Don't count the abort as a permanent failure of the experiment idea.
        lastResult = { ...(lastResult || {}), success: false, cancelled: true, killedByTimeout: false };
        break;
      }
      if (lastResult?.success) {
        succeeded = true;
        if (markAgentCompleted) {
          this.completedAgents.push(worker.name.toLowerCase());
          this.saveState();
        }
        break;
      }
      failures += 1;
      if (lastResult?.killedByTimeout) break;
      attempt += 1;
    }

    return { total, failures, success: succeeded, result: lastResult };
  }

  async _startScheduledWorker(step, workers, config, managerName, orchestration, options = {}) {
    if (step?.executionMode === 'param_patch') {
      return this._startDirectExecutorTask(step, config, managerName, options);
    }
    const workerSelection = options.worker
      ? { worker: options.worker, reason: null, wait: false }
      : findWorkerForStep(step, workers, options.busyWorkerNames || new Set());
    const worker = workerSelection.worker;
    if (!worker) {
      log(`Worker unavailable for "${step.id || step.agent || step.workerClass || 'task'}": ${workerSelection.reason}`, this.id);
      return {
        started: false,
        total: 1,
        failures: workerSelection.wait ? 0 : 1,
        success: false,
        blocked: true,
        wait: workerSelection.wait,
        reason: workerSelection.reason,
      };
    }

    let grantContext = null;
    if (this.isSingleManagerMode(config) && (step.resources?.gpus || 0) > 0) {
      grantContext = this._tryGrantWorkerResources(step, worker, managerName, config);
      if (grantContext.blocked) {
        appendTaskEvent(this.taskEventsDir, options.taskId, 'grant_blocked', {
          reason: grantContext.reason,
        });
        return {
          started: false,
          blocked: true,
          reason: grantContext.reason,
          workerName: worker.name,
        };
      }
      if (grantContext?.grant?.id) {
        appendTaskEvent(this.taskEventsDir, options.taskId, 'granted', {
          grant_id: grantContext.grant.id,
          tokens: grantContext.grant.tokenNames || [],
          worker: worker.name,
        });
      }
    }

    const promise = (async () => {
      let outcome = null;
      try {
        outcome = await this._runWorkerStepWithRetries(step, worker, config, managerName, grantContext, {
          markAgentCompleted: options.markAgentCompleted !== false,
          taskId: options.taskId || null,
          isCancelled: typeof options.isCancelled === 'function' ? options.isCancelled : () => false,
        });
        return outcome;
      } finally {
        if (grantContext?.grant?.id) {
          // Killed-by-timeout means SIGTERM/SIGKILL just propagated through
          // the bash + python tree. Pause 5s so any straggler releases its
          // CUDA context before we hand the GPU back, then snapshot
          // nvidia-smi for post-mortem.
          if (outcome?.result?.killedByTimeout) {
            await new Promise((resolve) => setTimeout(resolve, 5000));
            try {
              const procs = listComputeProcesses();
              if (procs.length > 0) {
                log(
                  `Hard-cap drain on ${options.taskId}: nvidia-smi still reports `
                  + `${procs.length} compute proc(s) — may include other workers`,
                  this.id,
                );
              }
            } catch {}
          }
          this._releaseWorkerGrant(grantContext.grant.id, managerName);
          appendTaskEvent(this.taskEventsDir, options.taskId, 'released', {
            grant_id: grantContext.grant.id,
          });
        }
      }
    })();

    return {
      started: true,
      step,
      workerName: worker.name,
      promise,
    };
  }

  /**
   * Dispatch a `param_patch` task without an LLM worker. Allocates a GPU
   * grant, registers the task in `activeRuns` so KILL_TASKS / shutdown
   * can abort it, and runs {@link runDirectTask}. The task plugin's
   * `directExecutor:` config block names the source repo, wrapper, and
   * sandbox/output roots; without that block we surface a blocking error
   * so the manager learns to stop emitting param_patch.
   *
   * @private
   */
  async _startDirectExecutorTask(step, config, managerName, options = {}) {
    const directConfig = (config?.directExecutor) || this.loadConfig().directExecutor || null;
    if (!directConfig?.enabled) {
      log(`Cannot run param_patch task ${step.id}: directExecutor not configured`, this.id);
      return {
        started: false,
        blocked: true,
        reason: 'directExecutor not configured for this project',
      };
    }
    const taskId = options.taskId || step.id;

    let grantContext = null;
    if (this.isSingleManagerMode(config) && (step.resources?.gpus || 0) > 0) {
      const syntheticWorker = { name: `direct:${step.id}` };
      grantContext = this._tryGrantWorkerResources(step, syntheticWorker, managerName, config);
      if (grantContext.blocked) {
        appendTaskEvent(this.taskEventsDir, taskId, 'grant_blocked', { reason: grantContext.reason });
        return { started: false, blocked: true, reason: grantContext.reason };
      }
      if (grantContext?.grant?.id) {
        appendTaskEvent(this.taskEventsDir, taskId, 'granted', {
          grant_id: grantContext.grant.id,
          tokens: grantContext.grant.tokenNames || [],
          worker: syntheticWorker.name,
        });
      }
    }

    // Token names follow the `gpu<index>` convention; map back to integer
    // CUDA device indices for CUDA_VISIBLE_DEVICES.
    const tokens = grantContext?.grant?.tokenNames || [];
    const cudaDevices = tokens
      .map((name) => Number.parseInt(String(name).replace(/^gpu/i, ''), 10))
      .filter(Number.isInteger);

    const abortController = new AbortController();
    const runState = {
      id: `direct-${step.id}-${Date.now()}`,
      taskId,
      abortController,
      agentName: 'direct-executor',
      isManager: false,
      isPrimary: false,
      startTime: Date.now(),
      log: [],
    };
    this.activeRuns.set(runState.id, runState);
    appendTaskEvent(this.taskEventsDir, taskId, 'worker_spawned', {
      worker: 'direct-executor',
      attempt: 1,
    });

    const outputDir = path.join(directConfig.outputRoot, step.id);

    const promise = (async () => {
      let result = null;
      try {
        const env = { ...directConfig.envOverrides };
        env.AUTORESEARCH_TIME_BUDGET_SECONDS = String(directConfig.timeBudgetSeconds);
        if (cudaDevices.length > 0) env.CUDA_VISIBLE_DEVICES = cudaDevices.join(',');

        result = await runDirectTask({
          sourceRepoPath: directConfig.sourceRepoPath,
          sandboxParentDir: directConfig.sandboxRoot,
          branchName: step.id,
          parentRef: step.baseRef || 'HEAD',
          edits: step.edits || [],
          wrapperScript: directConfig.wrapperScript,
          outputDir,
          env,
          abortSignal: abortController.signal,
          hardCapMs: directConfig.hardCapSeconds * 1000,
          metricKey: directConfig.metricKey,
        });

        if (!result.ok) {
          log(
            `Direct executor failed for ${taskId}: ${result.reason}`
            + (result.failurePath ? ` — see ${result.failurePath}` : ''),
            this.id,
          );
        }
        appendTaskEvent(this.taskEventsDir, taskId, 'worker_returned', {
          worker: 'direct-executor',
          attempt: 1,
          success: !!result.ok,
          killed_by_timeout: !!result.timedOut,
          cancelled: abortController.signal.aborted,
          exit_code: result.exitCode ?? null,
        });
        // The direct executor IS the canonical truth source — there's no
        // LLM stdout to disagree with — so mismatch is always false.
        appendTaskEvent(this.taskEventsDir, taskId, 'validated', {
          canonical_success: !!result.ok,
          mismatch: false,
          reason: result.reason || null,
          ...(result.metrics
            ? {
                [directConfig.metricKey]: result.metrics[directConfig.metricKey],
                training_seconds: result.metrics.training_seconds,
              }
            : {}),
        });
        return {
          success: !!result.ok,
          total: 1,
          failures: result.ok ? 0 : 1,
          result: { ...result, success: !!result.ok, killedByTimeout: !!result.timedOut },
        };
      } finally {
        this.activeRuns.delete(runState.id);
        if (grantContext?.grant?.id) {
          this._releaseWorkerGrant(grantContext.grant.id, managerName);
          appendTaskEvent(this.taskEventsDir, taskId, 'released', {
            grant_id: grantContext.grant.id,
          });
        }
      }
    })();

    return {
      started: true,
      step,
      workerName: 'direct-executor',
      promise,
    };
  }

  async _executeParallelWorkerBatch(steps, workers, config, managerName, orchestration) {
    const pending = [...steps];
    const running = [];
    let total = 0;
    let failures = 0;

    while ((pending.length > 0 || running.length > 0) && this.running && !this.abortCurrentCycle) {
      while (pending.length > 0 && running.length < orchestration.maxConcurrentWorkers && this.running && !this.abortCurrentCycle) {
        const availability = this.isSingleManagerMode(config)
          ? this._getParallelGpuAvailability()
          : { hasAllocationLease: true, totalGpuTokens: Number.MAX_SAFE_INTEGER, availableGpuTokens: Number.MAX_SAFE_INTEGER };
        const nextIndex = selectBackfillParallelStepIndex(
          pending,
          availability,
          orchestration.defaultWorkerResources,
        );
        if (nextIndex < 0) {
          if (running.length > 0) break;
          const blockedStep = pending.shift();
          const fit = classifyParallelStepGpuFit(
            blockedStep,
            availability,
            orchestration.defaultWorkerResources,
          );
          log(`Worker "${blockedStep?.agent || 'unknown'}" blocked: ${fit.reason || 'no runnable worker step'}`, this.id);
          total += 1;
          failures += 1;
          continue;
        }
        if (nextIndex > 0) {
          const headStep = pending[0];
          const chosenStep = pending[nextIndex];
          const headFit = classifyParallelStepGpuFit(
            headStep,
            availability,
            orchestration.defaultWorkerResources,
          );
          if (headFit.status === 'waiting') {
            log(
              `Backfilling ${chosenStep.agent} ahead of ${headStep.agent}: ${headFit.reason}`,
              this.id,
            );
          }
        }
        const [nextStep] = pending.splice(nextIndex, 1);
        const launched = await this._startScheduledWorker(nextStep, workers, config, managerName, orchestration);
        if (launched.started) {
          running.push(launched);
          continue;
        }
        if (launched.blocked) {
          log(`Worker "${launched.workerName}" blocked: ${launched.reason}`, this.id);
          total += 1;
          failures += 1;
          continue;
        }
        total += launched.total || 1;
        failures += launched.failures || 1;
      }

      if (running.length === 0) continue;

      const finished = await Promise.race(
        running.map((entry, index) => entry.promise.then((result) => ({ index, result }))),
      );
      const [entry] = running.splice(finished.index, 1);
      total += finished.result.total;
      failures += finished.result.failures;
      if (finished.result.success) {
        log(`Parallel worker ${entry.workerName} completed`, this.id);
      }
    }

    return { total, failures };
  }

  async _executeTaskGraph(plan, config, managerName, orchestration, workers, options = {}) {
    this._ensureTaskGraphRuntime(plan);
    const running = [];
    let total = 0;
    let failures = 0;
    let replanRequested = false;
    const managerAgent = options.managerAgent || null;
    const liveReplans = [];
    const queuedLiveReplanTaskIds = [];

    const removeLiveReplan = (liveEntry) => {
      const index = liveReplans.indexOf(liveEntry);
      if (index >= 0) liveReplans.splice(index, 1);
    };

    const startLiveReplan = (completedTaskId) => {
      if (replanRequested || !orchestration.liveReplanOnTaskComplete || !managerAgent || running.length === 0) return;
      // Watermark gate (Phase 5.1): only call the manager when the queue is
      // shallow. With a healthy backlog, the manager doesn't need to think
      // every time a worker returns — that just burns Opus calls. The
      // watermark defaults to 1.5× max workers (12 on an 8-GPU node), so we
      // don't wake the manager until the ready+running depth would force
      // GPUs to sit idle within the next dispatch round.
      const watermark = Math.max(
        running.length + 1,
        Math.ceil((orchestration.maxConcurrentWorkers || 1) * 1.5),
      );
      let readyCount = 0;
      for (const candidate of plan._tasks) {
        const state = plan._runtime.taskStates[candidate.id]?.status || 'pending';
        if (state !== 'pending') continue;
        const dep = this._getTaskGraphDependencyState(plan, candidate);
        if (dep.unmetDependencies.length === 0
            && dep.unmetTags.length === 0
            && dep.failedDependencies.length === 0) {
          readyCount += 1;
        }
      }
      if (readyCount + running.length >= watermark) {
        // Plenty of work queued; skip this completion's replan trigger.
        return;
      }
      if (liveReplans.length > 0 || plan._runtime.liveReplanInProgress) {
        // FIFO: accumulate every completion so the next replan pass sees the
        // full backlog instead of only the most recent task id.
        if (completedTaskId && !queuedLiveReplanTaskIds.includes(completedTaskId)) {
          queuedLiveReplanTaskIds.push(completedTaskId);
        }
        return;
      }

      const liveEntry = {};
      liveEntry.promise = this._maybeLiveReplanTaskGraph(plan, config, managerName, orchestration, managerAgent, {
        completedTaskId,
        running,
      })
        .then((result) => ({ type: 'live-replan', liveEntry, result }))
        .catch((error) => ({ type: 'live-replan', liveEntry, error }));
      liveReplans.push(liveEntry);
    };

    const startQueuedLiveReplan = () => {
      if (queuedLiveReplanTaskIds.length === 0) return;
      if (running.length === 0) return;
      const taskId = queuedLiveReplanTaskIds.shift();
      startLiveReplan(taskId);
    };

    const handleLiveReplanEvent = (event) => {
      removeLiveReplan(event.liveEntry);
      if (event.error) {
        log(`Live replanning failed: ${event.error?.message || event.error}`, this.id);
      }
      startQueuedLiveReplan();
    };

    while (this.running && !this.abortCurrentCycle) {
      while (!replanRequested && running.length < orchestration.maxConcurrentWorkers && this.running && !this.abortCurrentCycle) {
        const runnable = [];
        for (const task of plan._tasks) {
          const state = plan._runtime.taskStates[task.id];
          if (!state || state.status !== 'pending') continue;
          const deps = this._getTaskGraphDependencyState(plan, task);
          if (deps.failedDependencies.length > 0) {
            plan._runtime.taskStates[task.id] = {
              ...state,
              status: 'blocked',
              finishedAt: new Date().toISOString(),
              reason: `blocked by failed dependencies: ${deps.failedDependencies.join(', ')}`,
            };
            failures += 1;
            total += 1;
            this.saveState();
            continue;
          }
          if (deps.unmetDependencies.length > 0 || deps.unmetTags.length > 0) continue;
          runnable.push(task);
        }

        if (runnable.length === 0) break;

        const availability = this.isSingleManagerMode(config)
          ? this._getParallelGpuAvailability()
          : { hasAllocationLease: true, totalGpuTokens: Number.MAX_SAFE_INTEGER, availableGpuTokens: Number.MAX_SAFE_INTEGER };
        const selection = selectTaskGraphRunnableTask({
          tasks: runnable,
          availability,
          defaultWorkerResources: orchestration.defaultWorkerResources,
          workers,
          running,
          plan,
        });
        if (selection.index < 0) break;
        const task = runnable[selection.index];
        const assignedWorker = selection.worker;

        // Walltime admission gate: don't start a task that can't finish before
        // the Slurm allocation expires. Mark blocked (terminal) so the inner
        // loop continues looking for shorter ready tasks (e.g. analyst gpus=0
        // tasks). When all ready tasks are walltime-blocked, the cycle drains
        // and the manager gets a chance to schedule a finalization task.
        const wtRemaining = remainingWalltimeMs();
        if (wtRemaining !== null) {
          const requiredMs = (task.estimatedRuntimeSeconds && task.estimatedRuntimeSeconds > 0
            ? task.estimatedRuntimeSeconds
            : 300) * 1000 + 120 * 1000; // +2min slack for setup + tail
          if (wtRemaining < requiredMs) {
            log(`Task ${task.id} blocked: insufficient slurm walltime (${Math.floor(wtRemaining/1000)}s remaining < ${Math.floor(requiredMs/1000)}s required)`, this.id);
            appendTaskEvent(this.taskEventsDir, task.id, 'walltime_blocked', {
              remaining_seconds: Math.floor(wtRemaining / 1000),
              required_seconds: Math.floor(requiredMs / 1000),
            });
            plan._runtime.taskStates[task.id] = {
              ...plan._runtime.taskStates[task.id],
              status: 'blocked',
              finishedAt: new Date().toISOString(),
              reason: `insufficient slurm walltime: ${Math.floor(wtRemaining/1000)}s < ${Math.floor(requiredMs/1000)}s`,
            };
            failures += 1;
            total += 1;
            this.saveState();
            continue;
          }
        }

        const pendingAhead = plan._tasks.find((candidate) => {
          const state = plan._runtime.taskStates[candidate.id];
          if (!state || state.status !== 'pending' || candidate.id === task.id) return false;
          const deps = this._getTaskGraphDependencyState(plan, candidate);
          if (deps.failedDependencies.length > 0 || deps.unmetDependencies.length > 0 || deps.unmetTags.length > 0) return false;
          return true;
        });
        if (pendingAhead && pendingAhead.id !== task.id) {
          const headFit = classifyParallelStepGpuFit(pendingAhead, availability, orchestration.defaultWorkerResources);
          if (headFit.status === 'waiting') {
            log(`Backfilling ${task.id} ahead of ${pendingAhead.id}: ${headFit.reason}`, this.id);
          }
        }

        plan._runtime.taskStates[task.id] = {
          ...plan._runtime.taskStates[task.id],
          status: 'running',
          attempts: (plan._runtime.taskStates[task.id].attempts || 0) + 1,
          startedAt: new Date().toISOString(),
          reason: null,
        };
        this.saveState();

        appendTaskEvent(this.taskEventsDir, task.id, 'picked', {
          attempt: plan._runtime.taskStates[task.id].attempts,
          worker: assignedWorker?.name || null,
          requested_gpus: task.resources?.gpus || 0,
          priority: task.priority ?? 0,
        });

        const launched = await this._startScheduledWorker(task, workers, config, managerName, orchestration, {
          markAgentCompleted: false,
          worker: assignedWorker,
          busyWorkerNames: new Set(running.map((entry) => entry.workerName?.toLowerCase()).filter(Boolean)),
          taskId: task.id,
          isCancelled: () => plan._runtime.cancelledTaskIds?.has(task.id) === true,
        });
        if (!launched.started) {
          const fit = classifyParallelStepGpuFit(task, availability, orchestration.defaultWorkerResources);
          const status = launched.wait || (launched.blocked && fit.status === 'waiting') ? 'pending' : 'blocked';
          log(`Task ${task.id} ${status}: ${launched.reason || fit.reason || 'unknown'}`, this.id);
          plan._runtime.taskStates[task.id] = {
            ...plan._runtime.taskStates[task.id],
            status,
            finishedAt: status === 'blocked' ? new Date().toISOString() : null,
            reason: launched.reason || fit.reason || null,
          };
          if (status === 'blocked') {
            total += launched.total || 1;
            failures += launched.failures || 1;
          }
          this.saveState();
          if (status === 'pending') break;
          continue;
        }

        running.push({
          task,
          workerName: launched.workerName,
          promise: launched.promise,
        });
      }

      startQueuedLiveReplan();

      if (running.length === 0) {
        if (liveReplans.length > 0) {
          const event = await Promise.race(liveReplans.map((liveEntry) => liveEntry.promise));
          handleLiveReplanEvent(event);
          continue;
        }

        const blockedCount = this._finalizeBlockedTaskGraphTasks(plan);
        failures += blockedCount;
        total += blockedCount;
        let strandedCount = 0;
        for (const task of plan._tasks) {
          const state = plan._runtime.taskStates[task.id];
          if (!state || state.status !== 'pending') continue;
          const deps = this._getTaskGraphDependencyState(plan, task);
          const reasonParts = [];
          if (deps.unmetDependencies.length > 0) {
            reasonParts.push(`waiting on dependencies: ${deps.unmetDependencies.join(', ')}`);
          }
          if (deps.unmetTags.length > 0) {
            reasonParts.push(`waiting on tags: ${deps.unmetTags.join(', ')}`);
          }
          if (reasonParts.length === 0) {
            const fit = classifyParallelStepGpuFit(task, this._getParallelGpuAvailability(), orchestration.defaultWorkerResources);
            reasonParts.push(fit.reason || 'no runnable task remaining');
          }
          plan._runtime.taskStates[task.id] = {
            ...state,
            status: 'blocked',
            finishedAt: new Date().toISOString(),
            reason: reasonParts.join('; '),
          };
          strandedCount += 1;
        }
        if (strandedCount > 0) {
          this.saveState();
          failures += strandedCount;
          total += strandedCount;
        }
        const allTerminal = plan._tasks.every((task) => {
          const status = plan._runtime.taskStates[task.id]?.status;
          return ['completed', 'failed', 'blocked'].includes(status);
        });
        if (allTerminal && orchestration.refillOnGraphDrain && failures === 0) {
          plan._runtime.replanRequested = true;
          plan._runtime.replanTaskId = 'graph-drain';
          replanRequested = true;
          this.saveState();
        }
        break;
      }

      const finished = await Promise.race([
        ...running.map((entry, index) => entry.promise.then((result) => ({ type: 'worker', index, result }))),
        ...liveReplans.map((liveEntry) => liveEntry.promise),
      ]);
      if (finished.type === 'live-replan') {
        handleLiveReplanEvent(finished);
        continue;
      }

      const [entry] = running.splice(finished.index, 1);
      total += finished.result.total;
      failures += finished.result.failures;

      const currentState = plan._runtime.taskStates[entry.task.id] || {};
      // If the manager killed this task while it was running, keep the
      // 'cancelled' state and skip the success/failure write below.
      if (currentState.status === 'cancelled') {
        plan._runtime.lastCompletedTaskId = entry.task.id;
        this.saveState();
        startLiveReplan(entry.task.id);
        continue;
      }
      if (finished.result.success) {
        plan._runtime.taskStates[entry.task.id] = {
          ...currentState,
          status: 'completed',
          finishedAt: new Date().toISOString(),
          reason: null,
        };
        for (const tag of entry.task.producesTags || []) {
          if (!plan._runtime.producedTags.includes(tag)) {
            plan._runtime.producedTags.push(tag);
          }
        }
        if (entry.task.replanAfter) {
          plan._runtime.replanRequested = true;
          plan._runtime.replanTaskId = entry.task.id;
          replanRequested = true;
        }
      } else {
        plan._runtime.taskStates[entry.task.id] = {
          ...currentState,
          status: 'failed',
          finishedAt: new Date().toISOString(),
          reason: finished.result.result?.resultText || 'worker task failed',
        };
      }
      plan._runtime.lastCompletedTaskId = entry.task.id;
      this.saveState();

      startLiveReplan(entry.task.id);
    }

    return { total, failures, replanRequested };
  }

  async executeSchedule(schedule, config, managerName = null, options = {}) {
    if (!schedule) return { total: 0, failures: 0 };

    let total = 0;
    let failures = 0;
    const orchestration = this.getOrchestrationConfig(config);
    const workers = this._getManagedWorkers(managerName);

    if (this._isTaskGraphPlan(schedule)) {
      return this._executeTaskGraph(schedule, config, managerName, orchestration, workers, {
        managerAgent: options.managerAgent || null,
      });
    }
    if (!schedule._steps) return { total: 0, failures: 0 };

    for (const rawStep of schedule._steps) {
      if (!this.running || this.abortCurrentCycle) break;
      while (this.isPaused && this.running && !this.abortCurrentCycle) await sleep(1000);
      if (this.abortCurrentCycle) break;

      const step = this._normalizeScheduleStepForExecution(rawStep, config);
      if (!step) {
        failures += 1;
        total += 1;
        log(`Skipping invalid schedule step`, this.id);
        continue;
      }

      if (step.type === 'delay') {
        await this.sleepDelay(step.delay, 'schedule');
        continue;
      }

      if (step.type === 'parallel') {
        const result = await this._executeParallelWorkerBatch(step.steps, workers, config, managerName, orchestration);
        total += result.total;
        failures += result.failures;
        continue;
      }

      if (this.completedAgents.includes(step.agent.toLowerCase())) {
        log(`Skipping ${step.agent} (already completed this cycle)`, this.id);
        continue;
      }

      const launched = await this._startScheduledWorker(step, workers, config, managerName, orchestration);
      if (!launched.started) {
        total += launched.total || 1;
        failures += launched.failures || 1;
        if (launched.blocked) {
          log(`Worker "${launched.workerName}" blocked: ${launched.reason}`, this.id);
        }
        continue;
      }
      const result = await launched.promise;
      total += result.total;
      failures += result.failures;
    }

    return { total, failures };
  }

  async runSingleManagerCycle(config) {
    const orchestration = this.getOrchestrationConfig(config);
    const { managers } = this.loadAgents();
    const manager = managers.find((agent) => agent.name === orchestration.manager)
      || managers.find((agent) => agent.name === 'manager')
      || managers[0]
      || null;

    if (!manager) {
      log(`No manager agent found for single-manager mode`, this.id);
      return;
    }

    this.phase = 'orchestration';
    this.saveState();

    const maxManagerPasses = orchestration.maxManagerPasses || 8;
    const maxWallClockSeconds = orchestration.maxWallClockSeconds ?? null;
    let cycleFailures = 0;
    let cycleTotal = 0;

    // Phase 1.4: emit an aggregate report at cycle exit (regardless of which
    // return path fires) by reading task-events written during this window.
    const cycleStartSec = Date.now() / 1000;
    const cycleId = `${this.cycleCount}`;
    const emitReport = () => {
      try {
        const report = buildCycleReport({
          eventsDir: this.taskEventsDir,
          windowStart: cycleStartSec,
          windowEnd: Date.now() / 1000,
          gpuCount: this.loadConfig().resourceOrchestration?.gpuCount ?? 8,
          cycleId,
        });
        log(formatCycleReportLine(report), this.id);
        writeCycleReport({ reportsDir: path.join(this.projectDir, 'cycle-reports'), report });
      } catch (e) {
        try { log(`Cycle report failed: ${e.message}`, this.id); } catch {}
      }
    };

   try {

    for (let managerPass = 0; managerPass < maxManagerPasses && this.running && !this.abortCurrentCycle; managerPass += 1) {
      if (maxWallClockSeconds != null && (Date.now() - startTime) / 1000 >= maxWallClockSeconds) {
        log(`Wall-clock limit reached (${Math.floor((Date.now() - startTime) / 1000)}s >= ${maxWallClockSeconds}s), stopping`, this.id);
        return { cycleTotal, cycleFailures };
      }
      if (this.currentSchedule) {
        log(`Resuming interrupted single-manager schedule (${this.completedAgents.length} completed)`, this.id);
        const resumed = await this.executeSchedule(this.currentSchedule, config, manager.name, { managerAgent: manager });
        cycleTotal += resumed.total;
        cycleFailures += resumed.failures;
        const shouldReplan = resumed.replanRequested === true;
        this.currentSchedule = null;
        this.completedAgents = [];
        this.saveState();
        if (shouldReplan) {
          log(`Replanning after task-graph barrier`, this.id);
          continue;
        }
        return { cycleTotal, cycleFailures };
      }

      const result = await this.runAgent(manager, config, 'single-manager', this._buildSingleManagerContext(config, orchestration), { mode: 'full', issues: [] });
      cycleTotal += 1;
      if (!result?.success) cycleFailures += 1;

      let schedule = null;
      if (result?.resultText) {
        // Apply manager kill requests against the resumed schedule (if any) before
        // parsing the new graph, so dependency lineage stays consistent.
        const killIds = parseKillTasksDocument(result.resultText);
        if (killIds.length && this._isTaskGraphPlan(this.currentSchedule)) {
          this._killTaskGraphTasks(this.currentSchedule, killIds);
        }
        schedule = this.parseSchedule(result.resultText);
        if (schedule) {
          log(`Schedule: ${JSON.stringify(schedule)}`, this.id);
        }
        const completeMatch = result.resultText.match(/<!-- PROJECT_COMPLETE -->\s*([\s\S]*?)\s*<!-- \/PROJECT_COMPLETE -->/);
        if (completeMatch) {
          try {
            const completion = JSON.parse(completeMatch[1]);
            const success = completion.success !== false;
            const message = completion.message || 'Project completed';
            this.setState({
              phase: 'orchestration',
              isComplete: success,
              completionSuccess: success,
              completionMessage: message,
              isPaused: true,
              pauseReason: `Project completed: ${message}`,
              currentSchedule: null,
              completedAgents: [],
            });
            broadcastEvent({ type: 'project-complete', project: this.id, success, message });
            return { cycleTotal, cycleFailures };
          } catch (e) {
            log(`Failed to parse PROJECT_COMPLETE: ${e.message}`, this.id);
          }
        }
      }

      if (!schedule) {
        return { cycleTotal, cycleFailures };
      }

      this.currentSchedule = schedule;
      this.completedAgents = [];
      this.saveState();
      const executed = await this.executeSchedule(schedule, config, manager.name, { managerAgent: manager });
      cycleTotal += executed.total;
      cycleFailures += executed.failures;
      const shouldReplan = executed.replanRequested === true;
      this.currentSchedule = null;
      this.completedAgents = [];
      this.saveState();
      if (!shouldReplan) {
        return { cycleTotal, cycleFailures };
      }
      log(`Replanning after task-graph barrier`, this.id);
    }

    return { cycleTotal, cycleFailures };
   } finally {
     emitReport();
   }
  }

  async runLoop() {
    while (this.running) {
      while (this.isPaused && this.running) {
        await sleep(1000);
      }
      if (!this.running) break;

      const config = this.loadConfig();

      // Check budget before starting cycle
      const budgetStatus = this.getBudgetStatus();
      if (budgetStatus && budgetStatus.exhausted) {
        log(`Budget exhausted ($${budgetStatus.spent24h.toFixed(2)}/$${budgetStatus.budgetPer24h}), waiting for budget to roll off`, this.id);
        this.setState({ isPaused: true, pauseReason: `Budget exhausted: $${budgetStatus.spent24h.toFixed(2)} / $${budgetStatus.budgetPer24h} (24h)` });
        // Re-check every 2 hours until budget rolls off or manually resumed
        await this._autoPauseWait(2 * 60 * 60 * 1000, () => !this.getBudgetStatus().exhausted);
        if (!this.running) break;
        continue;
      }

      // Check if any API key is available before starting a cycle
      const preConfig = this.loadConfig();
      if (this.getAgentRuntime(preConfig) === 'api') {
        const poolCheck = getKeyPoolSafe();
        if (poolCheck.keys.filter(k => k.enabled).length === 0 && !preConfig.setupToken) {
          log(`No API keys configured. Pausing project. Add a key in Settings > Credentials.`, this.id);
          this.setState({ isPaused: true, pauseReason: 'No API keys configured. Add one in Settings > Credentials.' });
          await this._autoPauseWait(30_000, () => getKeyPoolSafe().keys.some(k => k.enabled));
          if (!this.running) break;
          continue;
        }
      }

      const { managers, workers } = this.loadAgents();

      // Start new cycle — preserve schedule state if resuming from reboot
      this.abortCurrentCycle = false;
      const resuming = !!this.currentSchedule;
      if (!resuming) {
        this.cycleCount++;
        this.completedAgents = [];
        this.saveState();
      }
      log(`===== CYCLE ${this.cycleCount} (phase: ${this.phase})${resuming ? ' [RESUMING]' : ''} =====`, this.id);

      let cycleFailures = 0;
      let cycleTotal = 0;

      if (this.isSingleManagerMode(config)) {
        const result = await this.runSingleManagerCycle(config);
        cycleFailures += result?.cycleFailures || 0;
        cycleTotal += result?.cycleTotal || 0;

        this.currentSchedule = null;
        this.completedAgents = [];
        this.saveState();

        if (!this.running) break;
        if (this.isPaused) continue;

        const sleepMs = this.computeSleepInterval();
        this.lastComputedSleepMs = sleepMs;
        if (sleepMs > 0 && this.running && !this.wakeNow) {
          log(`Sleeping ${Math.round(sleepMs / 1000)}s`, this.id);
          this.sleepUntil = Date.now() + sleepMs;
          let slept = 0;
          while (slept < sleepMs && !this.wakeNow && this.running) {
            await sleep(5000);
            slept += 5000;
            while (this.isPaused && !this.wakeNow && this.running) { await sleep(1000); }
          }
          this.sleepUntil = null;
          this.wakeNow = false;
        }
        continue;
      }

      // If no agent succeeded, don't count this cycle
      if (cycleTotal > 0 && cycleFailures === cycleTotal) {
        this.cycleCount--;
        this.saveState();
      }

      // Track consecutive agent failures — auto-pause after 10
      this.consecutiveFailures = (cycleTotal > 0 && cycleFailures === cycleTotal)
        ? this.consecutiveFailures + cycleFailures
        : 0;
      if (this.consecutiveFailures >= 10 && this.running) {
        log(`⚠️ ${this.consecutiveFailures} consecutive agent failures — auto-pausing (retry in 2h)`, this.id);
        broadcastEvent({ type: 'error', project: this.id, message: `${this.consecutiveFailures} consecutive failures — auto-paused` });
        this.setState({ isPaused: true, pauseReason: `${this.consecutiveFailures} consecutive agent failures` });
        this.consecutiveFailures = 0;
        await this._autoPauseWait(2 * 60 * 60 * 1000);
        if (!this.running) break;
        continue;
      }

      // Compute sleep: budget-derived or fixed interval
      const sleepMs = this.computeSleepInterval();
      this.lastComputedSleepMs = sleepMs; // Cache for status requests
      if (this.running) {
        log(`Sleeping ${Math.round(sleepMs / 1000)}s...`, this.id);
        this.wakeNow = false;
        this.sleepUntil = Date.now() + sleepMs;

        let sleptMs = 0;
        while (sleptMs < sleepMs && !this.wakeNow && this.running) {
          await sleep(5000);
          sleptMs += 5000;
          while (this.isPaused && !this.wakeNow && this.running) {
            await sleep(1000);
          }
        }
        this.sleepUntil = null;
      }
    }
  }

  // Build the full prompt for an agent (shared across CLI and API paths)
  _getAgentFilesystemPolicy(agent, visibility = null) {
    if (agent.name === 'doctor') {
      return null;
    }
    const visMode = visibility?.mode || 'full';
    const repoDir = this.path;
    const knowledgeDir = this.knowledgeDir;
    const ownWorkspaceDir = path.join(this.agentsDir, agent.name);
    const read = [repoDir];
    const write = [repoDir];
    if (visMode !== 'blind') {
      read.push(knowledgeDir);
      read.push(ownWorkspaceDir);
      write.push(ownWorkspaceDir);
    }
    if (agent.isManager && visMode !== 'blind') {
      read.push(this.workerSkillsDir);
      write.push(this.workerSkillsDir);
    }
    const denied = [
      this.agentsDir,
      this.responsesDir,
      this.uploadsDir,
      this.skillsDir,
      this.statePath,
      this.orchestratorLogPath,
      this.projectDbPath,
    ];
    return { read, write, denied, dbPath: this.projectDbPath };
  }

  _buildAgentPrompt(agent, task, visibility) {
    const skillPath = agent.isManager
      ? path.join(ROOT, 'agent', 'managers', `${agent.name}.md`)
      : path.join(this.workerSkillsDir, `${agent.name}.md`);

    if (!fs.existsSync(skillPath)) {
      return null;
    }

    let skillContent = fs.readFileSync(skillPath, 'utf-8');

    // Build shared rules: everyone.md + folder_structure.md + db.md + role-specific rules
    let sharedRules = '';
    try {
      const everyonePath = path.join(ROOT, 'agent', 'everyone.md');
      const folderStructurePath = path.join(ROOT, 'agent', 'folder_structure.md');
      sharedRules = fs.readFileSync(everyonePath, 'utf-8') + '\n\n---\n\n';
      try {
        sharedRules += fs.readFileSync(folderStructurePath, 'utf-8') + '\n\n---\n\n';
      } catch {}
      const visMode = visibility?.mode || 'full';
      if (visMode === 'full') {
        const dbPath = path.join(ROOT, 'agent', 'db.md');
        try {
          const dbContent = fs.readFileSync(dbPath, 'utf-8');
          sharedRules += dbContent + '\n\n---\n\n';
        } catch {}
      }
      if (visMode === 'focused') {
        sharedRules += '\n> **You are in focused mode.** You cannot read the issue tracker or PR board. Work only from the task, the repository, shared knowledge, and your own agent notes. If needed, you may create a new issue or PR record to report a blocker or finding.\n\n---\n\n';
      } else if (visMode === 'blind') {
        sharedRules += '\n> **You are in blind mode.** You cannot read the issue tracker or PR board, and you cannot rely on shared knowledge or any agent notes, including your own prior notes. Work only from the task and the repository. If needed, you may create a new issue or PR record to report a blocker or finding.\n\n---\n\n';
      }
      const rolePath = path.join(ROOT, 'agent', agent.isManager ? 'manager.md' : 'worker.md');
      sharedRules += fs.readFileSync(rolePath, 'utf-8') + '\n\n---\n\n';
      const activeRuntime = this.getAgentRuntime();
      if (activeRuntime === 'codex_cli') {
        sharedRules += '\n> **Runtime:** codex_cli\n> You are running through the local Codex CLI runtime, not the JSON API tool runtime. Use normal shell commands when you need to inspect files, run tests, call `tbc-db`, or attach to the shared Slurm allocation. If a task mentions `grant_id`, `lease_job_id`, `gpu_tokens`, or `recommended_cpus_per_task`, use them directly when constructing your `srun --jobid <job> --overlap --ntasks=1 --cpus-per-task <n>` command.\n\n---\n\n';
      } else if (activeRuntime === 'claude_cli') {
        sharedRules += '\n> **Runtime:** claude_cli (Claude Code)\n> You are running through the local Claude Code CLI runtime, not the JSON API tool runtime. Use normal shell commands (Bash/Read/Write/Edit/Grep) when you need to inspect files, run tests, call `tbc-db`, or attach to the shared Slurm allocation. If a task mentions `grant_id`, `lease_job_id`, `gpu_tokens`, or `recommended_cpus_per_task`, use them directly when constructing your `srun --jobid <job> --overlap --ntasks=1 --cpus-per-task <n>` command. Do not open GitHub PRs; use `tbc-db pr-create`.\n\n---\n\n';
      }
    } catch {}

    let taskHeader = '';
    if (task) {
      taskHeader = `> **Your assignment: ${task}**\n\n`;
    }

    // Strip YAML frontmatter (---...---) from skill content before building prompt
    skillContent = skillContent.replace(/^---[\s\S]*?---\n*/, '');
    skillContent = (taskHeader + sharedRules + skillContent).replaceAll('{project_dir}', this.projectDir);

    return skillContent;
  }

  // Post-processing shared by both CLI and API agent runs
  _postProcessAgentRun(agent, config, { resultText, cost, durationMs, killedByTimeout, exitCode, rawOutput, apiSuccess, usage }, runState) {
    const durationStr = `${Math.floor(durationMs / 60000)}m ${Math.floor((durationMs % 60000) / 1000)}s`;
    // For API runner: use apiSuccess if provided; for CLI runner: use exitCode
    const success = !killedByTimeout && (apiSuccess !== undefined ? apiSuccess : (exitCode === 0 || exitCode === undefined));

    // Build token info string for logging
    let tokenInfo = '';
    if (cost !== undefined) {
      tokenInfo = ` | cost: $${cost.toFixed(4)}`;
    }

    // Log response to agent-specific log file
    try {
      const responsesDir = this.responsesDir;
      fs.mkdirSync(responsesDir, { recursive: true });
      const timestamp = new Date().toLocaleString('sv-SE', { hour12: false }).replace(',', '');
      const header = `\n${'='.repeat(60)}\n[${timestamp}] Cycle ${this.cycleCount} | Success: ${success}\n${'='.repeat(60)}\n`;

      // Always log raw output for debugging
      const rawLogPath = path.join(responsesDir, `${agent.name}.raw.log`);
      fs.appendFileSync(rawLogPath, header + (rawOutput || resultText || '') + '\n');

      // Log parsed result if available
      if (resultText) {
        const agentLogPath = path.join(responsesDir, `${agent.name}.log`);
        fs.appendFileSync(agentLogPath, header + resultText + '\n');
      }
    } catch (e) {
      log(`Failed to log response for ${agent.name}: ${e.message}`, this.id);
    }

    // Write agent report to SQLite (cost data included — no longer writes to cost.csv)
    if (resultText || killedByTimeout || !success) {
      try {
        let reportBody;
        if (killedByTimeout || !success) {
          const errorType = killedByTimeout ? '⏰ Timeout' : '❌ Error';
          const errorMsg = killedByTimeout
            ? `Killed after exceeding the ${Math.floor(config.agentTimeoutMs / 60000)}m timeout limit.`
            : `Agent failed${exitCode !== undefined ? ` (exit code ${exitCode})` : ''}.`;
          // Capture partial work on timeout
          let partialWork = '';
          if (killedByTimeout) {
            try {
              const repoDir = path.join(this.projectDir, 'repo');
              if (fs.existsSync(path.join(repoDir, '.git'))) {
                const diffStat = execSync('git diff --stat HEAD 2>/dev/null || true', { cwd: repoDir, encoding: 'utf-8', timeout: 10000 }).trim();
                const stagedStat = execSync('git diff --stat --cached HEAD 2>/dev/null || true', { cwd: repoDir, encoding: 'utf-8', timeout: 10000 }).trim();
                if (diffStat || stagedStat) {
                  partialWork = `\n\n### Partial Work Detected\n\nUncommitted changes found in repo:\n\`\`\`\n${(stagedStat ? 'Staged:\n' + stagedStat + '\n' : '')}${(diffStat ? 'Unstaged:\n' + diffStat : '')}\n\`\`\``;
                }
              }
            } catch {}
          }
          reportBody = `## ${errorType}\n\n${errorMsg}\n\n- Duration: ${durationStr}${partialWork}`;
          // Include partial result text if we have it
          if (resultText) {
            reportBody += `\n\n### Partial Response\n\n${resultText.trim()}`;
          }
        } else {
          reportBody = resultText.trim();
        }
        // Prepend time log to all reports
        const agentStartTime = new Date(runState.startTime).toLocaleString('sv-SE');
        const endTime = new Date().toLocaleString('sv-SE');
        reportBody = `> ⏱ Started: ${agentStartTime} | Ended: ${endTime} | Duration: ${durationStr}\n\n${reportBody}`;
        const db = this.getDb();
        db.exec(`CREATE TABLE IF NOT EXISTS reports (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          cycle INTEGER NOT NULL,
          agent TEXT NOT NULL,
          body TEXT NOT NULL,
          summary TEXT,
          created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        )`);
        try { db.exec('ALTER TABLE reports ADD COLUMN summary TEXT'); } catch {}
        try { db.exec('ALTER TABLE reports ADD COLUMN cost REAL'); } catch {}
        try { db.exec('ALTER TABLE reports ADD COLUMN duration_ms INTEGER'); } catch {}
        try { db.exec('ALTER TABLE reports ADD COLUMN input_tokens INTEGER'); } catch {}
        try { db.exec('ALTER TABLE reports ADD COLUMN output_tokens INTEGER'); } catch {}
        try { db.exec('ALTER TABLE reports ADD COLUMN cache_read_tokens INTEGER'); } catch {}
        try { db.exec('ALTER TABLE reports ADD COLUMN success INTEGER'); } catch {}
        try { db.exec('ALTER TABLE reports ADD COLUMN model TEXT'); } catch {}
        try { db.exec('ALTER TABLE reports ADD COLUMN timed_out INTEGER'); } catch {}
        try { db.exec('ALTER TABLE reports ADD COLUMN key_id TEXT'); } catch {}
        try { db.exec('ALTER TABLE reports ADD COLUMN visibility_mode TEXT'); } catch {}
        try { db.exec('ALTER TABLE reports ADD COLUMN visibility_issues TEXT'); } catch {}
        db.prepare(`INSERT INTO reports (cycle, agent, body, created_at, cost, duration_ms, input_tokens, output_tokens, cache_read_tokens, success, model, timed_out, key_id, visibility_mode, visibility_issues)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(
          this.cycleCount, agent.name, reportBody, new Date().toISOString(),
          cost ?? null, durationMs ?? null,
          usage?.inputTokens ?? null, usage?.outputTokens ?? null, usage?.cacheReadTokens ?? null,
          success ? 1 : 0, runState.model ?? null, killedByTimeout ? 1 : 0,
          runState.keyId ?? null,
          runState.visibility?.mode || 'full', JSON.stringify(runState.visibility?.issues || [])
        );
        const lastId = db.prepare('SELECT last_insert_rowid() as id').get().id;
        db.close();
        log(`Saved report for ${agent.name}`, this.id);
        // Broadcast new report via SSE
        broadcastReportUpdate(this.id, lastId, agent.name, this.cycleCount);
      } catch (dbErr) {
        log(`Failed to write report: ${dbErr.message}`, this.id);
      }
    }

    log(`${agent.name} done (success: ${success})${tokenInfo}`, this.id);
    const summary = resultText ? stripMetaBlocks(resultText).slice(0, 500).replace(/\n+/g, ' ').trim() : '';
    broadcastEvent({ type: 'agent-done', project: this.id, agent: agent.name, success, summary });
    this._removeRunState(runState);

    return { success, resultText, killedByTimeout: !!killedByTimeout };
  }

  async runAgent(agent, config, mode = null, task = null, visibility = null, runOptions = {}) {
    const runAbortController = new AbortController();
    const runState = this._createRunState(agent, {
      mode,
      task,
      visibility,
      primary: runOptions.primary !== undefined ? !!runOptions.primary : !!agent.isManager,
      managerName: runOptions.managerName || null,
      resources: runOptions.resources || null,
      resourceGrant: runOptions.resourceGrant || null,
      taskId: runOptions.taskId || null,
    });
    runState.abortController = runAbortController;
    this._syncCurrentAgentSnapshot();
    const modeStr = mode ? ` [${mode}]` : '';
    log(`Running: ${agent.name}${agent.isManager ? ' (manager)' : ''}${modeStr}`, this.id);

    // Ensure agent notes directory exists for this agent
    const agentNotesDir = this.getAgentNotesDir(agent.name);
    fs.mkdirSync(agentNotesDir, { recursive: true });

    // Build the prompt (shared between CLI and API paths)
    const skillContent = this._buildAgentPrompt(agent, task, visibility);
    if (!skillContent) {
      const skillPath = agent.isManager
        ? path.join(ROOT, 'agent', 'managers', `${agent.name}.md`)
        : path.join(this.workerSkillsDir, `${agent.name}.md`);
      log(`Skill file not found: ${skillPath}, skipping ${agent.name}`, this.id);
      this._removeRunState(runState);
      return { success: false, resultText: '' };
    }

    const agentTierOrModel = agent.rawModel || config.model || 'mid';
    const agentRuntime = this.getAgentRuntime(config);
    const projectId = this.id;
    const hpcCfg = config.hpc || null;
    const hpcDefaults = {
      project_db_path: this.projectDbPath,
      agent_name: agent.name,
      shared_working_directory: this.path,
      shared_job_id: hpcCfg?.shared_job_id || null,
      shared_env: hpcCfg?.shared_env || null,
    };

    const agentEnv = {
      CLAUDE_CODE_ENTRYPOINT: 'cli',
      TBC_DB: this.projectDbPath,
      TBC_VISIBILITY: visibility?.mode || 'full',
      TBC_FOCUSED_ISSUES: visibility?.issues?.join(',') || '',
    };

    try {
      const sharedRunCallbacks = {
        log: (msg) => {
          log(`  [${agent.name}] ${msg}`, projectId);
          if (typeof msg === 'string' && msg.startsWith('Tool: ')) return;
          const event = { time: Date.now(), type: 'thinking', content: String(msg) };
          runState.log.push(event);
          if (runState.log.length > 500) runState.log.shift();
          broadcastLiveAgentEvent(projectId, event);
        },
        onEvent: (event) => {
          const enriched = { time: Date.now(), ...event };
          runState.log.push(enriched);
          if (runState.log.length > 500) runState.log.shift();
          broadcastLiveAgentEvent(projectId, enriched);
        },
        onProgress: ({ usage, cost }) => {
          runState.cost = cost;
          runState.usage = usage;
          if (this.currentRunId === runState.id) {
            this.currentAgentCost = cost;
            this.currentAgentUsage = usage;
          }
        },
      };

      if (agentRuntime === 'codex_cli') {
        const codexModel = agent.rawModel || config.models?.[agentTierOrModel] || config.model || null;
        runState.model = codexModel ? `codex-cli ${codexModel}` : 'codex-cli';
        this._syncCurrentAgentSnapshot();
        log(`Using Codex CLI runner for ${agent.name}${codexModel ? ` (model: ${codexModel})` : ''}`, this.id);

        const result = await runAgentWithCodexCLI({
          prompt: skillContent,
          model: codexModel,
          cwd: this.path,
          timeoutMs: config.agentTimeoutMs || 0,
          env: agentEnv,
          abortSignal: runAbortController.signal,
          projectDir: this.projectDir,
          ...sharedRunCallbacks,
        });

        return this._postProcessAgentRun(agent, config, {
          resultText: result.resultText,
          cost: result.cost,
          durationMs: result.durationMs,
          killedByTimeout: result.timedOut || false,
          apiSuccess: result.success,
          usage: result.usage,
          rawOutput: result.rawOutput,
        }, runState);
      }

      if (agentRuntime === 'claude_cli') {
        const claudeModel = agent.rawModel || config.models?.[agentTierOrModel] || config.model || null;
        runState.model = claudeModel ? `claude-cli ${claudeModel}` : 'claude-cli';
        this._syncCurrentAgentSnapshot();
        log(`Using Claude CLI runner for ${agent.name}${claudeModel ? ` (model: ${claudeModel})` : ''}`, this.id);

        const result = await runAgentWithClaudeCLI({
          prompt: skillContent,
          model: claudeModel,
          cwd: this.path,
          timeoutMs: config.agentTimeoutMs || 0,
          env: agentEnv,
          abortSignal: runAbortController.signal,
          projectDir: this.projectDir,
          ...sharedRunCallbacks,
        });

        return this._postProcessAgentRun(agent, config, {
          resultText: result.resultText,
          cost: result.cost,
          durationMs: result.durationMs,
          killedByTimeout: result.timedOut || false,
          apiSuccess: result.success,
          usage: result.usage,
          rawOutput: result.rawOutput,
        }, runState);
      }

      // Resolve token from key pool first — provider comes from the resolved key
      const oauthTokenGetter = async (authFile, provider) => {
        return getOAuthAccessToken(provider, this.id);
      };
      const keyResult = await resolveKeyForProject(config, null, oauthTokenGetter);
      let resolvedToken = keyResult?.token || null;
      let resolvedKeyId = keyResult?.keyId || null;
      const resolvedKeyType = keyResult?.type || 'api';
      runState.keyId = resolvedKeyId;
      this._syncCurrentAgentSnapshot();

      // Fallback: setupToken when project key selection is not configured
      if (!resolvedToken && config.setupToken) {
        resolvedToken = config.setupToken;
      }

      // Derive provider from the resolved key
      let providerHint;
      if (keyResult?.provider) {
        providerHint = keyResult.provider;
      } else if (config.setupTokenProvider) {
        providerHint = config.setupTokenProvider;
      } else if (resolvedToken) {
        providerHint = detectProviderFromToken(resolvedToken);
      } else {
        providerHint = 'anthropic';
      }

      const runtimeSelection = getProviderRuntimeSelection({
        provider: providerHint,
        modelTier: agentTierOrModel,
        keyResult,
        projectModels: config.models,
      });
      const agentModel = runtimeSelection.selectedModel;
      const reasoningEffort = runtimeSelection.reasoningEffort || null;
      const customConfig = runtimeSelection.customConfig || null;

      if (!resolvedToken) {
        log(`No API token configured for ${agent.name} (model: ${agentModel}). Skipping agent run. Add a key in Settings.`, this.id);
        this._removeRunState(runState);
        return { error: 'no_token', message: 'No API key configured. Add one in Settings > Credentials.' };
      }

      const tierLabel = runtimeSelection.reasoningEffort ? `${agentModel} (${runtimeSelection.reasoningEffort})` : agentModel;
      log(`Using API runner for ${agent.name} (model: ${tierLabel})`, this.id);
      runState.model = tierLabel;
      this._syncCurrentAgentSnapshot();

      const result = await runAgentWithAPI({
        prompt: skillContent,
        model: agentModel,
        token: resolvedToken,
        keyType: resolvedKeyType,
        provider: providerHint,
        customConfig,
        reasoningEffort,
        cwd: this.path,
        timeoutMs: config.agentTimeoutMs || 0,
        env: agentEnv,
        allowedRepo: agent.name === 'doctor' ? null : (this.repo || null),
        allowedPaths: this._getAgentFilesystemPolicy(agent, visibility),
        issuePolicy: { ...(visibility || { mode: 'full', issues: [] }), actor: agent.name },
        abortSignal: runAbortController.signal,
        keyId: resolvedKeyId,
        onRateLimited: (kid, cooldownMs) => markRateLimited(kid, cooldownMs || 5 * 60_000),
        resolveNewToken: async () => {
          const newKey = await resolveKeyForProject(config, null, oauthTokenGetter);
          if (newKey?.provider) {
            const newRuntimeSelection = getProviderRuntimeSelection({
              provider: newKey.provider,
              modelTier: agentTierOrModel,
              keyResult: newKey,
              projectModels: null,
            });
            newKey.model = newRuntimeSelection.selectedModel;
            newKey.reasoningEffort = newRuntimeSelection.reasoningEffort || null;
            newKey.customConfig = newRuntimeSelection.customConfig || null;
          }
          if (newKey?.keyId) {
            runState.keyId = newKey.keyId;
            this._syncCurrentAgentSnapshot();
          }
          return newKey;
        },
        ...sharedRunCallbacks,
        hpcDefaults,
      });

      if (result.success && resolvedKeyId) {
        markKeySucceeded(resolvedKeyId);
      }

      return this._postProcessAgentRun(agent, config, {
        resultText: result.resultText,
        cost: result.cost,
        durationMs: result.durationMs,
        killedByTimeout: result.timedOut || false,
        apiSuccess: result.success,
        usage: result.usage,
        rawOutput: JSON.stringify({ usage: result.usage, resultText: result.resultText }),
      }, runState);
    } catch (error) {
      log(`Agent ${agent.name} crashed: ${error.message}`, this.id);
      return this._postProcessAgentRun(agent, config, {
        resultText: '',
        cost: runState.cost || 0,
        durationMs: Date.now() - runState.startTime,
        killedByTimeout: false,
        exitCode: 1,
        apiSuccess: false,
        usage: runState.usage,
        rawOutput: error.stack || error.message,
      }, runState);
    }
  }
}

// --- Load Projects ---
function loadProjects() {
  const projectsPath = path.join(TBC_HOME, 'projects.yaml');
  try {
    if (!fs.existsSync(projectsPath)) {
      const defaultConfig = `# HPC Agent - Project Registry
projects:
  # Example:
  # m2sim:
  #   path: ~/dev/src/github.com/sarchlab/m2sim
  #   enabled: true
`;
      fs.writeFileSync(projectsPath, defaultConfig);
      log(`Created ${projectsPath}`);
      return {};
    }
    const raw = fs.readFileSync(projectsPath, 'utf-8');
    const config = yaml.load(raw) || {};
    return config.projects || {};
  } catch (e) {
    log(`Failed to load projects.yaml: ${e.message}`);
    return {};
  }
}

function syncProjects() {
  const config = loadProjects();
  
  for (const [id, cfg] of Object.entries(config)) {
    if (!projects.has(id)) {
      const runner = new ProjectRunner(id, cfg);
      projects.set(id, runner);
      if (runner.enabled) {
        runner.start();
      }
    }
  }
  
  for (const [id, runner] of projects) {
    if (!(id in config)) {
      runner.stop();
      projects.delete(id);
    }
  }
}

// --- Basic Auth ---
const TBC_PASSWORD = process.env.TBC_PASSWORD || null;

function isAuthenticated(req) {
  if (!TBC_PASSWORD) return true;
  const auth = req.headers.authorization;
  if (auth && auth.startsWith('Basic ')) {
    const decoded = Buffer.from(auth.slice(6), 'base64').toString();
    const [, pass] = decoded.split(':');
    if (pass === TBC_PASSWORD) return true;
  }
  return false;
}

function requireWrite(req, res) {
  if (isAuthenticated(req)) return true;
  res.writeHead(403, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'Authentication required for write operations' }));
  return false;
}

// --- HTTP API ---
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  const pathParts = url.pathname.split('/').filter(Boolean);
  
  // CORS: only allow requests from the same origin (the dashboard served by this server)
  const origin = req.headers.origin;
  const allowedOrigin = `http://localhost:${PORT}`;
  if (origin === allowedOrigin || origin === `http://127.0.0.1:${PORT}`) {
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  }

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  // --- VAPID public key ---
  if (req.method === 'GET' && url.pathname === '/api/push/vapid-key') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ key: VAPID_PUBLIC || null }));
    return;
  }

  // --- Push subscription ---
  if (req.method === 'POST' && url.pathname === '/api/push/subscribe') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try {
        const sub = JSON.parse(body);
        if (sub.endpoint) {
          pushSubscriptions.set(sub.endpoint, sub);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: true }));
        } else {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Missing endpoint' }));
        }
      } catch {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Invalid JSON' }));
      }
    });
    return;
  }

  if (req.method === 'POST' && url.pathname === '/api/push/unsubscribe') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try {
        const { endpoint } = JSON.parse(body);
        pushSubscriptions.delete(endpoint);
      } catch {}
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true }));
    });
    return;
  }

  // --- Notifications API ---
  if (req.method === 'GET' && url.pathname === '/api/notifications') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(notifications));
    return;
  }

  if (req.method === 'POST' && url.pathname === '/api/notifications/read-all') {
    for (const n of notifications) n.read = true;
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ok: true }));
    return;
  }

  if (req.method === 'POST' && /^\/api\/notifications\/[^/]+\/read$/.test(url.pathname)) {
    const id = url.pathname.split('/')[3];
    const n = notifications.find(x => x.id === id);
    if (n) n.read = true;
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ok: true }));
    return;
  }

  // --- SSE endpoint ---
  if (req.method === 'GET' && url.pathname === '/api/events') {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    });
    res.write('data: {"type":"connected"}\n\n');
    sseClients.add(res);
    req.on('close', () => sseClients.delete(res));
    return;
  }

  // --- Auth status ---
  if (req.method === 'GET' && url.pathname === '/api/auth') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ authenticated: isAuthenticated(req), passwordRequired: !!TBC_PASSWORD }));
    return;
  }

  // --- Settings (global token) ---

  if (req.method === 'GET' && url.pathname === '/api/settings') {
    const anthropicToken = process.env.ANTHROPIC_AUTH_TOKEN || process.env.ANTHROPIC_API_KEY || null;
    const openaiToken = process.env.OPENAI_API_KEY || null;
    const googleToken = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || null;
    const codexCreds = loadOAuthCredentials('openai-codex');
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      // Backward compat fields
      hasGlobalToken: !!anthropicToken,
      globalTokenPreview: anthropicToken ? maskToken(anthropicToken) : null,
      tokenType: anthropicToken ? (anthropicToken.startsWith('sk-ant-oat') ? 'oauth' : 'api_key') : null,
      providers: {
        anthropic: { hasToken: !!anthropicToken, preview: anthropicToken ? maskToken(anthropicToken) : null },
        openai: { hasToken: !!openaiToken, preview: openaiToken ? maskToken(openaiToken) : null },
        google: { hasToken: !!googleToken, preview: googleToken ? maskToken(googleToken) : null },
        'openai-codex': { hasToken: !!codexCreds?.access, type: 'oauth' },
      },
      // New: key pool
      keyPool: getKeyPoolSafe(),
    }));
    return;
  }

  if (req.method === 'POST' && url.pathname === '/api/settings/token') {
    if (!requireWrite(req, res)) return;
    let body = '';
    req.on('data', d => body += d);
    req.on('end', () => {
      try {
        const { token, provider: forceProvider } = JSON.parse(body);
        const envPath = path.join(TBC_HOME, '.env');
        let envContent = '';
        try { envContent = fs.readFileSync(envPath, 'utf-8'); } catch {}

        // Detect provider from token or explicit provider field
        const provider = forceProvider || detectTokenProvider(token) || 'anthropic';

        // Provider → env var mapping
        const providerEnvVars = {
          anthropic: ['ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_API_KEY'],
          openai: ['OPENAI_API_KEY'],
          google: ['GEMINI_API_KEY', 'GOOGLE_API_KEY'],
        };

        // Clear existing env vars for this provider
        const varsToClean = providerEnvVars[provider] || [];
        for (const v of varsToClean) {
          envContent = envContent.replace(new RegExp(`^${v}=.*\\n?`, 'm'), '');
          delete process.env[v];
        }

        if (token) {
          // Pick the right env var
          let envVar;
          if (provider === 'anthropic') {
            envVar = token.startsWith('sk-ant-oat') ? 'ANTHROPIC_AUTH_TOKEN' : 'ANTHROPIC_API_KEY';
          } else if (provider === 'openai') {
            envVar = 'OPENAI_API_KEY';
          } else if (provider === 'google') {
            envVar = 'GEMINI_API_KEY';
          } else {
            envVar = 'ANTHROPIC_API_KEY';
          }
          envContent = envContent.trimEnd() + `\n${envVar}=${token}\n`;
          process.env[envVar] = token;

          // Also add to key pool (backward compat)
          addKey({ label: `${provider.charAt(0).toUpperCase() + provider.slice(1)}`, token, provider });
        } else {
          // Token removal — remove matching pool entries for this provider
          const pool = loadKeyPool();
          const toRemove = pool.keys.filter(k => k.provider === provider && k.type === 'api_key');
          for (const k of toRemove) removeKey(k.id);
        }

        fs.writeFileSync(envPath, envContent);

        // Return updated status for all providers
        const anthropicToken = process.env.ANTHROPIC_AUTH_TOKEN || process.env.ANTHROPIC_API_KEY || null;
        const openaiToken = process.env.OPENAI_API_KEY || null;
        const googleToken = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || null;
        const codexCreds = loadOAuthCredentials('openai-codex');
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          success: true,
          provider,
          hasGlobalToken: !!anthropicToken,
          tokenType: anthropicToken ? (anthropicToken.startsWith('sk-ant-oat') ? 'oauth' : 'api_key') : null,
          providers: {
            anthropic: { hasToken: !!anthropicToken, preview: anthropicToken ? maskToken(anthropicToken) : null },
            openai: { hasToken: !!openaiToken, preview: openaiToken ? maskToken(openaiToken) : null },
            google: { hasToken: !!googleToken, preview: googleToken ? maskToken(googleToken) : null },
            'openai-codex': { hasToken: !!codexCreds?.access, type: 'oauth' },
          },
        }));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  // --- Key Pool CRUD endpoints ---

  if (req.method === 'GET' && url.pathname === '/api/keys') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ...getKeyPoolSafe(), allowCustomProvider: ALLOW_CUSTOM_PROVIDER }));
    return;
  }

  const keyGetMatch = url.pathname.match(/^\/api\/keys\/([^/]+)$/);
  if (req.method === 'GET' && keyGetMatch) {
    if (!requireWrite(req, res)) return;
    const key = getKeyPoolSafe().keys.find(k => k.id === keyGetMatch[1]);
    if (!key) {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Key not found' }));
      return;
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ key }));
    return;
  }

  if (req.method === 'POST' && url.pathname === '/api/keys') {
    if (!requireWrite(req, res)) return;
    let body = '';
    req.on('data', d => body += d);
    req.on('end', () => {
      try {
        const { label, token, provider, type, authFile, customConfig } = JSON.parse(body);
        if (provider === 'custom' && !ALLOW_CUSTOM_PROVIDER) {
          res.writeHead(403, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Custom provider is disabled on this instance (set TBC_ALLOW_CUSTOM_PROVIDER=true to enable)' }));
          return;
        }
        if (type === 'oauth' && authFile) {
          // OAuth credential (browser sign-in) — no token, has authFile
          addOAuthKey({ label, provider, authFile });
        } else if (token) {
          addKey({ label, token, provider, type, customConfig });
        } else {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Token is required (or authFile for OAuth)' }));
          return;
        }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(getKeyPoolSafe()));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  // PUT /api/keys/:id
  const keysPutMatch = url.pathname.match(/^\/api\/keys\/([^/]+)$/);
  if (req.method === 'PUT' && keysPutMatch) {
    if (!requireWrite(req, res)) return;
    let body = '';
    req.on('data', d => body += d);
    req.on('end', () => {
      try {
        const patch = JSON.parse(body);
        if (patch.customConfig && !ALLOW_CUSTOM_PROVIDER) {
          res.writeHead(403, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Custom provider is disabled on this instance' }));
          return;
        }
        const updated = updateKey(keysPutMatch[1], patch);
        if (!updated) {
          res.writeHead(404, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Key not found' }));
          return;
        }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(getKeyPoolSafe()));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  // DELETE /api/keys/:id
  const keysDeleteMatch = url.pathname.match(/^\/api\/keys\/([^/]+)$/);
  if (req.method === 'DELETE' && keysDeleteMatch) {
    if (!requireWrite(req, res)) return;
    removeKey(keysDeleteMatch[1]);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(getKeyPoolSafe()));
    return;
  }

  if (req.method === 'POST' && url.pathname === '/api/keys/reorder') {
    if (!requireWrite(req, res)) return;
    let body = '';
    req.on('data', d => body += d);
    req.on('end', () => {
      try {
        const { orderedIds } = JSON.parse(body);
        reorderKeys(orderedIds);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(getKeyPoolSafe()));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  // --- OAuth endpoints (generic, supports all pi-ai providers) ---

  // List available OAuth providers
  if (req.method === 'GET' && url.pathname === '/api/oauth/providers') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(listOAuthProviders()));
    return;
  }

  // Start OAuth login flow
  if (req.method === 'POST' && url.pathname === '/api/oauth/login') {
    if (!requireWrite(req, res)) return;
    let body = '';
    req.on('data', d => body += d);
    req.on('end', async () => {
      try {
        const { provider: providerId, project: projectId } = JSON.parse(body);
        if (!providerId) throw new Error('provider is required');
        const flow = await startOAuthLogin(providerId, projectId || null);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ authorization_url: flow.authorization_url, flowId: flow.flowId }));
      } catch (e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  // Submit manual code/redirect URL for active flow
  if (req.method === 'POST' && url.pathname === '/api/oauth/submit-code') {
    if (!requireWrite(req, res)) return;
    let body = '';
    req.on('data', d => body += d);
    req.on('end', async () => {
      try {
        const { flowId, code } = JSON.parse(body);
        if (!flowId || !code) throw new Error('flowId and code are required');
        submitManualCode(flowId, code);
        // Wait briefly for the flow to complete
        const { waitForFlow } = await import('./oauth.js');
        const completed = await waitForFlow(flowId, 10000);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, completed }));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  // Check OAuth status for a provider
  if (req.method === 'GET' && url.pathname === '/api/oauth/status') {
    const providerId = url.searchParams.get('provider');
    const projectId = url.searchParams.get('project') || null;
    if (!providerId) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'provider param required' }));
      return;
    }
    const status = await checkOAuthStatus(providerId, projectId);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(status));
    return;
  }

  // Logout / clear OAuth credentials
  if (req.method === 'POST' && url.pathname === '/api/oauth/logout') {
    if (!requireWrite(req, res)) return;
    let body = '';
    req.on('data', d => body += d);
    req.on('end', () => {
      try {
        const { provider: providerId, project: projectId } = JSON.parse(body);
        if (!providerId) throw new Error('provider is required');
        clearOAuthCredentials(providerId, projectId || null);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true }));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  // Backward-compatible OpenAI Codex endpoints
  if (req.method === 'POST' && url.pathname === '/api/openai-codex/login') {
    if (!requireWrite(req, res)) return;
    const projectId = url.searchParams.get('project') || null;
    try {
      const flow = await startOAuthLogin('openai-codex', projectId);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ authorization_url: flow.authorization_url, flowId: flow.flowId }));
    } catch (e) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: e.message }));
    }
    return;
  }

  if (req.method === 'GET' && url.pathname === '/api/openai-codex/status') {
    const projectId = url.searchParams.get('project') || null;
    const status = await checkOAuthStatus('openai-codex', projectId);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(status));
    return;
  }

  if (req.method === 'POST' && url.pathname === '/api/openai-codex/logout') {
    if (!requireWrite(req, res)) return;
    const projectId = url.searchParams.get('project') || null;
    clearOAuthCredentials('openai-codex', projectId);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ success: true }));
    return;
  }

  // --- Models API (fetch from Anthropic) ---

  if (req.method === 'GET' && url.pathname === '/api/models') {
    const token = process.env.ANTHROPIC_AUTH_TOKEN || process.env.ANTHROPIC_API_KEY || null;
    if (!token) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'No auth token configured' }));
      return;
    }
    try {
      const isOAuth = token.startsWith('sk-ant-oat');
      const headers = { 'anthropic-version': '2023-06-01' };
      if (isOAuth) {
        headers['Authorization'] = `Bearer ${token}`;
        headers['anthropic-beta'] = 'claude-code-20250219,oauth-2025-04-20';
      } else {
        headers['x-api-key'] = token;
      }
      const resp = await fetch('https://api.anthropic.com/v1/models?limit=100', { headers });
      if (!resp.ok) {
        const err = await resp.text();
        res.writeHead(resp.status, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: `Anthropic API error: ${resp.status}`, details: err }));
        return;
      }
      const data = await resp.json();
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(data));
    } catch (e) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: e.message }));
    }
    return;
  }

  // --- Global API ---
  
  if (req.method === 'GET' && url.pathname === '/api/status') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      uptime: Math.floor((Date.now() - startTime) / 1000),
      projectCount: projects.size,
      projects: Array.from(projects.values()).map(p => p.getStatus())
    }));
    return;
  }

  if (req.method === 'GET' && url.pathname === '/api/projects') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      projects: Array.from(projects.values()).map(p => p.getStatus())
    }));
    return;
  }

  if (req.method === 'POST' && url.pathname === '/api/reload') {
    if (!requireWrite(req, res)) return;
    syncProjects();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ success: true, projectCount: projects.size }));
    return;
  }

  // POST /api/local-projects/prepare - Prepare a local project without GitHub
  // Keep /api/projects/local as a compatibility alias.
  if (req.method === 'POST' && (url.pathname === '/api/local-projects/prepare' || url.pathname === '/api/projects/local')) {
    if (!requireWrite(req, res)) return;
    let body = '';
    req.on('data', c => body += c);
    req.on('end', () => {
      try {
        const { id: rawId, path: rawPath, createDirectory = true } = JSON.parse(body || '{}');
        const id = normalizeLocalProjectId(rawId);
        if (projects.has(id)) {
          res.writeHead(409, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: `Project "${id}" is already registered` }));
          return;
        }

        const projectPath = rawPath && String(rawPath).trim()
          ? expandHomePath(String(rawPath).trim())
          : path.join(TBC_HOME, 'local', id, 'repo');

        if (fs.existsSync(projectPath) && !fs.statSync(projectPath).isDirectory()) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: `Project path is not a directory: ${projectPath}` }));
          return;
        }

        if (!fs.existsSync(projectPath)) {
          if (!createDirectory) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: `Project path does not exist: ${projectPath}` }));
            return;
          }
          fs.mkdirSync(projectPath, { recursive: true });
        }

        const projectDataDir = path.join(TBC_HOME, 'local', id);
        const { hasSpec, specContent } = readInitialSpec(projectPath, projectDataDir);

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          success: true,
          id,
          path: projectPath,
          hasSpec,
          specContent,
          local: true,
        }));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  // GET /api/github/orgs - List GitHub orgs + current user
  if (req.method === 'GET' && url.pathname === '/api/github/orgs') {
    try {
      const user = execSync('gh api user --jq .login', { encoding: 'utf-8', timeout: 15000 }).trim();
      let orgs = [];
      try {
        orgs = execSync('gh api user/orgs --jq ".[].login"', { encoding: 'utf-8', timeout: 15000 })
          .trim().split('\n').filter(Boolean);
      } catch {}
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ available: true, user, orgs: [user, ...orgs] }));
    } catch (e) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        available: false,
        user: null,
        orgs: [],
        error: 'GitHub CLI is not configured. Local projects and manual public GitHub URL imports are still available.',
        details: e.message,
      }));
    }
    return;
  }

  // GET /api/github/repos?owner=xxx - List repos for an owner
  if (req.method === 'GET' && url.pathname === '/api/github/repos') {
    const owner = url.searchParams.get('owner');
    if (!owner) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Missing owner parameter' }));
      return;
    }
    try {
      const output = execSync(
        `gh repo list ${owner} --json nameWithOwner,name,description --limit 100`,
        { encoding: 'utf-8', timeout: 30000 }
      );
      const repos = JSON.parse(output);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ repos }));
    } catch (e) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: e.message }));
    }
    return;
  }

  // POST /api/github/create-repo - Create a new GitHub repo
  if (req.method === 'POST' && url.pathname === '/api/github/create-repo') {
    if (!requireWrite(req, res)) return;
    let body = '';
    req.on('data', c => body += c);
    req.on('end', () => {
      try {
        const { name, owner, isPrivate, description } = JSON.parse(body);
        if (!name) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Missing repo name' }));
          return;
        }
        
        // Get current user to check if owner is user or org
        const currentUser = execSync('gh api user --jq .login', { encoding: 'utf-8', timeout: 15000 }).trim();
        const isOrg = owner && owner !== currentUser;
        
        let cmd = `gh repo create`;
        if (isOrg) {
          cmd += ` ${owner}/${name}`;
        } else {
          cmd += ` ${name}`;
        }
        cmd += isPrivate ? ' --private' : ' --public';
        if (description) cmd += ` --description ${JSON.stringify(description)}`;
        // Create in TBC_HOME
        const repoId = `${owner || currentUser}/${name}`;
        const projectDir = path.join(TBC_HOME, 'dev', 'src', 'github.com', owner || currentUser, name);
        fs.mkdirSync(projectDir, { recursive: true });
        const repoDir = path.join(projectDir, 'repo');
        
        // Create the repo on GitHub (without --clone)
        execSync(cmd, { encoding: 'utf-8', timeout: 30000, stdio: 'pipe' });
        
        // Clone into the 'repo' subdirectory
        const cloneUrl = `https://github.com/${owner || currentUser}/${name}.git`;
        execSync(`git clone ${cloneUrl} repo`, { cwd: projectDir, encoding: 'utf-8', timeout: 60000, stdio: 'pipe' });
        
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, id: repoId, path: repoDir }));
      } catch (e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  // POST /api/projects/clone - Clone a GitHub repo and check for spec.md
  if (req.method === 'POST' && url.pathname === '/api/projects/clone') {
    if (!requireWrite(req, res)) return;
    let body = '';
    req.on('data', c => body += c);
    req.on('end', async () => {
      try {
        const { url: repoUrl } = JSON.parse(body);
        if (!repoUrl) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Missing url' }));
          return;
        }

        const parsed = parseGithubUrl(repoUrl);
        if (!parsed) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Invalid GitHub URL. Expected format: https://github.com/username/reponame' }));
          return;
        }

        if (projects.has(parsed.id)) {
          res.writeHead(409, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: `Project "${parsed.id}" is already registered` }));
          return;
        }

        fs.mkdirSync(parsed.projectDir, { recursive: true });

        if (fs.existsSync(path.join(parsed.repoDir, '.git'))) {
          try {
            execSync('git pull', { cwd: parsed.repoDir, encoding: 'utf-8', timeout: 60000, stdio: 'pipe' });
            log(`Pulled latest for ${parsed.id}`);
          } catch (e) {
            log(`Git pull failed for ${parsed.id}: ${e.message}`);
          }
        } else {
          try {
            execSync(`git clone ${parsed.cloneUrl} repo`, {
              cwd: parsed.projectDir,
              encoding: 'utf-8',
              timeout: 120000,
              stdio: 'pipe'
            });
            log(`Cloned ${parsed.id}`);
          } catch (e) {
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: `Failed to clone repository: ${e.message}` }));
            return;
          }
        }

        const knowledgeSpecPath = path.join(parsed.projectDir, 'knowledge', 'spec.md');
        const repoSpecPath = path.join(parsed.repoDir, 'spec.md');
        const specPath = fs.existsSync(knowledgeSpecPath) ? knowledgeSpecPath : repoSpecPath;
        const hasSpec = fs.existsSync(specPath);
        let specContent = null;
        if (hasSpec) {
          specContent = fs.readFileSync(specPath, 'utf-8');
        }

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          success: true,
          id: parsed.id,
          path: parsed.repoDir,
          hasSpec,
          specContent,
        }));
      } catch (e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  // POST /api/projects/add - Add a new project
  if (req.method === 'POST' && url.pathname === '/api/projects/add') {
    if (!requireWrite(req, res)) return;
    let body = '';
    req.on('data', c => body += c);
    req.on('end', () => {
      try {
        const { id, path: projectPath, spec, budgetPer24h } = JSON.parse(body);
        if (!id || !projectPath) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Missing id or path' }));
          return;
        }

        const resolvedPath = projectPath.replace(/^~/, process.env.HOME);

        const projectsPath = path.join(TBC_HOME, 'projects.yaml');
        const raw = fs.readFileSync(projectsPath, 'utf-8');
        const projConfig = yaml.load(raw) || {};
        if (!projConfig.projects) projConfig.projects = {};

        projConfig.projects[id] = { path: resolvedPath, enabled: true };

        fs.writeFileSync(projectsPath, yaml.dump(projConfig, { lineWidth: -1 }));

        const projectDataDir = resolveProjectDataDir(id, resolvedPath);
        writeProjectSpec(projectDataDir, spec);

        syncProjects();

        // Write initial project config with budget
        if (budgetPer24h !== undefined) {
          const runner = projects.get(id);
          if (runner) {
            const config = runner.loadConfig();
            config.budgetPer24h = parseFloat(budgetPer24h) || 0;
            fs.mkdirSync(runner.projectDir, { recursive: true });
            fs.writeFileSync(runner.configPath, yaml.dump(config, { lineWidth: -1 }));
          }
        }

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, id, path: resolvedPath }));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  // DELETE /api/projects/:id - Remove a project
  // DELETE /api/projects/:id — only match exact project path (no sub-routes like /chats/1)
  const isExactProjectDelete = req.method === 'DELETE' && pathParts[0] === 'api' && pathParts[1] === 'projects' && pathParts[2] && (
    pathParts.length === 3 || // single-segment: /api/projects/m2sim
    (pathParts.length === 4 && `${pathParts[2]}/${pathParts[3]}` && projects.has(`${pathParts[2]}/${pathParts[3]}`)) // two-segment: /api/projects/sarchlab/m2sim
  ) && !(pathParts.length > 4); // NOT a sub-route like /chats/1
  if (isExactProjectDelete) {
    const twoSegId = pathParts[3] ? `${pathParts[2]}/${pathParts[3]}` : null;
    const projectId = (twoSegId && projects.has(twoSegId)) ? twoSegId : pathParts[2];
    try {
      const projectsPath = path.join(TBC_HOME, 'projects.yaml');
      const raw = fs.readFileSync(projectsPath, 'utf-8');
      const config = yaml.load(raw) || {};
      
      if (config.projects && config.projects[projectId]) {
        // Stop the runner if running
        const runner = projects.get(projectId);
        if (runner) runner.stop();
        projects.delete(projectId);
        
        delete config.projects[projectId];
        fs.writeFileSync(projectsPath, yaml.dump(config, { lineWidth: -1 }));
        
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, id: projectId }));
      } else {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Project not found' }));
      }
    } catch (e) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: e.message }));
    }
    return;
  }

  // --- Project-scoped API ---
  // Support both single-segment (m2sim) and two-segment (sarchlab/m2sim) IDs

  if (pathParts[0] === 'api' && pathParts[1] === 'projects' && pathParts[2]) {
    const twoSegId = pathParts[3] ? `${pathParts[2]}/${pathParts[3]}` : null;
    let projectId, subPathStart;
    if (twoSegId && projects.has(twoSegId)) {
      projectId = twoSegId;
      subPathStart = 4;
    } else {
      projectId = pathParts[2];
      subPathStart = 3;
    }
    const runner = projects.get(projectId);

    if (!runner) {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Project not found' }));
      return;
    }

    const subPath = pathParts.slice(subPathStart).join('/');

    // GET /api/projects/:id/status
    if (req.method === 'GET' && subPath === 'status') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(runner.getStatus()));
      return;
    }

    // GET /api/projects/:id/logs
    if (req.method === 'GET' && subPath === 'logs') {
      const lines = parseInt(url.searchParams.get('lines')) || 50;
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ logs: runner.getLogs(lines) }));
      return;
    }

    // GET /api/projects/:id/agent-log
    if (req.method === 'GET' && subPath === 'agent-log') {
      const running = runner.currentAgent !== null;
      res.writeHead(200, { 'Content-Type': 'application/json' });
      // Resolve key label from pool
      let keyLabel = null;
      if (runner.currentAgentKeyId) {
        const pool = getKeyPoolSafe();
        keyLabel = (pool.keys || []).find(k => k.id === runner.currentAgentKeyId)?.label || null;
      }
      res.end(JSON.stringify({
        running,
        agent: runner.currentAgent,
        model: runner.currentAgentModel,
        keyId: runner.currentAgentKeyId || null,
        keyLabel,
        visibility: runner.currentAgentVisibility || { mode: 'full', issues: [] },
        startTime: runner.currentAgentStartTime,
        cost: runner.currentAgentCost || 0,
        usage: runner.currentAgentUsage || null,
        log: running ? runner.currentAgentLog : [],
        activeRuns: [...runner.activeRuns.values()].map((run) => ({
          id: run.id,
          agent: run.agentName,
          isManager: run.isManager,
          isPrimary: run.isPrimary,
          startTime: run.startTime,
          visibility: run.visibility || { mode: 'full', issues: [] },
          model: run.model || null,
          keyId: run.keyId || null,
          cost: run.cost || 0,
          usage: run.usage || null,
          resources: run.resources || null,
          grantId: run.resourceGrant?.id || null,
        })),
      }));
      return;
    }

    // GET /api/projects/:id/agents
    if (req.method === 'GET' && subPath === 'agents') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(runner.loadAgents()));
      return;
    }

    // GET /api/projects/:id/agents/:name
    if (req.method === 'GET' && subPath.startsWith('agents/') && subPath.split('/')[1]) {
      const agentName = subPath.split('/')[1];
      const details = runner.getAgentDetails(agentName);
      if (!details) {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Agent not found' }));
        return;
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(details));
      return;
    }

    // PATCH /api/projects/:id/agents/:name - Update agent settings
    if (req.method === 'PATCH' && subPath.startsWith('agents/') && subPath.split('/')[1]) {
      const agentName = subPath.split('/')[1];
      let body = '';
      req.on('data', chunk => body += chunk);
      req.on('end', () => {
        try {
          const { model, displayName } = JSON.parse(body);
          if (model === undefined && displayName === undefined) throw new Error('No fields to update');

          // Check if this is a manager or worker
          const managersDir = path.join(ROOT, 'agent', 'managers');
          const workersDir = runner.workerSkillsDir;
          const isManager = fs.existsSync(path.join(managersDir, `${agentName}.md`));
          const isWorker = fs.existsSync(path.join(workersDir, `${agentName}.md`));

          if (!isManager && !isWorker) {
            res.writeHead(404, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Agent not found' }));
            return;
          }

          // Both managers and workers store overrides in config.yaml under their respective keys.
          const config = runner.loadConfig();
          const bucket = isManager ? 'managers' : 'workers';
          if (!config[bucket]) config[bucket] = {};
          if (!config[bucket][agentName]) config[bucket][agentName] = {};

          if (model !== undefined) {
            if (model) config[bucket][agentName].model = model;
            else delete config[bucket][agentName].model;
          }
          if (displayName !== undefined) {
            const trimmed = String(displayName).trim();
            if (trimmed) config[bucket][agentName].displayName = trimmed;
            else delete config[bucket][agentName].displayName;
          }
          if (Object.keys(config[bucket][agentName]).length === 0) {
            delete config[bucket][agentName];
          }

          // Workers also previously stored model in skill-file frontmatter.
          // For backwards compat, keep writing model there when provided.
          if (isWorker && model !== undefined) {
            const skillPath = path.join(workersDir, `${agentName}.md`);
            let content = fs.readFileSync(skillPath, 'utf-8');
            if (content.startsWith('---')) {
              if (model) {
                content = content.replace(/^(---[\s\S]*?)model:\s*.+$/m, `$1model: ${model}`);
                if (!content.match(/^model:/m)) {
                  content = content.replace(/^---\n/, `---\nmodel: ${model}\n`);
                }
              }
            } else {
              content = `---\n${model ? `model: ${model}\n` : ''}---\n${content}`;
            }
            fs.writeFileSync(skillPath, content);
          }
          fs.writeFileSync(runner.configPath, yaml.dump(config, { lineWidth: -1 }));
          
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ success: true }));
        } catch (e) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: e.message }));
        }
      });
      return;
    }

    // GET /api/projects/:id/config
    if (req.method === 'GET' && subPath === 'config') {
      const raw = fs.existsSync(runner.configPath) ? fs.readFileSync(runner.configPath, 'utf-8') : '';
      const config = runner.loadConfig();
      const projectToken = config.setupToken;
      const hasProjectToken = !!projectToken;
      const safeConfig = { ...config };
      delete safeConfig.setupToken;
      // Include key pool and selection info
      const keyPool = getKeyPoolSafe();
      const keySelection = config.keySelection || null;

      // Determine effective provider from key selection
      let detectedKey = null;
      if (keySelection?.keyId) {
        detectedKey = keyPool.keys.find(k => k.id === keySelection.keyId) || null;
      }
      if (!detectedKey) {
        detectedKey = keyPool.keys.find(k => k.enabled) || null;
      }
      const detectedProvider = detectedKey?.provider || 'anthropic';
      const detectedTiers = detectedProvider === 'custom' && detectedKey?.customConfig
        ? buildCustomTierMap(detectedKey.customConfig)
        : (MODEL_TIERS[detectedProvider] || {});

      // Build available models list per provider from pi-ai
      // Only show recent/relevant models, not the full historical catalog
      const EFFORT_LEVELS = ['medium', 'high', 'xhigh'];
      const ALLOWED_MODELS = {
        anthropic: /^claude-(opus|sonnet)-4-6$|^claude-haiku-4-5-/,
        openai: /^(gpt-5\.[34]|o[34])/,
        'openai-codex': /^(gpt-5\.[34])/,
        google: /^gemini-[23]/,
        minimax: /MiniMax/,
      };
      const availableModels = {};
      for (const provider of Object.keys(MODEL_TIERS)) {
        try {
          const models = getPiModels(provider);
          const filter = ALLOWED_MODELS[provider];
          const entries = [];
          for (const m of models) {
            if (filter && !filter.test(m.id)) continue;
            if (m.id.includes('latest')) continue; // skip aliases, use exact versions
            if (m.reasoning) {
              for (const effort of EFFORT_LEVELS) {
                entries.push({ id: `${m.id}@${effort}`, name: `${m.name} (${effort})` });
              }
            } else {
              entries.push({ id: m.id, name: m.name });
            }
          }
          availableModels[provider] = entries;
        } catch { availableModels[provider] = []; }
      }
      availableModels.custom = [];

      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        config: safeConfig, raw, hasProjectToken,
        projectTokenPreview: projectToken ? maskToken(projectToken) : null,
        provider: detectedProvider,
        tiers: detectedTiers,
        allTiers: detectedProvider === 'custom' ? { ...MODEL_TIERS, custom: detectedTiers } : MODEL_TIERS,
        availableModels,
        keyPool,
        keySelection,
        allowCustomProvider: ALLOW_CUSTOM_PROVIDER,
      }));
      return;
    }

    // POST /api/projects/:id/token — set per-project key selection or direct token
    if (req.method === 'POST' && subPath === 'token') {
      if (!requireWrite(req, res)) return;
      let body = '';
      req.on('data', c => body += c);
      req.on('end', () => {
        try {
          const { keyId, fallback, token, provider: explicitProvider, customConfig } = JSON.parse(body);
          const configPath = runner.configPath;
          const existing = fs.existsSync(configPath) ? yaml.load(fs.readFileSync(configPath, 'utf-8')) || {} : {};

          if (keyId !== undefined) {
            // New key pool selection mode
            delete existing.setupToken;
            delete existing.setupTokenProvider;
            if (keyId) {
              existing.keySelection = { keyId, fallback: fallback !== false };
            } else {
              // Clear selection (use global default)
              delete existing.keySelection;
            }
          } else {
            // Legacy mode: raw token
            if (token) {
              existing.setupToken = token;
              if (explicitProvider) existing.setupTokenProvider = explicitProvider;
              // Also add to global pool
              const entry = addKey({ label: `${explicitProvider || 'API'} (from ${projectId})`, token, provider: explicitProvider, customConfig });
              existing.keySelection = { keyId: entry.id, fallback: true };
            } else {
              delete existing.setupToken;
              delete existing.setupTokenProvider;
              delete existing.keySelection;
            }
          }

          fs.writeFileSync(configPath, yaml.dump(existing));
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            success: true,
            hasProjectToken: !!(existing.setupToken || existing.keySelection?.keyId),
            keySelection: existing.keySelection || null,
          }));
        } catch (e) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: e.message }));
        }
      });
      return;
    }

    // POST /api/projects/:id/config
    if (req.method === 'POST' && subPath === 'config') {
      if (!requireWrite(req, res)) return;
      let body = '';
      req.on('data', c => body += c);
      req.on('end', () => {
        try {
          const { content } = JSON.parse(body);
          runner.saveConfig(content);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ success: true }));
        } catch (e) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: e.message }));
        }
      });
      return;
    }

    // POST /api/projects/:id/models — update project-level model overrides
    if (req.method === 'POST' && subPath === 'models') {
      if (!requireWrite(req, res)) return;
      let body = '';
      req.on('data', c => body += c);
      req.on('end', () => {
        try {
          const { models } = JSON.parse(body);
          // Read existing config, merge models, save
          const config = runner.loadConfig();
          if (models && (models.high || models.mid || models.low)) {
            config.models = {};
            if (models.high) config.models.high = models.high;
            if (models.mid) config.models.mid = models.mid;
            if (models.low) config.models.low = models.low;
          } else {
            delete config.models;
          }
          const newYaml = yaml.dump(config, { lineWidth: -1 });
          runner.saveConfig(newYaml);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ success: true, models: config.models || null }));
        } catch (e) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: e.message }));
        }
      });
      return;
    }

    // POST /api/projects/:id/hpc — update project shared allocation config
    if (req.method === 'POST' && subPath === 'hpc') {
      if (!requireWrite(req, res)) return;
      let body = '';
      req.on('data', c => body += c);
      req.on('end', () => {
        try {
          const { shared_job_id, shared_env } = JSON.parse(body);
          const cfgPath = runner.configPath;
          const existing = fs.existsSync(cfgPath) ? yaml.load(fs.readFileSync(cfgPath, 'utf-8')) || {} : {};
          if (shared_job_id || shared_env) {
            existing.hpc = {};
            if (shared_job_id) existing.hpc.shared_job_id = String(shared_job_id).trim();
            if (shared_env && typeof shared_env === 'object' && !Array.isArray(shared_env)) {
              existing.hpc.shared_env = shared_env;
            }
          } else {
            delete existing.hpc;
          }
          fs.writeFileSync(cfgPath, yaml.dump(existing, { lineWidth: -1 }));
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ success: true, hpc: existing.hpc || null }));
        } catch (e) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: e.message }));
        }
      });
      return;
    }

    // GET /api/projects/:id/comments
    if (req.method === 'GET' && subPath === 'comments') {
      const author = url.searchParams.get('author');
      const page = parseInt(url.searchParams.get('page')) || 1;
      const perPage = parseInt(url.searchParams.get('per_page')) || 20;
      const result = await runner.getComments(author, page, perPage);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(result));
      return;
    }

    // GET /api/projects/:id/prs
    if (req.method === 'GET' && subPath === 'prs') {
      const status = ['open', 'merged', 'closed', 'all'].includes(url.searchParams.get('status'))
        ? url.searchParams.get('status')
        : 'open';
      const prs = await runner.getPRs(status);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ prs }));
      return;
    }

    // GET /api/projects/:id/prs/:prId — single PR
    const prDetailMatch = req.method === 'GET' && subPath.match(/^prs\/(\d+)$/);
    if (prDetailMatch) {
      try {
        const prId = parseInt(prDetailMatch[1], 10);
        const pr = await runner.getPR(prId);
        if (!pr) {
          res.writeHead(404, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'PR not found' }));
        } else {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ pr }));
        }
      } catch (e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
      return;
    }

    // GET /api/projects/:id/reports — agent cycle reports (posted by orchestrator)
    if (req.method === 'GET' && subPath === 'reports') {
      try {
        const db = runner.getDb();
        db.exec(`CREATE TABLE IF NOT EXISTS reports (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          cycle INTEGER NOT NULL,
          agent TEXT NOT NULL,
          body TEXT NOT NULL,
          created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        )`);
        // Migrate: add summary/visibility columns if missing
        try { db.exec('ALTER TABLE reports ADD COLUMN summary TEXT'); } catch {}
        try { db.exec('ALTER TABLE reports ADD COLUMN visibility_mode TEXT'); } catch {}
        try { db.exec('ALTER TABLE reports ADD COLUMN visibility_issues TEXT'); } catch {}
        const agent = url.searchParams.get('agent');
        const page = parseInt(url.searchParams.get('page')) || 1;
        const perPage = parseInt(url.searchParams.get('per_page')) || 20;
        let query = 'SELECT * FROM reports';
        const params = [];
        if (agent) { query += ' WHERE agent = ?'; params.push(agent); }
        query += ' ORDER BY id DESC LIMIT ? OFFSET ?';
        params.push(perPage, (page - 1) * perPage);
        const reports = db.prepare(query).all(...params);
        const total = db.prepare(`SELECT COUNT(*) as count FROM reports${agent ? ' WHERE agent = ?' : ''}`).get(...(agent ? [agent] : [])).count;
        db.close();
        // Enrich reports with key labels from key pool
        const keyPool = getKeyPoolSafe();
        const keyMap = new Map((keyPool.keys || []).map(k => [k.id, k.label]));
        for (const r of reports) {
          if (r.key_id) r.key_label = keyMap.get(r.key_id) || null;
          try { r.visibility_issues = r.visibility_issues ? JSON.parse(r.visibility_issues) : []; } catch { r.visibility_issues = []; }
          if (!r.visibility_mode) r.visibility_mode = 'full';
        }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ reports, total, page, perPage }));
      } catch (e) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ reports: [], total: 0, page: 1, perPage: 20 }));
      }
      return;
    }

    // POST /api/projects/:id/reports/:reportId/summarize — lazy summarization
    const summarizeMatch = req.method === 'POST' && subPath.match(/^reports\/(\d+)\/summarize$/);
    if (summarizeMatch) {
      const reportId = parseInt(summarizeMatch[1], 10);
      let keyResult = null;
      try {
        const db = runner.getDb();
        try { db.exec('ALTER TABLE reports ADD COLUMN summary TEXT'); } catch {}
        const report = db.prepare('SELECT * FROM reports WHERE id = ?').get(reportId);
        if (!report) { db.close(); res.writeHead(404); res.end('Not found'); return; }
        if (report.summary) { db.close(); res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ summary: report.summary })); return; }

        // Use key pool for token resolution
        const config = runner.loadConfig() || {};
        const oauthGetter = async (authFile, provider) => {
          return getOAuthAccessToken(provider);
        };
        const poolSafe = getKeyPoolSafe();
        const firstKey = poolSafe.keys.find(k => k.enabled);
        const providerHintForSummary = config.setupTokenProvider || firstKey?.provider || 'anthropic';

        keyResult = await resolveKeyForProject(config, providerHintForSummary, oauthGetter);
        const token = keyResult?.token || config.setupToken || null;

        if (!token) { db.close(); res.writeHead(500, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ error: 'No API token configured' })); return; }

        // Use the resolved key's actual provider for model resolution (not the hint)
        const actualProvider = keyResult?.provider || providerHintForSummary;
        const runtimeSelection = getProviderRuntimeSelection({
          provider: actualProvider,
          modelTier: 'xlow',
          keyResult,
          projectModels: null,
        });
        const model = runtimeSelection.selectedModel;

        log(`Summarize report ${reportId}: provider=${actualProvider}, model=${model}`, runner.id);

        // Strip meta blocks from body for cleaner summarization
        const cleanBody = report.body
          .replace(/^>\s*⏱.*$/m, '')
          .replace(/<!--[\s\S]*?-->/g, '')
          .replace(/\n{3,}/g, '\n\n')
          .trim()
          .slice(0, 4000);

        const prompt = `Summarize this agent report in 5-8 words. Return ONLY the summary, nothing else.\n\n${cleanBody}`;

        // Use pi-ai adapter for summarization (same auth logic as agent calls)
        const { piModel } = resolveModel(model, actualProvider);
        const isOAuth = keyResult?.type === 'oauth';
        const summaryResponse = await callModel(
          piModel,
          'You are a helpful assistant. Return ONLY the summary, nothing else.',
          [buildUserMessage(prompt)],
          [], // no tools
          { token, isOAuth, provider: actualProvider, customConfig: runtimeSelection.customConfig || null },
        );
        const summary = summaryResponse.content?.trim() || null;

        if (summary) {
          db.prepare('UPDATE reports SET summary = ? WHERE id = ?').run(summary, reportId);
        }
        db.close();
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ summary }));
      } catch (e) {
        log(`Summarize error: ${e.message}`, runner.id);
        // Mark key as rate-limited if the error indicates a usage/rate limit
        if (keyResult?.keyId && /rate.limit|usage.limit|quota|429/i.test(e.message)) {
          const cooldownMs = parseSummarizeCooldown(e.message);
          markRateLimited(keyResult.keyId, cooldownMs);
          log(`Summarize: marked key ${keyResult.keyId} rate-limited for ${Math.ceil(cooldownMs / 60_000)}m`, runner.id);
        }
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
      return;
    }

    // GET /api/projects/:id/issues/:issueId — single issue + comments
    const issueDetailMatch = req.method === 'GET' && subPath.match(/^issues\/(\d+)$/);
    if (issueDetailMatch) {
      try {
        const issueId = parseInt(issueDetailMatch[1], 10);
        const db = runner.getDb();
        const issue = db.prepare('SELECT * FROM issues WHERE id = ?').get(issueId);
        const comments = issue ? db.prepare('SELECT * FROM comments WHERE issue_id = ? ORDER BY created_at ASC').all(issueId) : [];
        db.close();
        if (!issue) {
          res.writeHead(404, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Issue not found' }));
        } else {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ issue, comments }));
        }
      } catch (e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
      return;
    }

    // POST /api/projects/:id/issues/:issueId/comments — add comment
    const commentPostMatch = req.method === 'POST' && subPath.match(/^issues\/(\d+)\/comments$/);
    if (commentPostMatch) {
      if (!requireWrite(req, res)) return;
      let body = '';
      req.on('data', d => body += d);
      req.on('end', () => {
        try {
          const issueId = parseInt(commentPostMatch[1], 10);
          const { author, body: commentBody } = JSON.parse(body);
          if (!commentBody?.trim()) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Comment body required' }));
            return;
          }
          const db = runner.getDb();
          const now = new Date().toISOString();
          const result = db.prepare('INSERT INTO comments (issue_id, author, body, created_at) VALUES (?, ?, ?, ?)').run(issueId, author || 'human', commentBody.trim(), now);
          db.prepare('UPDATE issues SET updated_at = ? WHERE id = ?').run(now, issueId);
          db.close();
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ id: result.lastInsertRowid }));
        } catch (e) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: e.message }));
        }
      });
      return;
    }

    // PATCH /api/projects/:id/issues/:issueId — update issue status
    const issuePatchMatch = req.method === 'PATCH' && subPath.match(/^issues\/(\d+)$/);
    if (issuePatchMatch) {
      if (!requireWrite(req, res)) return;
      let body = '';
      req.on('data', d => body += d);
      req.on('end', () => {
        let db = null;
        try {
          const issueId = parseInt(issuePatchMatch[1], 10);
          const { status, actor } = JSON.parse(body);
          if (!['open', 'closed'].includes(status)) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Status must be "open" or "closed"' }));
            return;
          }
          db = runner.getDb();
          const issue = db.prepare('SELECT id, creator, status FROM issues WHERE id = ?').get(issueId);
          if (!issue) {
            res.writeHead(404, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Issue not found' }));
            return;
          }
          const actingAs = actor || 'human';
          if (status === 'closed' && issue.status !== 'closed') {
            const { allowed, special } = runner._resolveAllowedIssueClosers(db, issue.creator);
            if (!allowed.has(actingAs)) {
              const error = special === 'chat-human'
                ? `Issue #${issueId} was opened by ${issue.creator} and can only be closed by chat or human`
                : `Issue #${issueId} was opened by ${issue.creator} and can only be closed by the same creator`;
              res.writeHead(403, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ error }));
              return;
            }
          }
          const now = new Date().toISOString();
          const closedAt = status === 'closed' ? now : null;
          const closedBy = status === 'closed' ? actingAs : null;
          db.prepare('UPDATE issues SET status = ?, updated_at = ?, updated_by = ?, closed_at = ?, closed_by = ? WHERE id = ?').run(status, now, actingAs, closedAt, closedBy, issueId);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ success: true }));
        } catch (e) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: e.message }));
        } finally {
          try { db?.close(); } catch {}
        }
      });
      return;
    }

    // GET /api/projects/:id/issues
    if (req.method === 'GET' && subPath === 'issues') {
      const issues = await runner.getIssues();
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ issues }));
      return;
    }

    // POST /api/projects/:id/issues/create
    if (req.method === 'POST' && subPath === 'issues/create') {
      if (!requireWrite(req, res)) return;
      let body = '';
      req.on('data', c => body += c);
      req.on('end', async () => {
        try {
          const { title, body: issueBody, creator, assignee, text } = JSON.parse(body);
          // Support both structured and text input
          if (title) {
            const result = await runner.createIssue(title, issueBody, creator, assignee);
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(result));
          } else if (text) {
            const lines = text.trim().split('\n');
            const result = await runner.createIssue(lines[0], lines.slice(1).join('\n'), 'human');
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(result));
          } else {
            throw new Error('Missing title or text');
          }
        } catch (e) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: e.message }));
        }
      });
      return;
    }

    // GET /api/projects/:id/repo
    if (req.method === 'GET' && subPath === 'repo') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ 
        repo: runner.repo, 
        url: runner.repo ? `https://github.com/${runner.repo}` : null 
      }));
      return;
    }

    // GET /api/projects/:id/download - zip and download project data
    if (req.method === 'GET' && subPath === 'download') {
      try {
        const projectDataDir = runner.projectDir;
        if (!fs.existsSync(projectDataDir)) {
          res.writeHead(404, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Project data not found' }));
          return;
        }
        const filename = `${runner.id.replace(/\//g, '-')}-project.zip`;
        res.writeHead(200, {
          'Content-Type': 'application/zip',
          'Content-Disposition': `attachment; filename="${filename}"`,
        });
        const zip = spawn('zip', ['-r', '-q', '-', '.'], { cwd: projectDataDir, stdio: ['ignore', 'pipe', 'ignore'] });
        zip.stdout.pipe(res);
        zip.on('error', () => {
          // Fallback to tar if zip not available
          const tar = spawn('tar', ['-czf', '-', '-C', projectDataDir, '.'], { stdio: ['ignore', 'pipe', 'ignore'] });
          res.writeHead(200, {
            'Content-Type': 'application/gzip',
            'Content-Disposition': `attachment; filename="${filename.replace('.zip', '.tar.gz')}"`,
          });
          tar.stdout.pipe(res);
          tar.on('error', () => { res.end(); });
        });
      } catch (e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
      return;
    }

    // GET /api/projects/:id/bootstrap - preview what bootstrap will do
    if (req.method === 'GET' && subPath === 'bootstrap') {
      if (!requireWrite(req, res)) return;
      try {
        const result = runner.bootstrapPreview();
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
      } catch (e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
      return;
    }

    // POST /api/projects/:id/bootstrap - execute bootstrap
    if (req.method === 'POST' && subPath === 'bootstrap') {
      if (!requireWrite(req, res)) return;
      let body = '';
      req.on('data', chunk => body += chunk);
      req.on('end', () => {
        try {
          const options = body ? JSON.parse(body) : {};
          fs.mkdirSync(runner.chatsDir, { recursive: true });
          const result = runner.bootstrap(options);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ success: true, ...result }));
        } catch (e) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: e.message }));
        }
      });
      return;
    }

    // POST /api/projects/:id/doctor - run AI Doctor agent
    if (req.method === 'POST' && subPath === 'doctor') {
      if (!requireWrite(req, res)) return;
      try {
        if (!runner.isPaused || runner.currentAgent) {
          res.writeHead(409, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Doctor is only available when the project is fully paused.' }));
          return;
        }
        const result = await runner.runDoctor();
        if (!result.success) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Doctor agent failed', ...result }));
          return;
        }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, ...result }));
      } catch (e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
      return;
    }

    // --- Chat endpoints ---

    // GET /api/projects/:id/chats — list chat sessions
    if (req.method === 'GET' && subPath === 'chats') {
      try {
        const sessions = chatListSessions(runner.chatsDir);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ sessions }));
      } catch (e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
      return;
    }

    // POST /api/projects/:id/chats — create new session
    if (req.method === 'POST' && subPath === 'chats') {
      if (!requireWrite(req, res)) return;
      let body = '';
      req.on('data', chunk => body += chunk);
      req.on('end', () => {
        try {
          const data = body ? JSON.parse(body) : {};
          const session = chatCreateSession(runner.chatsDir, data.title, {
            selectedKeyId: typeof data.selectedKeyId === 'string' ? data.selectedKeyId : null,
            selectedModel: typeof data.selectedModel === 'string' ? data.selectedModel : null,
          });
          res.writeHead(201, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ session }));
        } catch (e) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: e.message }));
        }
      });
      return;
    }

    // GET /api/projects/:id/chats/:chatId — get session with messages
    const chatDetailMatch = req.method === 'GET' && subPath.match(/^chats\/(\d+)$/);
    if (chatDetailMatch) {
      try {
        const session = chatGetSession(runner.chatsDir, parseInt(chatDetailMatch[1]));
        if (!session) {
          res.writeHead(404, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Session not found' }));
        } else {
          const chatId = parseInt(chatDetailMatch[1]);
          const activeStream = getActiveStream(chatId);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            session,
            streaming: !!activeStream,
            streamingContent: activeStream ? { text: activeStream.text, toolCalls: activeStream.toolCalls } : null,
          }));
        }
      } catch (e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
      return;
    }

    // PATCH /api/projects/:id/chats/:chatId/preferences — persist session key/model selection
    const chatPreferencesMatch = req.method === 'PATCH' && subPath.match(/^chats\/(\d+)\/preferences$/);
    if (chatPreferencesMatch) {
      if (!requireWrite(req, res)) return;
      let body = '';
      req.on('data', chunk => body += chunk);
      req.on('end', () => {
        try {
          const data = body ? JSON.parse(body) : {};
          const chatId = parseInt(chatPreferencesMatch[1]);
          const session = chatUpdateSessionPreferences(runner.chatsDir, chatId, {
            selectedKeyId: typeof data.selectedKeyId === 'string' && data.selectedKeyId !== 'auto' ? data.selectedKeyId : null,
            selectedModel: typeof data.selectedModel === 'string' && data.selectedModel !== 'auto' ? data.selectedModel : null,
          });
          if (!session) {
            res.writeHead(404, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Session not found' }));
            return;
          }
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ session }));
        } catch (e) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: e.message }));
        }
      });
      return;
    }

    // GET /api/projects/:id/chats/:chatId/stream — reconnect to active SSE stream
    const chatStreamMatch = req.method === 'GET' && subPath.match(/^chats\/(\d+)\/stream$/);
    if (chatStreamMatch) {
      const chatId = parseInt(chatStreamMatch[1]);
      const activeStream = getActiveStream(chatId);
      if (!activeStream) {
        res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' });
        res.write(`data: ${JSON.stringify({ type: 'done' })}\n\n`);
        res.end();
        return;
      }
      res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' });
      activeStream.clients.add(res);
      res.on('close', () => { activeStream.clients.delete(res); });
      return;
    }

    // DELETE /api/projects/:id/chats/:chatId — delete session
    const chatDeleteMatch = req.method === 'DELETE' && subPath.match(/^chats\/(\d+)$/);
    if (chatDeleteMatch) {
      if (!requireWrite(req, res)) return;
      try {
        chatDeleteSession(runner.chatsDir, parseInt(chatDeleteMatch[1]));
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true }));
      } catch (e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
      return;
    }

    // POST /api/projects/:id/chats/upload — upload file for chat
    if (req.method === 'POST' && subPath === 'chats/upload') {
      if (!requireWrite(req, res)) return;
      const chunks = [];
      req.on('data', chunk => chunks.push(chunk));
      req.on('end', () => {
        try {
          const body = Buffer.concat(chunks);
          // Parse multipart boundary
          const contentType = req.headers['content-type'] || '';
          const boundaryMatch = contentType.match(/boundary=(.+)/);
          if (!boundaryMatch) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Missing multipart boundary' }));
            return;
          }
          const boundary = '--' + boundaryMatch[1];
          const parts = body.toString('binary').split(boundary).filter(p => p.trim() && p.trim() !== '--');
          
          let filename = null;
          let fileData = null;
          let mimeType = null;

          for (const part of parts) {
            const headerEnd = part.indexOf('\r\n\r\n');
            if (headerEnd === -1) continue;
            const headers = part.slice(0, headerEnd);
            const filenameMatch = headers.match(/filename="([^"]+)"/);
            const ctMatch = headers.match(/Content-Type:\s*(.+)/i);
            if (filenameMatch) {
              filename = filenameMatch[1];
              mimeType = ctMatch ? ctMatch[1].trim() : 'application/octet-stream';
              // Extract binary data (skip headers + \r\n\r\n, remove trailing \r\n)
              const dataStart = headerEnd + 4;
              const dataEnd = part.endsWith('\r\n') ? part.length - 2 : part.length;
              fileData = Buffer.from(part.slice(dataStart, dataEnd), 'binary');
            }
          }

          if (!filename || !fileData) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'No file in upload' }));
            return;
          }

          // Save to uploads/
          const uploadsDir = runner.uploadsDir;
          if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir, { recursive: true });
          const ext = path.extname(filename) || '.bin';
          const safeName = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}${ext}`;
          fs.writeFileSync(path.join(uploadsDir, safeName), fileData);

          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            filename: safeName,
            originalName: filename,
            mimeType,
            size: fileData.length,
            url: `/api/projects/${projectId}/uploads/${safeName}`,
          }));
        } catch (e) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: e.message }));
        }
      });
      return;
    }

    // GET /api/projects/:id/uploads/:filename — serve uploaded files
    const uploadMatch = req.method === 'GET' && subPath.match(/^uploads\/(.+)$/);
    if (uploadMatch) {
      const filename = uploadMatch[1];
      const filePath = path.join(runner.uploadsDir, filename);
      if (!fs.existsSync(filePath)) {
        res.writeHead(404);
        res.end('Not found');
        return;
      }
      const ext = path.extname(filename).toLowerCase();
      const mimeTypes = { '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.gif': 'image/gif', '.webp': 'image/webp', '.pdf': 'application/pdf', '.txt': 'text/plain', '.csv': 'text/csv' };
      res.writeHead(200, { 'Content-Type': mimeTypes[ext] || 'application/octet-stream' });
      fs.createReadStream(filePath).pipe(res);
      return;
    }

    // POST /api/projects/:id/chats/:chatId/message — send message (SSE streaming)
    const chatMessageMatch = req.method === 'POST' && subPath.match(/^chats\/(\d+)\/message$/);
    if (chatMessageMatch) {
      if (!requireWrite(req, res)) return;
      const chatId = parseInt(chatMessageMatch[1]);
      let body = '';
      req.on('data', chunk => body += chunk);
      req.on('end', async () => {
        const respondChatError = (statusCode, payload) => {
          try {
            chatSaveMessage(runner.chatsDir, chatId, 'assistant', formatStoredChatErrorMessage({
              error: payload.error,
              statusCode,
              source: payload.source || 'server',
              cooldownMs: payload.cooldownMs || 0,
            }), null, { success: false });
          } catch {}
          res.writeHead(statusCode, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(payload));
        };
        try {
          const data = JSON.parse(body);
          if (!data.message?.trim()) {
            respondChatError(400, { error: 'Message is required', source: 'validation' });
            return;
          }

          // Save user message once for this send attempt
          const imageUrls = (data.images || []).map(img => `/api/projects/${projectId}/uploads/${img.filename}`);
          chatSaveMessage(runner.chatsDir, chatId, 'user', data.message.trim(), imageUrls.length > 0 ? imageUrls : null);

          // Resolve API key / model selection
          const config = runner.loadConfig();
          const oauthTokenGetter = async (authFile, provider) => {
            return getOAuthAccessToken(provider, runner.id);
          };
          const explicitKeyId = typeof data.keyId === 'string' && data.keyId.trim() ? data.keyId.trim() : null;
          const { model: explicitModel, reasoningEffort: explicitReasoningEffort } = parseExplicitModelSelection(data.model);
          if (explicitModel && !explicitKeyId) {
            respondChatError(400, { error: 'Select a key before selecting a specific model.', source: 'validation' });
            return;
          }

          const selectedKeySafe = explicitKeyId
            ? (getKeyPoolSafe().keys || []).find(key => key.id === explicitKeyId) || null
            : null;

          chatUpdateSessionPreferences(runner.chatsDir, chatId, {
            selectedKeyId: explicitKeyId,
            selectedModel: explicitModel,
          });
          if (explicitKeyId && !selectedKeySafe) {
            respondChatError(404, { error: 'Selected API key was not found.', errorType: 'key_not_found', source: 'local_selection' });
            return;
          }
          if (selectedKeySafe && !selectedKeySafe.enabled) {
            respondChatError(400, { error: 'Selected API key is disabled.', errorType: 'key_disabled', source: 'local_selection' });
            return;
          }
          if (selectedKeySafe?.rateLimited) {
            respondChatError(429, {
              error: `Selected API key is currently rate limited${selectedKeySafe.cooldownMs ? ` for about ${Math.ceil(selectedKeySafe.cooldownMs / 60_000)}m` : ''}.`,
              errorType: 'key_rate_limited',
              source: 'local_cooldown',
              cooldownMs: selectedKeySafe.cooldownMs || 0,
            });
            return;
          }

          const keyConfig = explicitKeyId
            ? { ...config, keySelection: { keyId: explicitKeyId, fallback: false } }
            : config;
          const keyResult = await resolveKeyForProject(keyConfig, null, oauthTokenGetter);
          if (!keyResult?.token) {
            respondChatError(400, { error: explicitKeyId ? 'Selected API key is unavailable.' : 'No API key configured. Add one in Settings > Credentials.', source: explicitKeyId ? 'local_selection' : 'configuration' });
            return;
          }
          if (explicitKeyId && keyResult.keyId !== explicitKeyId) {
            respondChatError(400, { error: 'Selected API key is unavailable.', errorType: 'key_unavailable', source: 'local_selection' });
            return;
          }

          const modelTier = data.modelTier || 'high';
          const providerHint = keyResult.provider || detectProviderFromToken(keyResult.token);
          const runtimeSelection = explicitModel
            ? {
                selectedModel: explicitModel,
                reasoningEffort: explicitReasoningEffort,
                customConfig: keyResult.customConfig || null,
              }
            : getProviderRuntimeSelection({
                provider: providerHint,
                modelTier,
                keyResult,
                projectModels: config.models,
              });

          // SSE headers
          res.writeHead(200, {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
          });

          const chatOpts = {
            agentDir: runner.chatsDir,
            tbcDbPath: runner.projectDbPath,
            uploadsDir: runner.uploadsDir,
            projectPath: runner.path,
            chatId,
            userMessage: data.message.trim(),
            images: data.images || [],
            model: runtimeSelection.selectedModel,
            token: keyResult.token,
            provider: providerHint,
            customConfig: runtimeSelection.customConfig || null,
            res,
            reasoningEffort: runtimeSelection.reasoningEffort || null,
          };

          try {
            await streamChatMessage(chatOpts);
          } catch (chatErr) {
            // Check if rate-limited — try fallback key only in auto-key mode
            const isRateLimit = /rate.limit|usage.limit|quota|429/i.test(chatErr.message);
            if (isRateLimit && keyResult.keyId) {
              const cooldownMs = parseSummarizeCooldown(chatErr.message);
              markRateLimited(keyResult.keyId, cooldownMs);
              log(`Chat: marked key ${keyResult.keyId} rate-limited for ${Math.ceil(cooldownMs / 60_000)}m`, runner.id);

              if (explicitKeyId) {
                chatErr.errorType = 'provider_rate_limited';
                chatErr.source = 'provider_429';
                chatErr.cooldownMs = cooldownMs;
                chatErr.statusCode = 429;
                throw chatErr;
              }

              // Try fallback key
              const fallbackKey = await resolveKeyForProject(config, null, oauthTokenGetter);
              if (fallbackKey?.token && fallbackKey.token !== keyResult.token) {
                const fbProvider = fallbackKey.provider || detectProviderFromToken(fallbackKey.token);
                const fallbackSelection = getProviderRuntimeSelection({
                  provider: fbProvider,
                  modelTier,
                  keyResult: fallbackKey,
                  projectModels: null,
                });
                log(`Chat: falling back to key ${fallbackKey.keyId} (${fbProvider}), model → ${fallbackSelection.selectedModel}`, runner.id);
                chatOpts.token = fallbackKey.token;
                chatOpts.provider = fbProvider;
                chatOpts.model = fallbackSelection.selectedModel;
                chatOpts.reasoningEffort = fallbackSelection.reasoningEffort || null;
                chatOpts.customConfig = fallbackSelection.customConfig || null;
                await streamChatMessage(chatOpts);
              } else {
                throw chatErr; // no fallback available
              }
            } else {
              throw chatErr;
            }
          }

          res.end();
        } catch (e) {
          const errorPayload = {
            error: e.message,
            errorType: e.errorType || null,
            source: e.source || 'server',
            statusCode: e.statusCode || 500,
            cooldownMs: e.cooldownMs || 0,
          };
          try {
            chatSaveMessage(runner.chatsDir, chatId, 'assistant', formatStoredChatErrorMessage(errorPayload), null, { success: false });
          } catch {}
          if (!res.headersSent) {
            res.writeHead(errorPayload.statusCode, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(errorPayload));
          } else {
            res.write(`data: ${JSON.stringify({
              type: 'error',
              content: errorPayload.error,
              errorType: errorPayload.errorType,
              source: errorPayload.source,
              statusCode: errorPayload.statusCode,
              cooldownMs: errorPayload.cooldownMs,
            })}\n\n`);
            res.end();
          }
        }
      });
      return;
    }

    // POST /api/projects/:id/:action (pause, resume, skip, start, stop)
    // POST /api/projects/:id/archive or /unarchive
    if (req.method === 'POST' && (subPath === 'archive' || subPath === 'unarchive')) {
      if (!requireWrite(req, res)) return;
      const archive = subPath === 'archive';
      runner.archived = archive;
      // Update projects.yaml
      try {
        const configPath = path.join(TBC_HOME, 'projects.yaml');
        const raw = fs.readFileSync(configPath, 'utf-8');
        const config = yaml.load(raw) || {};
        if (config.projects && config.projects[projectId]) {
          if (archive) {
            config.projects[projectId].archived = true;
          } else {
            delete config.projects[projectId].archived;
          }
          fs.writeFileSync(configPath, yaml.dump(config));
        }
      } catch (e) {
        log(`Failed to update projects.yaml for archive: ${e.message}`);
      }
      if (archive && runner.running) {
        runner.pause('Archived');
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ success: true, archived: archive }));
      return;
    }

    if (req.method === 'POST' && ['pause', 'resume', 'skip', 'start', 'stop', 'kill-run', 'kill-cycle', 'kill-epoch'].includes(subPath)) {
      if (!requireWrite(req, res)) return;
      switch (subPath) {
        case 'pause': runner.pause(); break;
        case 'resume': runner.resume(); break;
        case 'skip': runner.skip(); break;
        case 'start': runner.start(); break;
        case 'stop': runner.stop(); break;
        case 'kill-run': runner.killRun(); break;
        case 'kill-cycle': runner.killCycle(); break;
        case 'kill-epoch': runner.killEpoch(); break;
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ success: true, action: subPath, projectId }));
      return;
    }
  }

  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'Not found' }));
});

// --- Helpers ---
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// --- Preflight checks ---
function checkPrerequisites() {
  const optional = [];
  let codexAvailable = false;
  let claudeAvailable = false;
  try { execSync('codex --version', { stdio: 'pipe' }); codexAvailable = true; } catch {}
  try { execSync('claude --version', { stdio: 'pipe' }); claudeAvailable = true; } catch {}
  try { execSync('gh --version', { stdio: 'pipe' }); } catch { optional.push('gh (GitHub CLI) — needed only for listing or creating GitHub repositories'); }
  if (!codexAvailable && !claudeAvailable) {
    log('WARNING: Neither codex nor claude CLI is available.');
    log('  - codex (Codex CLI) — required for agentRuntime: codex_cli');
    log('  - claude (Claude Code CLI) — required for agentRuntime: claude_cli');
  } else {
    if (!codexAvailable) log('Note: codex CLI not found; agentRuntime: codex_cli projects will fail.');
    if (!claudeAvailable) log('Note: claude CLI not found; agentRuntime: claude_cli projects will fail.');
  }
  if (optional.length > 0) {
    log('Optional integrations unavailable:');
    optional.forEach(m => log(`  - ${m}`));
  }
}

// --- Main ---
log('HPC Agent starting...');
checkPrerequisites();
syncProjects();
// Migrate existing keys from .env and project configs into key-pool.json
migrateFromEnv(projects);
server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    log(`Port ${PORT} is already in use. Stop the existing server or start this one with TBC_PORT=<port>.`);
    process.exit(1);
  }
  throw err;
});
server.listen(PORT, () => {
  log(`Server listening on http://localhost:${PORT}`);
});

process.on('SIGINT', () => {
  log('Shutting down...');
  for (const runner of projects.values()) {
    runner.stop();
  }
  process.exit(0);
});
process.on('SIGTERM', () => {
  log('Shutting down...');
  for (const runner of projects.values()) {
    runner.stop();
  }
  process.exit(0);
});
