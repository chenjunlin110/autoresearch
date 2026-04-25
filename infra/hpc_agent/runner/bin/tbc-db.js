#!/usr/bin/env node

/**
 * tbc-db — CLI for agent communication via SQLite
 * 
 * Usage: tbc-db <command> [options]
 * 
 * The DB_PATH environment variable must point to the project's database file.
 * The orchestrator sets this automatically when spawning agents.
 */

import Database from 'better-sqlite3';
import path from 'node:path';
import { parseArgs } from 'node:util';
import {
  createTokenRequest,
  ensureResourceOrchestrationTables,
  getAllocationLease,
  getResourceSummary,
  grantTokenRequest,
  listResourceTokens,
  listTokenGrants,
  listTokenRequests,
  releaseAllocationLease,
  releaseTokenGrant,
  upsertAllocationLease,
} from '../src/resource-orchestration.js';

const dbPath = process.env.TBC_DB;
if (!dbPath) {
  console.error('Error: TBC_DB environment variable not set');
  process.exit(1);
}
if (path.basename(dbPath) !== 'project.db') {
  console.error(`Error: TBC_DB must point to canonical project.db, got ${dbPath}`);
  process.exit(1);
}

const db = new Database(dbPath);
db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');

// Ensure schema exists
db.exec(`
  CREATE TABLE IF NOT EXISTS agents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    role TEXT,
    reports_to TEXT,
    model TEXT,
    disabled INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
  );

  CREATE TABLE IF NOT EXISTS issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    body TEXT DEFAULT '',
    status TEXT DEFAULT 'open' CHECK(status IN ('open', 'closed')),
    creator TEXT NOT NULL,
    assignee TEXT,
    labels TEXT DEFAULT '',
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_by TEXT,
    closed_at TEXT,
    closed_by TEXT
  );

  CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_id INTEGER NOT NULL REFERENCES issues(id),
    author TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
  );

  CREATE TABLE IF NOT EXISTS milestones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT NOT NULL,
    cycles_budget INTEGER DEFAULT 20,
    cycles_used INTEGER DEFAULT 0,
    phase TEXT DEFAULT 'implementation',
    status TEXT DEFAULT 'active' CHECK(status IN ('active', 'completed', 'failed')),
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    completed_at TEXT
  );

  CREATE TABLE IF NOT EXISTS tbc_prs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    summary TEXT DEFAULT '',
    base_branch TEXT NOT NULL,
    head_branch TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open', 'merged', 'closed')),
    issue_ids TEXT DEFAULT '[]',
    test_status TEXT DEFAULT 'unknown',
    github_pr_number INTEGER,
    github_pr_url TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
  );

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
ensureResourceOrchestrationTables(db, {
  gpuCount: process.env.TBC_GPU_COUNT,
  mode: process.env.TBC_RESOURCE_MODE,
  tokenPrefix: process.env.TBC_GPU_TOKEN_PREFIX,
  grantRequiresLease: process.env.TBC_GRANT_REQUIRES_LEASE !== 'false',
});

const command = process.argv[2];
try { db.exec('ALTER TABLE issues ADD COLUMN updated_by TEXT'); } catch {}
try { db.exec('ALTER TABLE issues ADD COLUMN closed_by TEXT'); } catch {}
try { db.exec('ALTER TABLE tbc_prs ADD COLUMN actor TEXT'); } catch {}
try { db.exec('ALTER TABLE tbc_prs ADD COLUMN updated_by TEXT'); } catch {}

const args = process.argv.slice(3);

// --- Visibility enforcement ---
const visibility = process.env.TBC_VISIBILITY || 'full';
const focusedIssues = (process.env.TBC_FOCUSED_ISSUES || '').split(',').map(s => s.trim()).filter(Boolean);

if (visibility === 'blind' || visibility === 'focused') {
  const allowedCommands = new Set(['issue-create', 'pr-create']);
  if (command && !allowedCommands.has(command)) {
    const scope = visibility === 'blind' ? 'blind' : 'focused';
    console.error(`Access denied: you are in ${scope} mode and cannot view the issue tracker or PR board.`);
    process.exit(1);
  }
}

const VALID_PR_STATUSES = new Set(['open', 'merged', 'closed']);

function normalizePrStatus(status) {
  if (!status) return 'open';
  if (status === 'merged') return 'merged';
  if (['closed', 'completed', 'superseded'].includes(status)) return 'closed';
  if (['open', 'draft', 'ready_for_review', 'ready_for_ci', 'in_progress'].includes(status)) return 'open';
  throw new Error(`Invalid PR status: ${status}. Allowed: open, merged, closed`);
}

function resolveAllowedIssueCloser(issueCreator) {
  if (issueCreator === 'human' || issueCreator === 'chat') {
    return { allowed: new Set(['human', 'chat']), special: 'chat-human' };
  }

  return { allowed: new Set([issueCreator, 'athena']), special: 'agent-athena' };
}

function jsonOut(data) {
  console.log(JSON.stringify(data, null, 2));
}

function textOut(rows, columns) {
  if (!rows || rows.length === 0) {
    console.log('(no results)');
    return;
  }
  // Simple table output
  for (const row of rows) {
    const parts = columns.map(c => `${c}: ${row[c] ?? ''}`);
    console.log(parts.join(' | '));
  }
}

function parseJsonOption(rawValue, fieldName) {
  if (rawValue === undefined || rawValue === null || rawValue === '') return undefined;
  try {
    return JSON.parse(rawValue);
  } catch (error) {
    console.error(`Invalid JSON for ${fieldName}: ${error.message}`);
    process.exit(1);
  }
}

const commands = {
  // ===== ISSUES =====
  'issue-create'() {
    const { values } = parseArgs({
      args,
      options: {
        actor: { type: 'string' },
        title: { type: 'string', short: 't' },
        body: { type: 'string', short: 'b', default: '' },
        actor: { type: 'string' },
        creator: { type: 'string', short: 'c' },
        assignee: { type: 'string', short: 'a' },
        labels: { type: 'string', short: 'l', default: '' },
      },
      strict: false,
    });
    const actor = values.actor || values.creator;
    if (!values.title || !actor) {
      console.error('Usage: tbc-db issue-create --title "..." --actor name [--body "..."] [--assignee name] [--labels "label1,label2"]');
      process.exit(1);
    }
    const now = new Date().toISOString();
    const stmt = db.prepare('INSERT INTO issues (title, body, creator, assignee, labels, created_at, updated_at, updated_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)');
    const result = stmt.run(values.title, values.body || '', actor, values.assignee || null, values.labels || '', now, now, actor);
    console.log(`Created issue #${result.lastInsertRowid}`);
  },

  'issue-list'() {
    const { values } = parseArgs({
      args,
      options: {
        status: { type: 'string', short: 's', default: 'open' },
        assignee: { type: 'string', short: 'a' },
        actor: { type: 'string' },
        creator: { type: 'string', short: 'c' },
        label: { type: 'string', short: 'l' },
        json: { type: 'boolean', default: false },
      },
      strict: false,
    });
    let query = 'SELECT * FROM issues WHERE 1=1';
    const params = [];
    if (values.status && values.status !== 'all') {
      query += ' AND status = ?'; params.push(values.status);
    }
    if (values.assignee) {
      query += ' AND assignee = ?'; params.push(values.assignee);
    }
    if (values.creator) {
      query += ' AND creator = ?'; params.push(values.creator);
    }
    if (values.label) {
      query += ' AND labels LIKE ?'; params.push(`%${values.label}%`);
    }
    // Focused mode: only show issues in the focused set
    if (visibility === 'focused' && focusedIssues.length > 0) {
      query += ` AND id IN (${focusedIssues.map(() => '?').join(',')})`;
      params.push(...focusedIssues);
    }
    query += ' ORDER BY id DESC';
    const rows = db.prepare(query).all(...params);
    if (values.json) {
      jsonOut(rows);
    } else {
      for (const r of rows) {
        const assignee = r.assignee ? ` → ${r.assignee}` : '';
        const labels = r.labels ? ` [${r.labels}]` : '';
        console.log(`#${r.id} [${r.status}] ${r.title}${assignee}${labels} (by ${r.creator})`);
      }
      if (rows.length === 0) console.log('(no issues)');
    }
  },

  'issue-view'() {
    const id = args[0];
    if (!id) { console.error('Usage: tbc-db issue-view <id>'); process.exit(1); }
    if (visibility === 'focused' && focusedIssues.length > 0 && !focusedIssues.includes(String(id))) {
      console.error(`Access denied: issue #${id} is not in your focused set.`);
      process.exit(1);
    }
    const issue = db.prepare('SELECT * FROM issues WHERE id = ?').get(id);
    if (!issue) { console.error(`Issue #${id} not found`); process.exit(1); }
    console.log(`# Issue #${issue.id}: ${issue.title}`);
    console.log(`Status: ${issue.status} | Creator: ${issue.creator} | Assignee: ${issue.assignee || 'none'}`);
    if (issue.labels) console.log(`Labels: ${issue.labels}`);
    console.log(`Created: ${issue.created_at} | Updated: ${issue.updated_at}`);
    if (issue.body) { console.log(`\n${issue.body}`); }
    // Show comments
    const comments = db.prepare('SELECT * FROM comments WHERE issue_id = ? ORDER BY created_at').all(id);
    if (comments.length > 0) {
      console.log(`\n--- Comments (${comments.length}) ---`);
      for (const c of comments) {
        console.log(`\n[${c.author}] (${c.created_at}):`);
        console.log(c.body);
      }
    }
  },

  'issue-close'() {
    const id = args[0];
    const { values } = parseArgs({
      args: args.slice(1),
      options: {
        actor: { type: 'string' },
        closer: { type: 'string', short: 'c' },
      },
      strict: false,
    });
    const actor = values.actor || values.closer;
    if (!id || !actor) { console.error('Usage: tbc-db issue-close <id> --actor name'); process.exit(1); }
    const issue = db.prepare('SELECT id, creator, status FROM issues WHERE id = ?').get(id);
    if (!issue) { console.error(`Issue #${id} not found`); process.exit(1); }
    if (issue.status === 'closed') { console.log(`Issue #${id} already closed`); return; }
    const closer = actor;
    const { allowed, special } = resolveAllowedIssueCloser(issue.creator);
    if (!allowed.has(closer)) {
      if (special === 'chat-human') {
        console.error(`Blocked: issue #${id} was opened by ${issue.creator} and can only be closed by chat or human`);
      } else {
        console.error(`Blocked: issue #${id} was opened by ${issue.creator} and can only be closed by ${issue.creator} or athena`);
      }
      process.exit(1);
    }
    const now = new Date().toISOString();
    db.prepare("UPDATE issues SET status = 'closed', closed_at = ?, closed_by = ?, updated_at = ?, updated_by = ? WHERE id = ?").run(now, closer, now, closer, id);
    console.log(`Closed issue #${id}`);
  },

  'issue-edit'() {
    const id = args[0];
    if (!id) { console.error('Usage: tbc-db issue-edit <id> --actor name [--title "..."] [--body "..."] [--assignee name] [--labels "..."]'); process.exit(1); }
    const { values } = parseArgs({
      args: args.slice(1),
      options: {
        actor: { type: 'string' },
        editor: { type: 'string', short: 'e' },
        title: { type: 'string', short: 't' },
        body: { type: 'string', short: 'b' },
        assignee: { type: 'string', short: 'a' },
        labels: { type: 'string', short: 'l' },
      },
      strict: false,
    });
    const actor = values.actor || values.editor;
    if (!actor) { console.error('Usage: tbc-db issue-edit <id> --actor name [--title "..."] [--body "..."] [--assignee name] [--labels "..."]'); process.exit(1); }
    if (!values.actor) { console.error('Usage: tbc-db pr-edit <id> --actor name [--title "..."] [--summary "..."] [--base main] [--head branch] [--status ready_for_review] [--issues "1,2"] [--test pass]'); process.exit(1); }
    const sets = [];
    const params = [];
    if (values.title) { sets.push('title = ?'); params.push(values.title); }
    if (values.body !== undefined) { sets.push('body = ?'); params.push(values.body); }
    if (values.assignee !== undefined) { sets.push('assignee = ?'); params.push(values.assignee); }
    if (values.labels !== undefined) { sets.push('labels = ?'); params.push(values.labels); }
    if (sets.length === 0) { console.error('Nothing to update'); process.exit(1); }
    sets.push("updated_at = ?"); params.push(new Date().toISOString());
    sets.push('updated_by = ?'); params.push(actor);
    params.push(id);
    db.prepare(`UPDATE issues SET ${sets.join(', ')} WHERE id = ?`).run(...params);
    console.log(`Updated issue #${id}`);
  },

  // ===== COMMENTS =====
  'comment'() {
    const { values } = parseArgs({
      args,
      options: {
        issue: { type: 'string', short: 'i' },
        actor: { type: 'string' },
        author: { type: 'string', short: 'a' },
        body: { type: 'string', short: 'b' },
      },
      strict: false,
    });
    const actor = values.actor || values.author;
    if (!values.issue || !actor || !values.body) {
      console.error('Usage: tbc-db comment --issue <id> --actor name --body "..."');
      process.exit(1);
    }
    if (visibility === 'focused' && focusedIssues.length > 0 && !focusedIssues.includes(String(values.issue))) {
      console.error(`Access denied: issue #${values.issue} is not in your focused set.`);
      process.exit(1);
    }
    const result = db.prepare('INSERT INTO comments (issue_id, author, body, created_at) VALUES (?, ?, ?, ?)').run(values.issue, actor, values.body, new Date().toISOString());
    db.prepare("UPDATE issues SET updated_at = ?, updated_by = ? WHERE id = ?").run(new Date().toISOString(), actor, values.issue);
    console.log(`Added comment #${result.lastInsertRowid} to issue #${values.issue}`);
  },

  'comments'() {
    const id = args[0];
    if (!id) { console.error('Usage: tbc-db comments <issue_id>'); process.exit(1); }
    if (visibility === 'focused' && focusedIssues.length > 0 && !focusedIssues.includes(String(id))) {
      console.error(`Access denied: issue #${id} is not in your focused set.`);
      process.exit(1);
    }
    const comments = db.prepare('SELECT * FROM comments WHERE issue_id = ? ORDER BY created_at').all(id);
    for (const c of comments) {
      console.log(`[${c.author}] (${c.created_at}):`);
      console.log(c.body);
      console.log('');
    }
    if (comments.length === 0) console.log('(no comments)');
  },

  // ===== TBC PRS =====
  'pr-create'() {
    const { values } = parseArgs({
      args,
      options: {
        title: { type: 'string', short: 't' },
        summary: { type: 'string', short: 's', default: '' },
        base: { type: 'string', default: 'main' },
        head: { type: 'string' },
        status: { type: 'string', default: 'draft' },
        issues: { type: 'string', default: '' },
        test: { type: 'string', default: 'unknown' },
        actor: { type: 'string' },
        github_number: { type: 'string' },
        github_url: { type: 'string' },
      },
      strict: false,
    });
    if (!values.title || !values.head || !values.actor) {
      console.error('Usage: tbc-db pr-create --title "..." --head branch --actor name [--summary "..."] [--base main] [--status draft] [--issues "1,2"] [--test unknown]');
      process.exit(1);
    }
    if ((values.base || 'main') === values.head) {
      console.error('Error: base and head branches must differ for a TBC PR record.');
      process.exit(1);
    }
    const now = new Date().toISOString();
    const normalizedStatus = normalizePrStatus(values.status);
    const issueIds = values.issues
      ? JSON.stringify(values.issues.split(',').map(s => s.trim()).filter(Boolean).map(Number).filter(Number.isFinite))
      : '[]';
    const result = db.prepare(`INSERT INTO tbc_prs
      (title, summary, base_branch, head_branch, status, issue_ids, test_status, actor, updated_by, github_pr_number, github_pr_url, created_at, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
      .run(
        values.title,
        values.summary || '',
        values.base || 'main',
        values.head,
        normalizedStatus,
        issueIds,
        values.test || 'unknown',
        values.actor,
        values.actor,
        values.github_number ? Number(values.github_number) : null,
        values.github_url || null,
        now,
        now,
      );
    console.log(`Created TBC PR #${result.lastInsertRowid}`);
  },

  'pr-list'() {
    const { values } = parseArgs({
      args,
      options: {
        status: { type: 'string', default: 'open' },
        json: { type: 'boolean', default: false },
      },
      strict: false,
    });
    let query = 'SELECT * FROM tbc_prs';
    const params = [];
    if (values.status && values.status !== 'all') {
      const normalizedStatus = normalizePrStatus(values.status);
      query += ' WHERE status = ?';
      params.push(normalizedStatus);
    }
    query += ' ORDER BY updated_at DESC, id DESC';
    const rows = db.prepare(query).all(...params).map(row => ({
      ...row,
      issue_ids: (() => { try { return JSON.parse(row.issue_ids || '[]'); } catch { return []; } })(),
    }));
    if (values.json) {
      jsonOut(rows);
    } else {
      for (const row of rows) {
        const issues = row.issue_ids.length ? ` issues=${row.issue_ids.join(',')}` : '';
        console.log(`#${row.id} [${row.status}] ${row.title} (${row.head_branch} -> ${row.base_branch}) test=${row.test_status}${issues}`);
      }
      if (rows.length === 0) console.log('(no TBC PRs)');
    }
  },

  'pr-view'() {
    const id = args[0];
    if (!id) { console.error('Usage: tbc-db pr-view <id>'); process.exit(1); }
    const pr = db.prepare('SELECT * FROM tbc_prs WHERE id = ?').get(id);
    if (!pr) { console.error(`TBC PR #${id} not found`); process.exit(1); }
    console.log(`# TBC PR #${pr.id}: ${pr.title}`);
    console.log(`Status: ${pr.status} | Base: ${pr.base_branch} | Head: ${pr.head_branch} | Test: ${pr.test_status}`);
    console.log(`Created: ${pr.created_at} | Updated: ${pr.updated_at}`);
    const issueIds = (() => { try { return JSON.parse(pr.issue_ids || '[]'); } catch { return []; } })();
    if (issueIds.length) console.log(`Issues: ${issueIds.join(', ')}`);
    if (pr.github_pr_number || pr.github_pr_url) {
      console.log(`GitHub mirror: #${pr.github_pr_number || '?'} ${pr.github_pr_url || ''}`.trim());
    }
    if (pr.summary) console.log(`\n${pr.summary}`);
  },

  'pr-edit'() {
    const id = args[0];
    if (!id) { console.error('Usage: tbc-db pr-edit <id> --actor name [--title "..."] [--summary "..."] [--base main] [--head branch] [--status ready_for_review] [--issues "1,2"] [--test pass]'); process.exit(1); }
    const { values } = parseArgs({
      args: args.slice(1),
      options: {
        actor: { type: 'string' },
        title: { type: 'string', short: 't' },
        summary: { type: 'string', short: 's' },
        base: { type: 'string' },
        head: { type: 'string' },
        status: { type: 'string' },
        issues: { type: 'string' },
        test: { type: 'string' },
        github_number: { type: 'string' },
        github_url: { type: 'string' },
      },
      strict: false,
    });
    if (!values.actor) { console.error('Usage: tbc-db pr-edit <id> --actor name [--title "..."] [--summary "..."] [--base main] [--head branch] [--status ready_for_review] [--issues "1,2"] [--test pass]'); process.exit(1); }
    const sets = [];
    const params = [];
    if (values.title !== undefined) { sets.push('title = ?'); params.push(values.title); }
    if (values.summary !== undefined) { sets.push('summary = ?'); params.push(values.summary); }
    if (values.base !== undefined) { sets.push('base_branch = ?'); params.push(values.base); }
    const nextBase = values.base;
    const nextHead = values.head;
    if (values.head !== undefined) { sets.push('head_branch = ?'); params.push(values.head); }
    if (values.status !== undefined) { sets.push('status = ?'); params.push(normalizePrStatus(values.status)); }
    if (values.issues !== undefined) {
      const issueIds = JSON.stringify(values.issues.split(',').map(s => s.trim()).filter(Boolean).map(Number).filter(Number.isFinite));
      sets.push('issue_ids = ?'); params.push(issueIds);
    }
    if (values.test !== undefined) { sets.push('test_status = ?'); params.push(values.test); }
    if (values.github_number !== undefined) { sets.push('github_pr_number = ?'); params.push(values.github_number ? Number(values.github_number) : null); }
    if (values.github_url !== undefined) { sets.push('github_pr_url = ?'); params.push(values.github_url || null); }
    if (sets.length === 0) { console.error('Nothing to update'); process.exit(1); }
    const current = db.prepare('SELECT base_branch, head_branch FROM tbc_prs WHERE id = ?').get(id);
    if (!current) { console.error(`TBC PR #${id} not found`); process.exit(1); }
    const finalBase = nextBase !== undefined ? nextBase : current.base_branch;
    const finalHead = nextHead !== undefined ? nextHead : current.head_branch;
    if (finalBase === finalHead) {
      console.error('Error: base and head branches must differ for a TBC PR record.');
      process.exit(1);
    }
    sets.push('updated_at = ?'); params.push(new Date().toISOString());
    sets.push('updated_by = ?'); params.push(values.actor);
    params.push(id);
    db.prepare(`UPDATE tbc_prs SET ${sets.join(', ')} WHERE id = ?`).run(...params);
    console.log(`Updated TBC PR #${id}`);
  },

  // ===== RESOURCE ORCHESTRATION =====
  'allocation-upsert'() {
    const { values } = parseArgs({
      args,
      options: {
        actor: { type: 'string' },
        owner: { type: 'string' },
        job_id: { type: 'string' },
        node: { type: 'string' },
        state: { type: 'string', default: 'active' },
        metadata: { type: 'string' },
        json: { type: 'boolean', default: false },
      },
      strict: false,
    });
    const owner = values.owner || values.actor;
    if (!owner || !values.job_id) {
      console.error('Usage: tbc-db allocation-upsert --job_id <job> --owner name [--node node001] [--state active|pending] [--metadata "{...}"]');
      process.exit(1);
    }
    const lease = upsertAllocationLease(db, {
      owner,
      jobId: values.job_id,
      nodeName: values.node || null,
      state: values.state || 'active',
      metadata: parseJsonOption(values.metadata, 'metadata'),
    });
    if (values.json) {
      jsonOut(lease);
    } else {
      console.log(`Lease ${lease.state}: job=${lease.jobId}${lease.nodeName ? ` node=${lease.nodeName}` : ''}`);
    }
  },

  'allocation-release'() {
    const { values } = parseArgs({
      args,
      options: {
        actor: { type: 'string' },
        note: { type: 'string' },
        json: { type: 'boolean', default: false },
      },
      strict: false,
    });
    const lease = releaseAllocationLease(db, {
      actor: values.actor || null,
      note: values.note || null,
    });
    if (values.json) {
      jsonOut(lease);
    } else {
      console.log(`Released lease for job ${lease.jobId}`);
    }
  },

  'allocation-status'() {
    const { values } = parseArgs({
      args,
      options: {
        json: { type: 'boolean', default: false },
      },
      strict: false,
    });
    const summary = getResourceSummary(db);
    if (values.json) {
      jsonOut(summary.allocation);
      return;
    }
    const lease = getAllocationLease(db, { includeInactive: true });
    if (!lease) {
      console.log('(no allocation lease)');
      return;
    }
    console.log(`state: ${lease.state} | owner: ${lease.owner || ''} | job: ${lease.jobId || ''} | node: ${lease.nodeName || ''}`);
  },

  'token-list'() {
    const { values } = parseArgs({
      args,
      options: {
        json: { type: 'boolean', default: false },
      },
      strict: false,
    });
    const tokens = listResourceTokens(db);
    if (values.json) {
      jsonOut(tokens);
    } else {
      textOut(tokens, ['tokenName', 'available']);
    }
  },

  'token-request'() {
    const { values } = parseArgs({
      args,
      options: {
        worker: { type: 'string' },
        count: { type: 'string', default: '1' },
        priority: { type: 'string', default: '0' },
        reason: { type: 'string', default: '' },
        metadata: { type: 'string' },
        json: { type: 'boolean', default: false },
      },
      strict: false,
    });
    if (!values.worker) {
      console.error('Usage: tbc-db token-request --worker worker-name [--count 1] [--priority 0] [--reason "..."] [--metadata "{...}"]');
      process.exit(1);
    }
    const request = createTokenRequest(db, {
      workerName: values.worker,
      requestedCount: values.count,
      priority: Number.parseInt(values.priority || '0', 10),
      rationale: values.reason || '',
      metadata: parseJsonOption(values.metadata, 'metadata'),
    });
    if (values.json) {
      jsonOut(request);
    } else {
      console.log(`Created token request #${request.id} for ${request.workerName} (${request.requestedCount} GPU)`);
    }
  },

  'token-requests'() {
    const { values } = parseArgs({
      args,
      options: {
        status: { type: 'string', default: 'all' },
        limit: { type: 'string', default: '50' },
        json: { type: 'boolean', default: false },
      },
      strict: false,
    });
    const requests = listTokenRequests(db, {
      status: values.status || 'all',
      limit: values.limit,
    });
    if (values.json) {
      jsonOut(requests);
    } else {
      textOut(requests, ['id', 'workerName', 'status', 'requestedCount', 'priority']);
    }
  },

  'token-grant'() {
    const { values } = parseArgs({
      args,
      options: {
        request: { type: 'string' },
        manager: { type: 'string' },
        tokens: { type: 'string' },
        job_id: { type: 'string' },
        node: { type: 'string' },
        metadata: { type: 'string' },
        resolution_note: { type: 'string' },
        json: { type: 'boolean', default: false },
      },
      strict: false,
    });
    if (!values.request || !values.manager || !values.tokens) {
      console.error('Usage: tbc-db token-grant --request <id> --manager manager-name --tokens gpu0,gpu1 [--job_id <job>] [--node node001] [--metadata "{...}"]');
      process.exit(1);
    }
    const grant = grantTokenRequest(db, {
      requestId: values.request,
      managerName: values.manager,
      tokenNames: values.tokens.split(',').map(part => part.trim()).filter(Boolean),
      leaseJobId: values.job_id || null,
      nodeName: values.node || null,
      metadata: parseJsonOption(values.metadata, 'metadata'),
      resolutionNote: values.resolution_note || null,
    });
    if (values.json) {
      jsonOut(grant);
    } else {
      console.log(`Granted request #${grant.requestId} to ${grant.workerName}: ${grant.tokenNames.join(', ')}`);
    }
  },

  'token-grants'() {
    const { values } = parseArgs({
      args,
      options: {
        status: { type: 'string', default: 'all' },
        limit: { type: 'string', default: '50' },
        json: { type: 'boolean', default: false },
      },
      strict: false,
    });
    const grants = listTokenGrants(db, {
      status: values.status || 'all',
      limit: values.limit,
    });
    if (values.json) {
      jsonOut(grants);
    } else {
      const rows = grants.map(grant => ({ ...grant, tokenNames: grant.tokenNames.join(',') }));
      textOut(rows, ['id', 'workerName', 'status', 'leaseJobId', 'tokenNames']);
    }
  },

  'token-release'() {
    const id = args[0];
    if (!id) {
      console.error('Usage: tbc-db token-release <grant_id> --actor manager-name [--note "..."]');
      process.exit(1);
    }
    const { values } = parseArgs({
      args: args.slice(1),
      options: {
        actor: { type: 'string' },
        note: { type: 'string' },
        json: { type: 'boolean', default: false },
      },
      strict: false,
    });
    if (!values.actor) {
      console.error('Usage: tbc-db token-release <grant_id> --actor manager-name [--note "..."]');
      process.exit(1);
    }
    const grant = releaseTokenGrant(db, {
      grantId: id,
      actor: values.actor,
      note: values.note || null,
    });
    if (values.json) {
      jsonOut(grant);
    } else {
      console.log(`Released token grant #${grant.id}`);
    }
  },

  // ===== QUERY =====
  'query'() {
    const sql = args.join(' ');
    if (!sql) { console.error('Usage: tbc-db query "SELECT ..."'); process.exit(1); }
    try {
      if (sql.trim().toUpperCase().startsWith('SELECT')) {
        const rows = db.prepare(sql).all();
        jsonOut(rows);
      } else {
        const result = db.prepare(sql).run();
        console.log(`Changes: ${result.changes}`);
      }
    } catch (e) {
      console.error(`SQL error: ${e.message}`);
      process.exit(1);
    }
  },

  'help'() {
    console.log(`tbc-db — Agent communication database

Issues:
  issue-create  --title "..." --creator name [--body "..."] [--assignee name] [--labels "..."]
  issue-list    [--status open|closed|all] [--assignee name] [--creator name] [--label name] [--json]
  issue-view    <id>
  issue-edit    <id> --editor name [--title "..."] [--body "..."] [--assignee name] [--labels "..."]
  issue-close   <id> --closer name

Comments:
  comment       --issue <id> --author name --body "..."
  comments      <issue_id>

TBC PRs:
  pr-create     --title "..." --head branch [--summary "..."] [--base main] [--status open|merged|closed] [--issues "1,2"] [--test unknown]
  pr-list       [--status open|merged|closed|all] [--json]
  pr-view       <id>
  pr-edit       <id> [--title "..."] [--summary "..."] [--base main] [--head branch] [--status open|merged|closed] [--issues "1,2"] [--test pass]

Resource orchestration:
  allocation-upsert   --job_id <job> --owner name [--node node001] [--state active|pending] [--metadata "{...}"]
  allocation-release  [--actor name] [--note "..."]
  allocation-status   [--json]
  token-list          [--json]
  token-request       --worker worker-name [--count 1] [--priority 0] [--reason "..."] [--metadata "{...}"] [--json]
  token-requests      [--status pending|granted|released|all] [--limit 50] [--json]
  token-grant         --request <id> --manager manager-name --tokens gpu0,gpu1 [--job_id <job>] [--node node001] [--metadata "{...}"] [--json]
  token-grants        [--status active|released|all] [--limit 50] [--json]
  token-release       <grant_id> --actor manager-name [--note "..."] [--json]

Advanced:
  query         "SQL statement"
  help          Show this help
`);
  }
};

if (!command || !commands[command]) {
  if (command) console.error(`Unknown command: ${command}`);
  commands.help();
  process.exit(command ? 1 : 0);
}

commands[command]();
db.close();
