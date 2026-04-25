const DEFAULT_RESOURCE_ORCHESTRATION = Object.freeze({
  enabled: true,
  mode: 'shared-slurm',
  gpuCount: 8,
  tokenPrefix: 'gpu',
  grantRequiresLease: true,
});

const REQUEST_STATUSES = new Set(['pending', 'granted', 'denied', 'cancelled', 'released']);
const GRANT_STATUSES = new Set(['active', 'released']);
const LEASE_STATES = new Set(['inactive', 'pending', 'active', 'released', 'failed']);

function nowIso() {
  return new Date().toISOString();
}

function coercePositiveInteger(value, fallback) {
  if (typeof value === 'number' && Number.isInteger(value) && value > 0) return value;
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number.parseInt(value, 10);
    if (Number.isInteger(parsed) && parsed > 0) return parsed;
  }
  return fallback;
}

function parseJsonField(value, fallback) {
  if (!value) return fallback;
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}

function stringifyMetadata(value) {
  if (value === undefined) return '{}';
  if (typeof value === 'string') {
    return value.trim() ? value : '{}';
  }
  return JSON.stringify(value ?? {});
}

function ensureNonEmptyString(value, name) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`${name} is required`);
  }
  return value.trim();
}

function getActiveLeaseRow(db) {
  return db.prepare(`
    SELECT slot, state, owner, job_id, node_name, metadata_json, created_at, updated_at, released_at
    FROM allocation_leases
    WHERE slot = 'primary'
  `).get();
}

function mapLeaseRow(row, { includeInactive = false } = {}) {
  if (!row) return null;
  if (!includeInactive && !['pending', 'active'].includes(row.state)) return null;
  return {
    slot: row.slot,
    state: row.state,
    owner: row.owner,
    jobId: row.job_id,
    nodeName: row.node_name,
    metadata: parseJsonField(row.metadata_json, {}),
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    releasedAt: row.released_at,
  };
}

function mapRequestRow(row) {
  if (!row) return null;
  return {
    id: row.id,
    workerName: row.worker_name,
    resourceType: row.resource_type,
    requestedCount: row.requested_count,
    priority: row.priority,
    rationale: row.rationale,
    status: row.status,
    metadata: parseJsonField(row.metadata_json, {}),
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    resolvedAt: row.resolved_at,
    resolvedBy: row.resolved_by,
    resolutionNote: row.resolution_note,
  };
}

function getGrantTokenNames(db, grantId) {
  return db.prepare(`
    SELECT token_name
    FROM token_grant_tokens
    WHERE grant_id = ?
    ORDER BY token_name
  `).all(grantId).map(row => row.token_name);
}

function mapGrantRow(db, row) {
  if (!row) return null;
  return {
    id: row.id,
    requestId: row.request_id,
    managerName: row.manager_name,
    workerName: row.worker_name,
    leaseJobId: row.lease_job_id,
    nodeName: row.node_name,
    status: row.status,
    metadata: parseJsonField(row.metadata_json, {}),
    grantedAt: row.granted_at,
    releasedAt: row.released_at,
    releasedBy: row.released_by,
    releaseNote: row.release_note,
    tokenNames: getGrantTokenNames(db, row.id),
  };
}

export function normalizeResourceOrchestrationConfig(input = {}) {
  const source = input && typeof input === 'object' ? input : {};
  const tokenPrefix = typeof source.tokenPrefix === 'string' && source.tokenPrefix.trim()
    ? source.tokenPrefix.trim()
    : DEFAULT_RESOURCE_ORCHESTRATION.tokenPrefix;

  return {
    enabled: source.enabled !== undefined ? Boolean(source.enabled) : DEFAULT_RESOURCE_ORCHESTRATION.enabled,
    mode: typeof source.mode === 'string' && source.mode.trim()
      ? source.mode.trim()
      : DEFAULT_RESOURCE_ORCHESTRATION.mode,
    gpuCount: coercePositiveInteger(source.gpuCount, DEFAULT_RESOURCE_ORCHESTRATION.gpuCount),
    tokenPrefix,
    grantRequiresLease: source.grantRequiresLease !== undefined
      ? Boolean(source.grantRequiresLease)
      : DEFAULT_RESOURCE_ORCHESTRATION.grantRequiresLease,
  };
}

export function ensureResourceOrchestrationTables(db, resourceConfig = {}) {
  const config = normalizeResourceOrchestrationConfig(resourceConfig);

  db.exec(`
    CREATE TABLE IF NOT EXISTS allocation_leases (
      slot TEXT PRIMARY KEY,
      state TEXT NOT NULL DEFAULT 'inactive' CHECK(state IN ('inactive', 'pending', 'active', 'released', 'failed')),
      owner TEXT,
      job_id TEXT,
      node_name TEXT,
      metadata_json TEXT DEFAULT '{}',
      created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
      updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
      released_at TEXT
    );

    CREATE TABLE IF NOT EXISTS resource_tokens (
      token_name TEXT PRIMARY KEY,
      resource_type TEXT NOT NULL CHECK(resource_type IN ('gpu')),
      ordinal INTEGER NOT NULL,
      capacity_units INTEGER NOT NULL DEFAULT 1,
      metadata_json TEXT DEFAULT '{}',
      created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    );

    CREATE TABLE IF NOT EXISTS token_requests (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      worker_name TEXT NOT NULL,
      resource_type TEXT NOT NULL CHECK(resource_type IN ('gpu')),
      requested_count INTEGER NOT NULL CHECK(requested_count > 0),
      priority INTEGER NOT NULL DEFAULT 0,
      rationale TEXT DEFAULT '',
      status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'granted', 'denied', 'cancelled', 'released')),
      metadata_json TEXT DEFAULT '{}',
      created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
      updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
      resolved_at TEXT,
      resolved_by TEXT,
      resolution_note TEXT
    );

    CREATE TABLE IF NOT EXISTS token_grants (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      request_id INTEGER NOT NULL UNIQUE REFERENCES token_requests(id),
      manager_name TEXT NOT NULL,
      worker_name TEXT NOT NULL,
      lease_job_id TEXT,
      node_name TEXT,
      status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'released')),
      metadata_json TEXT DEFAULT '{}',
      granted_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
      released_at TEXT,
      released_by TEXT,
      release_note TEXT
    );

    CREATE TABLE IF NOT EXISTS token_grant_tokens (
      grant_id INTEGER NOT NULL REFERENCES token_grants(id) ON DELETE CASCADE,
      token_name TEXT NOT NULL REFERENCES resource_tokens(token_name),
      PRIMARY KEY (grant_id, token_name)
    );
  `);

  const insertToken = db.prepare(`
    INSERT OR IGNORE INTO resource_tokens (token_name, resource_type, ordinal, capacity_units, metadata_json)
    VALUES (?, 'gpu', ?, 1, ?)
  `);
  const seedTokens = db.transaction(() => {
    for (let ordinal = 0; ordinal < config.gpuCount; ordinal += 1) {
      insertToken.run(
        `${config.tokenPrefix}${ordinal}`,
        ordinal,
        JSON.stringify({ gpuIndex: ordinal }),
      );
    }
  });
  seedTokens();

  return config;
}

export function upsertAllocationLease(db, lease) {
  const jobId = ensureNonEmptyString(lease.jobId, 'jobId');
  const owner = ensureNonEmptyString(lease.owner, 'owner');
  const state = lease.state ? String(lease.state).trim() : 'active';
  if (!LEASE_STATES.has(state)) {
    throw new Error(`invalid lease state: ${state}`);
  }
  const nodeName = typeof lease.nodeName === 'string' ? lease.nodeName.trim() || null : null;
  const metadataJson = stringifyMetadata(lease.metadata);
  const timestamp = nowIso();

  db.prepare(`
    INSERT INTO allocation_leases (slot, state, owner, job_id, node_name, metadata_json, created_at, updated_at, released_at)
    VALUES ('primary', ?, ?, ?, ?, ?, ?, ?, NULL)
    ON CONFLICT(slot) DO UPDATE SET
      state = excluded.state,
      owner = excluded.owner,
      job_id = excluded.job_id,
      node_name = excluded.node_name,
      metadata_json = excluded.metadata_json,
      updated_at = excluded.updated_at,
      released_at = NULL
  `).run(state, owner, jobId, nodeName, metadataJson, timestamp, timestamp);

  return getAllocationLease(db, { includeInactive: true });
}

export function releaseAllocationLease(db, { actor, note } = {}) {
  const current = getAllocationLease(db, { includeInactive: true });
  if (!current) {
    throw new Error('no allocation lease found');
  }
  if (current.state === 'released') {
    throw new Error('allocation lease is already released');
  }

  const timestamp = nowIso();
  const metadata = {
    ...(current.metadata || {}),
    release: {
      actor: actor || null,
      note: note || null,
      releasedAt: timestamp,
    },
  };

  db.prepare(`
    UPDATE allocation_leases
    SET state = 'released',
        metadata_json = ?,
        updated_at = ?,
        released_at = ?
    WHERE slot = 'primary'
  `).run(JSON.stringify(metadata), timestamp, timestamp);

  return getAllocationLease(db, { includeInactive: true });
}

export function getAllocationLease(db, { includeInactive = false } = {}) {
  return mapLeaseRow(getActiveLeaseRow(db), { includeInactive });
}

export function createTokenRequest(db, request) {
  const workerName = ensureNonEmptyString(request.workerName, 'workerName');
  const requestedCount = coercePositiveInteger(request.requestedCount, 0);
  if (!requestedCount) {
    throw new Error('requestedCount must be a positive integer');
  }
  const priority = Number.isInteger(request.priority) ? request.priority : 0;
  const rationale = typeof request.rationale === 'string' ? request.rationale : '';
  const metadataJson = stringifyMetadata(request.metadata);
  const timestamp = nowIso();

  const result = db.prepare(`
    INSERT INTO token_requests (
      worker_name,
      resource_type,
      requested_count,
      priority,
      rationale,
      status,
      metadata_json,
      created_at,
      updated_at
    )
    VALUES (?, 'gpu', ?, ?, ?, 'pending', ?, ?, ?)
  `).run(workerName, requestedCount, priority, rationale, metadataJson, timestamp, timestamp);

  return getTokenRequest(db, result.lastInsertRowid);
}

export function getTokenRequest(db, requestId) {
  const row = db.prepare(`
    SELECT id, worker_name, resource_type, requested_count, priority, rationale, status, metadata_json,
           created_at, updated_at, resolved_at, resolved_by, resolution_note
    FROM token_requests
    WHERE id = ?
  `).get(requestId);
  return mapRequestRow(row);
}

export function listTokenRequests(db, { status = 'all', limit = 50 } = {}) {
  const maxRows = coercePositiveInteger(limit, 50);
  const params = [];
  let whereClause = '';
  if (status !== 'all') {
    if (!REQUEST_STATUSES.has(status)) {
      throw new Error(`invalid request status: ${status}`);
    }
    whereClause = 'WHERE status = ?';
    params.push(status);
  }
  params.push(maxRows);

  return db.prepare(`
    SELECT id, worker_name, resource_type, requested_count, priority, rationale, status, metadata_json,
           created_at, updated_at, resolved_at, resolved_by, resolution_note
    FROM token_requests
    ${whereClause}
    ORDER BY created_at DESC, id DESC
    LIMIT ?
  `).all(...params).map(mapRequestRow);
}

export function getTokenGrant(db, grantId) {
  const row = db.prepare(`
    SELECT id, request_id, manager_name, worker_name, lease_job_id, node_name, status, metadata_json,
           granted_at, released_at, released_by, release_note
    FROM token_grants
    WHERE id = ?
  `).get(grantId);
  return mapGrantRow(db, row);
}

export function listTokenGrants(db, { status = 'all', limit = 50 } = {}) {
  const maxRows = coercePositiveInteger(limit, 50);
  const params = [];
  let whereClause = '';
  if (status !== 'all') {
    if (!GRANT_STATUSES.has(status)) {
      throw new Error(`invalid grant status: ${status}`);
    }
    whereClause = 'WHERE status = ?';
    params.push(status);
  }
  params.push(maxRows);

  return db.prepare(`
    SELECT id, request_id, manager_name, worker_name, lease_job_id, node_name, status, metadata_json,
           granted_at, released_at, released_by, release_note
    FROM token_grants
    ${whereClause}
    ORDER BY granted_at DESC, id DESC
    LIMIT ?
  `).all(...params).map(row => mapGrantRow(db, row));
}

export function listResourceTokens(db, { resourceType = 'gpu' } = {}) {
  const tokens = db.prepare(`
    SELECT token_name, resource_type, ordinal, capacity_units, metadata_json, created_at
    FROM resource_tokens
    WHERE resource_type = ?
    ORDER BY ordinal ASC, token_name ASC
  `).all(resourceType);

  const activeAssignments = db.prepare(`
    SELECT tgt.token_name, tg.id AS grant_id, tg.worker_name, tg.manager_name, tg.lease_job_id, tg.node_name, tg.granted_at
    FROM token_grant_tokens tgt
    INNER JOIN token_grants tg ON tg.id = tgt.grant_id
    WHERE tg.status = 'active'
  `).all();

  const assignmentByToken = new Map(activeAssignments.map(row => [row.token_name, row]));
  return tokens.map(token => {
    const assignment = assignmentByToken.get(token.token_name);
    return {
      tokenName: token.token_name,
      resourceType: token.resource_type,
      ordinal: token.ordinal,
      capacityUnits: token.capacity_units,
      metadata: parseJsonField(token.metadata_json, {}),
      createdAt: token.created_at,
      available: !assignment,
      activeGrant: assignment ? {
        grantId: assignment.grant_id,
        workerName: assignment.worker_name,
        managerName: assignment.manager_name,
        leaseJobId: assignment.lease_job_id,
        nodeName: assignment.node_name,
        grantedAt: assignment.granted_at,
      } : null,
    };
  });
}

export function grantTokenRequest(db, grant, resourceConfig = {}) {
  const config = normalizeResourceOrchestrationConfig(resourceConfig);
  const requestId = coercePositiveInteger(grant.requestId, 0);
  if (!requestId) {
    throw new Error('requestId is required');
  }
  const managerName = ensureNonEmptyString(grant.managerName, 'managerName');
  const requestedTokens = Array.isArray(grant.tokenNames)
    ? grant.tokenNames.map(tokenName => ensureNonEmptyString(tokenName, 'tokenName'))
    : [];
  if (requestedTokens.length === 0) {
    throw new Error('at least one token name is required');
  }

  const request = getTokenRequest(db, requestId);
  if (!request) {
    throw new Error(`token request ${requestId} not found`);
  }
  if (request.status !== 'pending') {
    throw new Error(`token request ${requestId} is not pending`);
  }
  if (request.requestedCount !== requestedTokens.length) {
    throw new Error(`token request ${requestId} expects ${request.requestedCount} tokens`);
  }

  const activeLease = getAllocationLease(db);
  let leaseJobId = typeof grant.leaseJobId === 'string' && grant.leaseJobId.trim() ? grant.leaseJobId.trim() : null;
  let nodeName = typeof grant.nodeName === 'string' && grant.nodeName.trim() ? grant.nodeName.trim() : null;
  if (!leaseJobId && activeLease) leaseJobId = activeLease.jobId;
  if (!nodeName && activeLease) nodeName = activeLease.nodeName;
  if (config.grantRequiresLease && !leaseJobId) {
    throw new Error('active allocation lease required to grant tokens');
  }

  const metadataJson = stringifyMetadata(grant.metadata);
  const txn = db.transaction(() => {
    const placeholders = requestedTokens.map(() => '?').join(', ');
    const existingTokens = db.prepare(`
      SELECT token_name
      FROM resource_tokens
      WHERE token_name IN (${placeholders})
    `).all(...requestedTokens).map(row => row.token_name);

    if (existingTokens.length !== requestedTokens.length) {
      const missing = requestedTokens.filter(tokenName => !existingTokens.includes(tokenName));
      throw new Error(`unknown token(s): ${missing.join(', ')}`);
    }

    const inUse = db.prepare(`
      SELECT tgt.token_name
      FROM token_grant_tokens tgt
      INNER JOIN token_grants tg ON tg.id = tgt.grant_id
      WHERE tg.status = 'active' AND tgt.token_name IN (${placeholders})
    `).all(...requestedTokens).map(row => row.token_name);

    if (inUse.length > 0) {
      throw new Error(`token(s) already granted: ${inUse.join(', ')}`);
    }

    const timestamp = nowIso();
    const grantResult = db.prepare(`
      INSERT INTO token_grants (
        request_id,
        manager_name,
        worker_name,
        lease_job_id,
        node_name,
        status,
        metadata_json,
        granted_at
      )
      VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
    `).run(requestId, managerName, request.workerName, leaseJobId, nodeName, metadataJson, timestamp);

    const insertGrantToken = db.prepare(`
      INSERT INTO token_grant_tokens (grant_id, token_name)
      VALUES (?, ?)
    `);
    for (const tokenName of requestedTokens) {
      insertGrantToken.run(grantResult.lastInsertRowid, tokenName);
    }

    db.prepare(`
      UPDATE token_requests
      SET status = 'granted',
          updated_at = ?,
          resolved_at = ?,
          resolved_by = ?,
          resolution_note = ?
      WHERE id = ?
    `).run(timestamp, timestamp, managerName, grant.resolutionNote || null, requestId);

    return grantResult.lastInsertRowid;
  });

  const grantId = txn();
  return getTokenGrant(db, grantId);
}

export function releaseTokenGrant(db, release) {
  const grantId = coercePositiveInteger(release.grantId, 0);
  if (!grantId) {
    throw new Error('grantId is required');
  }
  const actor = ensureNonEmptyString(release.actor, 'actor');
  const grant = getTokenGrant(db, grantId);
  if (!grant) {
    throw new Error(`token grant ${grantId} not found`);
  }
  if (grant.status !== 'active') {
    throw new Error(`token grant ${grantId} is not active`);
  }

  const note = typeof release.note === 'string' ? release.note : null;
  const timestamp = nowIso();
  const txn = db.transaction(() => {
    db.prepare(`
      UPDATE token_grants
      SET status = 'released',
          released_at = ?,
          released_by = ?,
          release_note = ?
      WHERE id = ?
    `).run(timestamp, actor, note, grantId);

    db.prepare(`
      UPDATE token_requests
      SET status = 'released',
          updated_at = ?,
          resolved_at = ?,
          resolved_by = ?,
          resolution_note = COALESCE(resolution_note, ?)
      WHERE id = ?
    `).run(timestamp, timestamp, actor, note, grant.requestId);
  });
  txn();

  return getTokenGrant(db, grantId);
}

export function buildEmptyResourceSummary(resourceConfig = {}) {
  const config = normalizeResourceOrchestrationConfig(resourceConfig);
  return {
    enabled: config.enabled,
    mode: config.mode,
    gpuCount: config.gpuCount,
    tokenPrefix: config.tokenPrefix,
    allocation: null,
    tokens: {
      total: 0,
      granted: 0,
      available: 0,
      availableTokens: [],
      grantedTokens: [],
    },
    requests: {
      total: 0,
      pending: 0,
      granted: 0,
      denied: 0,
      cancelled: 0,
      released: 0,
    },
    grants: {
      total: 0,
      active: 0,
      released: 0,
    },
  };
}

export function getResourceSummary(db, resourceConfig = {}) {
  const config = normalizeResourceOrchestrationConfig(resourceConfig);
  ensureResourceOrchestrationTables(db, config);

  const tokens = listResourceTokens(db);
  const requests = db.prepare(`
    SELECT status, COUNT(*) AS count
    FROM token_requests
    GROUP BY status
  `).all();
  const grants = db.prepare(`
    SELECT status, COUNT(*) AS count
    FROM token_grants
    GROUP BY status
  `).all();

  const requestCounts = { total: 0, pending: 0, granted: 0, denied: 0, cancelled: 0, released: 0 };
  for (const row of requests) {
    requestCounts[row.status] = row.count;
    requestCounts.total += row.count;
  }

  const grantCounts = { total: 0, active: 0, released: 0 };
  for (const row of grants) {
    grantCounts[row.status] = row.count;
    grantCounts.total += row.count;
  }

  const grantedTokens = tokens.filter(token => !token.available).map(token => token.tokenName);
  const availableTokens = tokens.filter(token => token.available).map(token => token.tokenName);

  return {
    enabled: config.enabled,
    mode: config.mode,
    gpuCount: config.gpuCount,
    tokenPrefix: config.tokenPrefix,
    allocation: getAllocationLease(db),
    tokens: {
      total: tokens.length,
      granted: grantedTokens.length,
      available: availableTokens.length,
      availableTokens,
      grantedTokens,
    },
    requests: requestCounts,
    grants: grantCounts,
  };
}
