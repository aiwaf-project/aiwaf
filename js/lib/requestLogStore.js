const path = require('path');
const db = require('../utils/db');
const csvStore = require('./csvStore');

const headers = [
  'id',
  'created_at',
  'ip_address',
  'method',
  'path',
  'status',
  'response_time_ms',
  'blocked',
  'reason',
  'country',
  'user_agent'
];
const csvPath = process.env.AIWAF_REQUEST_LOGS_CSV_PATH || path.join('logs', 'storage', 'request_logs.csv');

let initialized = false;
let dbAvailable = true;

async function initialize() {
  if (initialized) return dbAvailable;

  try {
    const hasLogs = await db.schema.hasTable('request_logs');
    if (!hasLogs) {
      await db.schema.createTable('request_logs', table => {
        table.increments('id').primary();
        table.timestamp('created_at').defaultTo(db.fn.now());
        table.string('ip_address');
        table.string('method');
        table.string('path');
        table.integer('status');
        table.integer('response_time_ms');
        table.boolean('blocked').defaultTo(false);
        table.string('reason');
        table.string('country', 2);
        table.string('user_agent');
      });
    }
    dbAvailable = true;
  } catch (err) {
    dbAvailable = false;
  }

  if (!dbAvailable) {
    csvStore.writeRows(csvPath, headers, csvStore.readRows(csvPath, headers));
  }

  initialized = true;
  return dbAvailable;
}

function rows() {
  return csvStore.readRows(csvPath, headers);
}

module.exports = {
  async initialize() {
    await initialize();
  },

  async insert(event) {
    const payload = {
      created_at: event.timestamp || new Date().toISOString(),
      ip_address: event.ip || '',
      method: event.method || '',
      path: event.path || '',
      status: Number(event.status || 0),
      response_time_ms: Number(event.responseTime || 0),
      blocked: event.blocked ? 1 : 0,
      reason: event.reason || '',
      country: event.country || '',
      user_agent: event.userAgent || ''
    };

    if (await initialize()) {
      try {
        await db('request_logs').insert({
          created_at: payload.created_at,
          ip_address: payload.ip_address,
          method: payload.method,
          path: payload.path,
          status: payload.status,
          response_time_ms: payload.response_time_ms,
          blocked: !!payload.blocked,
          reason: payload.reason,
          country: payload.country,
          user_agent: payload.user_agent
        });
        return;
      } catch (err) {
        dbAvailable = false;
      }
    }

    const current = rows();
    csvStore.appendRow(csvPath, headers, { id: csvStore.nextId(current), ...payload });
  },

  async recent(limit = 5000) {
    const normalizedLimit = Number(limit) > 0 ? Number(limit) : 5000;

    if (await initialize()) {
      try {
        return db('request_logs').select('*').orderBy('created_at', 'desc').limit(normalizedLimit);
      } catch (err) {
        dbAvailable = false;
      }
    }

    return rows()
      .sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)))
      .slice(0, normalizedLimit);
  },

  async geoSummary(limit = 25) {
    const normalizedLimit = Number(limit) > 0 ? Number(limit) : 25;

    if (await initialize()) {
      try {
        return db('request_logs')
          .select('country')
          .count({ requests: '*' })
          .sum({ blocked_count: db.raw('CASE WHEN blocked THEN 1 ELSE 0 END') })
          .whereNotNull('country')
          .groupBy('country')
          .orderBy('requests', 'desc')
          .limit(normalizedLimit);
      } catch (err) {
        dbAvailable = false;
      }
    }

    const map = new Map();
    for (const row of rows()) {
      const country = String(row.country || '').trim();
      if (!country) continue;
      const current = map.get(country) || { country, requests: 0, blocked_count: 0 };
      current.requests += 1;
      if (String(row.blocked) === '1' || String(row.blocked).toLowerCase() === 'true') {
        current.blocked_count += 1;
      }
      map.set(country, current);
    }

    return Array.from(map.values())
      .sort((a, b) => b.requests - a.requests)
      .slice(0, normalizedLimit);
  },

  async clear() {
    if (await initialize()) {
      try {
        return await db('request_logs').del();
      } catch (err) {
        dbAvailable = false;
      }
    }

    const current = rows();
    csvStore.writeRows(csvPath, headers, []);
    return current.length;
  }
};
