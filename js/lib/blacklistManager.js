const path = require('path');
const db = require('../utils/db');
const csvStore = require('./csvStore');

const headers = ['id', 'ip_address', 'reason', 'blocked_at'];
const csvPath = process.env.AIWAF_BLOCKED_IPS_CSV_PATH || path.join('logs', 'storage', 'blocked_ips.csv');

let initialized = false;
let dbAvailable = true;

async function initialize() {
  if (initialized) return dbAvailable;

  try {
    const exists = await db.schema.hasTable('blocked_ips');
    if (!exists) {
      await db.schema.createTable('blocked_ips', table => {
        table.increments('id').primary();
        table.string('ip_address').unique().notNullable();
        table.string('reason').defaultTo('WAF blocked');
        table.timestamp('blocked_at').defaultTo(db.fn.now());
      });
    }
    dbAvailable = true;
  } catch (err) {
    dbAvailable = false;
  }

  if (!dbAvailable) {
    const existing = csvStore.readRows(csvPath, headers);
    csvStore.writeRows(csvPath, headers, existing);
  }

  initialized = true;
  return dbAvailable;
}

function csvRows() {
  return csvStore.readRows(csvPath, headers);
}

module.exports = {
  async isBlocked(ip) {
    const normalizedIp = String(ip || '').trim();
    const useDb = await initialize();
    if (useDb) {
      try {
        const row = await db('blocked_ips').where('ip_address', normalizedIp).first();
        return !!row;
      } catch (err) {
        dbAvailable = false;
      }
    }

    return csvRows().some(row => String(row.ip_address) === normalizedIp);
  },

  async block(ip, reason = 'WAF blocked') {
    const normalizedIp = String(ip || '').trim();
    if (!normalizedIp) return false;

    const useDb = await initialize();
    if (useDb) {
      try {
        await db('blocked_ips').insert({ ip_address: normalizedIp, reason }).onConflict('ip_address').ignore();
        return true;
      } catch (err) {
        dbAvailable = false;
      }
    }

    const rows = csvRows();
    if (rows.some(row => String(row.ip_address) === normalizedIp)) return true;

    rows.push({
      id: csvStore.nextId(rows),
      ip_address: normalizedIp,
      reason,
      blocked_at: new Date().toISOString()
    });
    csvStore.writeRows(csvPath, headers, rows);
    return true;
  },

  async unblock(ip) {
    const normalizedIp = String(ip || '').trim();
    const useDb = await initialize();
    if (useDb) {
      try {
        const deleted = await db('blocked_ips').where('ip_address', normalizedIp).del();
        return deleted > 0;
      } catch (err) {
        dbAvailable = false;
      }
    }

    const rows = csvRows();
    const filtered = rows.filter(row => String(row.ip_address) !== normalizedIp);
    csvStore.writeRows(csvPath, headers, filtered);
    return filtered.length !== rows.length;
  },

  async getBlockedIPs() {
    const useDb = await initialize();
    if (useDb) {
      try {
        return await db('blocked_ips').select('*').orderBy('blocked_at', 'desc');
      } catch (err) {
        dbAvailable = false;
      }
    }

    return csvRows().sort((a, b) => String(b.blocked_at).localeCompare(String(a.blocked_at)));
  },

  async clear() {
    const useDb = await initialize();
    if (useDb) {
      try {
        return await db('blocked_ips').del();
      } catch (err) {
        dbAvailable = false;
      }
    }

    const rows = csvRows();
    csvStore.writeRows(csvPath, headers, []);
    return rows.length;
  },

  async initialize() {
    return initialize();
  }
};
