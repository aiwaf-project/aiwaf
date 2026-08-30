const path = require('path');
const db = require('../utils/db');
const csvStore = require('./csvStore');

const ipHeaders = ['id', 'ip_address', 'reason', 'created_at'];
const pathHeaders = ['id', 'path_prefix', 'reason', 'created_at'];

const ipCsvPath = process.env.AIWAF_IP_EXEMPTIONS_CSV_PATH || path.join('logs', 'storage', 'ip_exemptions.csv');
const pathCsvPath = process.env.AIWAF_PATH_EXEMPTIONS_CSV_PATH || path.join('logs', 'storage', 'path_exemptions.csv');

let initialized = false;
let dbAvailable = true;

async function initialize() {
  if (initialized) return dbAvailable;

  try {
    const hasIp = await db.schema.hasTable('ip_exemptions');
    if (!hasIp) {
      await db.schema.createTable('ip_exemptions', table => {
        table.increments('id').primary();
        table.string('ip_address').notNullable().unique();
        table.string('reason').defaultTo('manual');
        table.timestamp('created_at').defaultTo(db.fn.now());
      });
    }

    const hasPath = await db.schema.hasTable('path_exemptions');
    if (!hasPath) {
      await db.schema.createTable('path_exemptions', table => {
        table.increments('id').primary();
        table.string('path_prefix').notNullable().unique();
        table.string('reason').defaultTo('manual');
        table.timestamp('created_at').defaultTo(db.fn.now());
      });
    }

    dbAvailable = true;
  } catch (err) {
    dbAvailable = false;
  }

  if (!dbAvailable) {
    csvStore.writeRows(ipCsvPath, ipHeaders, csvStore.readRows(ipCsvPath, ipHeaders));
    csvStore.writeRows(pathCsvPath, pathHeaders, csvStore.readRows(pathCsvPath, pathHeaders));
  }

  initialized = true;
  return dbAvailable;
}

function ipRows() {
  return csvStore.readRows(ipCsvPath, ipHeaders);
}

function pathRows() {
  return csvStore.readRows(pathCsvPath, pathHeaders);
}

module.exports = {
  async initialize() {
    await initialize();
  },

  async addIp(ip, reason = 'manual') {
    const normalized = String(ip || '').trim();
    if (!normalized) return;

    if (await initialize()) {
      try {
        await db('ip_exemptions').insert({ ip_address: normalized, reason }).onConflict('ip_address').ignore();
        return;
      } catch (err) {
        dbAvailable = false;
      }
    }

    const rows = ipRows();
    if (rows.some(row => String(row.ip_address) === normalized)) return;
    rows.push({ id: csvStore.nextId(rows), ip_address: normalized, reason, created_at: new Date().toISOString() });
    csvStore.writeRows(ipCsvPath, ipHeaders, rows);
  },

  async removeIp(ip) {
    const normalized = String(ip || '').trim();
    if (await initialize()) {
      try {
        return db('ip_exemptions').where('ip_address', normalized).del();
      } catch (err) {
        dbAvailable = false;
      }
    }

    const rows = ipRows();
    const filtered = rows.filter(row => String(row.ip_address) !== normalized);
    csvStore.writeRows(ipCsvPath, ipHeaders, filtered);
    return rows.length - filtered.length;
  },

  async listIps() {
    if (await initialize()) {
      try {
        return db('ip_exemptions').select('*').orderBy('created_at', 'desc');
      } catch (err) {
        dbAvailable = false;
      }
    }

    return ipRows().sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
  },

  async isIpExempt(ip) {
    const normalized = String(ip || '').trim();
    if (await initialize()) {
      try {
        const row = await db('ip_exemptions').where('ip_address', normalized).first();
        return !!row;
      } catch (err) {
        dbAvailable = false;
      }
    }

    return ipRows().some(row => String(row.ip_address) === normalized);
  },

  async addPath(pathPrefix, reason = 'manual') {
    const normalized = String(pathPrefix || '').trim();
    if (!normalized) return;

    if (await initialize()) {
      try {
        await db('path_exemptions').insert({ path_prefix: normalized, reason }).onConflict('path_prefix').ignore();
        return;
      } catch (err) {
        dbAvailable = false;
      }
    }

    const rows = pathRows();
    if (rows.some(row => String(row.path_prefix) === normalized)) return;
    rows.push({ id: csvStore.nextId(rows), path_prefix: normalized, reason, created_at: new Date().toISOString() });
    csvStore.writeRows(pathCsvPath, pathHeaders, rows);
  },

  async removePath(pathPrefix) {
    const normalized = String(pathPrefix || '').trim();
    if (await initialize()) {
      try {
        return db('path_exemptions').where('path_prefix', normalized).del();
      } catch (err) {
        dbAvailable = false;
      }
    }

    const rows = pathRows();
    const filtered = rows.filter(row => String(row.path_prefix) !== normalized);
    csvStore.writeRows(pathCsvPath, pathHeaders, filtered);
    return rows.length - filtered.length;
  },

  async listPaths() {
    if (await initialize()) {
      try {
        return db('path_exemptions').select('*').orderBy('created_at', 'desc');
      } catch (err) {
        dbAvailable = false;
      }
    }

    return pathRows().sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
  },

  async isPathExempt(pathValue) {
    const normalizedPath = String(pathValue || '').toLowerCase();
    if (await initialize()) {
      try {
        const rows = await db('path_exemptions').select('path_prefix');
        return rows.some(row => normalizedPath.startsWith(String(row.path_prefix || '').toLowerCase()));
      } catch (err) {
        dbAvailable = false;
      }
    }

    return pathRows().some(row => normalizedPath.startsWith(String(row.path_prefix || '').toLowerCase()));
  },

  async clear() {
    let deleted = 0;
    if (await initialize()) {
      try {
        deleted += await db('ip_exemptions').del();
        deleted += await db('path_exemptions').del();
      } catch (err) {
        dbAvailable = false;
      }
    }
    if (!dbAvailable || !initialized) {
      csvStore.writeRows(ipCsvPath, ipHeaders, []);
      csvStore.writeRows(pathCsvPath, pathHeaders, []);
      deleted = -1; // Unknown count for CSV
    }
    return deleted;
  }
};
