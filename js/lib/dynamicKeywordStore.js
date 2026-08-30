const path = require('path');
const db = require('../utils/db');
const csvStore = require('./csvStore');

const headers = ['id', 'keyword', 'count', 'updated_at'];
const csvPath = process.env.AIWAF_DYNAMIC_KEYWORDS_CSV_PATH || path.join('logs', 'storage', 'dynamic_keywords.csv');

let initialized = false;
let dbAvailable = true;

async function initialize() {
  if (initialized) return dbAvailable;

  try {
    const hasTable = await db.schema.hasTable('dynamic_keywords');
    if (!hasTable) {
      await db.schema.createTable('dynamic_keywords', table => {
        table.increments('id').primary();
        table.string('keyword').notNullable().unique();
        table.integer('count').notNullable().defaultTo(0);
        table.timestamp('updated_at').defaultTo(db.fn.now());
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

  async list(limit = 1000) {
    const normalizedLimit = Number(limit) > 0 ? Number(limit) : 1000;
    if (await initialize()) {
      try {
        return db('dynamic_keywords')
          .select('*')
          .orderBy('count', 'desc')
          .limit(normalizedLimit);
      } catch (err) {
        dbAvailable = false;
      }
    }

    return rows()
      .map(row => ({ ...row, count: Number(row.count || 0) }))
      .sort((a, b) => b.count - a.count)
      .slice(0, normalizedLimit);
  },

  async increment(keyword) {
    const normalized = String(keyword || '').toLowerCase();
    if (!normalized) return;

    if (await initialize()) {
      try {
        const existing = await db('dynamic_keywords').where('keyword', normalized).first();
        if (existing) {
          await db('dynamic_keywords')
            .where('keyword', normalized)
            .update({ count: existing.count + 1, updated_at: db.fn.now() });
        } else {
          await db('dynamic_keywords').insert({ keyword: normalized, count: 1 });
        }
        return;
      } catch (err) {
        dbAvailable = false;
      }
    }

    const current = rows();
    const match = current.find(row => String(row.keyword) === normalized);
    if (match) {
      match.count = Number(match.count || 0) + 1;
      match.updated_at = new Date().toISOString();
    } else {
      current.push({
        id: csvStore.nextId(current),
        keyword: normalized,
        count: 1,
        updated_at: new Date().toISOString()
      });
    }
    csvStore.writeRows(csvPath, headers, current);
  },

  async add(keyword, count = 1) {
    const normalized = String(keyword || '').toLowerCase();
    if (!normalized) return;

    const normalizedCount = Number(count) > 0 ? Number(count) : 1;

    if (await initialize()) {
      try {
        const existing = await db('dynamic_keywords').where('keyword', normalized).first();
        if (existing) {
          await db('dynamic_keywords')
            .where('keyword', normalized)
            .update({ count: normalizedCount, updated_at: db.fn.now() });
        } else {
          await db('dynamic_keywords').insert({ keyword: normalized, count: normalizedCount });
        }
        return;
      } catch (err) {
        dbAvailable = false;
      }
    }

    const current = rows();
    const match = current.find(row => String(row.keyword) === normalized);
    if (match) {
      match.count = normalizedCount;
      match.updated_at = new Date().toISOString();
    } else {
      current.push({
        id: csvStore.nextId(current),
        keyword: normalized,
        count: normalizedCount,
        updated_at: new Date().toISOString()
      });
    }
    csvStore.writeRows(csvPath, headers, current);
  },

  async remove(keyword) {
    const normalized = String(keyword || '').toLowerCase();
    if (!normalized) return;

    if (await initialize()) {
      try {
        await db('dynamic_keywords').where('keyword', normalized).del();
        return;
      } catch (err) {
        dbAvailable = false;
      }
    }

    const current = rows().filter(row => String(row.keyword) !== normalized);
    csvStore.writeRows(csvPath, headers, current);
  },

  async clear() {
    if (await initialize()) {
      try {
        return await db('dynamic_keywords').del();
      } catch (err) {
        dbAvailable = false;
      }
    }

    const current = rows();
    csvStore.writeRows(csvPath, headers, []);
    return current.length;
  }
};
