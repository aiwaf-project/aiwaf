const path = require('path');
const db = require('../utils/db');
const csvStore = require('./csvStore');

const headers = ['id', 'country_code', 'reason', 'created_at'];
const csvPath = process.env.AIWAF_GEO_BLOCKED_COUNTRIES_CSV_PATH || path.join('logs', 'storage', 'geo_blocked_countries.csv');

let initialized = false;
let dbAvailable = true;

function normalizeCountry(value) {
  return String(value || '').trim().toUpperCase();
}

async function initialize() {
  if (initialized) return dbAvailable;

  try {
    const hasGeo = await db.schema.hasTable('geo_blocked_countries');
    if (!hasGeo) {
      await db.schema.createTable('geo_blocked_countries', table => {
        table.increments('id').primary();
        table.string('country_code', 2).notNullable().unique();
        table.string('reason').defaultTo('manual');
        table.timestamp('created_at').defaultTo(db.fn.now());
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

  async addBlockedCountry(countryCode, reason = 'manual') {
    const normalized = normalizeCountry(countryCode);
    if (!normalized) return;

    if (await initialize()) {
      try {
        await db('geo_blocked_countries').insert({ country_code: normalized, reason }).onConflict('country_code').ignore();
        return;
      } catch (err) {
        dbAvailable = false;
      }
    }

    const current = rows();
    if (current.some(row => String(row.country_code) === normalized)) return;
    current.push({ id: csvStore.nextId(current), country_code: normalized, reason, created_at: new Date().toISOString() });
    csvStore.writeRows(csvPath, headers, current);
  },

  async removeBlockedCountry(countryCode) {
    const normalized = normalizeCountry(countryCode);

    if (await initialize()) {
      try {
        return db('geo_blocked_countries').where('country_code', normalized).del();
      } catch (err) {
        dbAvailable = false;
      }
    }

    const current = rows();
    const filtered = current.filter(row => String(row.country_code) !== normalized);
    csvStore.writeRows(csvPath, headers, filtered);
    return current.length - filtered.length;
  },

  async listBlockedCountries() {
    if (await initialize()) {
      try {
        return db('geo_blocked_countries').select('*').orderBy('country_code', 'asc');
      } catch (err) {
        dbAvailable = false;
      }
    }

    return rows().sort((a, b) => String(a.country_code).localeCompare(String(b.country_code)));
  },

  async isBlockedCountry(countryCode) {
    const normalized = normalizeCountry(countryCode);
    if (await initialize()) {
      try {
        const row = await db('geo_blocked_countries').where('country_code', normalized).first();
        return !!row;
      } catch (err) {
        dbAvailable = false;
      }
    }

    return rows().some(row => String(row.country_code) === normalized);
  }
};
