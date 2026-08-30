const fs = require('fs');
const path = require('path');
const db = require('../utils/db');

const cache = new Map();

function getCacheKey(opts) {
  return opts.AIWAF_MODEL_CACHE_KEY || 'aiwaf:model';
}

function getModelPath(opts) {
  return opts.AIWAF_MODEL_PATH || path.join(__dirname, '..', 'resources', 'model.json');
}

async function ensureDb() {
  const hasTable = await db.schema.hasTable('model_artifacts');
  if (!hasTable) {
    await db.schema.createTable('model_artifacts', table => {
      table.increments('id').primary();
      table.timestamp('created_at').defaultTo(db.fn.now());
      table.string('storage').defaultTo('db');
      table.text('payload');
      table.text('metadata');
    });
  }
}

function cacheGet(key) {
  const entry = cache.get(key);
  if (!entry) return null;
  if (entry.expiresAt && Date.now() > entry.expiresAt) {
    cache.delete(key);
    return null;
  }
  return entry.value;
}

function cacheSet(key, value, ttlSeconds) {
  const ttl = Number(ttlSeconds) > 0 ? Number(ttlSeconds) : 0;
  const expiresAt = ttl > 0 ? Date.now() + ttl * 1000 : null;
  cache.set(key, { value, expiresAt });
}

async function loadFromFile(opts) {
  const modelPath = getModelPath(opts);
  if (!fs.existsSync(modelPath)) return null;
  const raw = fs.readFileSync(modelPath, 'utf8');
  return JSON.parse(raw);
}

async function saveToFile(opts, payload) {
  const modelPath = getModelPath(opts);
  fs.mkdirSync(path.dirname(modelPath), { recursive: true });
  fs.writeFileSync(modelPath, JSON.stringify(payload, null, 2), 'utf8');
}

async function loadFromDb() {
  await ensureDb();
  const row = await db('model_artifacts').select('*').orderBy('created_at', 'desc').first();
  if (!row || !row.payload) return null;
  return JSON.parse(row.payload);
}

async function saveToDb(payload, metadata) {
  await ensureDb();
  await db('model_artifacts').insert({
    payload: JSON.stringify(payload),
    metadata: JSON.stringify(metadata || {})
  });
}

module.exports = {
  async load(opts = {}) {
    const storage = opts.AIWAF_MODEL_STORAGE || 'file';
    const fallback = opts.AIWAF_MODEL_STORAGE_FALLBACK || 'file';
    const cacheKey = getCacheKey(opts);

    if (storage === 'cache') {
      const cached = cacheGet(cacheKey);
      if (cached) return cached;
    }

    try {
      if (storage === 'file') return await loadFromFile(opts);
      if (storage === 'db') return await loadFromDb();
      if (storage === 'cache') return null;
    } catch (err) {
      // fallthrough to fallback
    }

    if (fallback && fallback !== storage) {
      try {
        if (fallback === 'file') return await loadFromFile(opts);
        if (fallback === 'db') return await loadFromDb();
      } catch (err) {
        return null;
      }
    }

    return null;
  },

  async save(opts = {}, payload, metadata = {}) {
    const storage = opts.AIWAF_MODEL_STORAGE || 'file';
    const fallback = opts.AIWAF_MODEL_STORAGE_FALLBACK || 'file';
    const cacheKey = getCacheKey(opts);
    const cacheTtl = opts.AIWAF_MODEL_CACHE_TTL || 0;

    if (storage === 'cache') {
      cacheSet(cacheKey, payload, cacheTtl);
      return;
    }

    try {
      if (storage === 'file') return await saveToFile(opts, payload);
      if (storage === 'db') return await saveToDb(payload, metadata);
    } catch (err) {
      // fallthrough to fallback
    }

    if (fallback && fallback !== storage) {
      if (fallback === 'file') return saveToFile(opts, payload);
      if (fallback === 'db') return saveToDb(payload, metadata);
    }
  },

  cacheGet,
  cacheSet
};
