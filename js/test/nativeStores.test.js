// Host stores in the native pipeline, against the real WASM engine.
if (process.env.AIWAF_NATIVE_TEST_MODULE) {
  jest.doMock(require.resolve('aiwaf-wasm'), () => require(process.env.AIWAF_NATIVE_TEST_MODULE));
}
const fs = require('fs');
const os = require('os');
const path = require('path');
const express = require('express');
const request = require('supertest');
const createNative = require('../lib/nativePipeline');
const { createRequestPipeline } = require('../lib/wasmAdapter');
const { nativeOptionsFromSettings } = require('../lib/nativeSettings');

const roots = [];
const handles = [];
const temp = () => { const root = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-native-stores-')); roots.push(root); return root; };
afterAll(() => {
  for (const handle of handles) handle.close();
  for (const root of roots) fs.rmSync(root, { recursive: true, force: true });
});
const settle = () => new Promise(resolve => setTimeout(resolve, 20));

async function supported() {
  const probe = await createRequestPipeline({});
  const result = typeof probe.native.supports === 'function' && probe.native.supports('host_store');
  probe.native.free();
  return result;
}

// An in-memory store whose methods are async, as a database store's would be.
function memoryStores(seed = {}) {
  const state = { blocked: new Set(seed.blocked || []), exempt: new Set(seed.exempt || []),
    keywords: [...(seed.keywords || [])], countries: [...(seed.countries || [])], blocks: [], learned: [], reads: 0 };
  return {
    state,
    stores: {
      isBlocked: async ip => state.blocked.has(ip),
      isExempted: async ip => state.exempt.has(ip),
      topKeywords: async () => { state.reads++; return state.keywords; },
      blockedCountries: async () => state.countries,
      block: async (ip, reason, expiresAt) => { state.blocks.push([ip, reason, expiresAt]); state.blocked.add(ip); },
      learn: async keyword => { state.learned.push(keyword); state.keywords.push(keyword); }
    }
  };
}

async function appWith(options) {
  const app = express();
  app.set('trust proxy', true);
  const middleware = await createNative({ dataDir: temp(), manifestPath: path.join(temp(), 'none.json'),
    profile: { enabled: ['geo_block', 'ip_keyword_block'] }, ...options });
  handles.push(middleware);
  app.use(middleware);
  app.get('*', (_req, res) => res.status(404).json({ missing: true }));
  return { app, middleware };
}
const from = (app, ip, url) => request(app).get(url).set('X-Forwarded-For', ip);

test('stores decide blocks, exemptions, keywords and countries, and receive new blocks', async () => {
  if (!(await supported())) return;
  const { state, stores } = memoryStores({ blocked: ['203.0.113.1', '203.0.113.2'], exempt: ['203.0.113.2'],
    keywords: ['zzprobe'], countries: ['xx'] });
  const { app } = await appWith({ stores, countryLookup: req => (req.get('x-country') || '') });
  expect((await from(app, '203.0.113.1', '/a')).body.error).toBe('blocklist');
  // An exempt client isn't blocked, and its 404 is the app's.
  expect((await from(app, '203.0.113.2', '/a')).status).toBe(404);
  expect((await from(app, '203.0.113.3', '/x/zzprobe')).body.error).toBe('keyword');
  expect((await from(app, '203.0.113.4', '/a').set('x-country', 'XX')).body.error).toBe('geo_block');
  await settle();
  // Each 403 except a blocklist hit is written to the store, with the profile's expiry.
  expect(state.blocks.map(([ip, reason]) => [ip, reason])).toEqual([['203.0.113.3', 'keyword'], ['203.0.113.4', 'geo_block']]);
  expect(state.blocks[0][2]).toBeGreaterThan(Date.now() / 1000);
  // The store is the authority: the new block applies, and nothing goes to dataDir.
  expect((await from(app, '203.0.113.3', '/a')).body.error).toBe('blocklist');
  // Blocklist answers are cached for storeCacheSeconds, so an unblock elsewhere waits for it.
  state.blocked.delete('203.0.113.3');
  expect((await from(app, '203.0.113.3', '/a')).body.error).toBe('blocklist');
});

test('only the facts a request uses are read, and answers are cached briefly', async () => {
  if (!(await supported())) return;
  const reads = [];
  const { state, stores } = memoryStores({ exempt: ['203.0.113.50'] });
  for (const name of ['isBlocked', 'isExempted', 'topKeywords', 'blockedCountries']) {
    const original = stores[name];
    stores[name] = (...args) => { reads.push(name); return original(...args); };
  }
  const { app } = await appWith({ stores, storeCacheSeconds: 60, pathRules: [{ PREFIX: '/static/', DISABLE: ['ip_keyword_block', 'geo_block'] }],
    profile: { enabled: ['geo_block', 'ip_keyword_block', 'logging'] } });
  await from(app, '203.0.113.51', '/a');
  expect(reads.sort()).toEqual(['blockedCountries', 'isBlocked', 'isExempted', 'topKeywords']);
  reads.length = 0;
  // Cached: the same client and path read nothing again.
  await from(app, '203.0.113.51', '/a');
  expect(reads).toEqual([]);
  // Only logging runs here, so nothing is read.
  await from(app, '203.0.113.52', '/static/app.css');
  expect(reads).toEqual([]);
  // An exempt request runs logging alone, so only the exemption is read.
  await from(app, '203.0.113.50', '/a');
  expect(reads).toEqual(['isExempted']);
  expect(state.blocks).toEqual([]);
});

test('the anomaly stage sees stored blocks too', async () => {
  if (!(await supported())) return;
  const { stores } = memoryStores({ blocked: ['203.0.113.55'] });
  const { app } = await appWith({ stores, profile: { enabled: ['ai_anomaly'] } });
  expect((await from(app, '203.0.113.55', '/a')).body.error).toBe('blocklist');
});

test('cached answers are bounded, and storeCacheSeconds 0 keeps none', async () => {
  const { recentAnswers } = require('../lib/nativePipeline');
  const answers = recentAnswers(60, 2);
  for (const key of ['a', 'b', 'c']) await answers.get(key, () => true);
  expect([...answers.items.keys()]).toEqual(['b', 'c']);
  expect(await answers.get('b', () => false)).toBe(true);
  expect([...answers.items.keys()]).toEqual(['c', 'b']);
  const off = recentAnswers(0);
  off.set('a', true);
  expect(await off.get('a', () => false)).toBe(false);
  expect(off.items.size).toBe(0);
});

test('keywords are cached between reads, and learning refreshes them', async () => {
  if (!(await supported())) return;
  const { state, stores } = memoryStores();
  const { app } = await appWith({ stores, storeCacheSeconds: 60, routeExists: () => false,
    pathRules: [{ PREFIX: '/learn/', DISABLE: ['ip_keyword_block'] }],
    profile: { enabled: ['ip_keyword_block', 'ai_anomaly'], learning: true } });
  expect((await from(app, '203.0.113.10', '/a')).status).toBe(404);
  expect((await from(app, '203.0.113.11', '/b')).status).toBe(404);
  expect(state.reads).toBe(1);
  // A 404 probe is learned and written to the store, which clears the cached keywords.
  expect((await from(app, '203.0.113.12', '/learn/backup.php.bak')).status).toBe(404);
  await settle();
  expect(state.learned).toEqual(['backup.php.bak']);
  expect((await from(app, '203.0.113.13', '/c')).status).toBe(404);
  expect(state.reads).toBe(2);
});

test('a failing store read is reported and lets the request through', async () => {
  if (!(await supported())) return;
  const errors = [];
  const { stores } = memoryStores();
  stores.isBlocked = async () => { throw new Error('db down'); };
  stores.topKeywords = () => { throw new Error('db down'); };
  const { app } = await appWith({ stores, onError: error => errors.push(error.message) });
  expect((await from(app, '203.0.113.20', '/a')).status).toBe(404);
  expect(errors).toEqual(['db down', 'db down']);
  // A failed keyword read isn't cached.
  expect((await from(app, '203.0.113.20', '/a')).status).toBe(404);
  expect(errors).toHaveLength(4);
});

test('a failing store write is reported', async () => {
  if (!(await supported())) return;
  const errors = [];
  const { stores } = memoryStores({ keywords: ['zzprobe'] });
  stores.block = async () => { throw new Error('write failed'); };
  const { app } = await appWith({ stores, onError: error => errors.push(error.message) });
  expect((await from(app, '203.0.113.30', '/zzprobe')).status).toBe(403);
  await settle();
  expect(errors).toEqual(['write failed']);
});

test('stores options are checked at startup', async () => {
  await expect(createNative({ stores: { isBlocked() {} } })).rejects.toThrow(/stores is missing isExempted/);
  await expect(createNative({ stores: 5 })).rejects.toThrow(/stores must be/);
  await expect(createNative({ storesFallback: 'files' })).rejects.toThrow(/storesFallback must be "csv"/);
  await expect(createNative({ storeCacheSeconds: -1 })).rejects.toThrow(/storeCacheSeconds/);
});

describe('the regular stores', () => {
  const blocklistManager = require('../lib/blocklistManager');
  const exemptionStore = require('../lib/exemptionStore');
  const dynamicKeywordStore = require('../lib/dynamicKeywordStore');
  const geoStore = require('../lib/geoStore');

  test('hold the same blocks, exemptions, keywords and countries as the regular middleware', async () => {
    if (!(await supported())) return;
    await blocklistManager.block('203.0.113.40', 'manual', { checkExemptions: false });
    await exemptionStore.addIp('203.0.113.41', 'test');
    for (let i = 0; i < 11; i++) await dynamicKeywordStore.increment('zzlearned');
    await dynamicKeywordStore.increment('zzonce');
    await geoStore.addBlockedCountry('YY');
    const { app } = await appWith({ ...nativeOptionsFromSettings({}).options, profile: { enabled: ['geo_block', 'ip_keyword_block'] },
      countryLookup: req => (req.get('x-country') || '') });
    expect((await from(app, '203.0.113.40', '/a')).body.error).toBe('blocklist');
    expect((await from(app, '203.0.113.42', '/x/zzlearned')).body.error).toBe('keyword');
    // Below dynamicTopN, a learned keyword doesn't apply yet, as in the regular middleware.
    expect((await from(app, '203.0.113.43', '/x/zzonce')).status).toBe(404);
    expect((await from(app, '203.0.113.44', '/a').set('x-country', 'YY')).body.error).toBe('geo_block');
    await settle();
    expect(await blocklistManager.isBlocked('203.0.113.42')).toBe(true);
    expect((await from(app, '203.0.113.41', '/x/zzlearned')).status).toBe(404);
  });

  test('learn into the regular keyword store', async () => {
    if (!(await supported())) return;
    const { app } = await appWith({ stores: 'regular', routeExists: () => false, profile: { enabled: ['ai_anomaly'], learning: true } });
    expect((await from(app, '203.0.113.45', '/old/zzstore.php')).status).toBe(404);
    await settle();
    expect((await dynamicKeywordStore.list(2000)).map(row => row.keyword)).toContain('zzstore.php');
  });

  test('skip the exemption store when AIWAF_EXEMPTIONS_DB is off', async () => {
    const stores = nativeOptionsFromSettings({ AIWAF_EXEMPTIONS_DB: false }).options.stores;
    expect(await stores.isExempted('203.0.113.41', '/')).toBe(false);
  });
});
