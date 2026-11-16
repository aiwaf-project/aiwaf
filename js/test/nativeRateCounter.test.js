// Shared rate counts in the native pipeline, against the real WASM engine.
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
const { redisRateCounter, boundedKey } = require('../lib/nativeRateCounter');
const { nativeOptionsFromSettings } = require('../lib/nativeSettings');

const roots = [];
const handles = [];
const temp = () => { const root = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-native-rate-')); roots.push(root); return root; };
afterAll(() => {
  for (const handle of handles) handle.close();
  for (const root of roots) fs.rmSync(root, { recursive: true, force: true });
});

async function supported() {
  const probe = await createRequestPipeline({});
  const result = typeof probe.native.supports === 'function' && probe.native.supports('rate_count');
  probe.native.free();
  return result;
}

// One sliding window per key, shared by every pipeline given it, as Redis is.
function sharedCounter() {
  const windows = new Map();
  const calls = [];
  return {
    calls,
    async count(key, now, windowSeconds) {
      calls.push([key, windowSeconds]);
      const current = (windows.get(key) || []).filter(t => now - t < windowSeconds).concat(now);
      windows.set(key, current);
      return current.length;
    }
  };
}

async function worker(rateCounter, options = {}) {
  const app = express();
  app.set('trust proxy', true);
  const middleware = await createNative({ dataDir: temp(), manifestPath: path.join(temp(), 'none.json'), rateCounter,
    ...options, profile: { enabled: ['rate_limit'], rate_max: 2, rate_flood: 4, ...options.profile } });
  handles.push(middleware);
  app.use(middleware);
  app.get('*', (_req, res) => res.json({ ok: true }));
  return app;
}
const from = (app, url, ip = '198.51.100.20') => request(app).get(url).set('X-Forwarded-For', ip);

test('workers sharing a counter share one limit', async () => {
  if (!(await supported())) return;
  const counter = sharedCounter();
  const workers = [await worker(counter), await worker(counter)];
  const statuses = [];
  for (let i = 0; i < 5; i++) statuses.push((await from(workers[i % 2], '/api')).status);
  // Two requests pass in total, not two per worker; the fifth is a flood.
  expect(statuses).toEqual([200, 200, 429, 429, 403]);
  expect(counter.calls[0]).toEqual(['ratelimit:198.51.100.20:/api', 10]);
});

test('path-rule windows and the ip key mode reach the counter, and exempt requests are not counted', async () => {
  if (!(await supported())) return;
  const counter = sharedCounter();
  const app = await worker(counter, { pathRules: [{ PREFIX: '/api/', RATE_LIMIT: { WINDOW: 30 } }],
    profile: { rate_key_mode: 'ip', exempt_ips: ['198.51.100.99'] } });
  await from(app, '/api/x');
  await from(app, '/HEALTH');
  await from(app, '/a', '198.51.100.99');
  expect(counter.calls).toEqual([['ratelimit:198.51.100.20', 30]]);
});

test('a failing counter answers 503 by default, or lets requests through when open', async () => {
  if (!(await supported())) return;
  const errors = [];
  const broken = { count: async () => { throw new Error('redis down'); } };
  const closed = await worker(broken, { onError: error => errors.push(error.message) });
  expect((await from(closed, '/api')).status).toBe(503);
  const open = await worker(broken, { rateFailureMode: 'open', onError: error => errors.push(error.message) });
  expect((await from(open, '/api')).status).toBe(200);
  expect(errors).toEqual(['redis down', 'redis down']);
});

test('rate counter options are checked at startup', async () => {
  await expect(createNative({ rateCounter: {} })).rejects.toThrow(/rateCounter must have a count/);
  await expect(createNative({ rateCounterFallback: 'csv' })).rejects.toThrow(/rateCounterFallback must be "memory"/);
  await expect(createNative({ rateFailureMode: 'maybe' })).rejects.toThrow(/rateFailureMode must be/);
});

describe('redisRateCounter', () => {
  // Runs the Lua sliding window the way Redis would.
  function fakeRedis() {
    const values = new Map();
    const evalScript = ([command, , , key, now, window]) => {
      expect(command).toBe('EVAL');
      const current = JSON.parse(values.get(key) || '[]').filter(t => Number(now) - t < Number(window)).concat(Number(now));
      values.set(key, JSON.stringify(current));
      return current.length;
    };
    return { values, evalScript };
  }

  test('counts with node-redis and ioredis clients, under bounded keys', async () => {
    for (const shape of ['sendCommand', 'call']) {
      const redis = fakeRedis();
      const client = shape === 'call' ? { call: async (...args) => redis.evalScript(args) } : { sendCommand: async args => redis.evalScript(args) };
      const counter = redisRateCounter(client);
      expect(await counter.count('ratelimit:1.2.3.4:/x', 100, 10)).toBe(1);
      expect(await counter.count('ratelimit:1.2.3.4:/x', 101, 10)).toBe(2);
      expect([...redis.values.keys()]).toEqual([boundedKey('aiwaf:native:rate:v1', 'ratelimit:1.2.3.4:/x')]);
      expect([...redis.values.keys()][0]).toHaveLength('aiwaf:native:rate:v1:'.length + 64);
    }
  });

  test('rejects clients and namespaces it cannot use, and bad replies', async () => {
    expect(() => redisRateCounter({})).toThrow(/node-redis or ioredis/);
    expect(() => redisRateCounter({ call() {} }, { namespace: '' })).toThrow(/namespace/);
    await expect(redisRateCounter({ call: async () => 'OK' }).count('k', 1, 1)).rejects.toThrow(/Invalid Redis rate count/);
  });
});

describe('the converter', () => {
  test('carries a Redis cache and the failure mode', () => {
    const conversion = nativeOptionsFromSettings({ cache: { sendCommand: async () => 1 }, AIWAF_RATE_CACHE_FAILURE_MODE: 'open' });
    expect(typeof conversion.options.rateCounter.count).toBe('function');
    expect(conversion.options).toMatchObject({ rateCounterFallback: 'memory', rateFailureMode: 'open' });
    const notes = Object.fromEntries(conversion.notes.map(n => [n.setting, n.outcome]));
    expect(notes).toMatchObject({ cache: 'carried', AIWAF_RATE_CACHE_FAILURE_MODE: 'carried' });
  });

  test('reports a cache that is not a Redis client', () => {
    const conversion = nativeOptionsFromSettings({ cache: { lPush() {}, expire() {}, lLen() {}, lRange() {} } });
    expect(conversion.options.rateCounter).toBeUndefined();
    expect(conversion.differences().map(n => n.setting)).toContain('cache');
  });
});
