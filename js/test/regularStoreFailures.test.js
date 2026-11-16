// The regular middleware when its stores fail: no hung requests and no unhandled rejections.
const express = require('express');
const request = require('supertest');
const blocklistManager = require('../lib/blocklistManager');
const requestLogStore = require('../lib/requestLogStore');
const geoBlocker = require('../lib/geoBlocker');
const aiwaf = require('../lib/wafMiddleware');

const unhandled = [];
const onUnhandled = error => unhandled.push(error);
beforeAll(() => process.on('unhandledRejection', onUnhandled));
afterAll(() => process.off('unhandledRejection', onUnhandled));
afterEach(() => jest.restoreAllMocks());

function appWith(options) {
  jest.spyOn(console, 'warn').mockImplementation(() => {});
  const app = express();
  app.use(aiwaf({ MAX_REQ: 1000, FLOOD_REQ: 2000, AIWAF_PATH_MANIFEST: 'missing-manifest.json', ...options }));
  app.all('*', (_req, res) => res.json({ ok: true }));
  app.use((_error, _req, res, _next) => res.status(500).json({ error: 'app error handler' }));
  return app;
}
const boom = async () => { throw new Error('database is locked'); };

test('a failing blocklist read follows AIWAF_RATE_CACHE_FAILURE_MODE instead of hanging', async () => {
  jest.spyOn(blocklistManager, 'isBlocked').mockImplementation(boom);
  expect((await request(appWith({})).get('/items')).status).toBe(503);
  jest.spyOn(blocklistManager, 'isBlocked').mockImplementation(boom);
  expect((await request(appWith({ AIWAF_RATE_CACHE_FAILURE_MODE: 'open' })).get('/items')).status).toBe(200);
  expect(unhandled).toEqual([]);
});

test('a block that can\'t be written still denies the request', async () => {
  jest.spyOn(blocklistManager, 'block').mockImplementation(boom);
  const app = appWith({ AIWAF_METHOD_POLICY_ENABLED: true, AIWAF_ALLOWED_METHODS: ['GET'],
    AIWAF_MIDDLEWARES: ['honeypot'] });
  const response = await request(app).delete('/items');
  expect(response.status).toBeGreaterThanOrEqual(400);
  expect(response.status).toBeLessThan(500);
  expect(console.warn).toHaveBeenCalledWith(expect.stringContaining('blocklist write failed'));
  expect(unhandled).toEqual([]);
});

test('an error anywhere else reaches the app\'s error handler', async () => {
  jest.spyOn(geoBlocker, 'check').mockImplementation(boom);
  const app = appWith({ AIWAF_GEO_BLOCK_ENABLED: true, AIWAF_GEO_BLOCK_COUNTRIES: ['XX'], AIWAF_MIDDLEWARES: ['geo_block'] });
  expect((await request(app).get('/items')).body).toEqual({ error: 'app error handler' });
  expect(unhandled).toEqual([]);
});

test('a failing request log read is treated as no history', async () => {
  jest.spyOn(requestLogStore, 'recent').mockImplementation(boom);
  const app = appWith({ AIWAF_WASM_VALIDATE_RECENT: true, AIWAF_MIDDLEWARES: ['ai_anomaly'] });
  expect((await request(app).get('/items')).status).toBe(200);
  expect(unhandled).toEqual([]);
});

test('route paths with parameters, wildcards and literal dots are matched as Express matches them', async () => {
  // A "*" route made an invalid pattern, which ended the process before errors were caught.
  const app = appWith({ AIWAF_MIDDLEWARES: ['ai_anomaly'] });
  expect((await request(app).get('/anything/at/all')).status).toBe(200);
  expect(unhandled).toEqual([]);
});
