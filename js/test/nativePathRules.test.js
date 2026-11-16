// Path rules and route checks in the native pipeline, against the real WASM engine.
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

const ua = 'Mozilla/5.0 Chrome/130.0 Safari/537.36';
const roots = [];
const handles = [];
const temp = () => { const root = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-native-rules-')); roots.push(root); return root; };
const browser = (req, ip = '203.0.113.10') => req.set('User-Agent', ua).set('Accept', 'text/html')
  .set('Accept-Language', 'en').set('Accept-Encoding', 'gzip').set('Connection', 'keep-alive').set('X-Forwarded-For', ip);

async function appWith(profile, options = {}) {
  const app = express();
  app.set('trust proxy', 'loopback');
  const middleware = await createNative({ dataDir: temp(), manifestPath: path.join(temp(), 'missing.json'), profile, ...options });
  handles.push(middleware);
  app.use(middleware);
  for (const route of ['/health/live', '/api/items', '/home', '/files']) app.get(route, (_req, res) => res.json({ ok: true }));
  return app;
}

afterAll(() => {
  for (const handle of handles) handle.close();
  for (const root of roots) fs.rmSync(root, { recursive: true, force: true });
});

test('DISABLE turns stages off for a path prefix', async () => {
  const app = await appWith({ enabled: ['header_validation'] },
    { pathRules: [{ PREFIX: '/health/', DISABLE: ['header_validation'] }] });
  expect((await request(app).get('/health/live')).status).toBe(200);
  expect((await request(app).get('/home')).status).toBe(403);
});

test('RATE_LIMIT overrides the limits for a path prefix', async () => {
  const pathRules = [{ PREFIX: '/api/', RATE_LIMIT: { MAX: 1, FLOOD: 5 } }];
  const probe = await createRequestPipeline({});
  const supported = typeof probe.native.supports === 'function' && probe.native.supports('rate_overrides');
  probe.native.free();
  if (!supported) {
    // An older published aiwaf-wasm: the adapter must refuse at startup, not fail per request.
    await expect(appWith({ enabled: ['rate_limit'] }, { pathRules })).rejects.toThrow(/needs a newer aiwaf-wasm/);
    return;
  }
  const app = await appWith({ enabled: ['rate_limit'] }, { pathRules });
  expect([(await browser(request(app).get('/api/items'))).status, (await browser(request(app).get('/api/items'))).status])
    .toEqual([200, 429]);
  expect([(await browser(request(app).get('/home'))).status, (await browser(request(app).get('/home'))).status])
    .toEqual([200, 200]);
});

test('invalid path rules fail at startup', async () => {
  await expect(createNative({ dataDir: temp(), pathRules: { PREFIX: '/x/' } })).rejects.toThrow(/array/);
  await expect(createNative({ dataDir: temp(), pathRules: [{ PREFIX: '/x/', RATE_LIMIT: { WINDOW: 0 } }] }))
    .rejects.toThrow(/Invalid RATE_LIMIT/);
  await expect(createNative({ dataDir: temp(), pathRules: [{ PREFIX: '/x/', RATE_LIMIT: { MAX: 'many' } }] }))
    .rejects.toThrow(/must be a number/);
});

test('manifestPath and injectionRiskMode control route checks', async () => {
  const manifestPath = path.join(temp(), 'paths.json');
  fs.writeFileSync(manifestPath, JSON.stringify({ routes: { '/files': { injection_risks: [{ sink: 'path', input: 'name', confidence: 'high' }] } } }));
  const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
  try {
    const blocking = await appWith({ enabled: ['ip_keyword_block'] }, { manifestPath });
    const blocked = await browser(request(blocking).get('/files?name=../../etc/passwd'));
    expect([blocked.status, blocked.body.error]).toEqual([403, 'path_traversal']);
    expect((await browser(request(blocking).get('/files?name=report.pdf'))).status).toBe(200);
    const logging = await appWith({ enabled: ['ip_keyword_block'] }, { manifestPath, injectionRiskMode: 'log' });
    expect((await browser(request(logging).get('/files?name=../../etc/passwd'))).status).toBe(200);
    // With the keyword stage turned off for the path, the route check doesn't run either.
    const disabled = await appWith({ enabled: ['ip_keyword_block'] },
      { manifestPath, pathRules: [{ PREFIX: '/files', DISABLE: ['ip_keyword_block'] }] });
    expect((await browser(request(disabled).get('/files?name=../x'))).status).toBe(200);
  } finally {
    warn.mockRestore();
  }
  await expect(createNative({ dataDir: temp(), manifestPath, injectionRiskMode: 'loud' })).rejects.toThrow(/block, log, or off/);
});

test('a limit above the flood threshold acts as the threshold, as in the regular middleware', async () => {
  // MAX above FLOOD in a rule, and a rule MAX above the profile's default flood of 40.
  for (const [rule, flood] of [[{ MAX: 9, FLOOD: 3 }, 3], [{ WINDOW: 60, MAX: 100 }, 40]]) {
    const app = await appWith({ enabled: ['rate_limit'] }, { pathRules: [{ PREFIX: '/', RATE_LIMIT: rule }] });
    const statuses = [];
    for (let i = 0; i <= flood; i++) statuses.push((await browser(request(app).get('/home'))).status);
    expect(statuses.slice(-2)).toEqual([200, 403]);
  }
  const app = await appWith({ enabled: ['rate_limit'], rate_max: 100 }, {});
  const statuses = [];
  for (let i = 0; i <= 40; i++) statuses.push((await browser(request(app).get('/home'))).status);
  expect(statuses.slice(-2)).toEqual([200, 403]);
});
