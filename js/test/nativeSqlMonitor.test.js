// SQL monitor mode in the native pipeline, against the real WASM engine.
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

const roots = [];
const handles = [];
const temp = () => { const root = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-native-monitor-')); roots.push(root); return root; };

afterAll(() => {
  for (const handle of handles) handle.close();
  for (const root of roots) fs.rmSync(root, { recursive: true, force: true });
});

async function supported() {
  const probe = await createRequestPipeline({});
  const result = typeof probe.native.supports === 'function' && probe.native.supports('sql_monitor');
  probe.native.free();
  return result;
}

async function appWith(sqlMode, logger) {
  const app = express();
  app.use(express.json());
  const middleware = await createNative({ dataDir: temp(), manifestPath: path.join(temp(), 'missing.json'), logger,
    profile: { enabled: ['ip_keyword_block'], sql_mode: sqlMode } });
  handles.push(middleware);
  app.use(middleware);
  app.get('/search', (_req, res) => res.json({ ok: true }));
  return { app, middleware };
}

test('monitor mode logs SQL findings and lets the request through', async () => {
  const logger = { warn: jest.fn() };
  if (!(await supported())) {
    // An older published aiwaf-wasm: refuse at startup with an upgrade message.
    await expect(appWith('monitor', logger)).rejects.toThrow(/needs a newer aiwaf-wasm/);
    return;
  }
  const { app, middleware } = await appWith('monitor', logger);
  expect((await request(app).get("/search?q=%27%20OR%201%3D1--")).status).toBe(200);
  expect(logger.warn).toHaveBeenCalledWith({ event: 'aiwaf.payload', rule: 'sql_breakout_single_quote', mode: 'monitor' });
  logger.warn.mockClear();
  // Framework plugins block on any finding, so a monitored one comes back as null.
  expect(middleware.inspectPayload({ url: '/search', headers: {}, body: { q: "' OR 1=1--" } })).toBeNull();
  expect(logger.warn).toHaveBeenCalledWith(expect.objectContaining({ event: 'aiwaf.payload', mode: 'monitor' }));
  logger.warn.mockClear();
  expect((await request(app).get('/search?q=hello')).status).toBe(200);
  expect(logger.warn).not.toHaveBeenCalled();
});

test('block mode still blocks the same request', async () => {
  const { app, middleware } = await appWith('block', { warn: jest.fn() });
  expect((await request(app).get("/search?q=%27%20OR%201%3D1--")).status).toBe(403);
  expect(middleware.inspectPayload({ url: '/search', headers: {}, body: { q: "' OR 1=1--" } }).status).toBe(403);
});
