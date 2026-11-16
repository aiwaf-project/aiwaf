// uuidResolver in the native pipeline, against the real WASM engine.
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

const VALID = '550e8400-e29b-41d4-a716-446655440000';
const roots = [];
const handles = [];
const temp = () => { const root = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-native-uuid-')); roots.push(root); return root; };
afterAll(() => {
  for (const handle of handles) handle.close();
  for (const root of roots) fs.rmSync(root, { recursive: true, force: true });
});

async function supported() {
  const probe = await createRequestPipeline({});
  const result = typeof probe.native.supports === 'function' && probe.native.supports('uuid_resolver');
  probe.native.free();
  return result;
}

async function appWith(uuidResolver) {
  const app = express();
  const middleware = await createNative({ dataDir: temp(), manifestPath: path.join(temp(), 'none.json'), uuidResolver,
    profile: { enabled: ['uuid_tamper'] } });
  handles.push(middleware);
  app.use(middleware);
  app.get('/user/:id', (_req, res) => res.json({ ok: true }));
  app.get('/other', (_req, res) => res.json({ ok: true }));
  return app;
}

test('a resolver that says the record is missing blocks, as in the regular middleware', async () => {
  if (!(await supported())) {
    await expect(appWith(async () => true)).rejects.toThrow(/uuidResolver needs a newer aiwaf-wasm/);
    return;
  }
  const calls = [];
  const app = await appWith(async uid => { calls.push(uid); return uid !== VALID; });
  const missing = await request(app).get(`/user/${VALID}`);
  expect([missing.status, missing.body.error]).toEqual([403, 'uuid_not_found']);
  const other = '11111111-2222-3333-4444-555555555555';
  expect((await request(app).get(`/user/${other}`)).status).toBe(200);
  expect(calls).toEqual([VALID, other]);
  // Not asked for other paths or malformed UUIDs; the engine rejects malformed ones itself.
  expect((await request(app).get('/other')).status).toBe(200);
  expect((await request(app).get('/user/not-a-uuid')).status).toBe(403);
  expect(calls).toHaveLength(2);
});

test('sync resolvers work, and a failing resolver lets the request through', async () => {
  if (!(await supported())) return;
  expect((await request(await appWith(() => false)).get(`/user/${VALID}`)).status).toBe(403);
  expect((await request(await appWith(async () => { throw new Error('db down'); })).get(`/user/${VALID}`)).status).toBe(200);
  expect((await request(await appWith(() => { throw new Error('db down'); })).get(`/user/${VALID}`)).status).toBe(200);
});

test('the converter carries uuidResolver and uuidRoutePrefix over', () => {
  const resolver = async () => true;
  const conversion = nativeOptionsFromSettings({ uuidResolver: resolver, uuidRoutePrefix: '/accounts' });
  expect(conversion.options.uuidResolver).toBe(resolver);
  expect(conversion.profile.uuid_prefix).toBe('/accounts/');
  expect(nativeOptionsFromSettings({}).profile.uuid_prefix).toBe('/user/');
  expect(conversion.notes.filter(n => n.outcome === 'carried').map(n => n.setting))
    .toEqual(expect.arrayContaining(['uuidResolver', 'uuidRoutePrefix']));
});

test('a uuidResolver that is not a function is refused', async () => {
  await expect(createNative({ dataDir: temp(), uuidResolver: 'yes' })).rejects.toThrow(/must be a function/);
});
