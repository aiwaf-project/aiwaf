// aiwaf(options) runs the native pipeline by default when every option carries over and the
// installed aiwaf-wasm has every native feature.
if (process.env.AIWAF_NATIVE_TEST_MODULE) {
  jest.doMock(require.resolve('aiwaf-wasm'), () => require(process.env.AIWAF_NATIVE_TEST_MODULE));
}
const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');
const express = require('express');
const request = require('supertest');
const aiwaf = require('../lib/wafMiddleware');
const createNative = require('../lib/nativePipeline');
const { createIsolationForest } = require('../lib/wasmAdapter');
const { nativeOptionsFromSettings, settingsModel, SHIPPED_MODEL_SHA256 } = require('../lib/nativeSettings');

const roots = [];
const handles = [];
const temp = () => { const root = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-native-default-')); roots.push(root); return root; };
const saved = process.env.AIWAF_NATIVE_PIPELINE;
afterEach(() => {
  jest.restoreAllMocks();
  if (saved === undefined) delete process.env.AIWAF_NATIVE_PIPELINE; else process.env.AIWAF_NATIVE_PIPELINE = saved;
});
afterAll(() => {
  for (const handle of handles) handle.close();
  for (const root of roots) fs.rmSync(root, { recursive: true, force: true });
});

// Reports this aiwaf-wasm version as installed, leaving every other file read alone.
function installed(version) {
  const read = fs.readFileSync;
  jest.spyOn(fs, 'readFileSync').mockImplementation((file, ...rest) => (
    String(file).endsWith(path.join('aiwaf-wasm', 'package.json')) ? JSON.stringify({ version }) : read.call(fs, file, ...rest)));
}
const regular = middleware => middleware.ready === undefined;

describe('the automatic choice', () => {
  beforeEach(() => { delete process.env.AIWAF_NATIVE_PIPELINE; jest.spyOn(console, 'info').mockImplementation(() => {}); });

  test('is native when every option carries over and aiwaf-wasm is new enough', async () => {
    installed(aiwaf.NATIVE_DEFAULT_WASM);
    const middleware = aiwaf({ MAX_REQ: 1, FLOOD_REQ: 5, AIWAF_PATH_MANIFEST: path.join(temp(), 'none.json') });
    expect(regular(middleware)).toBe(false);
    const native = await middleware.ready;
    handles.push(native);
    expect(console.info).toHaveBeenCalledWith(expect.stringContaining('AIWAF_NATIVE_PIPELINE: false'));
    const app = express();
    app.use(middleware);
    app.get('/items', (_req, res) => res.json({ ok: true }));
    expect([(await request(app).get('/items')).status, (await request(app).get('/items')).status]).toEqual([200, 429]);
  });

  test('stays regular when an option would not carry over', () => {
    installed('9.0.0');
    expect(regular(aiwaf({ AIWAF_LOGIN_PATH_PREFIXES: ['/signin'] }))).toBe(true);
    expect(console.info).not.toHaveBeenCalled();
  });

  test('stays regular with an older or missing aiwaf-wasm', () => {
    installed('0.2.6');
    expect(regular(aiwaf({}))).toBe(true);
    jest.restoreAllMocks();
    const read = fs.readFileSync;
    jest.spyOn(fs, 'readFileSync').mockImplementation((file, ...rest) => {
      if (String(file).endsWith(path.join('aiwaf-wasm', 'package.json'))) throw new Error('ENOENT');
      return read.call(fs, file, ...rest);
    });
    expect(regular(aiwaf({}))).toBe(true);
  });

  test('can be turned off in options or the environment', () => {
    installed('9.0.0');
    expect(regular(aiwaf({ AIWAF_NATIVE_PIPELINE: false }))).toBe(true);
    for (const value of ['false', '0', 'off', 'NO']) {
      process.env.AIWAF_NATIVE_PIPELINE = value;
      expect(regular(aiwaf({}))).toBe(true);
    }
  });

  test('compares versions numerically', () => {
    for (const [version, expected] of [['0.2.10', true], ['0.3.0', true], ['1.0.0-rc.1', true], ['0.2.6', false], ['0.1.99', false]]) {
      installed(version);
      const conversion = nativeOptionsFromSettings({});
      expect(conversion.differences()).toEqual([]);
      expect(regular(aiwaf({}))).toBe(!expected);
      jest.restoreAllMocks();
      jest.spyOn(console, 'info').mockImplementation(() => {});
    }
  });
});

async function supports(feature) {
  const probe = await require('../lib/wasmAdapter').createRequestPipeline({});
  const result = typeof probe.native.supports === 'function' && probe.native.supports(feature);
  probe.native.free();
  return result;
}

describe('client addresses', () => {
  // The rate counter's key shows the address the pipeline resolved.
  async function keysFor(options) {
    const keys = [];
    const app = express();
    const middleware = await createNative({ dataDir: temp(), manifestPath: path.join(temp(), 'none.json'), ...options,
      rateCounter: { count: async key => { keys.push(key); return 1; } }, profile: { enabled: ['rate_limit'], rate_key_mode: 'ip' } });
    handles.push(middleware);
    app.use(middleware);
    app.get('*', (_req, res) => res.json({ ok: true }));
    await request(app).get('/a').set('X-Forwarded-For', '203.0.113.9');
    return keys;
  }

  test('trustedProxies resolves the client as the regular middleware does', async () => {
    if (!(await supports('rate_count'))) return;
    // The test client connects from loopback, a default trusted proxy.
    expect(await keysFor({ trustedProxies: 'default' })).toEqual(['ratelimit:203.0.113.9']);
    // Without it, req.ip: Express doesn't trust proxies by default.
    expect((await keysFor({}))[0]).toMatch(/^ratelimit:(127\.0\.0\.1|::1)$/);
  });

  test('the converter carries AIWAF_TRUSTED_PROXIES, with the regular default', () => {
    expect(nativeOptionsFromSettings({}).options.trustedProxies).toBe('default');
    const conversion = nativeOptionsFromSettings({ AIWAF_TRUSTED_PROXIES: ['10.1.2.3'] });
    expect(conversion.options.trustedProxies).toEqual(['10.1.2.3']);
    expect(conversion.notes.find(n => n.setting === 'AIWAF_TRUSTED_PROXIES').outcome).toBe('carried');
  });
});

describe('the anomaly model', () => {
  async function wasmModelFile() {
    const model = await createIsolationForest({ nTrees: 10, sampleSize: 16 });
    model.fit(Array.from({ length: 32 }, (_, i) => [i % 7, 0, 0, 10, 1, 0]));
    const file = path.join(temp(), 'model.json');
    fs.writeFileSync(file, JSON.stringify({ model_type: 'aiwaf_wasm.IsolationForest', model_state: model.toJSON(), metadata: {} }));
    return file;
  }

  test('the shipped sample model is recognised by its checksum', () => {
    const shipped = fs.readFileSync(path.join(__dirname, '..', 'resources', 'model.json'));
    expect(crypto.createHash('sha256').update(shipped).digest('hex')).toBe(SHIPPED_MODEL_SHA256);
    expect(settingsModel({}).kind).toBe('shipped');
    expect(nativeOptionsFromSettings({}).options.modelFile).toBeUndefined();
  });

  test('a model saved by aiwaf train is carried and loaded', async () => {
    const file = await wasmModelFile();
    const conversion = nativeOptionsFromSettings({ AIWAF_MODEL_PATH: file });
    expect(conversion.options.modelFile).toBe(file);
    expect(conversion.differences()).toEqual([]);
    const middleware = await createNative({ dataDir: temp(), manifestPath: path.join(temp(), 'none.json'), modelFile: file });
    handles.push(middleware);
    expect(middleware.diagnostics().model).toBe(true);
  });

  test('models the native pipeline cannot load are reported', async () => {
    const dir = temp();
    const old = path.join(dir, 'old.json');
    fs.writeFileSync(old, JSON.stringify({ nTrees: 1, trees: [] }));
    const broken = path.join(dir, 'broken.json');
    fs.writeFileSync(broken, '{');
    for (const file of [old, broken]) {
      const note = nativeOptionsFromSettings({ AIWAF_MODEL_PATH: file }).notes.find(n => n.setting === 'AIWAF_MODEL_PATH');
      expect([note.outcome, note.detail]).toEqual(['not carried', expect.stringContaining('aiwaf train')]);
    }
    await expect(createNative({ dataDir: temp(), modelFile: old })).rejects.toThrow(/Retrain with: aiwaf train/);
    await expect(createNative({ modelFile: 5 })).rejects.toThrow(/modelFile must be a path/);
    expect(settingsModel({ AIWAF_MODEL_STORAGE: 'db' }).kind).toBe('storage');
    const missing = nativeOptionsFromSettings({ AIWAF_MODEL_PATH: path.join(dir, 'none.json') });
    expect(missing.notes.find(n => n.setting === 'AIWAF_MODEL_PATH').outcome).toBe('carried');
  });
});
