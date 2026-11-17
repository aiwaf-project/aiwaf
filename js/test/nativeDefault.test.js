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
    expect(regular(aiwaf({ AIWAF_FORCE_JSON_ERRORS: false }))).toBe(true);
    // The reason is said, once however many times the app builds the middleware.
    expect(regular(aiwaf({ AIWAF_FORCE_JSON_ERRORS: false }))).toBe(true);
    expect(console.info).toHaveBeenCalledTimes(1);
    expect(console.info).toHaveBeenCalledWith(expect.stringMatching(
      /using the regular middleware, not the native pipeline, because[^]*AIWAF_FORCE_JSON_ERRORS: not carried/));
  });

  test('stays regular with an older or missing aiwaf-wasm', () => {
    installed('0.2.6');
    expect(regular(aiwaf({}))).toBe(true);
    expect(console.info).toHaveBeenCalledWith(expect.stringContaining('the installed aiwaf-wasm 0.2.6 is older than'));
    jest.restoreAllMocks();
    jest.spyOn(console, 'info').mockImplementation(() => {});
    const read = fs.readFileSync;
    jest.spyOn(fs, 'readFileSync').mockImplementation((file, ...rest) => {
      if (String(file).endsWith(path.join('aiwaf-wasm', 'package.json'))) throw new Error('ENOENT');
      return read.call(fs, file, ...rest);
    });
    expect(regular(aiwaf({}))).toBe(true);
    expect(console.info).toHaveBeenCalledWith(expect.stringContaining("aiwaf-wasm isn't installed"));
  });

  test('header caps, blocked user agents and login forms carry over', async () => {
    installed('9.0.0');
    const browser = { 'user-agent': 'Mozilla/5.0 Chrome/130 Safari/537.36', accept: 'text/html',
      'accept-language': 'en', 'accept-encoding': 'gzip', connection: 'keep-alive' };
    let client = 0;
    const app = async options => {
      const middleware = aiwaf({ AIWAF_HEADER_VALIDATION: true, AIWAF_PATH_MANIFEST: path.join(temp(), 'none.json'),
        MAX_REQ: 1000, FLOOD_REQ: 2000, ...options });
      expect(regular(middleware)).toBe(false);
      handles.push(await middleware.ready);
      const server = express();
      server.use(express.urlencoded({ extended: false }));
      server.use(middleware);
      server.use((_req, res) => res.json({ ok: true }));
      return (method, url, headers = {}) => request(server)[method](url)
        .set({ ...browser, 'x-forwarded-for': `192.0.2.${++client}`, ...headers });
    };
    // A long User-Agent the app allows, and its own blocked agents alongside the regular defaults.
    let send = await app({ AIWAF_MAX_USER_AGENT_LENGTH: 1000, AIWAF_BLOCKED_USER_AGENTS: ['ExampleBrowser'] });
    expect((await send('get', '/docs', { 'user-agent': `Mozilla/5.0 Chrome/130 ${'a'.repeat(700)}` })).status).toBe(200);
    expect((await send('get', '/docs', { 'user-agent': 'Mozilla/5.0 examplebrowser/2' })).status).toBe(403);
    // The regular defaults block scanners such as sqlmap.
    send = await app({});
    expect((await send('get', '/docs', { 'user-agent': 'Mozilla/5.0 (sqlmap/1.7)' })).status).toBe(403);
    expect((await send('get', '/docs', { 'user-agent': `Mozilla/5.0 Chrome/130 ${'a'.repeat(700)}` })).status).toBe(403);
    // Header bytes are counted as Node.js names the headers, as in the regular check: a limit of
    // exactly the request's bytes passes, one byte less blocks.
    const echo = express();
    echo.use((req, res) => res.json(req.headers));
    const sent = (await request(echo).get('/docs').set({ ...browser, 'x-forwarded-for': '192.0.2.200' })).body;
    // The regular check counts the protocol too, as server-protocol.
    const bytes = Object.entries({ ...sent, 'server-protocol': 'HTTP/1.1' })
      .reduce((total, [key, value]) => total + key.length + String(value).length, 0);
    send = await app({ AIWAF_MAX_HEADER_BYTES: bytes });
    expect((await send('get', '/docs', { 'x-forwarded-for': '192.0.2.200' })).status).toBe(200);
    send = await app({ AIWAF_MAX_HEADER_BYTES: bytes - 1 });
    expect((await send('get', '/docs', { 'x-forwarded-for': '192.0.2.201' })).status).toBe(403);
    // A form posted 0.2 s after it was fetched is too fast, except under the app's login prefixes,
    // where 0.1 s is enough, as in the regular check.
    send = await app({ AIWAF_MIN_FORM_TIME: 5, AIWAF_HEADER_VALIDATION: false, AIWAF_LOGIN_PATH_PREFIXES: ['/members/signin/'] });
    const post = async url => {
      const headers = { 'x-forwarded-for': `192.0.2.${100 + ++client}` };
      await send('get', url, headers);
      await new Promise(resolve => setTimeout(resolve, 200));
      return (await send('post', url, headers).type('form').send('a=1')).status;
    };
    expect(await post('/members/signin/')).toBe(200);
    expect(await post('/login/')).toBe(403);
    // Unset, default login prefixes relax timing; an explicit empty list disables that.
    send = await app({ AIWAF_MIN_FORM_TIME: 5, AIWAF_HEADER_VALIDATION: false });
    expect(await post('/login/')).toBe(200);
    send = await app({ AIWAF_MIN_FORM_TIME: 5, AIWAF_HEADER_VALIDATION: false, AIWAF_LOGIN_PATH_PREFIXES: [] });
    expect(await post('/login/')).toBe(403);
  });

  test('chosen automatically, it runs only on an engine that answers exactly as the regular middleware', async () => {
    const wasm = require('aiwaf-wasm');
    const supports = wasm.NativeRequestPipeline.prototype.supports;
    jest.spyOn(wasm.NativeRequestPipeline.prototype, 'supports')
      .mockImplementation(function (feature) { return feature !== 'header_policy' && supports.call(this, feature); });
    jest.spyOn(console, 'warn').mockImplementation(() => {});
    const middleware = aiwaf({ AIWAF_PATH_MANIFEST: path.join(temp(), 'none.json') });
    expect((await middleware.ready).diagnostics()).toMatchObject({ native: false });
    expect(console.warn).toHaveBeenCalledWith(expect.stringContaining('need a newer aiwaf-wasm'));
  });

  test('an older aiwaf-wasm runs without them only at the regular defaults', async () => {
    const { LEGACY_DEFAULTS } = require('../lib/nativeSettings');
    const converted = nativeOptionsFromSettings({});
    for (const key of ['header_bytes_without_prefix', 'blocked_user_agents', 'login_prefixes']) {
      expect(converted.profile[key]).toEqual(LEGACY_DEFAULTS[key][1]);
    }
    const wasm = require('aiwaf-wasm');
    const supports = wasm.NativeRequestPipeline.prototype.supports;
    jest.spyOn(wasm.NativeRequestPipeline.prototype, 'supports')
      .mockImplementation(function (feature) {
        return !['header_limits', 'login_prefixes', 'header_policy', 'keyword_exemptions', 'post_only_suffixes', 'method_headers']
          .includes(feature) && supports.call(this, feature);
      });
    const start = profile => createNative({ dataDir: temp(), profile });
    // The regular defaults, with header validation off (aiwaf-wasm's User-Agent check) and on.
    expect(converted.profile.header_policy).toBe('node_ua');
    const defaults = await start({ ...converted.profile });
    handles.push(defaults);
    const checked = nativeOptionsFromSettings({ AIWAF_HEADER_VALIDATION: true }).profile;
    handles.push(await start({ ...checked }));
    await expect(start({ ...converted.profile, exempt_keywords: ['.php'] }))
      .rejects.toThrow('exempt_keywords need a newer aiwaf-wasm');
    await expect(start({ ...converted.profile, header_max_user_agent: 1000 }))
      .rejects.toThrow('header_max_user_agent need a newer aiwaf-wasm');
    await expect(start({ ...converted.profile, login_prefixes: ['/signin/'] }))
      .rejects.toThrow('login_prefixes need a newer aiwaf-wasm');
  });

  test('geo blocking finds countries natively where the regular check finds them', async () => {
    installed('9.0.0');
    let client = 0;
    const blocked = async (options, headers = {}) => {
      const middleware = aiwaf({ AIWAF_GEO_BLOCK_ENABLED: true, AIWAF_GEO_BLOCK_COUNTRIES: ['CN'],
        AIWAF_PATH_MANIFEST: path.join(temp(), 'none.json'), ...options });
      expect(regular(middleware)).toBe(false);
      handles.push(await middleware.ready);
      const app = express();
      app.use(middleware);
      app.get('/items', (_req, res) => res.json({ ok: true }));
      // A blocked client is blocklisted, as in the regular check, so each case is a new client.
      return (await request(app).get('/items').set({ 'x-forwarded-for': `198.51.100.${++client}`, ...headers })).status;
    };
    // The x-country-code header, as the regular check reads it without a database match.
    expect(await blocked({}, { 'x-country-code': 'cn' })).toBe(403);
    expect(await blocked({}, { 'x-country-code': 'FR' })).toBe(200);
    // The database and cache settings carry over.
    expect(await blocked({ AIWAF_GEO_MMDB_PATH: path.join(temp(), 'missing.mmdb'), AIWAF_GEO_CACHE_SECONDS: 0,
      AIWAF_GEO_CACHE_PREFIX: 'geo:' }, { 'x-country-code': 'CN' })).toBe(403);
    // The app's geoResolver, which may be asynchronous, as the regular check uses it.
    let calls = 0;
    const geoResolver = async req => { calls += 1; return req.headers['x-test-country']; };
    expect(await blocked({ geoResolver }, { 'x-test-country': 'cn', 'x-country-code': 'FR' })).toBe(403);
    expect(await blocked({ geoResolver }, { 'x-test-country': 'FR', 'x-country-code': 'CN' })).toBe(200);
    expect(await blocked({ geoResolver: () => { throw new Error('down'); } }, { 'x-country-code': 'CN' })).toBe(200);
    expect(calls).toBe(2);
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
    // Converted, the regular anomaly check runs beside the engine and loads the model itself.
    const conversion = nativeOptionsFromSettings({ AIWAF_MODEL_PATH: file });
    expect(conversion.options.regularAnomaly).toMatchObject({ AIWAF_MODEL_PATH: file });
    expect(conversion.profile.enabled).not.toContain('ai_anomaly');
    expect(conversion.differences()).toEqual([]);
    // Set up explicitly, the engine loads it.
    const middleware = await createNative({ dataDir: temp(), manifestPath: path.join(temp(), 'none.json'), modelFile: file });
    handles.push(middleware);
    expect(middleware.diagnostics().model).toBe(true);
  });

  test('models the engine cannot load still carry over, since the regular check loads them', async () => {
    const dir = temp();
    const old = path.join(dir, 'old.json');
    fs.writeFileSync(old, JSON.stringify({ nTrees: 1, trees: [] }));
    const broken = path.join(dir, 'broken.json');
    fs.writeFileSync(broken, '{');
    for (const file of [old, broken]) {
      expect(nativeOptionsFromSettings({ AIWAF_MODEL_PATH: file }).differences()).toEqual([]);
    }
    // Set up explicitly, the engine refuses them.
    await expect(createNative({ dataDir: temp(), modelFile: old })).rejects.toThrow(/Retrain with: aiwaf train/);
    await expect(createNative({ modelFile: 5 })).rejects.toThrow(/modelFile must be a path/);
    expect(settingsModel({ AIWAF_MODEL_STORAGE: 'db' }).kind).toBe('storage');
    const missing = nativeOptionsFromSettings({ AIWAF_MODEL_PATH: path.join(dir, 'none.json') });
    expect(missing.notes.find(n => n.setting === 'AIWAF_MODEL_PATH').outcome).toBe('carried');
  });
});

describe('when the native pipeline can\'t start', () => {
  test('an automatic switch falls back to the regular middleware', async () => {
    delete process.env.AIWAF_NATIVE_PIPELINE;
    installed(aiwaf.NATIVE_DEFAULT_WASM);
    jest.spyOn(console, 'info').mockImplementation(() => {});
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
    // An engine without a feature the options need: only startup finds out.
    const wasm = require('aiwaf-wasm');
    const supports = wasm.NativeRequestPipeline.prototype.supports;
    jest.spyOn(wasm.NativeRequestPipeline.prototype, 'supports')
      .mockImplementation(function (feature) { return feature !== 'user_agent_verdict' && supports.call(this, feature); });
    const middleware = aiwaf({ AIWAF_HEADER_VALIDATION: true, AIWAF_SUSPICIOUS_USER_AGENTS: ['(?<=x)y'],
      AIWAF_PATH_MANIFEST: path.join(temp(), 'none.json') });
    const fallback = await middleware.ready;
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('the regular middleware runs instead'));
    expect(fallback.diagnostics()).toMatchObject({ native: false });
    expect(fallback.inspectBody()).toBeNull();
    fallback.setModel();
    fallback.close();
    const app = express();
    app.use(middleware);
    app.get('/items', (_req, res) => res.json({ ok: true }));
    expect((await request(app).get('/items').set({ 'user-agent': 'Mozilla/5.0 Chrome/130 Safari/537.36',
      accept: 'text/html', 'accept-language': 'en', 'accept-encoding': 'gzip', connection: 'keep-alive' })).status).toBe(200);
  });

  test('an explicit native pipeline reports the error to each request instead of crashing', async () => {
    const middleware = aiwaf({ AIWAF_NATIVE_PIPELINE: { profile: { sql_mode: 'loud' } } });
    await expect(middleware.ready).rejects.toThrow();
    const app = express();
    app.use(middleware);
    app.get('/items', (_req, res) => res.json({ ok: true }));
    app.use((error, _req, res, _next) => res.status(500).json({ error: 'startup' }));
    expect((await request(app).get('/items')).body).toEqual({ error: 'startup' });
  });
});

describe('the regular header check', () => {
  test('uses the app\'s User-Agent cap in its aiwaf-wasm check too', async () => {
    const browser = { 'user-agent': `Mozilla/5.0 Chrome/130 ${'a'.repeat(700)}`, accept: 'text/html',
      'accept-language': 'en', 'accept-encoding': 'gzip', connection: 'keep-alive' };
    const status = async (options, ip) => {
      const app = express();
      app.use(aiwaf({ AIWAF_NATIVE_PIPELINE: false, AIWAF_HEADER_VALIDATION: true, MAX_REQ: 1000, FLOOD_REQ: 2000,
        AIWAF_PATH_MANIFEST: path.join(temp(), 'none.json'), ...options }));
      app.use((_req, res) => res.json({ ok: true }));
      return (await request(app).get('/docs').set({ ...browser, 'x-forwarded-for': ip })).status;
    };
    expect(await status({ AIWAF_MAX_USER_AGENT_LENGTH: 1000 }, '192.0.2.230')).toBe(200);
    expect(await status({}, '192.0.2.231')).toBe(403);
  });
});

describe('the regular and native pipelines answer alike', () => {
  const browser = { 'user-agent': 'Mozilla/5.0 Chrome/130 Safari/537.36', accept: 'text/html',
    'accept-language': 'en', 'accept-encoding': 'gzip', connection: 'keep-alive' };
  beforeEach(() => {
    delete process.env.AIWAF_NATIVE_PIPELINE;
    installed('9.0.0');
    jest.spyOn(console, 'info').mockImplementation(() => {});
  });
  // Each mode gets its own client addresses, since blocks are shared through the regular stores.
  const statuses = async (mode, base, options, requests, handler = (_req, res) => res.json({ ok: true })) => {
    const middleware = aiwaf({ MAX_REQ: 1000, FLOOD_REQ: 2000, AIWAF_PATH_MANIFEST: path.join(temp(), 'none.json'),
      ...(mode === 'regular' ? { AIWAF_NATIVE_PIPELINE: false } : {}), ...options });
    expect(regular(middleware)).toBe(mode === 'regular');
    if (middleware.ready) handles.push(await middleware.ready);
    const app = express();
    app.use(express.urlencoded({ extended: false }));
    app.use(middleware);
    app.use(handler);
    const out = [];
    for (const [index, [method, url, headers]] of requests.entries()) {
      out.push((await request(app)[method](url).set({ ...browser, 'x-forwarded-for': `198.18.${base}.${index + 1}`, ...headers })).status);
    }
    return out;
  };
  const alike = async (base, options, requests) => {
    const answers = [await statuses('regular', base, options, requests), await statuses('native', base + 1, options, requests)];
    expect(answers[1]).toEqual(answers[0]);
    return answers[0];
  };

  test('keyword exemptions and allowed path fragments', async () => {
    const answers = await alike(1, { staticKeywords: ['.php', 'shell'], AIWAF_EXEMPT_KEYWORDS: ['.php'],
      AIWAF_ALLOWED_PATH_KEYWORDS: ['legacy'] }, [['get', '/x.php'], ['get', '/shell'], ['get', '/legacy/shell'], ['get', '/docs']]);
    expect(answers).toEqual([200, 403, 200, 200]);
  });

  test('post-only paths under the method policy', async () => {
    const answers = await alike(3, { AIWAF_METHOD_POLICY_ENABLED: true, AIWAF_ALLOWED_METHODS: ['GET', 'POST'],
      AIWAF_POST_ONLY_SUFFIXES: ['/submit/'] }, [['get', '/form/submit/'], ['get', '/form/'], ['delete', '/form/']]);
    expect(answers).toEqual([405, 200, 405]);
  });

  test('required headers per method', async () => {
    const answers = await alike(5, { AIWAF_HEADER_VALIDATION: true,
      AIWAF_REQUIRED_HEADERS: { GET: ['user-agent'], POST: ['user-agent', 'x-token'], DEFAULT: ['user-agent', 'accept'] } },
    [['get', '/docs', { accept: '' }], ['post', '/docs'], ['post', '/docs', { 'x-token': 't' }], ['put', '/docs', { accept: '' }]]);
    expect(answers).toEqual([403, 403, 200, 403]);
  });

  test('the header check, with the regular defaults and the app\'s own lists', async () => {
    const cases = [['get', '/docs'], ['get', '/docs', { accept: '' }], ['get', '/docs', { 'user-agent': '' }],
      ['get', '/docs', { 'accept-language': '', 'accept-encoding': '', connection: '' }],
      ['get', '/docs', { accept: '*/*', 'accept-language': '', 'accept-encoding': '' }],
      ['get', '/docs', { 'user-agent': 'Googlebot/2.1', accept: '' }], ['get', '/docs', { 'user-agent': 'Mozilla/4.0' }],
      ['get', '/docs', { 'user-agent': 'Mozilla/5' }], ['get', '/static/app.css', { 'user-agent': 'curl/8' }],
      ['get', '/docs', { 'user-agent': 'curl/8' }]];
    const on = await alike(11, { AIWAF_HEADER_VALIDATION: true }, cases);
    expect(on.slice(0, 3)).toEqual([200, 403, 403]);
    await alike(13, { AIWAF_HEADER_VALIDATION: true, AIWAF_REQUIRED_HEADERS: ['user-agent', 'accept'], AIWAF_HEADER_QUALITY_MIN_SCORE: 7 }, cases);
    await alike(15, { AIWAF_HEADER_VALIDATION: true, AIWAF_STATIC_PATHS: ['/public/'], AIWAF_WASM_HEADER_VALIDATION: false }, cases);
    await alike(17, { AIWAF_HEADER_VALIDATION: false }, cases);
  });

  test('login forms, with the default prefixes and the app\'s own', async () => {
    // A form posted 0.2 s after it was fetched: too fast, except on a login form.
    const posts = async (mode, base, options) => {
      const middleware = aiwaf({ MAX_REQ: 1000, FLOOD_REQ: 2000, AIWAF_PATH_MANIFEST: path.join(temp(), 'none.json'),
        AIWAF_MIN_FORM_TIME: 5, ...(mode === 'regular' ? { AIWAF_NATIVE_PIPELINE: false } : {}), ...options });
      if (middleware.ready) handles.push(await middleware.ready);
      const app = express();
      app.use(express.urlencoded({ extended: false }));
      app.use(middleware);
      app.use((_req, res) => res.json({ ok: true }));
      const out = [];
      for (const [index, url] of ['/login/', '/accounts/login/', '/Login/', '/form/', '/signin/next'].entries()) {
        const headers = { ...browser, 'x-forwarded-for': `198.18.${base}.${index + 1}` };
        await request(app).get(url).set(headers);
        await new Promise(resolve => setTimeout(resolve, 200));
        out.push((await request(app).post(url).set(headers).type('form').send('a=1')).status);
      }
      return out;
    };
    for (const [options, base] of [[{}, 33], [{ AIWAF_LOGIN_PATH_PREFIXES: [] }, 35], [{ AIWAF_LOGIN_PATH_PREFIXES: ['/form/'] }, 37]]) {
      const regularAnswers = await posts('regular', base, options);
      expect(await posts('native', base + 1, options)).toEqual(regularAnswers);
      if (!options.AIWAF_LOGIN_PATH_PREFIXES) expect(regularAnswers).toEqual([200, 200, 200, 403, 200]);
    }
  }, 30000);

  test('an asynchronous geoResolver', async () => {
    const geoResolver = async req => req.headers['x-test-country'];
    const answers = await alike(7, { AIWAF_GEO_BLOCK_ENABLED: true, AIWAF_GEO_BLOCK_COUNTRIES: ['CN'], geoResolver },
      [['get', '/docs', { 'x-test-country': 'cn' }], ['get', '/docs', { 'x-test-country': 'FR' }]]);
    expect(answers).toEqual([403, 200]);
  });

  test('requests are logged by the regular logger, blocks included', async () => {
    const logs = {};
    for (const [mode, base] of [['regular', 9], ['native', 10]]) {
      const file = path.join(temp(), 'requests.jsonl');
      await statuses(mode, base, { AIWAF_MIDDLEWARE_LOGGING: true, AIWAF_MIDDLEWARE_LOG_PATH: file, staticKeywords: ['.php'] },
        [['get', '/docs'], ['get', '/x.php']]);
      await new Promise(resolve => setTimeout(resolve, 50));
      logs[mode] = fs.readFileSync(file, 'utf8').trim().split('\n').map(line => JSON.parse(line))
        .map(event => [event.path, event.status, event.blocked]);
    }
    expect(logs.native).toEqual(logs.regular);
    expect(logs.native).toEqual([['/docs', 200, false], ['/x.php', 403, true]]);
  });

  test('the app\'s own User-Agent lists', async () => {
    const lists = { AIWAF_HEADER_VALIDATION: true, AIWAF_SUSPICIOUS_USER_AGENTS: ['^evil', /scanner/i], AIWAF_LEGITIMATE_BOTS: ['mymonitor'] };
    const cases = [['get', '/docs', { 'user-agent': 'curl/8.0 Mozilla/5.0 Chrome' }],
      ['get', '/docs', { 'user-agent': 'Evil/1.0 Mozilla/5.0' }], ['get', '/docs', { 'user-agent': 'Mozilla/5.0 Scanner' }],
      ['get', '/docs', { 'user-agent': 'MyMonitor/1 evil', accept: '' }]];
    // The regular middleware's aiwaf-wasm check still uses its own lists afterwards.
    expect(await alike(19, lists, cases)).toEqual([403, 403, 403, 403]);
    expect(await alike(23, { ...lists, AIWAF_WASM_HEADER_VALIDATION: false }, cases)).toEqual([200, 403, 403, 200]);
    // Patterns are tested in JavaScript, so flags and JavaScript-only syntax mean the same in both.
    const flagged = { AIWAF_HEADER_VALIDATION: true, AIWAF_WASM_HEADER_VALIDATION: false,
      AIWAF_SUSPICIOUS_USER_AGENTS: [/^evil/m, /scan.ner/s, /\u{e9}clair/u, '(?<=bad-)agent', /tool/g], AIWAF_LEGITIMATE_BOTS: [/^monitor/m] };
    // Header values can't hold newlines, so m and s match as they do without them.
    const agents = ['Evil/1 Mozilla/5.0', 'Mozilla/5.0 scan-ner', 'Mozilla/5.0 \u00e9clair Chrome', 'Mozilla/5.0 bad-agent x',
      'Mozilla/5.0 tool/1', 'Mozilla/5.0 tool/2', 'Mozilla/5.0 Chrome/130 Safari/537.36', 'monitor/2 tool'];
    const answers = await alike(41, flagged, agents.map(agent => ['get', '/docs', { 'user-agent': agent }]));
    expect(answers).toContain(403);
    expect(answers[6]).toBe(200);
  });

  test('a block records the request\'s extended info when the app captures it', async () => {
    const blocklistManager = require('../lib/blocklistManager');
    const info = {};
    for (const [mode, base] of [['regular', 21], ['native', 22]]) {
      await statuses(mode, base, { staticKeywords: ['.php'], AIWAF_CAPTURE_EXTENDED_REQUEST_INFO: true },
        [['get', '/x.php', { 'x-custom': 'yes', cookie: 'session=secret' }]]);
      await new Promise(resolve => setTimeout(resolve, 50));
      // Each request has its own address and test server port.
      const { ip: _ip, headers: { host: _host, 'x-forwarded-for': _forwarded, ...headers }, ...rest } =
        (await blocklistManager.getBlockInfo(`198.18.${base}.1`)).extended_request_info;
      info[mode] = { ...rest, headers };
    }
    expect(info.native).toEqual(info.regular);
    expect(JSON.stringify(info.native)).toContain('x-custom');
    expect(JSON.stringify(info.native)).not.toContain('secret');
  });

  test('the model waits for AIWAF_MIN_AI_LOGS recent logs, as the regular check does', async () => {
    const anomalyDetector = require('../lib/anomalyDetector');
    const forest = await createIsolationForest({ nTrees: 5, sampleSize: 16 });
    forest.fit(Array.from({ length: 16 }, (_, i) => [i, 0, 0, 1, 1, 0]));
    const file = path.join(temp(), 'model.json');
    fs.writeFileSync(file, JSON.stringify({ model_type: 'aiwaf_wasm.IsolationForest', model_state: forest.toJSON(), metadata: {} }));
    // The regular anomaly check runs beside the engine, with its own log minimum.
    const model = async options => {
      const middleware = aiwaf({ AIWAF_MODEL_PATH: file, AIWAF_PATH_MANIFEST: path.join(temp(), 'none.json'), ...options });
      handles.push(await middleware.ready);
      return anomalyDetector.isModelSufficientlyTrained();
    };
    // The regular options default to no minimum; a new app has fewer logs than one it sets.
    expect(await model({})).toBe(true);
    expect(await anomalyDetector.logsSufficient({ AIWAF_MIN_AI_LOGS: 5 })).toBe(false);
    expect(await model({ AIWAF_MIN_AI_LOGS: 5 })).toBe(false);
    // Set up explicitly, the engine's model waits for the same minimum.
    const gated = await createNative({ dataDir: temp(), modelFile: file, modelGate: () => anomalyDetector.logsSufficient({ AIWAF_MIN_AI_LOGS: 5 }) });
    handles.push(gated);
    expect(gated.diagnostics().model).toBe(false);
  });

  test('the regular stores report learned keywords past the threshold, for custom-store callers', async () => {
    const dynamicKeywordStore = require('../lib/dynamicKeywordStore');
    await dynamicKeywordStore.clear();
    await dynamicKeywordStore.add('often', 3);
    await dynamicKeywordStore.add('once', 1);
    const stores = createNative.regularStores({ dynamicTopN: 2 });
    expect(await stores.topKeywords(10)).toEqual(['often']);
    expect((await stores.keywordCounts(10)).sort()).toEqual([['often', 3], ['once', 1]]);
    await dynamicKeywordStore.clear();
  });

  test('keyword learning from 404 probes, and the blocks it causes', async () => {
    const dynamicKeywordStore = require('../lib/dynamicKeywordStore');
    const learned = {};
    const probes = [['get', '/backup-old/config.php.bak'], ['get', '/wordpress-setup/install.php'],
      ['get', '/oldsite/admin-panel'], ['get', '/temp-files/..%2f..%2fetc'], ['get', '/catalog-pages/browse'],
      ['get', '/legacy/shell'], ['get', '/legacy-tools/exec-runner'],
      // A later request reusing a learned word is blocked; one under an allowed fragment isn't learned from.
      ['get', '/nothing/oldsite'], ['get', '/legacy-tools'], ['get', '/catalog-pages'], ['get', '/safe-zone/admin-x']];
    const answers = {};
    for (const [mode, base] of [['regular', 25], ['native', 26]]) {
      await dynamicKeywordStore.clear();
      // Only /docs exists; everything else is a 404.
      answers[mode] = await statuses(mode, base, { dynamicTopN: 0, AIWAF_ALLOWED_PATH_KEYWORDS: ['safe-zone'] }, probes,
        (req, res) => (req.path === '/docs' ? res.json({ ok: true }) : res.status(404).json({ error: 'not found' })));
      await new Promise(resolve => setTimeout(resolve, 50));
      learned[mode] = (await dynamicKeywordStore.list()).map(row => [row.keyword, Number(row.count)]).sort();
    }
    if (process.env.AIWAF_PARITY_PRINT) console.log(JSON.stringify(answers), JSON.stringify(learned));
    expect(learned.native).toEqual(learned.regular);
    expect(answers.native).toEqual(answers.regular);
    expect(learned.native.length).toBeGreaterThan(0);
    if (process.env.AIWAF_PARITY_PRINT) console.log(JSON.stringify(answers.regular), JSON.stringify(learned.regular));
  });

  test('the anomaly model\'s blocks', async () => {
    const forest = await createIsolationForest({ nTrees: 50, sampleSize: 64 });
    forest.fit(Array.from({ length: 200 }, (_, i) => [8 + (i % 12), 0, 0, 20 + (i % 5) * 10, 1 + (i % 3), 0]));
    const file = path.join(temp(), 'model.json');
    fs.writeFileSync(file, JSON.stringify({ model_type: 'aiwaf_wasm.IsolationForest', model_state: forest.toJSON(), metadata: {} }));
    const words = ['backup', 'database', 'mysql', 'dump', 'setup', 'install', 'installer', 'adminer', 'cgi-bin', 'scripts',
      'base64', 'backup', 'database', 'mysql', 'dump', 'setup'];
    // One client's scan: every request comes from the same address.
    // The regular anomaly check loads a model once per process, so each scenario gets its own
    // copy of the modules, shared by its regular and native runs.
    let fresh = aiwaf;
    // Run a scenario's regular and native scans with modules of their own, including the ones
    // the native pipeline loads as it starts.
    const scenario = async run => {
      let result;
      await jest.isolateModulesAsync(async () => {
        fresh = require('../lib/wafMiddleware');
        result = await run();
      });
      return result;
    };
    const scan = async (mode, base, modelPath, extra = {}) => {
      const middleware = fresh({ MAX_REQ: 1000, FLOOD_REQ: 2000, AIWAF_PATH_MANIFEST: path.join(temp(), 'none.json'),
        AIWAF_MODEL_PATH: modelPath, AIWAF_ENABLE_KEYWORD_LEARNING: false, ...extra,
        ...(mode === 'regular' ? { AIWAF_NATIVE_PIPELINE: false } : {}) });
      if (middleware.ready) handles.push(await middleware.ready);
      const app = express();
      app.use(middleware);
      // A real route: an app without any is treated as a proxy, and isn't checked for anomalies.
      app.get('/docs', (_req, res) => res.json({ ok: true }));
      app.use((_req, res) => res.status(404).json({ error: 'not found' }));
      const out = [];
      for (const [n, word] of words.entries()) {
        out.push((await request(app).get(`/${word}/area${n}`).set({ ...browser, 'x-forwarded-for': `198.18.${base}.1` })).status);
      }
      return out;
    };
    const withModel = await scenario(async () => [await scan('regular', 27, file), await scan('native', 28, file)]);
    const without = await scenario(async () => [await scan('regular', 29, path.join(temp(), 'none.json')),
      await scan('native', 30, path.join(temp(), 'none.json'))]);
    // The regular check judges recent behavior from the request log, so it can block once
    // requests are logged there.
    const logged = { AIWAF_MIDDLEWARE_LOGGING: true, AIWAF_MIDDLEWARE_LOG_DB: true, AIWAF_MIDDLEWARE_LOG_PATH: path.join(temp(), 'r.jsonl') };
    const withLog = await scenario(async () => [await scan('regular', 31, file, logged), await scan('native', 32, file, logged)]);
    if (process.env.AIWAF_PARITY_PRINT) console.log(JSON.stringify({ withModel, without, withLog }));
    expect(withModel[1]).toEqual(withModel[0]);
    expect(without[1]).toEqual(without[0]);
    expect(withLog[1]).toEqual(withLog[0]);
    expect(withLog[0]).toContain(403);
  });

  test('a model kept in the app\'s model store', async () => {
    const forest = await createIsolationForest({ nTrees: 5, sampleSize: 16 });
    forest.fit(Array.from({ length: 16 }, (_, i) => [i, 0, 0, 1, 1, 0]));
    const modelStore = require('../lib/modelStore');
    const anomalyDetector = require('../lib/anomalyDetector');
    jest.spyOn(modelStore, 'load').mockResolvedValue({ model_type: 'aiwaf_wasm.IsolationForest', model_state: forest.toJSON() });
    // The regular anomaly check reads it from the store, as the regular middleware does. (It
    // loads a model once per process, so this process may already hold another test's.)
    expect(nativeOptionsFromSettings({ AIWAF_MODEL_STORAGE: 'db' }).options.regularAnomaly)
      .toMatchObject({ AIWAF_MODEL_STORAGE: 'db' });
    const middleware = aiwaf({ AIWAF_MODEL_STORAGE: 'db', AIWAF_PATH_MANIFEST: path.join(temp(), 'none.json') });
    expect(regular(middleware)).toBe(false);
    handles.push(await middleware.ready);
    expect(typeof anomalyDetector.hasModel()).toBe('boolean');
    // Set up explicitly, the engine reads it with modelLoader; one in the older format is refused.
    const loaded = await createNative({ dataDir: temp(), modelLoader: () => modelStore.load() });
    handles.push(loaded);
    expect(loaded.diagnostics().model).toBe(true);
    modelStore.load.mockResolvedValue({ metadata: {}, trees: [] });
    await expect(createNative({ dataDir: temp(), modelLoader: () => modelStore.load() })).rejects.toThrow('older format');
  });
});
