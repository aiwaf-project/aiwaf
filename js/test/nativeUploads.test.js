// Unparsed bodies in the native pipeline: uploads pass unread, inspectable bodies are read
// up to max_bytes and put back for the parsers after it. Uses the real WASM engine.
if (process.env.AIWAF_NATIVE_TEST_MODULE) {
  jest.doMock(require.resolve('aiwaf-wasm'), () => require(process.env.AIWAF_NATIVE_TEST_MODULE));
}
const fs = require('fs');
const os = require('os');
const path = require('path');
const zlib = require('zlib');
const express = require('express');
const request = require('supertest');
const http = require('http');
const createNative = require('../lib/nativePipeline');

// Each request from its own address, so one block doesn't affect the next request.
let client = 0;
const post = (app, url) => request(app).post(url).set('X-Forwarded-For', `198.51.100.${++client % 250}`);

const roots = [];
const handles = [];
const temp = () => { const root = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-native-uploads-')); roots.push(root); return root; };
afterAll(() => {
  for (const handle of handles) handle.close();
  for (const root of roots) fs.rmSync(root, { recursive: true, force: true });
});

// The native middleware first, then the app's own parsers, as when AIWAF is mounted early.
async function appWith(profile = {}, options = {}) {
  const app = express();
  app.set('trust proxy', true);
  const middleware = await createNative({ dataDir: temp(), manifestPath: path.join(temp(), 'none.json'),
    ...options, profile: { enabled: ['ip_keyword_block'], max_bytes: 1024, ...profile } });
  handles.push(middleware);
  app.use(middleware);
  app.use(express.json({ limit: '10mb' }));
  app.use(express.urlencoded({ extended: false }));
  app.use(express.raw({ type: ['application/octet-stream', 'multipart/form-data'], limit: '10mb' }));
  app.post('*', (req, res) => res.json(Buffer.isBuffer(req.body)
    ? { bytes: req.body.length, sum: req.body.reduce((a, b) => (a + b) % 65521, 0) } : { body: req.body }));
  return app;
}

test('uploads pass unread and reach the app intact', async () => {
  const app = await appWith();
  const file = Buffer.alloc(5 * 1024 * 1024, 7);
  file.write("' OR 1=1--", 100);
  const binary = await post(app, '/upload').set('content-type', 'application/octet-stream').send(file);
  expect(binary.status).toBe(200);
  expect(binary.body).toEqual({ bytes: file.length, sum: file.reduce((a, b) => (a + b) % 65521, 0) });
  const multipart = await post(app, '/upload').attach('file', file, 'data.bin');
  expect(multipart.status).toBe(200);
  expect(multipart.body.bytes).toBeGreaterThan(file.length);
});

test('unparsed JSON and form bodies are inspected, then parsed by the app as sent', async () => {
  const app = await appWith();
  const clean = { name: 'Ada', nested: { tags: ['a', 'b'] } };
  expect((await post(app, '/api').send(clean)).body).toEqual({ body: clean });
  expect((await post(app, '/api').send({ q: "' OR 1=1--" })).status).toBe(403);
  expect((await post(app, '/api').type('form').send("q=1' union select password from users--")).status).toBe(403);
  expect((await post(app, '/api').type('form').send('q=hello')).body).toEqual({ body: { q: 'hello' } });
  // Up to the limit, an unparsed body is read in full and still reaches the parser.
  const large = { text: 'x'.repeat(1000) };
  expect((await post(app, '/api').send(large)).body).toEqual({ body: large });
});

test('a body over max_bytes is refused, or logged in monitor mode, as in Python', async () => {
  const big = { text: 'x'.repeat(5000) };
  expect((await post(await appWith(), '/api').send(big)).body).toEqual({ error: 'payload_inspection_limit' });
  const probe = await require('../lib/wasmAdapter').createRequestPipeline({});
  const monitorSupported = typeof probe.native.supports === 'function' && probe.native.supports('sql_monitor');
  probe.native.free();
  if (!monitorSupported) return;
  const logger = { warn: jest.fn() };
  const monitored = await appWith({ sql_mode: 'monitor' }, { logger });
  expect((await post(monitored, '/api').send(big)).body).toEqual({ body: big });
  expect(logger.warn).toHaveBeenCalledWith(expect.objectContaining({ rule: 'payload_inspection_limit', mode: 'monitor' }));
});

test('chunked and empty bodies work', async () => {
  const app = await appWith();
  const server = app.listen(0);
  try {
    // No content-length: the body arrives in chunks.
    const send = body => new Promise((resolve, reject) => {
      const req = http.request({ port: server.address().port, method: 'POST', path: '/api',
        headers: { 'content-type': 'application/json', 'x-forwarded-for': `203.0.113.${++client % 250}` } }, res => {
        let data = '';
        res.on('data', chunk => { data += chunk; });
        res.on('end', () => resolve([res.statusCode, JSON.parse(data)]));
      });
      req.on('error', reject);
      for (let i = 0; i < body.length; i += 7) req.write(body.slice(i, i + 7));
      req.end();
    });
    expect((await send(JSON.stringify({ q: "' OR 1=1--" })))[0]).toBe(403);
    expect(await send(JSON.stringify({ q: 'fine' }))).toEqual([200, { body: { q: 'fine' } }]);
  } finally { server.close(); }
  const empty = await post(app, '/api').set('content-type', 'application/json').send('');
  expect(empty.status).toBe(200);
});

test('a body it cannot read is refused as before', async () => {
  const app = await appWith();
  const compressed = await post(app, '/api').set('content-type', 'application/json')
    .set('content-encoding', 'gzip').send(zlib.gzipSync(JSON.stringify({ q: 1 })));
  expect(compressed.body).toEqual({ error: 'payload_inspection_limit' });
});

test('bodies are left alone when payload inspection is off', async () => {
  for (const [profile, rules] of [[{ sql_mode: 'off' }, []], [{}, [{ PREFIX: '/raw/', DISABLE: ['ip_keyword_block'] }]]]) {
    const app = await appWith(profile, { pathRules: rules });
    const url = rules.length ? '/raw/api' : '/api';
    expect((await post(app, url).send({ q: "' OR 1=1--" })).status).toBe(200);
  }
});

test('malformed JSON is checked as text, and the app parser still answers', async () => {
  const app = await appWith();
  expect((await post(app, '/api').set('content-type', 'application/json').send("{oops' union select password from users--")).status).toBe(403);
  expect((await post(app, '/api').set('content-type', 'application/json').send('{oops')).status).toBe(400);
});

test('a body stream that fails while being read goes to next(error)', async () => {
  const { PassThrough } = require('stream');
  const middleware = await createNative({ dataDir: temp(), manifestPath: path.join(temp(), 'none.json'),
    profile: { enabled: ['ip_keyword_block'] } });
  handles.push(middleware);
  const req = Object.assign(new PassThrough(), { method: 'POST', url: '/api', path: '/api', ip: '198.51.100.250',
    headers: { 'content-type': 'application/json', 'content-length': '100' } });
  const res = { on() {}, status() { return this; }, json() {} };
  const failed = new Promise(resolve => middleware(req, res, resolve));
  req.write('{"a":');
  req.destroy(new Error('client went away'));
  expect((await failed).message).toBe('client went away');
});

test('a request log that can\'t be written is reported once and requests carry on', async () => {
  if (!fs.existsSync('/dev/full')) return;
  const dir = temp();
  const errors = [];
  const logPath = path.join(dir, 'requests.csv');
  const middleware = await createNative({ dataDir: dir, manifestPath: path.join(dir, 'none.json'), logPath,
    onError: error => errors.push(error.message), profile: { enabled: ['logging'] } });
  handles.push(middleware);
  const app = express();
  app.use(middleware);
  app.get('/items', (_req, res) => res.json({ ok: true }));
  expect((await request(app).get('/items')).status).toBe(200);
  // The adapter keeps its file open; swapping the file doesn't affect it, so make writes fail directly.
  const fd = fs.openSync('/dev/full', 'w');
  const write = fs.writeSync;
  jest.spyOn(fs, 'writeSync').mockImplementation((target, ...rest) => write(fd, ...rest));
  for (let i = 0; i < 3; i++) expect((await request(app).get('/items')).status).toBe(200);
  jest.restoreAllMocks();
  fs.closeSync(fd);
  expect(errors).toHaveLength(1);
});
