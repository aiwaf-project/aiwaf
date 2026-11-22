const express = require('express');
const request = require('supertest');
const aiwaf = require('../index');

function appFor(options = {}) {
  const app = express();
  app.use(express.json());
  app.use(aiwaf({ AIWAF_MIDDLEWARES: ['ip_keyword_block'], AIWAF_WASM_VALIDATION: false,
    AIWAF_EXEMPTIONS_DB: false, ...options }));
  app.post('/login', (req, res) => res.status(401).json({ error: 'invalid credentials' }));
  return app;
}

test.each(["admin@juice-sh.op' OR 1=1--", "admin@juice-sh.op'--"])
('SQL login bypass is denied before the handler: %s', async email => {
  const result = await request(appFor()).post('/login').set('X-Forwarded-For', '93.184.217.20').send({ email, password: 'x' });
  expect(result.status).toBe(403);
  expect(result.body.error).toMatch(/^sql_/);
});

test('ordinary credentials reach the handler', async () => {
  await request(appFor()).post('/login').send({ email: "o'reilly@example.com", password: 'ordinary#123' }).expect(401);
});

test('monitor logs only rule metadata and forwards the request', async () => {
  const warn = jest.fn();
  await request(appFor({ AIWAF_SQL_INJECTION_MODE: 'monitor', logger: { warn } }))
    .post('/login').send({ email: "admin'--", password: 'private-secret' }).expect(401);
  expect(warn).toHaveBeenCalledWith({ event: 'aiwaf.payload', rule: 'sql_breakout_single_quote', mode: 'monitor' });
  expect(JSON.stringify(warn.mock.calls)).not.toContain('private-secret');
});

test('respects explicit path exemptions', async () => {
  await request(appFor({ AIWAF_EXEMPT_PATHS: ['/login'] })).post('/login').send({ email: "admin'--" }).expect(401);
});

test('respects off mode and per-route middleware selection', async () => {
  await request(appFor({ AIWAF_SQL_INJECTION_MODE: 'off' })).post('/login').send({ email: "admin'--" }).expect(401);
  await request(appFor({ AIWAF_PATH_RULES: [{ PREFIX: '/login', DISABLE: ['ip_keyword_block'] }] }))
    .post('/login').send({ email: "admin'--" }).expect(401);
});

test('rejects over-budget payloads', async () => {
  await request(appFor({ AIWAF_PAYLOAD_MAX_BYTES: 32 })).post('/login').send({ email: 'a'.repeat(64) }).expect(413);
});

// AIWAF before the upload parser, as when it's mounted first: it reads the start of a multipart
// body and puts it back, so the parser still gets all of it.
function uploadApp(options = {}) {
  const app = express();
  app.use(aiwaf({ AIWAF_MIDDLEWARES: ['ip_keyword_block'], AIWAF_WASM_VALIDATION: false,
    AIWAF_EXEMPTIONS_DB: false, ...options }));
  app.use(express.raw({ type: 'multipart/form-data', limit: '10mb' }));
  app.post('/upload', (req, res) => res.json({ bytes: req.body.length }));
  return app;
}

test('multipart forms are checked part by part before the upload parser', async () => {
  const send = (app, build) => build(request(app).post('/upload').set('X-Forwarded-For', '93.184.217.21'));
  expect((await send(uploadApp(), r => r.field('email', "admin'--"))).status).toBe(403);
  const shell = await send(uploadApp(), r => r.attach('file', Buffer.from('GIF89a'), 'shell.php'));
  expect([shell.status, shell.body.error]).toEqual([403, 'rce_script_upload']);
  const polyglot = await send(uploadApp(), r => r.attach('file', Buffer.from('\xff\xd8<?php system($_GET["c"]); ?>'), 'a.jpg'));
  expect(polyglot.body.error).toBe('rce_php');
  // A large ordinary upload is checked by its start and reaches the parser whole.
  const file = Buffer.alloc(2 * 1024 * 1024, 7);
  const big = await send(uploadApp({ AIWAF_PAYLOAD_MAX_BYTES: 4096 }), r => r.field('title', 'ok').attach('file', file, 'data.bin'));
  expect(big.status).toBe(200);
  expect(big.body.bytes).toBeGreaterThan(file.length);
  // Off mode doesn't read the body at all.
  expect((await send(uploadApp({ AIWAF_SQL_INJECTION_MODE: 'off' }), r => r.field('email', "admin'--"))).status).toBe(200);
});

test('the route schema saved by init lets prose through in enforce mode, and monitor only logs', async () => {
  const fs = require('fs');
  const os = require('os');
  const path = require('path');
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-schema-mw-'));
  try {
    const manifest = path.join(directory, 'paths.json');
    fs.writeFileSync(manifest, JSON.stringify({ routes: { '/comments': {
      view: 'comments', inputs: { body: 'text', post_id: 'int' }, inputs_complete: true } } }));
    const app = options => {
      const server = express();
      server.use(express.json());
      server.use(aiwaf({ AIWAF_MIDDLEWARES: ['ip_keyword_block'], AIWAF_WASM_VALIDATION: false, AIWAF_EXEMPTIONS_DB: false,
        AIWAF_PATH_MANIFEST: manifest, logger: { warn() {} }, ...options }));
      server.post('/comments', (req, res) => res.status(201).json(req.body));
      return server;
    };
    const post = (server, body) => request(server).post('/comments').set('X-Forwarded-For', '93.184.217.30').send(body);
    const prose = { body: 'use sleep(5) in the query', post_id: 7 };
    const enforced = app({ AIWAF_ROUTE_SCHEMA_MODE: 'enforce' });
    expect((await post(enforced, prose)).body).toEqual(prose);
    expect((await post(enforced, { post_id: '7 OR 1=1' })).body).toEqual({ error: 'schema_type_mismatch' });
    expect((await post(enforced, { body: "x' OR '1'='1" })).status).toBe(403);
    expect((await post(app({}), prose)).status).toBe(403);
    // A path's finding is reviewed by the route's parameters.
    fs.writeFileSync(manifest, JSON.stringify({ routes: { '/posts/:slug': { view: 'post', inputs: {}, inputs_complete: true } } }));
    const pathApp = express();
    pathApp.use(aiwaf({ AIWAF_MIDDLEWARES: ['ip_keyword_block'], AIWAF_WASM_VALIDATION: false, AIWAF_EXEMPTIONS_DB: false,
      AIWAF_PATH_MANIFEST: manifest, AIWAF_ROUTE_SCHEMA_MODE: 'enforce', logger: { warn() {} } }));
    pathApp.get('/posts/:slug', (req, res) => res.json({ slug: req.params.slug }));
    expect((await request(pathApp).get('/posts/use-sleep(5)-in-queries').set('X-Forwarded-For', '93.184.217.31')).status).toBe(200);
    expect((await request(pathApp).get("/posts/x'%20OR%20'1'='1").set('X-Forwarded-For', '93.184.217.32')).status).toBe(403);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});
