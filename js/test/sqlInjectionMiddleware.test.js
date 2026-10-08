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
  expect(warn).toHaveBeenCalledWith({ event: 'aiwaf.payload', rule: 'sql_quote_comment', mode: 'monitor' });
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
