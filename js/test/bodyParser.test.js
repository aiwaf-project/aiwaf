const express = require('express');
const request = require('supertest');
const bodyParser = require('../lib/bodyParser');
const { createInspector } = require('../lib/sqlInjection');

function app(limit) {
  const server = express();
  server.use(bodyParser(limit ? { AIWAF_PAYLOAD_MAX_BYTES: limit } : {}));
  const inspect = createInspector();
  server.post('/login', (req, res) => {
    const finding = inspect(req);
    if (finding) return res.status(finding.status).json({ error: finding.rule });
    res.json({ body: req.body, original: req.aiwafRawBody.toString() });
  });
  return server;
}

test.each(['json', 'form'])('parses %s before inspection and retains forwarding bytes', async type => {
  const result = await request(app()).post('/login').type(type).send({ email: 'normal@example.invalid' });
  expect(result.status).toBe(200);
  expect(result.body.body.email).toBe('normal@example.invalid');
  expect(result.body.original).toContain('normal');
});

test.each(['json', 'form'])('blocks a %s SQL login bypass after parsing', async type => {
  const result = await request(app()).post('/login').type(type).send({ email: "admin' OR 1=1--", password: 'x' });
  expect(result.status).toBe(403);
  expect(result.text).not.toContain('admin');
});

test('rejects an oversized body without exposing it', async () => {
  const result = await request(app(20)).post('/login').send({ email: 'a'.repeat(50) });
  expect(result.status).toBe(413);
  expect(result.body.error).toBe('payload_inspection_limit');
});
