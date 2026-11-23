const fs = require('fs');
const os = require('os');
const path = require('path');
const express = require('express');
const request = require('supertest');
const aiwaf = require('../index');
const recorder = require('../lib/inputRecorder');

const lines = file => fs.readFileSync(file, 'utf8').trim().split('\n').map(line => JSON.parse(line));

test('destination follows the environment', () => {
  const saved = process.env[recorder.ENVIRONMENT];
  delete process.env[recorder.ENVIRONMENT];
  try {
    expect(recorder.destination()).toBeNull();
    expect(recorder.destination('off')).toBeNull();
    expect(recorder.destination('1')).toBe(recorder.DEFAULT_PATH);
    expect(recorder.destination('logs/seen.jsonl')).toBe('logs/seen.jsonl');
  } finally {
    if (saved !== undefined) process.env[recorder.ENVIRONMENT] = saved;
  }
});

test('records what Express handlers read, not what clients send', async () => {
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-record-')), 'out', 'seen.jsonl');
  const app = express();
  app.use(aiwaf.recordInputs({ path: file }));
  app.use(express.json());
  const items = express.Router();
  items.get('/:id', (req, res) => res.json({ q: req.query.q, page: req.query.page, has: 'x' in req.query }));
  app.use('/items', items);
  app.post('/save', (req, res) => {
    const { title } = req.body;
    res.json({ title, csrf: req.body._csrf, text: req.body.toString() });
  });
  app.post('/dump', (req, res) => res.json({ ...req.body, after: req.body.late }));
  app.get('/nothing', (req, res) => res.end());
  await request(app).get('/items/1?q=a&extra=2');
  await request(app).post('/save').send({ title: 't', _csrf: 'x', unread: 1 });
  await request(app).post('/dump').send({ a: 1 });
  await request(app).get('/nothing?x=1');
  await request(app).get('/missing');
  const found = lines(file).map(line => `${line.method} ${line.route} ${line.key}`).sort();
  expect(found).toEqual(['GET /items/:id page', 'GET /items/:id q', 'POST /dump *', 'POST /save title']);
});

test('recording off only calls next', () => {
  const next = jest.fn();
  recorder.recordInputs({ path: '0' })({}, {}, next);
  expect(next).toHaveBeenCalled();
});

test('a write that fails is reported, not thrown', async () => {
  const folder = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-record-'));
  const blocked = path.join(folder, 'file');
  fs.writeFileSync(blocked, '');
  const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
  try {
    const app = express();
    app.use(recorder.recordInputs({ path: path.join(blocked, 'seen.jsonl') }));
    app.get('/q', (req, res) => res.json(req.query.q || null));
    await request(app).get('/q?q=1');
    expect(warn.mock.calls[0][0]).toMatch(/couldn't record inputs/);
  } finally {
    warn.mockRestore();
  }
});

test('the AIWAF middleware records what handlers read, and not its own reads', async () => {
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-record-')), 'seen.jsonl');
  const app = express();
  app.use(express.json());
  app.use(aiwaf({ AIWAF_RECORD_INPUTS: file, AIWAF_MIDDLEWARES: ['ip_keyword_block'], AIWAF_WASM_VALIDATION: false }));
  app.post('/save', (req, res) => res.json({ title: req.body.title }));
  await request(app).post('/save?debug=1').set('X-Forwarded-For', '203.0.113.77').send({ title: 't', other: 1 });
  expect(lines(file).map(line => [line.route, line.key])).toEqual([['/save', 'title']]);
});
