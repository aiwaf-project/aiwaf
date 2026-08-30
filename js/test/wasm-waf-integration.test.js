const express = require('express');
const request = require('supertest');

describe('WASM validation integration', () => {
  afterEach(() => {
    jest.resetModules();
    jest.clearAllMocks();
  });

  it('blocks when wasm validators return reasons', async () => {
    jest.doMock('aiwaf-wasm', () => ({
      default: jest.fn(async () => {}),
      validate_headers: jest.fn(() => null),
      validate_url: jest.fn(() => 'wasm_url_bad'),
      validate_content: jest.fn(() => null),
      validate_recent: jest.fn(() => null),
      AiwafIsolationForest: class {
        constructor() {}
        fit() {}
        anomaly_score() { return 0.1; }
      }
    }));

    const aiwaf = require('../index');
    const app = express();
    app.use(express.json());
    app.use(aiwaf({ AIWAF_WASM_VALIDATION: true }));
    app.post('/safe', (req, res) => res.json({ ok: true }));

    await request(app)
      .post('/safe')
      .set('host', 'example.com')
      .set('accept', 'application/json')
      .send({ ok: true })
      .expect(403, { error: 'blocked' });
  });

  it('allows traffic when wasm validation is disabled', async () => {
    jest.doMock('aiwaf-wasm', () => ({
      default: jest.fn(async () => {}),
      validate_headers: jest.fn(() => 'wasm_header_bad'),
      validate_url: jest.fn(() => 'wasm_url_bad'),
      validate_content: jest.fn(() => 'wasm_content_bad'),
      validate_recent: jest.fn(() => 'wasm_recent_bad'),
      AiwafIsolationForest: class {
        constructor() {}
        fit() {}
        anomaly_score() { return 0.1; }
      }
    }));

    const aiwaf = require('../index');
    const app = express();
    app.use(express.json());
    app.use(aiwaf({ AIWAF_WASM_VALIDATION: false }));
    app.post('/safe', (req, res) => res.json({ ok: true }));

    await request(app)
      .post('/safe')
      .set('host', 'example.com')
      .set('accept', 'application/json')
      .send({ ok: true })
      .expect(200, { ok: true });
  });
});
