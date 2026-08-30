const request = require('supertest');
const express = require('express');
const aiwaf = require('../index');
const db = require('../utils/db');

describe('AIWAF middleware behavior', () => {
  it('returns text response when JSON errors are disabled and request is non-JSON', async () => {
    const app = express();
    app.use(express.json());
    app.use(aiwaf({
      staticKeywords: ['.php'],
      AIWAF_FORCE_JSON_ERRORS: false,
      WINDOW_SEC: 60,
      MAX_REQ: 1000,
      FLOOD_REQ: 2000
    }));
    app.get('/safe', (req, res) => res.send('ok'));

    const res = await request(app)
      .get('/a.php')
      .set('Accept', 'text/plain')
      .set('X-Forwarded-For', '198.51.120.1')
      .expect(403);

    expect(res.text).toBe('blocked');
    expect(res.headers['content-type']).toMatch(/text\/html|text\/plain/);
  });

  it('enforces geo allowlist mode', async () => {
    const app = express();
    app.use(express.json());
    app.use(aiwaf({
      AIWAF_GEO_BLOCK_ENABLED: true,
      AIWAF_GEO_ALLOW_COUNTRIES: ['US'],
      WINDOW_SEC: 60,
      MAX_REQ: 1000,
      FLOOD_REQ: 2000
    }));
    app.get('/safe', (req, res) => res.send('ok'));

    await request(app)
      .get('/safe')
      .set('x-country-code', 'CA')
      .set('X-Forwarded-For', '198.51.120.2')
      .expect(403);

    await request(app)
      .get('/safe')
      .set('x-country-code', 'US')
      .set('X-Forwarded-For', '198.51.120.3')
      .expect(200, 'ok');
  });

  it('respects allowed path keywords and skips static keyword blocking', async () => {
    const app = express();
    app.use(express.json());
    app.use(aiwaf({
      staticKeywords: ['.php'],
      AIWAF_ALLOWED_PATH_KEYWORDS: ['public'],
      WINDOW_SEC: 60,
      MAX_REQ: 1000,
      FLOOD_REQ: 2000
    }));
    app.get('/public/readme.php', (req, res) => res.send('ok'));

    await request(app)
      .get('/public/readme.php')
      .set('X-Forwarded-For', '198.51.120.4')
      .expect(200, 'ok');
  });
});

afterAll(async () => {
  await db.destroy();
});
