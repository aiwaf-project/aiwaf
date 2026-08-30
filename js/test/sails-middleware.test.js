const express = require('express');
const request = require('supertest');
const aiwaf = require('../index');

process.env.NODE_ENV = 'test';

const testCache = (() => {
  const store = new Map();
  return {
    async get(key) {
      return store.has(key) ? store.get(key) : null;
    },
    async set(key, val) {
      store.set(key, val);
    }
  };
})();

function buildApp(opts = {}) {
  const app = express();
  app.use(express.json());
  app.use(aiwaf.sails({
    cache: testCache,
    WINDOW_SEC: 60,
    MAX_REQ: 1000,
    FLOOD_REQ: 2000,
    ...opts
  }));
  app.get('/safe', (req, res) => res.send('ok'));
  return app;
}

describe('AIWAF Sails middleware', () => {
  it('allows safe requests', async () => {
    const app = buildApp();
    const res = await request(app)
      .get('/safe')
      .set('X-Forwarded-For', '198.51.100.70');
    expect(res.status).toBe(200);
    expect(res.text).toBe('ok');
  });

  it('blocks static keyword paths', async () => {
    const app = buildApp({ staticKeywords: ['.php'] });
    const res = await request(app)
      .get('/wp-config.php')
      .set('X-Forwarded-For', '198.51.100.71')
      .set('Accept', 'application/json');
    expect(res.status).toBe(403);
    expect(res.body).toEqual({ error: 'blocked' });
  });

  it('enforces method policy when enabled', async () => {
    const app = buildApp({
      AIWAF_METHOD_POLICY_ENABLED: true,
      AIWAF_ALLOWED_METHODS: ['GET']
    });
    const res = await request(app)
      .post('/safe')
      .set('X-Forwarded-For', '198.51.100.72')
      .set('Accept', 'application/json');
    expect(res.status).toBe(405);
    expect(res.body).toEqual({ error: 'blocked' });
  });

  it('enforces header validation requirements', async () => {
    const app = buildApp({
      AIWAF_HEADER_VALIDATION: true,
      AIWAF_REQUIRED_HEADERS: ['x-required-security-header']
    });
    const res = await request(app)
      .get('/safe')
      .set('X-Forwarded-For', '198.51.100.73')
      .set('Accept', 'application/json');
    expect(res.status).toBe(403);
    expect(res.body).toEqual({ error: 'blocked' });
  });

  it('rate limits repeated requests', async () => {
    const app = buildApp({
      WINDOW_SEC: 60,
      MAX_REQ: 1,
      FLOOD_REQ: 2
    });
    const agent = request(app);
    const headers = {
      'X-Forwarded-For': '198.51.100.74',
      Accept: 'application/json'
    };

    const first = await agent.get('/safe').set(headers);
    expect(first.status).toBe(200);

    const second = await agent.get('/safe').set(headers);
    expect(second.status).toBe(429);
    expect(second.body).toEqual({ error: 'too_many_requests' });
  });
});
