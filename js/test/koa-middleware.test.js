const Koa = require('koa');
const bodyParser = require('koa-bodyparser');
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
  const app = new Koa();
  app.use(bodyParser());
  app.use(aiwaf.koa({
    cache: testCache,
    WINDOW_SEC: 60,
    MAX_REQ: 1000,
    FLOOD_REQ: 2000,
    ...opts
  }));
  app.use(ctx => {
    if (ctx.path === '/safe') {
      ctx.body = 'ok';
      return;
    }
    ctx.status = 404;
    ctx.body = 'not found';
  });
  return app;
}

describe('AIWAF Koa middleware', () => {
  it('allows safe requests', async () => {
    const app = buildApp();
    const res = await request(app.callback())
      .get('/safe')
      .set('X-Forwarded-For', '198.51.100.30');
    expect(res.status).toBe(200);
    expect(res.text).toBe('ok');
  });

  it('blocks static keyword paths', async () => {
    const app = buildApp({ staticKeywords: ['.php'] });
    const res = await request(app.callback())
      .get('/wp-config.php')
      .set('X-Forwarded-For', '198.51.100.31');
    expect(res.status).toBe(403);
    expect(res.body).toEqual({ error: 'blocked' });
  });

  it('enforces method policy when enabled', async () => {
    const app = buildApp({
      AIWAF_METHOD_POLICY_ENABLED: true,
      AIWAF_ALLOWED_METHODS: ['GET']
    });
    const res = await request(app.callback())
      .post('/safe')
      .set('X-Forwarded-For', '198.51.100.32');
    expect(res.status).toBe(405);
    expect(res.body).toEqual({ error: 'blocked' });
  });

  it('enforces header validation requirements', async () => {
    const app = buildApp({
      AIWAF_HEADER_VALIDATION: true,
      AIWAF_REQUIRED_HEADERS: ['x-required-security-header']
    });
    const res = await request(app.callback())
      .get('/safe')
      .set('X-Forwarded-For', '198.51.100.33')
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
    const agent = request(app.callback());
    const headers = {
      'X-Forwarded-For': '198.51.100.34',
      Accept: 'application/json'
    };

    const first = await agent.get('/safe').set(headers);
    expect(first.status).toBe(200);

    const second = await agent.get('/safe').set(headers);
    expect(second.status).toBe(429);
    expect(second.body).toEqual({ error: 'too_many_requests' });
  });
});
