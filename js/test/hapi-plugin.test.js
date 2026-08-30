const Hapi = require('@hapi/hapi');
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

async function buildServer(opts = {}) {
  const server = Hapi.server({ port: 0 });
  await server.register({
    plugin: aiwaf.hapi,
    options: {
      cache: testCache,
      WINDOW_SEC: 60,
      MAX_REQ: 1000,
      FLOOD_REQ: 2000,
      ...opts
    }
  });

  server.route({ method: 'GET', path: '/safe', handler: () => 'ok' });
  server.route({ method: 'POST', path: '/safe', handler: () => 'ok' });
  return server;
}

describe('AIWAF Hapi plugin', () => {
  it('allows safe requests', async () => {
    const server = await buildServer();
    const res = await server.inject({
      method: 'GET',
      url: '/safe',
      headers: { 'x-forwarded-for': '198.51.100.20' }
    });
    expect(res.statusCode).toBe(200);
    await server.stop();
  });

  it('blocks static keyword paths', async () => {
    const server = await buildServer({ staticKeywords: ['.php'] });
    const res = await server.inject({
      method: 'GET',
      url: '/wp-config.php',
      headers: { 'x-forwarded-for': '198.51.100.21' }
    });
    expect(res.statusCode).toBe(403);
    const payload = JSON.parse(res.payload);
    expect(payload).toEqual({ error: 'blocked' });
    await server.stop();
  });

  it('enforces method policy when enabled', async () => {
    const server = await buildServer({
      AIWAF_METHOD_POLICY_ENABLED: true,
      AIWAF_ALLOWED_METHODS: ['GET']
    });
    const res = await server.inject({
      method: 'POST',
      url: '/safe',
      headers: { 'x-forwarded-for': '198.51.100.22' }
    });
    expect(res.statusCode).toBe(405);
    const payload = JSON.parse(res.payload);
    expect(payload).toEqual({ error: 'blocked' });
    await server.stop();
  });

  it('enforces header validation requirements', async () => {
    const server = await buildServer({
      AIWAF_HEADER_VALIDATION: true,
      AIWAF_REQUIRED_HEADERS: ['x-required-security-header']
    });
    const res = await server.inject({
      method: 'GET',
      url: '/safe',
      headers: {
        'x-forwarded-for': '198.51.100.23',
        accept: 'application/json'
      }
    });
    expect(res.statusCode).toBe(403);
    const payload = JSON.parse(res.payload);
    expect(payload).toEqual({ error: 'blocked' });
    await server.stop();
  });

  it('rate limits repeated requests', async () => {
    const server = await buildServer({
      WINDOW_SEC: 60,
      MAX_REQ: 1,
      FLOOD_REQ: 2
    });
    const headers = {
      'x-forwarded-for': '198.51.100.24',
      accept: 'application/json'
    };
    const first = await server.inject({ method: 'GET', url: '/safe', headers });
    expect(first.statusCode).toBe(200);

    const second = await server.inject({ method: 'GET', url: '/safe', headers });
    expect(second.statusCode).toBe(429);
    const payload = JSON.parse(second.payload);
    expect(payload).toEqual({ error: 'too_many_requests' });
    await server.stop();
  });
});
