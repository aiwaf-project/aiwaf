const fastifyFactory = require('fastify');
const aiwafFastify = require('../lib/fastifyPlugin');

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
  const fastify = fastifyFactory();
  fastify.register(aiwafFastify, {
    cache: testCache,
    WINDOW_SEC: 60,
    MAX_REQ: 1000,
    FLOOD_REQ: 2000,
    ...opts
  });

  fastify.get('/safe', async () => ({ ok: true }));
  fastify.post('/safe', async () => ({ ok: true }));
  return fastify;
}

describe('AIWAF Fastify plugin', () => {
  it('allows safe requests', async () => {
    const fastify = await buildServer();
    const res = await fastify.inject({
      method: 'GET',
      url: '/safe',
      headers: { 'x-forwarded-for': '198.51.100.10' }
    });
    expect(res.statusCode).toBe(200);
    await fastify.close();
  });

  it('blocks static keyword paths', async () => {
    const fastify = await buildServer({ staticKeywords: ['.php'] });
    const res = await fastify.inject({
      method: 'GET',
      url: '/wp-config.php',
      headers: { 'x-forwarded-for': '198.51.100.11' }
    });
    expect(res.statusCode).toBe(403);
    expect(res.json()).toEqual({ error: 'blocked' });
    await fastify.close();
  });

  it('enforces method policy when enabled', async () => {
    const fastify = await buildServer({
      AIWAF_METHOD_POLICY_ENABLED: true,
      AIWAF_ALLOWED_METHODS: ['GET']
    });
    const res = await fastify.inject({
      method: 'POST',
      url: '/safe',
      headers: { 'x-forwarded-for': '198.51.100.12' }
    });
    expect(res.statusCode).toBe(405);
    expect(res.json()).toEqual({ error: 'blocked' });
    await fastify.close();
  });

  it('enforces header validation requirements', async () => {
    const fastify = await buildServer({
      AIWAF_HEADER_VALIDATION: true,
      AIWAF_REQUIRED_HEADERS: ['x-required-security-header']
    });
    const res = await fastify.inject({
      method: 'GET',
      url: '/safe',
      headers: { 'x-forwarded-for': '198.51.100.13' }
    });
    expect(res.statusCode).toBe(403);
    expect(res.json()).toEqual({ error: 'blocked' });
    await fastify.close();
  });

  it('rate limits repeated requests', async () => {
    const fastify = await buildServer({
      WINDOW_SEC: 60,
      MAX_REQ: 1,
      FLOOD_REQ: 2
    });
    const headers = {
      'x-forwarded-for': '198.51.100.14',
      accept: 'application/json'
    };

    const first = await fastify.inject({ method: 'GET', url: '/safe', headers });
    expect(first.statusCode).toBe(200);

    const second = await fastify.inject({ method: 'GET', url: '/safe', headers });
    expect(second.statusCode).toBe(429);
    expect(second.json()).toEqual({ error: 'too_many_requests' });
    await fastify.close();
  });
});
