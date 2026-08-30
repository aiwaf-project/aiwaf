require('reflect-metadata');

const request = require('supertest');
const { NestFactory } = require('@nestjs/core');
const { Module } = require('@nestjs/common');
const aiwaf = require('../index');

process.env.NODE_ENV = 'test';

class AppModule {}
Module({})(AppModule);

async function buildApp(opts = {}) {
  const app = await NestFactory.create(AppModule, { logger: false });
  app.use(aiwaf({
    WINDOW_SEC: 60,
    MAX_REQ: 1000,
    FLOOD_REQ: 2000,
    ...opts
  }));
  app.use((req, res) => {
    if (req.url === '/safe') {
      res.status(200).send('ok');
      return;
    }
    res.status(404).send('not found');
  });
  await app.init();
  return app;
}

describe('AIWAF NestJS integration', () => {
  it('blocks static keyword paths', async () => {
    const app = await buildApp({ staticKeywords: ['.php'] });
    const res = await request(app.getHttpServer())
      .get('/wp-config.php')
      .set('X-Forwarded-For', '198.51.100.40')
      .set('Accept', 'application/json');
    expect(res.status).toBe(403);
    expect(res.body).toEqual({ error: 'blocked' });
    await app.close();
  });

  it('enforces method policy when enabled', async () => {
    const app = await buildApp({
      AIWAF_METHOD_POLICY_ENABLED: true,
      AIWAF_ALLOWED_METHODS: ['GET']
    });
    const res = await request(app.getHttpServer())
      .post('/safe')
      .set('X-Forwarded-For', '198.51.100.41')
      .set('Accept', 'application/json');
    expect(res.status).toBe(405);
    expect(res.body).toEqual({ error: 'blocked' });
    await app.close();
  });

  it('rate limits repeated requests', async () => {
    const app = await buildApp({
      WINDOW_SEC: 60,
      MAX_REQ: 1,
      FLOOD_REQ: 2
    });
    const agent = request(app.getHttpServer());
    const headers = {
      'X-Forwarded-For': '198.51.100.42',
      Accept: 'application/json'
    };

    const first = await agent.get('/safe').set(headers);
    expect(first.status).toBe(200);

    const second = await agent.get('/safe').set(headers);
    expect(second.status).toBe(429);
    expect(second.body).toEqual({ error: 'too_many_requests' });
    await app.close();
  });

  it('enforces header validation requirements', async () => {
    const app = await buildApp({
      AIWAF_HEADER_VALIDATION: true,
      AIWAF_REQUIRED_HEADERS: ['x-required-security-header']
    });
    const res = await request(app.getHttpServer())
      .get('/safe')
      .set('X-Forwarded-For', '198.51.100.43')
      .set('Accept', 'application/json');
    expect(res.status).toBe(403);
    expect(res.body).toEqual({ error: 'blocked' });
    await app.close();
  });
});
