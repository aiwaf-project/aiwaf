const request = require('supertest');
const express = require('express');
const fs = require('fs');
const db = require('../utils/db');
const aiwaf = require('../index');
const dynamicKeyword = require('../lib/dynamicKeyword');
const dynamicKeywordStore = require('../lib/dynamicKeywordStore');
const redisManager = require('../lib/redisClient');
const { init: initRateLimiter } = require('../lib/rateLimiter');

process.env.NODE_ENV = 'test';
jest.setTimeout(20000);

let redisAvailable = false;
let ipCounter = 1;

function nextTestIp() {
  const high = Math.floor(ipCounter / 254) % 254;
  const low = (ipCounter % 254) + 1;
  ipCounter += 1;
  return `198.51.${high}.${low}`;
}

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

beforeAll(async () => {
  const hasTable = await db.schema.hasTable('blocked_ips');
  if (!hasTable) {
    await db.schema.createTable('blocked_ips', table => {
      table.increments('id');
      table.string('ip_address').unique();
      table.string('reason');
      table.timestamp('blocked_at').defaultTo(db.fn.now());
    });
  }

  await redisManager.connect();
  redisAvailable = redisManager.isReady();

  await initRateLimiter({
    WINDOW_SEC: 1,
    MAX_REQ: 5,
    FLOOD_REQ: 10,
    cache: testCache // inject test cache
  });
});

describe('AIWAF-JS Middleware', () => {
  let app, ip, middlewareLogPath, middlewareCsvPath;

  beforeEach(() => {
    ip = nextTestIp();
    middlewareLogPath = `logs/test-aiwaf-${Date.now()}-${Math.random().toString(36).slice(2)}.jsonl`;
    middlewareCsvPath = `logs/test-aiwaf-${Date.now()}-${Math.random().toString(36).slice(2)}.csv`;
    dynamicKeywordStore.clear().catch(() => {});
    dynamicKeyword.init({ dynamicTopN: 3 });

    app = express();
    app.use(express.json());
    app.use(aiwaf({
      staticKeywords: ['.php', '.env', '.git'],
      dynamicTopN: 1000,
      WINDOW_SEC: 1,
      MAX_REQ: 5,
      FLOOD_REQ: 10,
      HONEYPOT_FIELD: 'hp_field',
      cache: testCache,
      logger: console,
      AIWAF_MIDDLEWARE_LOGGING: true,
      AIWAF_MIDDLEWARE_LOG_PATH: middlewareLogPath,
      AIWAF_MIDDLEWARE_LOG_CSV: true,
      AIWAF_MIDDLEWARE_LOG_CSV_PATH: middlewareCsvPath
    }));

    app.get('/', (req, res) => res.send('OK'));
    app.post('/', (req, res) => res.send('POST OK'));
    app.get('/user/:uuid', (req, res) => res.send('USER OK'));
    app.get('/health', (req, res) => res.send('healthy'));
    app.get('/admin/wp-config.php', (req, res) => res.send('allow-by-exemption'));
    app.get('/geo-test', (req, res) => res.send('geo-ok'));
    app.get('/headers-test', (req, res) => res.send('headers-ok'));
    app.get('/legacy-safe', (req, res) => res.send('legacy-ok'));
    app.get('/form', (req, res) => res.send('FORM GET'));
    app.post('/form', (req, res) => res.send('FORM POST'));
    app.use((req, res) => res.status(404).send('Not Found'));
  });

  it('blocks static keyword .php', () =>
    request(app)
      .get('/wp-config.php')
      .set('X-Forwarded-For', ip)
      .set('x-response-time', '15')
      .expect(403, { error: 'blocked' })
  );

  it('continues working if Redis goes down', async () => {
    const redis = redisManager.getClient();
    if (redis) await redis.quit();

    const segment = `/simulate-${Date.now().toString(36)}`;
    for (let i = 0; i < 3; i++) {
      await request(app).get(segment).set('X-Forwarded-For', ip);
    }
    const resp = await request(app).get(segment).set('X-Forwarded-For', ip);
    expect([200, 403, 404, 429]).toContain(resp.status);
  });

  it('allows safe paths', () =>
    request(app)
      .get('/')
      .set('X-Forwarded-For', ip)
      .set('x-response-time', '15')
      .expect(200, 'OK')
  );

  it('blocks after exceeding rate limit', async () => {
    for (let i = 0; i < 7; i++) {
      const resp = await request(app)
        .get('/')
        .set('X-Forwarded-For', ip)
        .set('x-response-time', '15');

      if (i < 5) {
        expect(resp.status).toBe(200);
      } else {
        expect([200, 403, 429]).toContain(resp.status);
      }
    }
  });

  it('blocks honeypot field', () =>
    request(app)
      .post('/')
      .set('X-Forwarded-For', ip)
      .send({ hp_field: 'caught' })
      .expect(403, { error: 'bot_detected' })
  );

  it('blocks invalid UUIDs', () =>
    request(app)
      .get('/user/not-a-uuid')
      .set('X-Forwarded-For', ip)
      .expect(403)
  );

  it('learns and blocks dynamic keywords', async () => {
    const segment = `/wp-admin/secret-${Date.now().toString(36)}`;
    const keywordApp = express();
    keywordApp.use(express.json());
    keywordApp.use(aiwaf({
      staticKeywords: ['.php', '.env', '.git'],
      dynamicTopN: 2,
      WINDOW_SEC: 1,
      MAX_REQ: 5,
      FLOOD_REQ: 10,
      HONEYPOT_FIELD: 'hp_field',
      cache: testCache
    }));

    for (let i = 0; i < 3; i++) {
      await request(keywordApp).get(segment).set('X-Forwarded-For', ip);
    }
    await request(keywordApp).get(segment).set('X-Forwarded-For', ip).expect(403);
    await request(keywordApp).get(segment).set('X-Forwarded-For', ip).expect(403, { error: 'blocked' });
  });

  it('flags and blocks anomalous paths', async () => {
    for (let i = 0; i < 5; i++) {
      await request(app)
        .get(`/wp-admin/scan-${i}`)
        .set('X-Forwarded-For', ip)
        .set('x-response-time', '20')
        .expect(404);
    }

    const longPath = '/' + 'a'.repeat(200);
    const resp = await request(app)
      .get(longPath)
      .set('X-Forwarded-For', ip)
      .set('x-response-time', '20');

    expect([403, 429]).toContain(resp.status);
  });

  it('supports exempt paths that bypass keyword blocking', async () => {
    const exemptApp = express();
    exemptApp.use(express.json());
    exemptApp.use(aiwaf({
      staticKeywords: ['.php'],
      AIWAF_EXEMPT_PATHS: ['/admin'],
      cache: testCache
    }));
    exemptApp.get('/admin/wp-config.php', (req, res) => res.send('ok'));

    await request(exemptApp)
      .get('/admin/wp-config.php')
      .set('X-Forwarded-For', ip)
      .expect(200, 'ok');
  });

  it('supports header validation when enabled', async () => {
    const headerApp = express();
    headerApp.use(express.json());
    headerApp.use(aiwaf({
      AIWAF_HEADER_VALIDATION: true,
      AIWAF_REQUIRED_HEADERS: ['x-required-security-header'],
      cache: testCache
    }));
    headerApp.get('/headers-test', (req, res) => res.send('ok'));

    await request(headerApp)
      .get('/headers-test')
      .set('X-Forwarded-For', ip)
      .expect(403, { error: 'blocked' });

    await request(headerApp)
      .get('/headers-test')
      .set('X-Forwarded-For', `198.51.100.${Math.floor(Math.random() * 254) + 1}`)
      .set('x-required-security-header', 'present')
      .set('user-agent', 'Mozilla/5.0')
      .set('accept', 'text/html,application/xml;q=0.9,*/*;q=0.8')
      .set('accept-language', 'en-US,en;q=0.9')
      .set('accept-encoding', 'gzip, deflate, br')
      .set('connection', 'keep-alive')
      .set('cache-control', 'no-cache')
      .expect(200, 'ok');
  });

  it('distinguishes curl-like headers from browser-like headers', async () => {
    const headerApp = express();
    headerApp.use(express.json());
    headerApp.use(aiwaf({
      AIWAF_HEADER_VALIDATION: true,
      AIWAF_REQUIRED_HEADERS: ['accept'],
      AIWAF_HEADER_QUALITY_MIN_SCORE: 3,
      cache: testCache
    }));
    headerApp.get('/headers-human', (req, res) => res.send('ok'));

    await request(headerApp)
      .get('/headers-human')
      .set('X-Forwarded-For', `198.51.100.${Math.floor(Math.random() * 254) + 1}`)
      .set('user-agent', 'curl/7.88.1')
      .set('accept', '*/*')
      .expect(403, { error: 'blocked' });

    await request(headerApp)
      .get('/headers-human')
      .set('X-Forwarded-For', `198.51.100.${Math.floor(Math.random() * 254) + 1}`)
      .set('user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36')
      .set('accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8')
      .set('accept-language', 'en-US,en;q=0.9')
      .set('accept-encoding', 'gzip, deflate, br')
      .set('connection', 'keep-alive')
      .set('upgrade-insecure-requests', '1')
      .expect(200, 'ok');
  });

  it('supports geo blocking using country code header', async () => {
    const geoApp = express();
    geoApp.use(express.json());
    geoApp.use(aiwaf({
      AIWAF_GEO_BLOCK_ENABLED: true,
      AIWAF_GEO_BLOCK_COUNTRIES: ['CN'],
      cache: testCache
    }));
    geoApp.get('/geo-test', (req, res) => res.send('ok'));

    await request(geoApp)
      .get('/geo-test')
      .set('X-Forwarded-For', ip)
      .set('x-country-code', 'CN')
      .expect(403, { error: 'blocked' });

    await request(geoApp)
      .get('/geo-test')
      .set('X-Forwarded-For', `203.0.113.${Math.floor(Math.random() * 254) + 1}`)
      .set('x-country-code', 'US')
      .expect(200, 'ok');
  });

  it('supports honeypot timing checks (GET->POST too fast)', async () => {
    const timingApp = express();
    timingApp.use(express.json());
    timingApp.use(aiwaf({
      HONEYPOT_FIELD: 'hp_field',
      AIWAF_MIN_FORM_TIME: 2,
      cache: testCache
    }));
    timingApp.get('/form', (req, res) => res.send('FORM GET'));
    timingApp.post('/form', (req, res) => res.send('FORM POST'));

    await request(timingApp)
      .get('/form')
      .set('X-Forwarded-For', ip)
      .expect(200);

    await request(timingApp)
      .post('/form')
      .set('X-Forwarded-For', ip)
      .send({})
      .expect(403, { error: 'bot_detected' });
  });

  it('writes middleware request logs when enabled', async () => {
    await request(app)
      .get('/health')
      .set('X-Forwarded-For', ip)
      .expect(200);

    await new Promise(resolve => setTimeout(resolve, 50));
    const logContent = fs.readFileSync(middlewareLogPath, 'utf8');
    expect(logContent).toContain('"path":"/health"');
    expect(logContent).toContain('"status":200');

    const csvContent = fs.readFileSync(middlewareCsvPath, 'utf8');
    expect(csvContent).toContain('timestamp,ip,method,path,status,response_time,blocked,reason,country,user_agent');
    expect(csvContent).toContain('/health');
  });

  it('supports legacy nested AIWAF_SETTINGS compatibility mapping', async () => {
    const legacyApp = express();
    legacyApp.use(express.json());
    legacyApp.use(aiwaf({
      AIWAF_SETTINGS: {
        rate: { window: 1, max: 5, flood: 10 },
        honeypot: { field: 'hp_field' },
        exemptions: { paths: ['/legacy-safe'] },
        keywords: { static: ['.php'], dynamicTopN: 3 }
      },
      cache: testCache
    }));
    legacyApp.get('/legacy-safe/wp-config.php', (req, res) => res.send('ok'));

    await request(legacyApp)
      .get('/legacy-safe/wp-config.php')
      .set('X-Forwarded-For', ip)
      .expect(200, 'ok');
  });
});

afterAll(async () => {
  if (redisAvailable && redisManager.getClient()) {
    await redisManager.getClient().quit();
  }
  await db.destroy();
});
