const request = require('supertest');
const express = require('express');
const aiwaf = require('../index');
const exemptionStore = require('../lib/exemptionStore');
const db = require('../utils/db');

describe('exemptions DB integration', () => {
  afterAll(async () => {
    try {
      await db('ip_exemptions').del();
      await db('path_exemptions').del();
    } catch (err) {
      // noop
    }
    await db.destroy();
  });

  it('allows requests for DB-exempt IPs and paths', async () => {
    await exemptionStore.addIp('198.51.120.20', 'test');
    await exemptionStore.addPath('/admin', 'test');

    const app = express();
    app.use(express.json());
    app.use(aiwaf({
      staticKeywords: ['.php'],
      AIWAF_EXEMPTIONS_DB: true,
      WINDOW_SEC: 60,
      MAX_REQ: 1000,
      FLOOD_REQ: 2000
    }));
    app.get('/admin/wp-config.php', (req, res) => res.send('ok'));

    await request(app)
      .get('/admin/wp-config.php')
      .set('X-Forwarded-For', '198.51.120.20')
      .expect(200, 'ok');
  });
});
