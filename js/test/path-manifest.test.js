const fs = require('fs');
const os = require('os');
const path = require('path');
const express = require('express');
const request = require('supertest');
const aiwaf = require('../index');
const {
  buildManifest,
  buildRouteEntry,
  compileManifestToPathRules,
  extractExpressRoutes,
  generateExpressManifest
} = require('../lib/pathManifest');

describe('path manifest', () => {
  it('classifies API routes and compiles protections to path rules', () => {
    const [routePath, entry] = buildRouteEntry({
      routePath: '/api/users/',
      methods: ['GET', 'POST'],
      view: 'users',
      metadata: { response_type: 'json', payload_type: 'json', request_body: true }
    });
    const manifest = buildManifest({ framework: 'test', routes: { [routePath]: entry } });

    expect(entry.category).toBe('api');
    expect(entry.protections.rate_limit.requests).toBe(120);
    expect(compileManifestToPathRules(manifest)).toEqual([
      {
        PREFIX: '/api/users/',
        DISABLE: ['honeypot'],
        RATE_LIMIT: { WINDOW: 60, MAX: 120 }
      }
    ]);
  });

  it('extracts Express routes with source-based API and form signals', () => {
    const app = express();
    app.get('/api/users', (_req, res) => res.json([]));
    app.post('/contact', (req, res) => {
      const name = req.body?.name;
      return res.render('contact', { name });
    });

    const routes = extractExpressRoutes(app);

    expect(routes['/api/users'].category).toBe('api');
    expect(routes['/api/users'].response_type).toBe('json');
    expect(routes['/contact'].category).toBe('form');
    expect(routes['/contact'].payload_type).toBe('form');
  });

  it('uses manifest-compiled rules in middleware', async () => {
    const app = express();
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-manifest-'));
    const manifestPath = path.join(tempDir, 'paths.json');
    generateExpressManifest(app, manifestPath);
    fs.writeFileSync(manifestPath, JSON.stringify(buildManifest({
      framework: 'express',
      routes: {
        '/manifest-allowed.php': {
          methods: ['GET'],
          view: 'manifest',
          protections: {
            ip_keyword_block: { enabled: false },
            rate_limit: { requests: 10, window_seconds: 60 }
          }
        }
      }
    }), null, 2));

    app.use(aiwaf({
      staticKeywords: ['.php'],
      AIWAF_WASM_VALIDATION: false,
      AIWAF_PATH_MANIFEST: manifestPath
    }));
    app.get('/manifest-allowed.php', (_req, res) => res.send('ok'));

    await request(app)
      .get('/manifest-allowed.php')
      .set('X-Forwarded-For', '203.0.113.44')
      .expect(200, 'ok');
  });
});
