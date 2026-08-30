const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const {
  extractFastifyRoutes,
  extractHapiRoutes,
  extractKoaRoutes,
  extractNestRoutes,
  extractNextRoutes,
  generateFrameworkManifest
} = require('../lib/pathManifest');

function apiHandler(req, res) {
  return res.json({ ok: Boolean(req.body?.name) });
}

function formHandler(req, res) {
  return res.render('contact', { name: req.body?.name });
}

describe('path manifest framework extractors', () => {
  it('extracts explicit Fastify route lists', () => {
    const routes = extractFastifyRoutes(null, [
      { method: 'GET', path: '/api/users', handler: apiHandler },
      { method: 'POST', path: '/contact', handler: formHandler }
    ]);

    expect(routes['/api/users'].category).toBe('api');
    expect(routes['/contact'].payload_type).toBe('form');
  });

  it('extracts Hapi server table routes', () => {
    const routes = extractHapiRoutes({
      table: () => [
        { method: 'get', path: '/api/items', settings: { handler: apiHandler } }
      ]
    });

    expect(routes['/api/items'].category).toBe('api');
    expect(routes['/api/items'].methods).toEqual(['GET']);
  });

  it('extracts explicit Koa, Next, and Nest route lists', () => {
    expect(extractKoaRoutes(null, [{ method: 'POST', path: '/contact', handler: formHandler }])['/contact'].category).toBe('form');
    expect(extractNextRoutes([{ method: 'GET', path: '/api/next', handler: apiHandler }])['/api/next'].category).toBe('api');
    expect(extractNestRoutes(null, [{ method: 'GET', path: '/api/nest', handler: apiHandler }])['/api/nest'].category).toBe('api');
  });

  it('generates framework manifests from explicit route lists', () => {
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-fw-manifest-'));
    const output = path.join(tempDir, 'paths.json');
    const manifest = generateFrameworkManifest('fastify', null, output, {
      routes: [{ method: 'GET', path: '/api/users', handler: apiHandler }]
    });

    expect(fs.existsSync(output)).toBe(true);
    expect(manifest.framework).toBe('fastify');
    expect(manifest.routes['/api/users'].category).toBe('api');
  });

  it('CLI manifest command writes route JSON manifests', () => {
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-cli-manifest-'));
    const routesFile = path.join(tempDir, 'routes.json');
    const output = path.join(tempDir, 'paths.json');
    fs.writeFileSync(routesFile, JSON.stringify([
      { method: 'GET', path: '/api/cli' }
    ]));

    const result = spawnSync(process.execPath, [
      path.join(__dirname, '..', 'bin', 'aiwaf.js'),
      'manifest',
      '--framework',
      'express',
      '--routes',
      routesFile,
      '--output',
      output
    ], { encoding: 'utf8' });

    expect(result.status).toBe(0);
    expect(fs.existsSync(output)).toBe(true);
    expect(JSON.parse(fs.readFileSync(output, 'utf8')).routes['/api/cli'].category).toBe('api');
  });
});
