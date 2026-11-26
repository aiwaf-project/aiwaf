const { execFileSync, spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { runInit, runManifest, parseArgs, describeAudit, EmptyManifestError, INIT_HELP } = require('../lib/initCommand');

const FIXTURES = path.join(__dirname, 'fixtures', 'init');
const CLI = path.join(__dirname, '..', 'bin', 'aiwaf.js');
const fixture = (...parts) => path.join(FIXTURES, ...parts);

let dir;
beforeEach(() => { dir = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-init-')); });
afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true });
  delete global.aiwafFixtureServer;
  delete global.aiwafFixtureListened;
});

function capture() {
  const lines = { out: [], err: [] };
  return { lines, io: { out: text => lines.out.push(text), err: text => lines.err.push(text) } };
}

const readRoutes = file => JSON.parse(fs.readFileSync(file, 'utf8')).routes;

describe('aiwaf init', () => {
  test('writes every route the Express app registers, mounted routers included', async () => {
    const output = path.join(dir, 'paths.json');
    const { lines, io } = capture();
    const code = await runInit(['--app', fixture('express', 'app.js'), '--output', output, '--source', fixture('express')], io);
    expect(code).toBe(0);
    const routes = readRoutes(output);
    expect(Object.keys(routes).sort()).toEqual(['/', '/account/login', '/account/profile', '/api/items/:id', '/contact', '/health', '/orders']);
    expect(routes['/orders'].response_type).toBe('json');
    expect(routes['/account/login'].inputs).toEqual({ password: 'text', username: 'text' });
    expect(routes['/contact'].response_type).toBe('mixed');
    const text = lines.out.join('\n');
    expect(text).toMatch(/Framework: express/);
    expect(text).toMatch(/App: .*app\.js/);
    expect(text).toMatch(/Routes: 7 \(5 API\)/);
    expect(text).toMatch(/Injection checks: 0 routes from \d+ files; inputs and types for 7 routes, complete for 7/);
  });

  test('refuses to write a manifest with no routes, leaving the old one', async () => {
    const output = path.join(dir, 'paths.json');
    fs.writeFileSync(output, '{"routes": {"/kept": {}}}');
    const { lines, io } = capture();
    expect(await runInit(['--app', fixture('express', 'empty.js'), '--output', output, '--no-audit'], io)).toBe(1);
    expect(lines.err.join('\n')).toMatch(/No routes found in .*empty\.js \(express\), so .*paths\.json was not written/);
    expect(fs.readFileSync(output, 'utf8')).toBe('{"routes": {"/kept": {}}}');
  });

  test('reads a Koa router and names the factory export it called', async () => {
    const output = path.join(dir, 'paths.json');
    const { lines, io } = capture();
    expect(await runInit(['--app', fixture('koa', 'app.js'), '-o', output, '--no-audit'], io)).toBe(0);
    const routes = readRoutes(output);
    expect(routes['/items/:id'].methods).toEqual(['GET']);
    expect(routes['/items'].methods).toEqual(['POST']);
    expect(routes['/ignored']).toBeUndefined();
    expect(await runInit([fixture('express', 'factory.js'), '-e', 'createApp', '-o', output, '--no-audit'], io)).toBe(0);
    expect(lines.out.join('\n')).toMatch(/App: .*factory\.js \(export createApp\)/);
  });

  test('says why when a Koa app has no router', async () => {
    const app = path.join(FIXTURES, 'koa', 'bare.js');
    fs.writeFileSync(app, "const Koa = require('koa');\nmodule.exports = new Koa();\n");
    try {
      const { lines, io } = capture();
      expect(await runInit(['--app', app, '-o', path.join(dir, 'p.json'), '--no-audit'], io)).toBe(1);
      expect(lines.err.join('\n')).toMatch(/Koa keeps routes only in @koa\/router or koa-router middleware/);
    } finally {
      fs.rmSync(app, { force: true });
    }
  });

  test('takes a route list, and the source supplies its handlers', async () => {
    const list = path.join(dir, 'routes.json');
    fs.writeFileSync(list, JSON.stringify([
      { method: 'POST', path: '/orders' }, { method: 'GET', path: '/account/profile' }, { method: 'GET', path: '/' }
    ]));
    const output = path.join(dir, 'paths.json');
    const { io } = capture();
    expect(await runInit(['--routes', list, '-o', output, '--source', fixture('express'), '--framework', 'express'], io)).toBe(0);
    const routes = readRoutes(output);
    expect(routes['/orders'].response_type).toBe('json');
    expect(routes['/account/profile'].response_type).toBe('json');
    expect(routes['/'].response_type).toBe('html');
  });

  test('explains bad arguments and load failures', async () => {
    const { lines, io } = capture();
    expect(await runInit(['--wait'], io)).toBe(2);
    expect(lines.err[0]).toMatch(/--wait needs a value[\s\S]*Usage: aiwaf init/);
    expect(await runInit(['--help'], io)).toBe(0);
    expect(lines.out).toContain(INIT_HELP);
    expect(await runInit(['--app', fixture('express', 'broken.js')], io)).toBe(1);
    expect(lines.err.pop()).toMatch(/database unreachable/);
  });
});

describe('aiwaf manifest', () => {
  test('without a route list, loads the app as init does', async () => {
    const output = path.join(dir, 'paths.json');
    const { lines, io } = capture();
    expect(await runManifest(['--app', fixture('express', 'app.js'), '--output', output, '--no-audit'], io)).toBe(0);
    expect(JSON.parse(lines.out[0])).toMatchObject({ ok: true, framework: 'express', routes: 7, unlinked_findings: null });
  });

  test('with a route list, refuses an empty one', async () => {
    const list = path.join(dir, 'routes.json');
    fs.writeFileSync(list, '[]');
    const { lines, io } = capture();
    expect(await runManifest(['--routes', list, '--output', path.join(dir, 'p.json'), '--no-audit'], io)).toBe(1);
    expect(JSON.parse(lines.err[0]).error).toMatch(/No routes found in the --routes list/);
    expect(fs.existsSync(path.join(dir, 'p.json'))).toBe(false);
    fs.writeFileSync(list, JSON.stringify([{ method: 'GET', path: '/api/a' }]));
    expect(await runManifest(['--routes', list, '--output', path.join(dir, 'p.json'), '--source', dir], io)).toBe(0);
    expect(JSON.parse(lines.out[0])).toMatchObject({ ok: true, routes: 1, injection_check_routes: 0 });
    expect(await runManifest(['--bogus'], io)).toBe(2);
  });
});

describe('parseArgs and describeAudit', () => {
  test('parse every option, and reject a route list that is not an array', () => {
    const list = path.join(dir, 'r.json');
    fs.writeFileSync(list, '[]');
    expect(parseArgs(['-a', 'x.js', '-f', 'Fastify', '--wait', '10', '-r', list])).toMatchObject({ app: 'x.js', framework: 'fastify', waitMs: 10, routes: [] });
    fs.writeFileSync(list, '{}');
    expect(() => parseArgs(['--routes', list])).toThrow(/JSON array/);
    expect(() => parseArgs(['--wait', '-1'])).toThrow(/milliseconds/);
    expect(() => parseArgs(['a.js', 'b.js'])).toThrow(/Unknown option: b\.js/);
  });

  test('describe an audit as Python does', () => {
    expect(describeAudit({})).toBe('Injection checks: not run');
    expect(describeAudit({
      routes: { '/a': { injection_risks: {} }, '/b': {} },
      injection_audit: { files_scanned: 3, routes_with_inputs: 2, routes_with_complete_inputs: 1, incomplete_input_reasons: { helper: 1, dynamic_key: 2 }, unlinked_findings: 4 }
    })).toBe('Injection checks: 1 routes from 3 files; inputs and types for 2 routes, complete for 1 (incomplete because of dynamic_key 2, helper 1) (4 findings couldn\'t be tied to a route)');
    expect(new EmptyManifestError('x', 'o.json').message).toMatch(/No routes found in x, so o\.json was not written/);
  });
});

describe('the aiwaf command', () => {
  test('loads an ES module app, exits on its own, and binds no port', () => {
    const output = path.join(dir, 'paths.json');
    const out = execFileSync(process.execPath, [CLI, 'init', '--app', fixture('esm', 'app.mjs'), '--output', output, '--no-audit'], {
      encoding: 'utf8', timeout: 30000
    });
    expect(out).toMatch(/Routes: 1 \(1 API\)/);
    expect(Object.keys(readRoutes(output))).toEqual(['/esm']);
  });

  test('exits 1 and writes nothing for an app with no routes', () => {
    const output = path.join(dir, 'paths.json');
    const result = spawnSync(process.execPath, [CLI, 'manifest', '--app', fixture('express', 'empty.js'), '--output', output, '--no-audit'], {
      encoding: 'utf8', timeout: 30000
    });
    expect(result.status).toBe(1);
    expect(JSON.parse(result.stderr).ok).toBe(false);
    expect(fs.existsSync(output)).toBe(false);
  });
});
