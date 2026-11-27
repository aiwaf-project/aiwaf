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

describe('route inventory: init lists what the framework serves', () => {
  test('Express routers mounted at several prefixes, and twice', async () => {
    const output = path.join(dir, 'paths.json');
    const { io } = capture();
    expect(await runInit(['--app', fixture('express', 'multi.js'), '--output', output, '--no-audit'], io)).toBe(0);
    const listed = Object.keys(readRoutes(output)).sort();
    expect(listed).toEqual(['/account/profile', '/me/profile', '/members/:member/posts', '/one/list', '/two/list', '/users/:user/posts']);
    // Every listed route answers through the app itself, and an unlisted one doesn't.
    const app = require(fixture('express', 'multi.js'));
    const server = await new Promise(resolve => { const started = app.listen(0, () => resolve(started)); });
    try {
      const status = async route => (await fetch(`http://127.0.0.1:${server.address().port}${route.replace(/:\w+/g, 'x')}`)).status;
      for (const route of listed) expect([route, await status(route)]).toEqual([route, 200]);
      expect(await status('/three/list')).toBe(404);
    } finally {
      await new Promise(resolve => server.close(resolve));
    }
  });

  test('Fastify routes registered before the AIWAF plugin', async () => {
    const output = path.join(dir, 'paths.json');
    const { io } = capture();
    const app = fixture('fastify-before', 'app.js');
    expect(await runInit(['--app', app, '--output', output, '--source', fixture('fastify-before')], io)).toBe(0);
    const routes = readRoutes(output);
    expect(Object.keys(routes).sort()).toEqual(['/early', '/early-order', '/early-route', '/late']);
    expect(routes['/late'].inputs).toEqual({ n: 'int' });
    // The schemas of routes registered before the plugin come from the source.
    expect(routes['/early-order']).toMatchObject({ inputs: { quantity: 'int' }, inputs_complete: true });
    expect(routes['/early-route']).toMatchObject({ inputs: { sku: 'text', quantity: 'int' }, inputs_complete: true });
    const fastify = require(app);
    await fastify.ready();
    try {
      for (const [route, entry] of Object.entries(routes)) {
        for (const method of entry.methods) expect(fastify.hasRoute({ url: route, method })).toBe(true);
      }
    } finally {
      await fastify.close();
    }
  });
});

describe('Fastify input rules', () => {
  test('types stay with their method, $refs resolve, and a raw text body is no field', async () => {
    const output = path.join(dir, 'paths.json');
    const { io } = capture();
    expect(await runInit(['--app', fixture('fastify-refs', 'app.js'), '--output', output, '--source', fixture('fastify-refs')], io)).toBe(0);
    const routes = readRoutes(output);
    // GET sends ref as any text; POST's integer can't hold for both.
    expect(routes['/orders'].inputs).toEqual({ customer: 'text', ref: 'text' });
    expect(routes['/orders'].inputs_by_method).toEqual({ GET: ['ref'], POST: ['customer', 'ref'] });
    expect(routes['/orders'].inputs_nested).toEqual({ customer: ['email', 'tier'] });
    expect(routes['/orders'].inputs_complete).toBe(true);
    expect(routes['/notes'].inputs).toEqual({});
    expect(routes['/notes'].inputs_complete).toBe(false);
    expect(routes['/notes'].inputs_incomplete.map(item => item.code)).toEqual(['raw_body']);
  });
});

describe('Fastify named handlers', () => {
  test('each named handler\'s reading belongs to its own method', async () => {
    const output = path.join(dir, 'paths.json');
    const { io } = capture();
    expect(await runInit(['--app', fixture('fastify-named', 'app.js'), '--output', output, '--source', fixture('fastify-named')], io)).toBe(0);
    const route = readRoutes(output)['/plans'];
    expect(route.inputs).toEqual({ dryRun: 'bool', force: 'bool', mode: 'text', weeks: 'int' });
    expect(route.inputs_by_method).toEqual({ DELETE: ['force'], GET: ['mode'], POST: ['dryRun', 'mode', 'weeks'] });
    // POST's own integer mode, for engines that check a method's own type.
    expect(route.inputs_types_by_method).toEqual({ POST: { mode: 'int' } });
    expect(route.inputs_complete).toBe(true);
  });
});

describe('more exact input rules', () => {
  const run = async (name, ...flags) => {
    const output = path.join(dir, `${name}.json`);
    const { io, lines } = capture();
    expect(await runInit(['--app', fixture(name, 'app.js'), '--output', output, '--source', fixture(name), ...flags], io)).toBe(0);
    return { routes: readRoutes(output), lines };
  };

  test('query and body keep their own types for one name', async () => {
    const { routes } = await run('fastify-locations');
    expect(routes['/reports'].inputs).toEqual({ limit: 'text', since: 'int' });
    expect(routes['/reports'].inputs_types_by_location).toEqual({ POST: { query: { limit: 'int' } } });
  });

  test('an imported schema is the same before and after the plugin', async () => {
    const { routes } = await run('fastify-imported');
    expect(routes['/webhooks/payments'].inputs).toEqual({ attempt: 'int', eventId: 'uuid', successful: 'bool' });
    expect(routes['/webhooks/payments']).toEqual(expect.objectContaining({ inputs: routes['/webhooks/after'].inputs, inputs_complete: true }));
    // A route list can't load the schema: the route says so and isn't complete.
    const list = path.join(dir, 'routes.json');
    fs.writeFileSync(list, JSON.stringify([{ method: 'POST', path: '/webhooks/payments' }]));
    const output = path.join(dir, 'listed.json');
    const { io } = capture();
    expect(await runManifest(['--framework', 'fastify', '--routes', list, '--source', fixture('fastify-imported'), '--output', output], io)).toBe(0);
    const listed = readRoutes(output)['/webhooks/payments'];
    expect(listed.inputs_complete).toBe(false);
    expect(listed.inputs_incomplete.map(item => item.code)).toContain('external_class');
  });

  test('an app-level hook\'s reads are every route\'s', async () => {
    const { routes } = await run('fastify-hooks');
    expect(routes['/invoices']).toMatchObject({ inputs: { page: 'int', tenant: 'text' }, inputs_complete: true });
    expect(routes['/scoped'].inputs.tenant).toBe('text');
    expect(routes['/scoped'].inputs_complete).toBe(false);
  });

  test('closed schemas give nested keys by path, open ones none', async () => {
    const { routes } = await run('fastify-deep');
    expect(routes['/orders'].inputs_nested).toEqual({
      customer: ['address', 'email'], 'customer.address': ['city', 'zip'], 'lines[]': ['qty', 'sku']
    });
  });

  test('init says which checks the installed engine can\'t make', () => {
    const { engineGaps } = require('../lib/initCommand');
    const manifest = { routes: {
      '/a': { inputs_types_by_method: { POST: { x: 'int' } }, inputs_types_by_location: { POST: { query: { x: 'int' } } }, inputs_nested: { 'a.b': ['c'] } },
      '/b': { inputs_nested: { top: ['x'] } }, '/c': null
    } };
    expect(engineGaps(manifest, null)).toEqual([
      'query and body types (inputs_types_by_location)', 'per-method types (inputs_types_by_method)', 'nested keys below the first level (inputs_nested)'
    ]);
    expect(engineGaps(manifest, () => true)).toEqual([]);
    expect(engineGaps({ routes: { '/b': { inputs_nested: { top: ['x'] } } } }, null)).toEqual([]);
  });

  test('a failed write keeps the old manifest', () => {
    const { writeManifest } = require('../lib/pathManifest');
    const output = path.join(dir, 'kept.json');
    writeManifest({ routes: { '/a': {} } }, output);
    const loop = {};
    loop.self = loop;
    expect(() => writeManifest({ routes: { '/b': loop } }, output)).toThrow();
    expect(JSON.parse(fs.readFileSync(output, 'utf8'))).toEqual({ routes: { '/a': {} } });
    expect(fs.readdirSync(dir).filter(name => name.endsWith('.tmp'))).toEqual([]);
  });
});

describe('the audit', () => {
  test('a route whose handler the audit never read is not complete', async () => {
    const output = path.join(dir, 'paths.json');
    const unrelated = path.join(dir, 'unrelated');
    fs.mkdirSync(unrelated);
    fs.writeFileSync(path.join(unrelated, 'other.js'), 'module.exports = () => 1;\n');
    const { io } = capture();
    expect(await runInit(['--app', fixture('fastify-schema', 'app.js'), '--output', output, '--source', unrelated], io)).toBe(0);
    const route = readRoutes(output)['/orders'];
    expect(route).toMatchObject({ inputs: { quantity: 'int', sku: 'text' }, inputs_complete: false });
    expect(route.inputs_incomplete.map(item => item.code)).toEqual(['unread_view']);
  });

  test('a missing audit source fails init and keeps the old manifest', async () => {
    const output = path.join(dir, 'paths.json');
    fs.writeFileSync(output, '{"routes": {"/kept": {}}}');
    const { lines, io } = capture();
    expect(await runInit(['--app', fixture('express', 'app.js'), '--output', output, '--source', path.join(dir, 'nowhere')], io)).toBe(1);
    expect(lines.err.join('\n')).toMatch(/The audit source .*nowhere doesn't exist/);
    expect(fs.readFileSync(output, 'utf8')).toBe('{"routes": {"/kept": {}}}');
  });

  test('an audit that reads no files counts as no audit', async () => {
    const output = path.join(dir, 'paths.json');
    const empty = path.join(dir, 'empty');
    fs.mkdirSync(empty);
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const { io } = capture();
      expect(await runInit(['--app', fixture('fastify-schema', 'app.js'), '--output', output, '--source', empty], io)).toBe(0);
      expect(warn.mock.calls.flat().join('\n')).toMatch(/found no JavaScript or TypeScript files/);
    } finally {
      warn.mockRestore();
    }
    const manifest = JSON.parse(fs.readFileSync(output, 'utf8'));
    expect(manifest.injection_audit).toBeUndefined();
    expect(manifest.routes['/orders']).toMatchObject({ inputs: { quantity: 'int', sku: 'text' }, inputs_complete: false });
  });

  test('a scan that fails is an error', () => {
    const pathManifest = require('../lib/pathManifest');
    const analysis = require('../lib/injectionAnalysis');
    const spy = jest.spyOn(analysis, 'analyzePath').mockImplementation(() => { throw new Error('parser crashed'); });
    try {
      expect(() => pathManifest.auditIntoManifest({ routes: {} }, dir)).toThrow(pathManifest.AuditError);
    } finally {
      spy.mockRestore();
    }
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

  test('saves a Fastify schema\'s integer, with and without the audit', () => {
    const app = fixture('fastify-schema', 'app.js');
    const run = (...flags) => {
      const output = path.join(dir, `paths${flags.length}.json`);
      execFileSync(process.execPath, [CLI, 'init', '--app', app, '--output', output, ...flags], { cwd: dir, encoding: 'utf8', timeout: 30000 });
      return readRoutes(output)['/orders'];
    };
    const audited = run('--source', fixture('fastify-schema'));
    expect(audited.inputs).toEqual({ quantity: 'int', sku: 'text' });
    expect(audited.inputs_complete).toBe(true);
    expect(audited.response_type).toBe('json');
    const unread = run('--no-audit', '--source', fixture('fastify-schema'));
    expect(unread.inputs).toEqual({ quantity: 'int', sku: 'text' });
    expect(unread.inputs_complete).toBe(false);
    expect(unread.inputs_incomplete.map(item => item.code)).toEqual(['not_audited']);
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
