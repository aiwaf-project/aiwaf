const fs = require('fs');
const os = require('os');
const path = require('path');
const { analyzeSource, runAuditCommand } = require('../lib/injectionAnalysis');
const { loadManifest, compileManifestToPathRules } = require('../lib/pathManifest');
const {
  applyReport, compileRisks, compileRoute, createObserver, formatCounts, inputFromSource, inspectRisks, requestInputs, ruleFor, updateManifest
} = require('../lib/routeRisks');

const CASES = JSON.parse(fs.readFileSync(path.join(__dirname, '../../tests/fixtures/route_risk_cases.json'), 'utf8'));
const manifestWith = (risks, route = '/search') => ({ routes: { [route]: { view: 'search', injection_risks: risks }, '/health': { view: 'health' } } });
const inspect = (manifest, ...args) => inspectRisks(compileRisks(manifest), ...args);
const report = (source, file = 'app.js') => ({ files_scanned: 1, findings: analyzeSource(source, file) });

describe('route risks', () => {
  let directory;
  beforeEach(() => { directory = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-risk-')); });
  afterEach(() => fs.rmSync(directory, { recursive: true, force: true }));
  const save = manifest => {
    const file = path.join(directory, 'paths.json');
    fs.writeFileSync(file, JSON.stringify(manifest));
    return file;
  };

  test.each(CASES.rules.map(item => [item.sink, item.value, item.rule]))('shared rule %s: %s', (sink, value, rule) => {
    expect(ruleFor(sink, value)).toBe(rule);
  });

  test.each(CASES.routes.map(item => [item.route, item.path, item.values]))('shared route %s: %s', (route, target, values) => {
    const { pattern, names } = compileRoute(route);
    const match = pattern.exec(target);
    if (values === null) {
      expect(match).toBeNull();
    } else {
      const found = {};
      names.forEach((name, index) => { if (name) found[name] = match[index + 1]; });
      expect(found).toEqual(values);
    }
  });

  test('only the named input is checked', () => {
    const manifest = manifestWith([{ sink: 'path', input: 'file', confidence: 'medium' }]);
    expect(inspect(manifest, '/search', { other: '../x' })).toEqual([]);
    expect(inspect(manifest, '/search', { file: '../x' })).toEqual([
      { route: '/search', sink: 'path', input: 'file', rule: 'path_traversal', confidence: 'medium' }
    ]);
  });

  test('an unknown input checks everything', () => {
    const manifest = manifestWith([{ sink: 'command', input: '' }]);
    expect(inspect(manifest, '/search', {}, { deep: { cmd: '$(id)' } })[0].rule).toBe('command_substitution');
    expect(inspect(manifest, '/search', 'a=$(id)')[0].input).toBe('');
  });

  test('nested values count as the field that holds them', () => {
    const manifest = manifestWith([{ sink: 'nosql', input: 'user' }]);
    expect(inspect(manifest, '/search', {}, { user: { $ne: 1 } })[0].rule).toBe('nosql_operator');
    expect(inspect(manifest, '/search', 'user[$ne]=1')[0].rule).toBe('nosql_operator');
    expect(inspect(manifest, '/search', {}, '{"user": {"$ne": 1}}')[0].rule).toBe('nosql_operator');
  });

  test('only the rules for the operation apply, and route values are inputs', () => {
    expect(inspect(manifestWith([{ sink: 'path', input: 'q' }]), '/search', { q: '$(id)' })).toEqual([]);
    expect(inspect(manifestWith([{ sink: 'path', input: 'name' }], '/files/:name*'), '/files/../x')[0].rule).toBe('path_traversal');
  });

  test('routes without saved risks get nothing', () => {
    const manifest = manifestWith([{ sink: 'path', input: '' }]);
    expect(inspect(manifest, '/health', { x: '../y' })).toEqual([]);
    expect(inspect(manifest, '/other', { x: '../y' })).toEqual([]);
    expect(compileRisks(null)).toEqual([]);
    expect(compileRisks({ routes: 'bad' })).toEqual([]);
  });

  test('exact routes win and repeated matches report once', () => {
    const manifest = { routes: {
      '/items/:id': { injection_risks: [{ sink: 'path', input: '' }] },
      '/items/new': { injection_risks: [{ sink: 'command', input: '' }] }
    } };
    expect(inspect(manifest, '/items/new', { x: '$(id)' })[0].route).toBe('/items/new');
    expect(inspect(manifestWith([{ sink: 'path', input: '' }]), '/search', { a: '../x', b: '../y' })).toHaveLength(1);
  });

  test('request inputs read each format and are bounded', () => {
    expect(requestInputs(null, Buffer.from('a=b'))).toContainEqual(['a', 'b']);
    expect(requestInputs(null, 'plain')).toContainEqual(['', 'plain']);
    expect(requestInputs(null, '{broken')).toContainEqual(['', '{broken']);
    expect(requestInputs({ a: ['b'] })).toContainEqual(['a', 'b']);
    expect(requestInputs(null, { pipe() {} })).toEqual([]);
    expect(requestInputs(null, null)).toEqual([]);
    expect(requestInputs(Array.from({ length: 5000 }, (_, i) => `k${i}=v`).join('&'), null).length).toBeLessThanOrEqual(512);
    let deep = { a: 'x' };
    for (let index = 0; index < 20; index += 1) deep = { a: deep };
    expect(requestInputs(null, deep)).not.toContainEqual(['a', 'x']);
  });

  test('log mode logs names, never values, and each event once per request', () => {
    const logger = { warn: jest.fn() };
    const observe = createObserver({ AIWAF_INJECTION_RISK_MODE: 'log', AIWAF_PATH_MANIFEST: save(manifestWith([{ sink: 'path', input: '', confidence: 'medium' }])), logger });
    const req = { method: 'GET', url: '/search?evil_name=../topsecret' };
    expect(observe(req)).toBeUndefined();
    observe(req);
    expect(logger.warn.mock.calls).toEqual([[{
      event: 'aiwaf.injection_risk', mode: 'log', method: 'GET', route: '/search', sink: 'path', input: '', rule: 'path_traversal', confidence: 'medium'
    }]]);
  });

  test.each(['off', '', 'block', 'monitor'])('mode %s stays off', mode => {
    expect(createObserver({ AIWAF_INJECTION_RISK_MODE: mode, AIWAF_PATH_MANIFEST: save(manifestWith([{ sink: 'path', input: '' }])) })).toBeNull();
  });

  test('a missing or empty manifest turns it off', () => {
    expect(createObserver({ AIWAF_INJECTION_RISK_MODE: 'log', AIWAF_PATH_MANIFEST: path.join(directory, 'nope.json') })).toBeNull();
    expect(createObserver({ AIWAF_INJECTION_RISK_MODE: 'log', AIWAF_PATH_MANIFEST: save({ routes: { '/x': {} } }) })).toBeNull();
  });

  test('a failing check never reaches the request', () => {
    const observe = createObserver({
      AIWAF_INJECTION_RISK_MODE: 'log',
      AIWAF_PATH_MANIFEST: save(manifestWith([{ sink: 'path', input: '' }])),
      logger: { warn() { throw new Error('logger failed'); } }
    });
    expect(() => observe({ url: '/search', query: { a: '../x' } })).not.toThrow();
    expect(() => observe({ get url() { throw new Error('bad request'); } })).not.toThrow();
  });

  test('the middleware logs without changing the outcome', async () => {
    const logger = { warn: jest.fn() };
    const middleware = require('../lib/wafMiddleware')({
      AIWAF_INJECTION_RISK_MODE: 'log',
      AIWAF_PATH_MANIFEST: save(manifestWith([{ sink: 'command', input: 'q', confidence: 'high' }])),
      AIWAF_MIDDLEWARES: ['ip_keyword_block'],
      logger
    });
    const finding = await middleware.inspectPayload({ method: 'GET', url: '/search', headers: {}, query: { q: '$(id)' } });
    expect(finding).toBeNull();
    expect(logger.warn).toHaveBeenCalledWith(expect.objectContaining({ event: 'aiwaf.injection_risk', rule: 'command_substitution', input: 'q' }));
  });

  test('the native pipeline logs from the environment settings', async () => {
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
    process.env.AIWAF_INJECTION_RISK_MODE = 'log';
    process.env.AIWAF_PATH_MANIFEST = save(manifestWith([{ sink: 'command', input: 'q', confidence: 'high' }]));
    try {
      const native = await require('../lib/nativePipeline')({ dataDir: path.join(directory, 'native') });
      expect(native.inspectPayload({ method: 'GET', url: '/search', headers: {}, query: { q: '$(id)' } })?.status).toBeFalsy();
      expect(warn).toHaveBeenCalledWith(expect.objectContaining({ event: 'aiwaf.injection_risk', input: 'q' }));
      warn.mockClear();
      native.inspectPayload({ url: '/search', headers: {}, query: { q: '$(id)' }, aiwafRoute: { fullyExempt: true } });
      expect(warn).not.toHaveBeenCalled();
      native.close();
    } finally {
      delete process.env.AIWAF_INJECTION_RISK_MODE;
      delete process.env.AIWAF_PATH_MANIFEST;
      warn.mockRestore();
    }
  });

  test.each([
    ['req.query.id', 'req.query.id', 'id'],
    ['req.body.user', 'req.body.user', 'user'],
    ['req.params.id', 'req.params.id', 'id'],
    ["req.param('id')", "req.param('id')", 'id'],
    ['req.query', 'req.query', ''],
    ['req.query[name]', 'req.query', '']
  ])('findings name the parameter for %s', (expression, source, input) => {
    const [finding] = analyzeSource(`function view(req) { exec(${expression}); }`);
    expect(finding.source).toBe(source);
    expect(inputFromSource(finding.source)).toBe(input);
  });

  test('other sources name the parameter', () => {
    expect(inputFromSource(analyzeSource('function v(req) { const { term } = req.body; exec(term); }')[0].source)).toBe('term');
    expect(inputFromSource(analyzeSource('app.use(async ctx => { exec(ctx.query.id); });')[0].source)).toBe('id');
    expect(inputFromSource(analyzeSource('app.use(async ctx => { exec(ctx.request.body.id); });')[0].source)).toBe('id');
    expect(inputFromSource(analyzeSource("class A { async s({ request }) { exec(request.input('id')); } }")[0].source)).toBe('id');
    expect(inputFromSource(analyzeSource("class A { f(@Query('q') value: string) { exec(value); } }")[0].source)).toBe('q');
    expect(inputFromSource(analyzeSource('class A { f(@Body() dto: Item) { exec(dto); } }')[0].source)).toBe('');
  });

  const linked = (source, routes, file) => {
    const manifest = { routes };
    return { counts: applyReport(report(source, file), manifest), routes: manifest.routes };
  };

  test('findings link to routes by the path in the source', () => {
    const { counts, routes } = linked("app.get('/x', (req, res) => { exec(req.query.id); });", { '/x': { view: '' }, '/y': { view: '' } });
    expect(counts).toEqual({ routes: 1, linked: 1, unlinked: 0 });
    expect(routes['/x'].injection_risks).toEqual([{ sink: 'command', input: 'id', confidence: 'high' }]);
    expect(routes['/y'].injection_risks).toBeUndefined();
  });

  test('routers link under one mount prefix, and never by a guess', () => {
    const source = "router.get('/x', (req, res) => { exec(req.query.id); });";
    expect(linked(source, { '/api/x': {} }).counts.linked).toBe(1);
    expect(linked(source, { '/a/x': {}, '/b/x': {} }).counts).toEqual({ routes: 0, linked: 0, unlinked: 1 });
  });

  test('Hapi routes link by their path', () => {
    const source = "server.route({ method: 'GET', path: '/x', handler: (request, h) => exec(request.query.id) });";
    expect(linked(source, { '/x': {} }).counts.linked).toBe(1);
  });

  test('named handlers link when the name is used once', () => {
    const source = 'function search(req, res) { exec(req.query.q); }';
    expect(linked(source, { '/s': { view: 'search' } }).counts.linked).toBe(1);
    expect(linked(source, { '/s': { view: 'bound search' } }).counts.linked).toBe(1);
    expect(linked(source, { '/s': { view: 'search' }, '/t': { view: 'search' } }).counts.unlinked).toBe(1);
    expect(linked(source, { '/s': { view: 'other' } }).counts.unlinked).toBe(1);
  });

  test('Next.js routes link by file', () => {
    const source = 'export default function handler(req, res) { exec(req.query.q); }';
    expect(linked(source, { '/api/items/:id': {} }, 'pages/api/items/[id].js').counts.linked).toBe(1);
    const appSource = 'export async function GET(req) { exec(req.body.q); }';
    expect(linked(appSource, { '/api/a': {} }, 'src/app/api/a/route.ts').counts.linked).toBe(1);
  });

  test('helpers link through the handler that called them', () => {
    const source = "function run(v) { exec(v); } app.get('/x', (req, res) => { run(req.query.id); });";
    expect(linked(source, { '/x': {} }).counts.unlinked).toBe(0);
  });

  test('a clean audit removes old risks and nothing else', () => {
    const manifest = { routes: { '/x': { view: 'a', protections: { rate_limit: { requests: 1 } }, injection_risks: [{ sink: 'sql' }] } } };
    applyReport({ findings: [], files_scanned: 3 }, manifest);
    expect(manifest.routes['/x']).toEqual({ view: 'a', protections: { rate_limit: { requests: 1 } } });
    expect(manifest.injection_audit.files_scanned).toBe(3);
  });

  test('saved risks never change path rules', () => {
    const manifest = { routes: { '/x': { protections: { rate_limit: { requests: 5, window_seconds: 60 }, honeypot: { enabled: false } } } } };
    const before = compileManifestToPathRules(manifest);
    applyReport(report("app.get('/x', (req, res) => { exec(req.query.id); });"), manifest);
    expect(manifest.routes['/x'].injection_risks).toBeDefined();
    expect(compileManifestToPathRules(manifest)).toEqual(before);
  });

  test('updating a manifest writes the file and needs one to exist', () => {
    const file = save({ schema_version: '1.0', routes: { '/x': {} } });
    const counts = updateManifest(report("app.get('/x', (req, res) => { exec(req.query.id); });"), file);
    expect(loadManifest(file).routes['/x'].injection_risks[0].sink).toBe('command');
    expect(formatCounts(counts, file)).toContain('Saved 1 possible injection paths to 1 routes');
    expect(formatCounts({ linked: 0, routes: 0, unlinked: 2 }, 'x')).toContain("couldn't be tied");
    expect(() => updateManifest(report(''), path.join(directory, 'nope.json'))).toThrow('Manifest not found');
    expect(() => applyReport({ findings: [] }, { routes: [] })).toThrow('no routes');
  });

  test('the audit command saves to a manifest', () => {
    const source = path.join(directory, 'app.js');
    fs.writeFileSync(source, "app.get('/x', (req, res) => { exec(req.query.id); });");
    const file = save({ routes: { '/x': {} } });
    const output = [];
    expect(runAuditCommand([source, '--manifest', file], text => output.push(text))).toBe(0);
    expect(output.join('\n')).toContain('Saved 1 possible injection paths');
    expect(loadManifest(file).routes['/x'].injection_risks).toHaveLength(1);
    expect(runAuditCommand([source, '--manifest', path.join(directory, 'nope.json')], () => {})).toBe(2);
  });
});
