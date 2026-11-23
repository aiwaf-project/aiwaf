const fs = require('fs');
const os = require('os');
const path = require('path');
const { analyzeSource, runAuditCommand } = require('../lib/injectionAnalysis');
const { loadManifest, compileManifestToPathRules } = require('../lib/pathManifest');
const {
  applyReport, compileRisks, createObserver, formatCounts, inputFromSource, inspectRisks, reviewRequest, ruleFor, strictInputs,
  updateManifest, withStrict
} = require('../lib/routeRisks');

const CASES = JSON.parse(fs.readFileSync(path.join(__dirname, '../../tests/fixtures/route_risk_cases.json'), 'utf8'));
const manifestWith = (risks, route = '/search') => ({ routes: { [route]: { view: 'search', injection_risks: risks }, '/health': { view: 'health' } } });
const inspect = (manifest, ...args) => inspectRisks(compileRisks(manifest), ...args);
const report = (source, file = 'app.js') => ({ files_scanned: 1, findings: analyzeSource(source, file) });

beforeAll(() => require('../lib/wasmChecks').load());

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
    expect(compileRisks(null).empty).toBe(true);
    expect(compileRisks({ routes: 'bad' }).empty).toBe(true);
  });

  test('exact routes win and repeated matches report once', () => {
    const manifest = { routes: {
      '/items/:id': { injection_risks: [{ sink: 'path', input: '' }] },
      '/items/new': { injection_risks: [{ sink: 'command', input: '' }] }
    } };
    expect(inspect(manifest, '/items/new', { x: '$(id)' })[0].route).toBe('/items/new');
    expect(inspect(manifestWith([{ sink: 'path', input: '' }]), '/search', { a: '../x', b: '../y' })).toHaveLength(1);
  });

  test('request formats are inputs', () => {
    const manifest = manifestWith([{ sink: 'path', input: 'a' }]);
    for (const [query, body] of [['a=../x', null], [{ a: '../x' }, null], [{ a: ['../x'] }, null], ['', Buffer.from('a=../x')], ['', '{"a": ["../x"]}'], ['', { a: '../x' }]]) {
      expect(inspect(manifest, '/search', query, body)[0].rule).toBe('path_traversal');
    }
    expect(inspect(manifestWith([{ sink: 'path', input: '' }]), '/search', '', { pipe() {} })).toEqual([]);
  });

  test('log mode logs names, never values, and each event once per request', () => {
    const logger = { warn: jest.fn() };
    const observe = createObserver({ AIWAF_INJECTION_RISK_MODE: 'log', AIWAF_PATH_MANIFEST: save(manifestWith([{ sink: 'path', input: '', confidence: 'medium' }])), logger });
    const req = { method: 'GET', url: '/search?evil_name=../topsecret' };
    expect(observe(req)).toBeNull();
    observe(req);
    expect(logger.warn.mock.calls).toEqual([[{
      event: 'aiwaf.injection_risk', mode: 'log', method: 'GET', route: '/search', sink: 'path', input: '', rule: 'path_traversal', confidence: 'medium'
    }]]);
  });

  test('block mode is the default and blocks', () => {
    const logger = { warn: jest.fn() };
    const observe = createObserver({ AIWAF_PATH_MANIFEST: save(manifestWith([{ sink: 'path', input: '' }])), logger });
    expect(observe({ method: 'GET', url: '/search?x=../y' })).toEqual({ rule: 'path_traversal', status: 403, mode: 'block' });
    expect(logger.warn.mock.calls[0][0].mode).toBe('block');
    expect(observe({ url: '/search?x=fine' })).toBeNull();
  });

  test('off turns it off and unknown modes are rejected', () => {
    const file = save(manifestWith([{ sink: 'path', input: '' }]));
    expect(createObserver({ AIWAF_INJECTION_RISK_MODE: 'off', AIWAF_PATH_MANIFEST: file })).toBeNull();
    for (const mode of ['', 'monitor', 'enforce']) {
      expect(() => createObserver({ AIWAF_INJECTION_RISK_MODE: mode, AIWAF_PATH_MANIFEST: file })).toThrow('AIWAF_INJECTION_RISK_MODE');
    }
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
    expect(observe({ url: '/search', query: { a: '../x' } })).toBeNull();
    expect(() => observe({ get url() { throw new Error('bad request'); } })).not.toThrow();
  });

  test('the middleware blocks by default and only logs in log mode', async () => {
    const blocking = require('../lib/wafMiddleware')({
      AIWAF_PATH_MANIFEST: save(manifestWith([{ sink: 'command', input: 'q', confidence: 'high' }])),
      AIWAF_MIDDLEWARES: ['ip_keyword_block'],
      logger: { warn() {} }
    });
    expect(await blocking.inspectPayload({ method: 'GET', url: '/search', headers: {}, query: { q: 'x; ls -la' } }))
      .toEqual({ rule: 'command_chain', status: 403, mode: 'block' });

    const logger = { warn: jest.fn() };
    const middleware = require('../lib/wafMiddleware')({
      AIWAF_INJECTION_RISK_MODE: 'log',
      AIWAF_PATH_MANIFEST: save(manifestWith([{ sink: 'command', input: 'q', confidence: 'high' }])),
      AIWAF_MIDDLEWARES: ['ip_keyword_block'],
      logger
    });
    const finding = await middleware.inspectPayload({ method: 'GET', url: '/search', headers: {}, query: { q: 'x; ls -la' } });
    expect(finding).toBeNull();
    expect(logger.warn).toHaveBeenCalledWith(expect.objectContaining({ event: 'aiwaf.injection_risk', rule: 'command_chain', input: 'q' }));
  });

  test('the native pipeline logs from the environment settings', async () => {
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
    process.env.AIWAF_INJECTION_RISK_MODE = 'log';
    process.env.AIWAF_PATH_MANIFEST = save(manifestWith([{ sink: 'command', input: 'q', confidence: 'high' }]));
    try {
      const native = await require('../lib/nativePipeline')({ dataDir: path.join(directory, 'native') });
      expect(native.inspectPayload({ method: 'GET', url: '/search', headers: {}, query: { q: 'x; ls -la' } })?.status).toBeFalsy();
      process.env.AIWAF_INJECTION_RISK_MODE = 'block';
      const blocking = await require('../lib/nativePipeline')({ dataDir: path.join(directory, 'native-block') });
      expect(blocking.inspectPayload({ method: 'GET', url: '/search', headers: {}, query: { q: 'x; ls -la' } }).rule).toBe('command_chain');
      const res = { status: jest.fn(() => res), json: jest.fn(() => res) };
      blocking({ method: 'GET', url: '/search', headers: {}, query: { q: 'x; ls -la' }, socket: {} }, res, () => {});
      expect(res.status).toHaveBeenCalledWith(403);
      blocking.close();
      expect(warn).toHaveBeenCalledWith(expect.objectContaining({ event: 'aiwaf.injection_risk', input: 'q' }));
      warn.mockClear();
      native.inspectPayload({ url: '/search', headers: {}, query: { q: 'x; ls -la' }, aiwafRoute: { fullyExempt: true } });
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
    expect(counts).toMatchObject({ routes: 1, linked: 1, unlinked: 0 });
    expect(routes['/x'].injection_risks).toEqual([{ sink: 'command', input: 'id', confidence: 'high' }]);
    expect(routes['/y'].injection_risks).toBeUndefined();
  });

  test('findings in a helper file link through the Next.js route file that called it', () => {
    const finding = {
      file: 'lib/sql.js', entry_file: 'pages/api/users/[id].js', line: 2, function: 'handler -> search',
      sink: 'sql', call: 'db.query', source: 'req.query.id', confidence: 'high'
    };
    const manifest = { routes: { '/api/users/:id': {}, '/other': {} } };
    expect(applyReport({ findings: [finding] }, manifest)).toMatchObject({ routes: 1, linked: 1, unlinked: 0 });
    expect(manifest.routes['/api/users/:id'].injection_risks).toHaveLength(1);
  });

  test('routers link under one mount prefix, and never by a guess', () => {
    const source = "router.get('/x', (req, res) => { exec(req.query.id); });";
    expect(linked(source, { '/api/x': {} }).counts.linked).toBe(1);
    expect(linked(source, { '/a/x': {}, '/b/x': {} }).counts).toMatchObject({ routes: 0, linked: 0, unlinked: 1 });
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

  test('manifest generation saves audit findings, and a failed scan saves none', () => {
    const { generateFrameworkManifest, auditIntoManifest } = require('../lib/pathManifest');
    fs.writeFileSync(path.join(directory, 'app.js'), "app.get('/x', (req, res) => { db.query('SELECT ' + req.query.id); });");
    const output = path.join(directory, 'out', 'paths.json');
    const routes = [{ path: '/x', methods: ['GET'] }, { path: '/y', methods: ['GET'] }];
    const manifest = generateFrameworkManifest('express', null, output, { routes, auditSource: directory });
    expect(manifest.routes['/x'].injection_risks).toEqual([{ sink: 'sql', input: 'id', confidence: 'high' }]);
    expect(loadManifest(output).routes['/y'].injection_risks).toBeUndefined();
    expect(generateFrameworkManifest('express', null, output, { routes }).injection_audit).toBeUndefined();
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      expect(auditIntoManifest({ routes: {} }, path.join(directory, 'missing'))).toBeNull();
      expect(warn.mock.calls[0][0]).toContain('Injection audit skipped');
    } finally {
      warn.mockRestore();
    }
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

describe('ordinary text', () => {
  const values = JSON.parse(fs.readFileSync(path.join(__dirname, '../../tests/fixtures/ordinary_text.json'), 'utf8')).values;
  const sinks = ['sql', 'nosql', 'command', 'path', 'ssrf', 'code', 'template', 'deserialization'];

  test('no rule blocks ordinary text', () => {
    const blocked = values.flatMap(value => sinks.filter(sink => ruleFor(sink, value)).map(sink => [sink, value]));
    expect(blocked).toEqual([]);
  });

  test('the middleware passes ordinary text on a route checked for every operation', async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-ordinary-'));
    try {
      const manifest = path.join(directory, 'paths.json');
      fs.writeFileSync(manifest, JSON.stringify({ routes: { '/login': { injection_risks: sinks.map(sink => ({ sink, input: '', confidence: 'high' })) } } }));
      const middleware = require('../lib/wafMiddleware')({ AIWAF_PATH_MANIFEST: manifest, AIWAF_MIDDLEWARES: ['ip_keyword_block'], logger: { warn() {} } });
      const blocked = [];
      for (const value of values) {
        const req = { method: 'POST', url: `/login?q=${encodeURIComponent(value)}`, headers: {}, query: { q: value }, body: { email: value } };
        if (await middleware.inspectPayload(req)) blocked.push(value);
      }
      expect(blocked).toEqual([]);
    } finally {
      fs.rmSync(directory, { recursive: true, force: true });
    }
  });
});

describe('route schemas saved by init', () => {
  const SCHEMA = JSON.parse(fs.readFileSync(path.join(__dirname, '../../tests/fixtures/route_schema_cases.json'), 'utf8'));
  const wasm = () => require('../lib/wasmChecks').get();
  let directory;
  beforeEach(() => { directory = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-schema-')); });
  afterEach(() => fs.rmSync(directory, { recursive: true, force: true }));
  const save = manifest => {
    const file = path.join(directory, 'paths.json');
    fs.writeFileSync(file, JSON.stringify(manifest));
    return file;
  };

  test.each(SCHEMA.cases.map(item => [item.path, item.expect, item]))('shared case %s %s', (_path, _expect, item) => {
    const general = JSON.parse(wasm().sql_inspect(Buffer.from(item.query), Buffer.from(item.body), 65536));
    const review = reviewRequest(compileRisks(SCHEMA.manifest), item.path, item.query, Buffer.from(item.body), 65536, general && general.rule);
    const outcome = review ? review.action : (general ? 'block' : 'pass');
    expect(outcome).toBe(item.expect);
    if (outcome === 'mismatch') expect(review.input).toBe(item.input);
  });

  test.each(SCHEMA.path_cases.map(item => [item.path, item.expect, item]))('shared path case %s %s', (_path, _expect, item) => {
    const general = JSON.parse(wasm().sql_inspect_path(item.path, 65536));
    const review = general ? JSON.parse(compileRisks(SCHEMA.manifest).review_path(item.path, 65536, general.rule)) : null;
    const outcome = review ? review.action : (general ? 'block' : 'pass');
    expect(outcome).toBe(item.expect);
    if (outcome === 'mismatch') expect(review.input).toBe(item.input);
  });

  test('path findings are reviewed by the route\'s parameters', () => {
    const enforce = createObserver({ AIWAF_PATH_MANIFEST: save(SCHEMA.manifest), AIWAF_ROUTE_SCHEMA_MODE: 'enforce', logger: { warn() {} } });
    const finding = { rule: 'sql_function', status: 403, mode: 'block', source: 'path' };
    expect(enforce.reviewPath({ path: '/posts/use-sleep(5)-in-queries' }, finding, 65536)).toBeNull();
    expect(enforce.reviewPath({ path: '/orders/char(65)' }, finding, 65536)).toEqual({ rule: 'schema_type_mismatch', status: 403, mode: 'block' });
    expect(enforce.reviewPath({ path: '/x' }, { rule: 'payload_inspection_limit', status: 413 }, 1)).toEqual({ rule: 'payload_inspection_limit', status: 413 });
  });

  test('review modes log names and never values', () => {
    const warn = jest.fn();
    const finding = { rule: 'sql_function', status: 403, mode: 'block' };
    const req = body => ({ method: 'POST', url: '/comments', path: '/comments', body });
    const prose = req({ body: 'use sleep(5) in the query' });
    const monitor = createObserver({ AIWAF_PATH_MANIFEST: save(SCHEMA.manifest), logger: { warn } });
    expect(monitor.review(prose, finding, 65536)).toBe(finding);
    expect(warn).toHaveBeenCalledWith({ event: 'aiwaf.route_schema', mode: 'monitor', method: 'POST', route: '/comments', action: 'relax', rule: 'sql_function' });
    const enforce = createObserver({ AIWAF_PATH_MANIFEST: save(SCHEMA.manifest), AIWAF_ROUTE_SCHEMA_MODE: 'enforce', AIWAF_INJECTION_RISK_MODE: 'off', logger: { warn } });
    expect(enforce(prose)).toBeNull();
    expect(enforce.review(prose, finding, 65536)).toBeNull();
    expect(enforce.review(req({ post_id: '1 OR 1=1' }), null, 65536)).toEqual({ rule: 'schema_type_mismatch', status: 403, mode: 'block' });
    expect(enforce.review(req({ body: 'hello' }), null, 65536)).toBeNull();
    expect(JSON.stringify(warn.mock.calls)).not.toContain('sleep');
    // A failing review keeps the finding.
    expect(enforce.review({ get path() { throw new Error('boom'); } }, finding, 65536)).toBe(finding);
    expect(createObserver({ AIWAF_PATH_MANIFEST: save(SCHEMA.manifest), AIWAF_ROUTE_SCHEMA_MODE: 'off' })?.review).toBeUndefined();
    expect(createObserver({ AIWAF_PATH_MANIFEST: save(manifestWith([])), AIWAF_INJECTION_RISK_MODE: 'off' })).toBeNull();
    expect(() => createObserver({ AIWAF_ROUTE_SCHEMA_MODE: 'strict' })).toThrow('AIWAF_ROUTE_SCHEMA_MODE');
  });

  const strictManifest = () => ({ ...SCHEMA.manifest, strict_inputs: { allow: SCHEMA.strict_cases.allow } });
  test.each(SCHEMA.strict_cases.cases.map(item => [item.path, item.expect, item]))('shared strict case %s %s', (_path, _expect, item) => {
    const general = JSON.parse(wasm().sql_inspect(Buffer.from(item.query), Buffer.from(item.body), 65536));
    const review = reviewRequest(compileRisks(strictManifest()), item.path, item.query, Buffer.from(item.body), 65536, general && general.rule);
    const outcome = review ? review.action : (general ? 'block' : 'pass');
    expect(outcome).toBe(item.expect);
    if (item.input) expect(review.input).toBe(item.input);
  });

  test('strict inputs log or block names a complete route never reads', () => {
    expect(strictInputs({})).toBeNull();
    expect(strictInputs({ AIWAF_STRICT_INPUTS: 'no' })).toBeNull();
    expect(strictInputs({ AIWAF_STRICT_INPUTS: true, AIWAF_STRICT_INPUTS_ALLOW: 'a, b*,' })).toEqual({ allow: ['a', 'b*'] });
    expect(strictInputs({ AIWAF_STRICT_INPUTS: 'on', AIWAF_STRICT_INPUTS_ALLOW: ['x'] })).toEqual({ allow: ['x'] });
    expect(withStrict(null, { allow: [] })).toBeNull();
    expect(withStrict({ routes: {} }, null)).toEqual({ routes: {} });
    const warn = jest.fn();
    const req = query => ({ method: 'GET', url: `/search?${query}`, originalUrl: `/search?${query}`, path: '/search', query: {} });
    const options = { AIWAF_PATH_MANIFEST: save(SCHEMA.manifest), AIWAF_INJECTION_RISK_MODE: 'off', AIWAF_STRICT_INPUTS: '1', logger: { warn } };
    const monitor = createObserver(options);
    expect(monitor.review(req('q=a&debug=1'), null, 65536)).toBeNull();
    expect(warn).toHaveBeenCalledWith({ event: 'aiwaf.route_schema', mode: 'monitor', method: 'GET', route: '/search', action: 'unexpected', input: 'debug' });
    const enforce = createObserver({ ...options, AIWAF_ROUTE_SCHEMA_MODE: 'enforce' });
    expect(enforce.review(req('q=a&debug=1'), null, 65536)).toEqual({ rule: 'schema_unexpected_input', status: 403, mode: 'block' });
    expect(enforce.review(req('q=a&utm_source=x'), null, 65536)).toBeNull();
    // Off unless turned on.
    const plain = createObserver({ ...options, AIWAF_STRICT_INPUTS: undefined, AIWAF_ROUTE_SCHEMA_MODE: 'enforce' });
    expect(plain.review(req('q=a&debug=1'), null, 65536)).toBeNull();
  });

  test('strict inputs warn when the engine is too old', () => {
    const wasmChecks = require('../lib/wasmChecks');
    const real = wasmChecks.get();
    const warn = jest.fn();
    const spy = jest.spyOn(wasmChecks, 'get').mockReturnValue({ ...real, RouteRiskChecker: class { review() { return 'null'; } } });
    try {
      const observer = createObserver({ AIWAF_PATH_MANIFEST: save(SCHEMA.manifest), AIWAF_INJECTION_RISK_MODE: 'off', AIWAF_STRICT_INPUTS: true, logger: { warn } });
      observer.review({ path: '/search', url: '/search?x=1' }, null, 65536);
      observer.review({ path: '/search', url: '/search?x=1' }, null, 65536);
      expect(warn).toHaveBeenCalledTimes(1);
      expect(warn.mock.calls[0][0]).toContain('strict inputs need aiwaf-wasm 0.2.12');
      const throwing = jest.fn(() => { throw new Error('log'); });
      const quiet = createObserver({ AIWAF_PATH_MANIFEST: save(SCHEMA.manifest), AIWAF_INJECTION_RISK_MODE: 'off', AIWAF_STRICT_INPUTS: true, logger: { warn: throwing } });
      expect(quiet.review({ path: '/search', url: '/search?x=1' }, null, 65536)).toBeNull();
    } finally {
      spy.mockRestore();
    }
  });

  test('init saves each route\'s inputs from the handlers that serve it', () => {
    const { analyzePath } = require('../lib/injectionAnalysis');
    fs.writeFileSync(path.join(directory, 'app.js'), [
      "app.get('/search', (req, res) => res.json({ q: req.query.q, page: parseInt(req.query.page) }));",
      "app.post('/save', auth, (req, res) => res.json(req.body.x));"
    ].join('\n'));
    const manifest = { routes: { '/search': { view: 'x' }, '/save': { view: 'y' }, '/none': { view: 'z' } } };
    expect(applyReport(analyzePath(directory), manifest).inputs).toBe(2);
    expect(manifest.routes['/search']).toEqual({ view: 'x', inputs: { q: 'text', page: 'int' }, inputs_complete: true });
    expect(manifest.routes['/save'].inputs_complete).toBe(false);
    expect(manifest.routes['/none'].inputs).toBeUndefined();
    expect(manifest.injection_audit.routes_with_inputs).toBe(2);
    applyReport({ findings: [] }, manifest);
    expect(manifest.routes['/search']).toEqual({ view: 'x' });
  });
});
