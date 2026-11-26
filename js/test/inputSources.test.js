const fs = require('fs');
const os = require('os');
const path = require('path');
const express = require('express');
const aiwaf = require('../index');
const sources = require('../lib/inputSources');

const SPEC = {
  openapi: '3.0.3',
  paths: {
    '/items/{id}': {
      parameters: [{ $ref: '#/components/parameters/Verbose' }],
      get: { parameters: [
        { name: 'id', in: 'path', schema: { type: 'integer' } },
        { name: 'page', in: 'query', schema: { type: 'integer' } },
        { name: 'tags', in: 'query', schema: { type: 'array', items: { type: 'string', format: 'uuid' } } }
      ] },
      put: { requestBody: { content: { 'application/json': { schema: { $ref: '#/components/schemas/Item' } } } } },
      summary: 'not an operation'
    },
    '/open': { post: {
      parameters: [{ name: 'filter', in: 'query', schema: { type: 'object' } }, { $ref: '#/missing' }],
      requestBody: { content: {
        'application/json': { schema: { type: 'object', additionalProperties: true, properties: { a: { type: ['number', 'null'] } } } },
        'text/plain': { schema: { type: 'string' } }
      } }
    } },
    '/union': { post: { requestBody: { content: {
      'application/x-www-form-urlencoded': { schema: { oneOf: [{ type: 'object', properties: { x: { type: 'boolean' } } }, { $ref: '#/components/schemas/Item' }] } },
      'application/merge-patch+json': { schema: { type: 'array' } }
    } } } },
    '/broken': { post: { requestBody: { content: { 'application/json': { schema: { $ref: 'other.json#/X' } } } } } }
  },
  components: {
    parameters: { Verbose: { name: 'verbose', in: 'query', schema: { type: 'boolean' } } },
    schemas: { Item: { allOf: [{ type: 'object', properties: { name: { type: 'string' } } }, { properties: { qty: { type: 'integer' } } }] } }
  }
};

const temp = () => fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-inputs-'));

test('kinds and canonical paths', () => {
  expect([Number, Boolean, String, 'integer', 'bigint', 'number', 'boolean', 'uuid', 'other', undefined].map(sources.kindOf))
    .toEqual(['float', 'bool', 'text', 'int', 'int', 'float', 'bool', 'uuid', 'text', 'text']);
  expect(sources.canonical('/posts/:id')).toBe('/posts/{}');
  expect(sources.canonical('posts/<int:id>/')).toBe('/posts/{}');
  expect(sources.canonical('/files/[slug]?x=1')).toBe('/files/{}');
  expect(sources.canonical()).toBe('/');
});

test('aiwaf.inputs declares a handler, read from any handler on an Express route', () => {
  const handler = aiwaf.inputs({ q: 'text', page: Number })((req, res) => res.end());
  expect(sources.declaredOn([handler])).toEqual({ inputs: { q: 'text', page: 'float' }, complete: true });
  expect(aiwaf.inputs({ a: 'int' }, { complete: false })('not a function')).toBe('not a function');
  expect(sources.declaredOn([], null)).toBeNull();
  const app = express();
  app.get('/search', (req, res, next) => next(), handler);
  const { extractExpressRoutes } = require('../lib/pathManifest');
  expect(extractExpressRoutes(app)['/search'].input_declaration).toEqual({ inputs: { q: 'text', page: 'float' }, complete: true });
});

test('OpenAPI operations list query and body inputs', () => {
  const found = sources.openapiInputs(SPEC);
  expect(found['/items/{}'].GET).toEqual({ inputs: { verbose: 'bool', page: 'int', tags: 'uuid' }, complete: true, reasons: [] });
  expect(found['/items/{}'].PUT.inputs).toEqual({ verbose: 'bool', name: 'text', qty: 'int' });
  const opened = found['/open'].POST;
  expect(opened.inputs).toEqual({ a: 'float' });
  expect(opened.reasons.map(item => item.code)).toEqual(['container_whole', 'external_class', 'container_whole', 'raw_body']);
  expect(found['/union'].POST.inputs).toEqual({ x: 'bool', name: 'text', qty: 'int' });
  expect(found['/union'].POST.reasons.map(item => item.code)).toEqual(['container_whole']);
  expect(found['/broken'].POST.reasons[0].code).toBe('external_class');
  expect(sources.openapiInputs({})).toEqual({});
});

test('specs, recordings and declaration files load', () => {
  const folder = temp();
  const spec = path.join(folder, 'openapi.json');
  fs.writeFileSync(spec, JSON.stringify(SPEC));
  expect(sources.loadSpec(spec).paths).toBeTruthy();
  fs.writeFileSync(path.join(folder, 'bad.json'), '{}');
  expect(() => sources.loadSpec(path.join(folder, 'bad.json'))).toThrow(/no paths/);
  fs.writeFileSync(path.join(folder, 'spec.yaml'), 'paths: {}\n');
  let yamlInstalled = true;
  try { require.resolve('yaml'); } catch (_error) { yamlInstalled = false; }
  if (yamlInstalled) expect(sources.loadSpec(path.join(folder, 'spec.yaml'))).toEqual({ paths: {} });
  else expect(() => sources.loadSpec(path.join(folder, 'spec.yaml'))).toThrow(/yaml package/);
  const seen = path.join(folder, 'seen.jsonl');
  fs.writeFileSync(seen, [
    JSON.stringify({ route: '/search/', key: 'q' }), JSON.stringify({ route: '/search', key: '*' }),
    JSON.stringify({ route: '/posts/:id', key: 'page', type: 'int' }), 'not json', JSON.stringify(['x']), JSON.stringify({ route: '/x' })
  ].join('\n'));
  expect(sources.loadObservations(seen)).toEqual({ '/search': { q: 'text', '*': '*' }, '/posts/{}': { page: 'int' } });
  const declared = path.join(folder, 'inputs.json');
  fs.writeFileSync(declared, JSON.stringify({ routes: { '/a/:id': { inputs: { n: 'int' } }, '/b': { complete: false }, '/c': 'skipped' } }));
  expect(sources.loadDeclarations(declared)).toEqual({ '/a/{}': { inputs: { n: 'int' }, complete: true }, '/b': { inputs: {}, complete: false } });
  fs.writeFileSync(declared, 'null');
  expect(sources.loadDeclarations(declared)).toEqual({});
});

test('finalize combines every source', () => {
  const routes = {
    '/read': { inputs: { q: 'text' }, inputs_complete: true },
    '/decorated': { inputs: { q: 'text' }, inputs_complete: false, inputs_incomplete: [{ code: 'helper', detail: 'x' }],
      input_declaration: { inputs: { q: 'int' }, complete: true } },
    '/filed/:id': { methods: ['GET'] },
    '/spec': { methods: ['GET', 'POST'], inputs: {}, inputs_complete: false, inputs_incomplete: [{ code: 'helper', detail: 'x' }] },
    '/partial': { methods: ['GET', 'POST'] },
    '/seen': { inputs: { q: 'text' }, inputs_complete: true },
    '/whole': { inputs: {}, inputs_complete: true },
    '/only-seen': {},
    '/untouched': { view: 'x' },
    '/odd': 'not an object'
  };
  const counts = sources.finalize(routes, {
    declarations: { '/filed/{}': { inputs: { n: 'int' }, complete: true } },
    openapi: {
      '/spec': { GET: { inputs: { a: 'int' }, complete: true, reasons: [] }, POST: { inputs: { b: 'text' }, complete: true, reasons: [] } },
      '/partial': { GET: { inputs: { a: 'int' }, complete: true, reasons: [] } }
    },
    observations: { '/seen': { q: 'text', extra: 'text' }, '/whole': { '*': '*' }, '/only-seen': { k: 'text' } }
  });
  expect(routes['/decorated']).toEqual({ inputs: { q: 'int' }, inputs_complete: true });
  expect(routes['/filed/:id'].inputs).toEqual({ n: 'int' });
  expect(routes['/spec'].inputs_complete).toBe(true);
  expect(routes['/partial'].inputs).toEqual({ a: 'int' });
  expect(routes['/seen'].inputs_incomplete[0].code).toBe('observed');
  expect(routes['/whole'].inputs_incomplete[0].code).toBe('container_whole');
  expect(routes['/only-seen']).toEqual({ inputs: { k: 'text' }, inputs_complete: true });
  expect(routes['/untouched'].inputs).toBeUndefined();
  expect(counts).toMatchObject({ declared: 2, openapi: 2, observed: 3, reasons: { observed: 1, container_whole: 1 } });
  expect(sources.finalize({ '/x': { methods: [] } }, { openapi: { '/x': { GET: { inputs: {}, complete: true } } } }).complete).toBe(1);
});

test('sources come from the environment', () => {
  const folder = temp();
  expect(sources.sourcesFromEnvironment(folder, {})).toEqual({});
  fs.mkdirSync(path.join(folder, '.aiwaf'));
  fs.writeFileSync(path.join(folder, '.aiwaf', 'inputs.json'), JSON.stringify({ '/a': { inputs: { q: 'text' } } }));
  fs.writeFileSync(path.join(folder, '.aiwaf', 'observed_inputs.jsonl'), JSON.stringify({ route: '/a', key: 'q' }));
  fs.writeFileSync(path.join(folder, 'spec.json'), JSON.stringify(SPEC));
  const found = sources.sourcesFromEnvironment(folder, { AIWAF_OPENAPI_SPEC: path.join(folder, 'spec.json') });
  expect(Object.keys(found).sort()).toEqual(['declarations', 'observations', 'openapi']);
});

test('init applies declarations and reports completeness', () => {
  const folder = temp();
  fs.writeFileSync(path.join(folder, 'app.js'), "app.get('/search', function search(req, res) { return helper(req); });\n");
  const { auditIntoManifest } = require('../lib/pathManifest');
  const manifest = { routes: {
    '/search': { view: 'search', methods: ['GET'], input_declaration: { inputs: { q: 'text' }, complete: true } },
    '/other': { view: 'other', input_declaration: { inputs: {}, complete: true } }
  } };
  const previous = process.cwd();
  process.chdir(folder);
  try {
    auditIntoManifest(manifest, folder);
  } finally {
    process.chdir(previous);
  }
  expect(manifest.routes['/search']).toMatchObject({ inputs: { q: 'text' }, inputs_complete: true });
  expect(manifest.routes['/search'].input_declaration).toBeUndefined();
  // /other's handler isn't in the audited source, so its risks weren't looked for.
  expect(manifest.injection_audit.routes_with_complete_inputs).toBe(1);
  expect(manifest.routes['/other']).toMatchObject({ inputs: {}, inputs_complete: false });
  expect(manifest.routes['/other'].inputs_incomplete.map(item => item.code)).toEqual(['unread_view']);
  const skipped = { routes: { '/x': { input_declaration: {} } } };
  expect(auditIntoManifest(skipped, null)).toBeNull();
  expect(skipped.routes['/x'].input_declaration).toBeUndefined();
});

test('a bad source file is reported, and the audit still runs', () => {
  const folder = temp();
  fs.writeFileSync(path.join(folder, 'app.js'), "app.get('/search', function search(req, res) { return req.query.q; });\n");
  fs.mkdirSync(path.join(folder, '.aiwaf'));
  fs.writeFileSync(path.join(folder, '.aiwaf', 'inputs.json'), 'not json');
  const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
  const previous = process.cwd();
  process.chdir(folder);
  try {
    const manifest = { routes: { '/search': { view: 'search' } } };
    require('../lib/pathManifest').auditIntoManifest(manifest, folder);
    expect(manifest.routes['/search'].inputs_complete).toBe(true);
    expect(warn.mock.calls[0][0]).toMatch(/Declared, OpenAPI or recorded inputs skipped/);
  } finally {
    process.chdir(previous);
    warn.mockRestore();
  }
});

test('path parameters are never query or body inputs', () => {
  expect(sources.pathParameters('/a/<int:id>/{item_id}/{x:int}/:name/:opt?/[page]/[...rest]')).toEqual(['id', 'item_id', 'x', 'name', 'opt', 'page', 'rest']);
  const routes = { '/items/:id': { inputs: { id: 'text', q: 'text' }, inputs_complete: true } };
  sources.finalize(routes);
  expect(routes['/items/:id'].inputs).toEqual({ q: 'text' });
});

test('a type the framework validates holds over an untyped read of the same name', () => {
  const { finalize, finalizeWithoutCode } = require('../lib/inputSources');
  const routes = {
    '/orders': {
      inputs: { sku: 'text', quantity: 'text', note: 'int' }, inputs_complete: true,
      declared_inputs: { inputs: { sku: 'text', quantity: 'int', note: 'text' }, covers: [], reasons: [], standalone: false }
    }
  };
  finalize(routes);
  expect(routes['/orders'].inputs).toEqual({ sku: 'text', quantity: 'int', note: 'text' });
  expect(routes['/orders'].inputs_complete).toBe(true);

  const unread = {
    '/orders': { declared_inputs: { inputs: { quantity: 'int' }, covers: [], reasons: [], standalone: false } },
    '/items': { declared_inputs: { inputs: { id: 'int' }, covers: [], reasons: [], standalone: true } },
    '/plain': {},
    '/bad': null
  };
  finalizeWithoutCode(unread);
  expect(unread['/orders']).toMatchObject({ inputs: { quantity: 'int' }, inputs_complete: false });
  expect(unread['/orders'].inputs_incomplete.map(item => item.code)).toEqual(['not_audited']);
  // Even a list complete on its own isn't complete unaudited: its risks weren't looked for.
  expect(unread['/items']).toMatchObject({ inputs: { id: 'int' }, inputs_complete: false });
  expect(unread['/plain'].inputs).toBeUndefined();
});

