const parser = require('@babel/parser');
const { fileInputs, mergeInto } = require('../lib/inputSchema');

const inputs = source => {
  const ast = parser.parse(source, { sourceType: 'unambiguous', plugins: ['typescript'] });
  const label = (node, parent) => (parent && parent.arguments ? String(parent.arguments[0].value) : node.id && node.id.name);
  return Object.fromEntries(fileInputs(ast, 'app.js', source, label).map(entry => [entry.function, [entry.inputs, entry.complete]]));
};

test('reads, types, destructuring and aliases', () => {
  const found = inputs([
    "app.get('/search', (req, res) => res.json({ q: req.query.q, page: parseInt(req.query.page || '1'), n: +req.query['n'] }));",
    "app.post('/items', express.json(), (req, res) => { const { title, note = '' } = req.body; res.json({ title }); });",
    'function legacy(req, res) { const b = req.body; return b.a + Number(b.n) + Boolean(b.ok); }',
    "app.get('/param', (req, res) => res.json(req.param('id'), 'q' in req.query));",
    "app.get('/redirect', (req, res) => res.redirect(req.originalUrl));",
    "router.get('/koa', async ctx => { ctx.body = ctx.query.q + ctx.request.body.n; });"
  ].join('\n'));
  expect(found['/search']).toEqual([{ q: 'text', page: 'int', n: 'float' }, true]);
  expect(found['/items']).toEqual([{ title: 'text', note: 'text' }, true]);
  expect(found.legacy).toEqual([{ a: 'text', n: 'float', ok: 'bool' }, true]);
  expect(found['/param']).toEqual([{ id: 'text' }, true]);
  expect(found['/redirect']).toEqual([{}, true]);
  expect(found['/koa']).toEqual([{ q: 'text', n: 'text' }, true]);
});

test('zod object schemas give their fields', () => {
  const found = inputs([
    "const z = require('zod');",
    'const Body = z.object({ name: z.string(), age: z.number().int(), id: z.string().uuid(), ok: z.boolean(), score: z.number() }).strict();',
    "const Loose = z.object({ a: z.string() }).passthrough();",
    "app.post('/a', (req, res) => res.json(Body.parse(req.body)));",
    "app.post('/b', (req, res) => res.json(Loose.safeParse(req.body)));",
    "app.post('/c', (req, res) => res.json(Unknown.parse(req.body)));"
  ].join('\n'));
  expect(found['/a']).toEqual([{ name: 'text', age: 'int', id: 'uuid', ok: 'bool', score: 'float' }, true]);
  expect(found['/b'][1]).toBe(false);
  expect(found['/c'][1]).toBe(false);
});

test('anything not understood makes the list incomplete', () => {
  const found = inputs([
    "app.post('/guarded', auth, (req, res) => res.json(req.body.x));",
    "app.post('/whole', (req, res) => save(req.body));",
    "app.get('/dynamic', (req, res) => res.json(req.query[res.locals.key]));",
    "app.get('/helper', (req, res) => helper(req));",
    "app.get('/url', (req, res) => new URL(req.url, 'http://x').searchParams.get('q'));",
    "app.get('/rest', (req, res) => { const { a, ...rest } = req.query; res.json(rest); });",
    "app.post('/raw', (req, res) => req.on('data', () => {}));",
    'function pattern({ body }, res) { return body.x; }'
  ].join('\n'));
  for (const name of ['/guarded', '/whole', '/dynamic', '/helper', '/url', '/rest', '/raw']) expect([name, found[name][1]]).toEqual([name, false]);
  expect(found.pattern).toBeUndefined();
});

test('a route served by several handlers takes all their inputs', () => {
  const routes = { '/x': {} };
  mergeInto(routes, '/x', { inputs: { a: 'int', b: 'text' }, complete: true });
  mergeInto(routes, '/x', { inputs: { a: 'text' }, complete: false });
  expect(routes['/x']).toEqual({ inputs: { a: 'text', b: 'text' }, inputs_complete: false, _flat: ['a', 'b'] });
});

test('the request is followed into functions in the same file, and constant keys are known', () => {
  const found = inputs([
    "const PAGE = 'page';",
    'function render(request, res) { return [parseInt(request.query[PAGE]), request.body.title]; }',
    'const deeper = (r) => r.query.q;',
    "app.get('/a', (req, res) => render(req, res));",
    "app.get('/b', (req, res) => deeper(req));",
    "app.get('/c', (req, res) => elsewhere(req));",
    "app.get('/d', (req, res) => res.json(req.query[OTHER]));",
    'function loop(req, res) { return loop(req, res); }'
  ].join('\n'));
  expect(found['/a']).toEqual([{ page: 'int', title: 'text' }, true]);
  expect(found['/b']).toEqual([{ q: 'text' }, true]);
  expect(found['/c'][1]).toBe(false);
  expect(found['/d'][1]).toBe(false);
  expect(found.loop[1]).toBe(false);
});

const reasonsOf = source => {
  const ast = parser.parse(source, { sourceType: 'unambiguous', plugins: ['typescript'] });
  const label = (node, parent) => (parent && parent.arguments ? String(parent.arguments[0].value) : node.id && node.id.name);
  return Object.fromEntries(fileInputs(ast, 'app.js', source, label).map(entry => [entry.function, (entry.reasons || []).map(item => item.code)]));
};

test('each gap says why', () => {
  const found = reasonsOf([
    "app.get('/helper', (req, res) => helper(req));",
    "app.get('/stored', (req, res) => { list.push = req; });",
    "app.get('/dynamic', (req, res) => res.json(req.query[key], req.param(key)));",
    "app.get('/raw', (req, res) => { req.on('data', f); return new URL(req.url); });",
    "app.get('/whole', (req, res) => res.json(Object.keys(req.query), req.body));",
    "app.get('/rest', (req, res) => { const { a, ...rest } = req.query; });",
    "app.post('/schema', (req, res) => Missing.parse(req.body));",
    "app.post('/guarded', auth, (req, res) => res.json(req.body.a));"
  ].join('\n'));
  expect(found['/helper']).toEqual(['helper']);
  expect(found['/stored']).toEqual(['helper']);
  expect(found['/dynamic']).toEqual(['dynamic_key', 'dynamic_key']);
  expect(found['/raw']).toEqual(['raw_body', 'raw_body']);
  expect(found['/whole']).toEqual(['container_whole', 'container_whole']);
  expect(found['/rest']).toEqual(['container_whole']);
  expect(found['/schema']).toEqual(['external_class']);
  expect(found['/guarded']).toEqual(['middleware']);
});

test('schemas, helpers and constants are followed into imported files', () => {
  const { moduleScope } = require('../lib/inputSchema');
  const files = {
    '/p/schemas.js': "const z = require('zod');\nexport const Body = z.object({ name: z.string(), n: z.number().int() });\n"
      + "export const Loose = z.object({ a: z.string() }).passthrough();\nexport const PAGE = 'page';\nexport { Body as Renamed };\n",
    '/p/helpers.js': "function paginate(req) { return parseInt(req.query.page); }\n"
      + "module.exports = { paginate, other: (req) => req.query.o, method(req) { return req.query.m; } };\n"
      + "exports.direct = function (req) { return req.query.d; };\nmodule.exports.alias = paginate;\n",
    '/p/single.js': 'module.exports = function (req) { return req.body.s; };\n',
    '/p/named.js': 'module.exports = handle;\nfunction handle(req) { return req.body.h; }\n',
    '/p/dflt.js': 'export default function dflt(req) { return req.query.x; }\n',
    '/p/arrow.js': 'export default (req) => req.query.arrow;\nconst y = (req) => req.query.y;\nexport { y };\nexport const CONST = "k";\nexport const notString = 3;\n',
    '/p/reexport.js': "export { Body } from './schemas';\nexport { paginate as pages } from './helpers';\n",
    '/p/app.js': [
      "import { Body, Loose, PAGE, Renamed } from './schemas';",
      "import * as schemas from './schemas';",
      "import dflt from './dflt';",
      "import arrow, { y, CONST } from './arrow';",
      "import { Body as Again, pages } from './reexport';",
      "import { nothing } from 'external';",
      "const { paginate, other, method, direct, alias } = require('./helpers');",
      "const helpers = require('./helpers');",
      "const single = require('./single');",
      "const named = require('./named');",
      "const fromMember = require('./helpers').paginate;",
      "app.post('/body', (req, res) => Body.parse(req.body) && req.query[PAGE] && req.query[CONST]);",
      "app.post('/ns', (req, res) => schemas.Body.parse(req.body) && Renamed.parse(req.body) && Again.parse(req.body));",
      "app.post('/loose', (req, res) => Loose.parse(req.body));",
      "app.get('/helpers', (req, res) => paginate(req) + other(req) + method(req) + direct(req) + alias(req) + helpers.paginate(req));",
      "app.get('/defaults', (req, res) => single(req) + named(req) + dflt(req) + arrow(req) + y(req) + fromMember(req) + pages(req));",
      "app.get('/external', (req, res) => nothing(req));"
    ].join('\n')
  };
  const scopes = new Map();
  const asts = {};
  for (const [file, source] of Object.entries(files)) {
    asts[file] = parser.parse(source, { sourceType: 'unambiguous' });
    const scope = moduleScope(asts[file], source, file, spec => (spec.startsWith('./') ? `/p/${spec.slice(2)}.js` : null));
    scope.project = scopes;
    scopes.set(file, scope);
  }
  const label = (node, parent) => (parent && parent.arguments ? String(parent.arguments[0].value) : node.id && node.id.name);
  const found = Object.fromEntries(fileInputs(asts['/p/app.js'], 'app.js', files['/p/app.js'], label, scopes.get('/p/app.js'))
    .map(entry => [entry.function, entry]));
  expect(found['/body'].inputs).toEqual({ name: 'text', n: 'int', page: 'text', k: 'text' });
  expect(found['/body'].complete).toBe(true);
  expect(found['/ns'].inputs).toEqual({ name: 'text', n: 'int' });
  expect(found['/ns'].complete).toBe(true);
  expect(found['/loose'].reasons[0].code).toBe('external_class');
  expect(found['/helpers'].inputs).toEqual({ page: 'int', o: 'text', m: 'text', d: 'text' });
  expect(found['/helpers'].complete).toBe(true);
  expect(found['/defaults'].inputs).toEqual({ s: 'text', h: 'text', x: 'text', arrow: 'text', y: 'text', page: 'int' });
  expect(found['/external'].reasons[0].code).toBe('helper');
});

test('mergeInto keeps each handler\'s reasons', () => {
  const routes = { '/a': {} };
  mergeInto(routes, '/a', { function: 'one', inputs: { q: 'text' }, complete: true });
  expect(routes['/a'].inputs_incomplete).toBeUndefined();
  const reason = { code: 'helper', detail: 'x' };
  mergeInto(routes, '/a', { function: 'two', inputs: {}, complete: false, reasons: [reason, reason] });
  expect(routes['/a'].inputs_incomplete).toEqual([{ code: 'helper', detail: 'x', function: 'two' }]);
});

test('validation middleware in route source names inputs, and checkExact closes the route', () => {
  const parse = source => {
    const ast = parser.parse(source, { sourceType: 'unambiguous' });
    const label = (node, parent) => String(parent.arguments[0].value);
    return Object.fromEntries(fileInputs(ast, 'app.js', source, label).map(entry => [entry.function, entry]));
  };
  const found = parse([
    "app.get('/ev', query('page').isInt().toInt(), query(['q', 'user.name', '*']), check('both'), param('id').isUUID(), header('x-key'), (req, res) => res.json(req.query));",
    "app.post('/exact', checkExact([body('name').isString(), body('age').isInt()]), (req, res) => res.json(helper(req.body)));",
    "app.post('/exact-one', checkExact(body('a')), checkExact(), (req, res) => res.json(req.body));",
    "app.post('/schema', checkSchema({ email: { in: ['body'], isEmail: true }, page: { in: 'query', isInt: true }, id: { in: ['params'] }, 'tags[0]': {}, [dynamic]: {}, ...rest, plain: 1 }), checkSchema(rules), (req, res) => res.json(req.body.email));",
    "app.post('/celebrate', celebrate({ [Segments.QUERY]: Joi.object({ page: Joi.number().integer() }), body: Joi.object().keys({ id: Joi.string().guid(), on: Joi.boolean(), n: Joi.number(), s: Joi.string(), [k]: Joi.any() }), params: Joi.object({ x: Joi.string() }), headers: other }), (req, res) => res.json(helper(req.body)));",
    "app.post('/celebrate-open', celebrate({ query: { q: Joi.string() }, body: Joi.object({ a: Joi.string() }).unknown(), cookies: Joi.object(), other: x }), (req, res) => res.json(helper(req.body)));",
    "app.post('/not', [body('b'), auth], obj[k].body('c'), make()[k]('d'), (req, res) => res.json(req.body.b));",
    "app.post('/this', this.validators.body('c').isInt(), make().query('d'), this.body('e'), (req, res) => res.json(req.body.c));",
    "app.post('/namespaced', ev.body('a').isInt(), ev.checkExact([ev.query('q')]), (req, res) => res.json(helper(req.body)));"
  ].join('\n'));
  expect(found['/ev'].inputs).toEqual({ page: 'int', q: 'text', user: 'text', both: 'text' });
  expect(found['/ev'].complete).toBe(false);
  expect(found['/ev'].reasons.map(item => item.code)).toEqual(['container_whole']);
  expect(found['/exact']).toEqual(expect.objectContaining({ inputs: { name: 'text', age: 'int' }, complete: true }));
  expect(found['/exact-one']).toEqual(expect.objectContaining({ inputs: { a: 'text' }, complete: true }));
  expect(found['/schema'].inputs).toEqual({ email: 'text', page: 'int', tags: 'text', plain: 'text' });
  expect(found['/schema'].complete).toBe(true);
  expect(found['/celebrate']).toEqual(expect.objectContaining({ inputs: { page: 'int', id: 'uuid', on: 'bool', n: 'float', s: 'text' }, complete: true }));
  expect(found['/celebrate-open'].inputs).toEqual({ q: 'text', a: 'text' });
  expect(found['/celebrate-open'].reasons.map(item => item.code)).toEqual(['container_whole']);
  expect(found['/not'].reasons.map(item => item.code)).toEqual(['middleware', 'middleware', 'middleware']);
  expect(found['/this']).toEqual(expect.objectContaining({ inputs: { c: 'text', d: 'text', e: 'text' }, complete: true }));
  expect(found['/namespaced']).toEqual(expect.objectContaining({ inputs: { a: 'int', q: 'text' }, complete: true }));
});

test('route middleware of the project\'s own is read like the handler', () => {
  const found = inputs([
    "const z = require('zod');",
    'const Shared = z.object({ title: z.string(), n: z.number().int() });',
    'const checkBody = (req, res, next) => { Shared.parse(req.body); next(); };',
    "function audit(req, res, next) { log(req.query.trace); next(); }",
    "function headersOnly(req, res, next) { if (!req.headers.authorization) return res.end(); next(); }",
    "function leaky(req, res, next) { helper(req.body); next(); }",
    "function notMiddleware(options) { return options; }",
    "app.post('/save', checkBody, audit, headersOnly, (req, res) => res.json(req.body.title));",
    "app.post('/leaky', leaky, (req, res) => res.json(req.body.x));",
    "app.post('/other', notMiddleware, imported, (req, res) => res.json(req.body.x));"
  ].join('\n'));
  expect(found['/save']).toEqual([{ title: 'text', n: 'int', trace: 'text' }, true]);
  expect(found['/leaky']).toEqual([{ x: 'text' }, false]);
  expect(found['/other']).toEqual([{ x: 'text' }, false]);
});


test('raw parsers, helpers given a container, route options and handlers that read nothing', () => {
  const found = inputs([
    "function readAddress(body) { return { street: body.street, city: body.city }; }",
    "function dump(b) { return JSON.stringify(b); }",
    "app.post('/orders', (req, res) => { const a = readAddress(req.body); res.json({ ...a, qty: req.body.qty }); });",
    "app.post('/dump', (req, res) => res.json(dump(req.body)));",
    "app.post('/raw', express.raw({ type: '*/*' }), (req, res) => res.json(req.body.length));",
    "app.post('/text', bodyParser.text(), (req, res) => res.json(req.body));",
    "app.get('/items', { schema: { querystring: {} } }, async request => request.query.q);",
    "app.get('/hooked', { preHandler: check }, async request => request.query.q);",
    "app.get('/spread', { ...options }, async request => request.query.q);",
    "app.get('/health', async () => ({ ok: true }));",
    "app.get(path, () => 1);"
  ].join('\n'));
  expect(found['/orders']).toEqual([{ street: 'text', city: 'text', qty: 'text' }, true]);
  expect(found['/dump'][1]).toBe(false);
  expect(found['/raw']).toEqual([{}, false]); // req.body.length is the raw bytes', not a field.
  expect(found['/text'][1]).toBe(false);
  expect(found['/items']).toEqual([{ q: 'text' }, true]);
  expect(found['/hooked']).toEqual([{ q: 'text' }, false]);
  expect(found['/spread']).toEqual([{ q: 'text' }, false]);
  expect(found['/health']).toEqual([{}, true]);
});

test('keys read inside an input are saved when every read of it reads constant keys', () => {
  const parser2 = require('@babel/parser');
  const src = [
    "app.post('/p', (req, res) => res.json([req.body.user.name, req.body.user.email, req.body.title, req.query.q]));",
    "app.post('/q', (req, res) => { save(req.body.user); return req.body.user.name; });",
    "app.post('/v', body('user'), (req, res) => req.body.user.name);",
    "function readUser(body) { return body.user.id; }",
    "app.post('/h', (req, res) => readUser(req.body));",
    "app.post('/m', check, (req, res) => req.body.user.name);",
    "function check(req, res, next) { req.body.user.role; next(); }"
  ].join('\n');
  const label = (node, parent) => (parent && parent.arguments ? String(parent.arguments[0].value) : node.id.name);
  const found = Object.fromEntries(fileInputs(parser2.parse(src), 'app.js', src, label).map(entry => [entry.function, entry]));
  expect(found['/p'].nested).toEqual({ user: ['email', 'name'] });
  expect(found['/q'].nested).toBeUndefined();
  expect(found['/v'].nested).toBeUndefined();
  expect(found['/h'].nested).toEqual({ user: ['id'] });
  expect(found['/m'].nested).toBeUndefined();
  const routes = { '/x': { view: 'x' } };
  mergeInto(routes, '/x', { function: 'a', inputs: { user: 'text' }, complete: true, nested: { user: ['name'] } });
  mergeInto(routes, '/x', { function: 'b', inputs: { user: 'text' }, complete: true, nested: { user: ['email'] } });
  expect(routes['/x']._nested).toEqual({ user: ['email', 'name'] });
  mergeInto(routes, '/x', { function: 'c', inputs: { user: 'text' }, complete: true });
  expect(routes['/x']._flat).toEqual(['user']);
});

