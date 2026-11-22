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
  expect(routes['/x']).toEqual({ inputs: { a: 'text', b: 'text' }, inputs_complete: false });
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
