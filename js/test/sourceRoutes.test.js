const fs = require('fs');
const os = require('os');
const path = require('path');
const { HandlerSource, handlerText, routeHandlers, handlerFor } = require('../lib/sourceRoutes');
const pathManifest = require('../lib/pathManifest');
const { analyzeHandlerAst } = require('../lib/sourceAst');

let dir;
beforeEach(() => { dir = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-source-routes-')); });
afterEach(() => fs.rmSync(dir, { recursive: true, force: true }));

function write(name, text) {
  fs.mkdirSync(path.dirname(path.join(dir, name)), { recursive: true });
  fs.writeFileSync(path.join(dir, name), text);
}

describe('routeHandlers', () => {
  test('finds inline, named and chained handlers', () => {
    write('app.js', `
      function createOrder(req, res) { res.json({ sku: req.body.sku }); }
      const remove = (req, res) => res.json({ ok: true });
      app.post('/orders', auth, createOrder);
      app.get(\`/health\`, (req, res) => res.send('ok'));
      router.route('/items/:id').get(async (req, res) => res.json({})).delete(remove);
      router.del('/old', remove);
      app.get('/unknown', missingHandler);
      app.get(path, handler);
      app.get('relative', (req, res) => res.json({}));
      app.listen(3000);
    `);
    write('broken.js', 'this is not javascript (');
    const found = routeHandlers(dir);
    expect([...found.keys()].sort()).toEqual(['DELETE /items/:id', 'DELETE /old', 'GET /health', 'GET /items/:id', 'POST /orders']);
    expect(found.get('POST /orders')).toBeInstanceOf(HandlerSource);
    expect(found.get('POST /orders').name).toBe('createOrder');
    expect(found.get('POST /orders').text).toMatch(/^function createOrder/);
    expect(routeHandlers(path.join(dir, 'nowhere')).size).toBe(0);
  });

  test('matches a listed route exactly, through ALL, or by the one path it ends with', () => {
    const handlers = new Map([
      ['GET /profile', new HandlerSource('a')],
      ['ALL /any', new HandlerSource('b')],
      ['GET /', new HandlerSource('root')],
      ['POST /a/save', new HandlerSource('c')],
      ['POST /b/save', new HandlerSource('d')]
    ]);
    expect(handlerFor(handlers, '/profile').text).toBe('a');
    expect(handlerFor(handlers, '/any', ['PUT']).text).toBe('b');
    expect(handlerFor(handlers, '/account/profile', ['GET']).text).toBe('a');
    expect(handlerFor(handlers, '/x/save', ['POST'])).toBeNull();
    expect(handlerFor(handlers, '/account/profile', ['POST'])).toBeNull();
  });

  test('handlerText reads functions and HandlerSources only', () => {
    expect(handlerText(function named() { return 1; })).toMatch(/function named/);
    expect(handlerText(new HandlerSource('(req, res) => res.json({})'))).toBe('(req, res) => res.json({})');
    expect(handlerText({ text: 'not a handler' })).toBeNull();
    expect(new HandlerSource(null).text).toBe('');
  });
});

describe('handlers from source in manifests', () => {
  test('a HandlerSource gives the same API and auth signals as the function', () => {
    const handler = async (req, res) => {
      const { username, password } = req.body;
      res.json({ token: Boolean(username && password) });
    };
    const text = new HandlerSource(handler.toString());
    expect(pathManifest.detectApiEndpoint(text, '/session', ['POST']).isApi).toBe(true);
    expect(pathManifest.detectApiEndpoint(text, '/session', ['POST'])).toEqual(pathManifest.detectApiEndpoint(handler, '/session', ['POST']));
    expect(pathManifest.detectAuthEndpoint(text, '/login')).toEqual(pathManifest.detectAuthEndpoint(handler, '/login'));
  });

  test('an arrow handler whose body is an object returns JSON; an inner one does not count', () => {
    expect(analyzeHandlerAst(async () => ({ ok: true })).apiSignals).toContain('return:json_literal');
    const inner = analyzeHandlerAst(function list(req, res) { res.send(req.query.ids.map(id => ({ id }))); });
    expect(inner.apiSignals).not.toContain('return:json_literal');
  });

  test('Koa ctx.body objects and class methods read as JSON', () => {
    const koa = async ctx => { ctx.body = { id: ctx.params.id }; };
    expect(analyzeHandlerAst(koa).apiSignals).toContain('ctx.body:json');
    const text = async ctx => { ctx.body = 'plain'; };
    expect(analyzeHandlerAst(text).apiSignals).not.toContain('ctx.body:json');
    class OrdersController {
      list(query) { return { items: [], page: query.page }; }
      async show() { return [1]; }
    }
    expect(analyzeHandlerAst(OrdersController.prototype.list).available).toBe(true);
    expect(pathManifest.detectApiEndpoint(OrdersController.prototype.list, '/orders', ['GET']).isApi).toBe(true);
    expect(pathManifest.detectApiEndpoint(OrdersController.prototype.show, '/orders/1', ['GET']).isApi).toBe(true);
    const [, entry] = pathManifest.routeEntryFor({ routePath: '/x', methods: ['GET'], handler: () => 'x', responseType: 'json' });
    expect(entry.response_type).toBe('json');
    expect(entry.api_signals).toContain('framework:json_response');
    const listed = pathManifest.buildFrameworkManifest('koa', null, { routes: [{ path: '/k', method: 'GET', response_type: 'json' }] });
    expect(listed.routes['/k'].response_type).toBe('json');
  });

  test('a route list keeps handlers it already has, and paths it has none for', () => {
    write('app.js', "app.get('/a', (req, res) => res.json({}));");
    const own = () => 1;
    const manifest = pathManifest.buildFrameworkManifest('express', null, {
      routes: [{ method: 'GET', path: '/a' }, { method: 'GET', path: '/b', handler: own }, { method: 'GET' }, null],
      handlerSource: dir
    });
    expect(Object.keys(manifest.routes).sort()).toEqual(['/a', '/b']);
    expect(manifest.routes['/a'].response_type).toBe('json');
    expect(pathManifest.buildFrameworkManifest('express', null, { routes: [{ path: '/c' }], handlerSource: path.join(dir, 'gone') }).routes['/c']).toBeDefined();
  });
});

describe('live route reading', () => {
  test('a ready Fastify app lists every route with its methods', async () => {
    const fastify = require('fastify')();
    fastify.get('/health', async () => ({ ok: true }));
    fastify.post('/orders', async request => ({ sku: request.body.sku }));
    fastify.register(async scope => { scope.get('/profile', async () => ({})); }, { prefix: '/account' });
    await fastify.ready();
    try {
      expect(pathManifest.fastifyRouteList(fastify)).toEqual([
        { path: '/health', methods: ['GET'] }, { path: '/orders', methods: ['POST'] }, { path: '/account/profile', methods: ['GET'] }
      ]);
      const manifest = pathManifest.buildFrameworkManifest('fastify', fastify, {});
      expect(Object.keys(manifest.routes).sort()).toEqual(['/account/profile', '/health', '/orders']);
      expect(Object.keys(pathManifest.extractFastifyRoutes(fastify)).sort()).toEqual(['/account/profile', '/health', '/orders']);
    } finally {
      await fastify.close();
    }
    expect(pathManifest.fastifyRouteList({ printRoutes() { throw new Error('not ready'); } })).toEqual([]);
    expect(pathManifest.fastifyRouteList({})).toEqual([]);
  });

  test('a Koa app with router middleware gives its routes; one without gives none', () => {
    const handler = async ctx => { ctx.body = { ok: true }; };
    const routes = async () => {};
    routes.router = { stack: [{ path: '/k', methods: ['HEAD', 'GET'], stack: [handler] }, { path: 3 }, null] };
    expect(Object.keys(pathManifest.extractKoaRoutes({ middleware: [() => {}, routes] }))).toEqual(['/k']);
    expect(pathManifest.extractKoaRoutes({ middleware: [] })).toEqual({});
    expect(pathManifest.extractKoaRoutes(null)).toEqual({});
  });
});
