require('reflect-metadata');

const fastifyFactory = require('fastify');
const Hapi = require('@hapi/hapi');
const Validate = require('@hapi/validate');
const { NestFactory } = require('@nestjs/core');
const {
  Body, Controller, Get, Module, Next, ParseIntPipe, Post, Query, Redirect, Render, Req, Res, createParamDecorator
} = require('@nestjs/common');
const framework = require('../lib/frameworkInputs');
const { finalize } = require('../lib/inputSources');
const {
  auditIntoManifest, extractExpressRoutes, extractFastifyRoutes, extractHapiRoutes, extractNestRoutes
} = require('../lib/pathManifest');

process.env.NODE_ENV = 'test';

const codes = declared => declared.reasons.map(item => item.code);

function decorate(cls, name, decorators) {
  const descriptor = Object.getOwnPropertyDescriptor(cls.prototype, name);
  for (const decorator of decorators) decorator(cls.prototype, name, descriptor);
}

describe('framework inputs', () => {
  it('reads JSON Schema, Joi and shorthand schemas', () => {
    const reasons = [];
    expect(framework.fieldsOf({
      allOf: [{ type: 'object', properties: { a: { type: 'integer' } } }],
      properties: { b: { type: 'string', format: 'uuid' }, c: { $ref: '#/x' } }
    }, 'body', reasons)).toEqual({ a: 'int', b: 'uuid', c: 'text' });
    expect(reasons).toEqual([]);

    expect(framework.fieldsOf({ oneOf: [{ type: 'object', properties: { x: { type: 'boolean' } } }] }, 'body', reasons)).toEqual({ x: 'bool' });
    expect(framework.fieldsOf({ $ref: 'shared#' }, 'body', reasons)).toEqual({});
    expect(framework.fieldsOf({ type: 'array' }, 'body', reasons)).toEqual({});
    expect(framework.fieldsOf({ type: 'object', additionalProperties: true, properties: { y: { type: 'number' } } }, 'query', reasons)).toEqual({ y: 'float' });
    expect(reasons.map(item => item.code)).toEqual(['external_class', 'container_whole', 'container_whole']);

    const joi = [];
    expect(framework.fieldsOf(Validate.object({
      n: Validate.number().integer(), f: Validate.number(), b: Validate.boolean(), id: Validate.string().guid(),
      list: Validate.array().items(Validate.number().integer()), any: Validate.array(), s: Validate.string()
    }), 'body', joi)).toEqual({ n: 'int', f: 'float', b: 'bool', id: 'uuid', list: 'int', any: 'text', s: 'text' });
    expect(joi).toEqual([]);
    expect(framework.fieldsOf(Validate.object({ a: Validate.string() }).unknown(), 'body', joi)).toEqual({ a: 'text' });
    expect(framework.fieldsOf(Validate.object(), 'body', joi)).toEqual({});
    expect(framework.fieldsOf(Validate.string(), 'body', joi)).toEqual({});
    expect(joi.map(item => item.code)).toEqual(['container_whole', 'container_whole', 'container_whole']);

    expect(framework.fieldsOf({ q: Validate.string() }, 'query', [])).toEqual({ q: 'text' });
    expect(framework.fieldsOf({ q: 'nope' }, 'query', [])).toBeNull();
    expect(framework.fieldsOf(null, 'query', [])).toBeNull();
    expect(framework.fieldsOf([], 'query', [])).toBeNull();
  });

  it('reads DTO classes from Swagger and class-validator metadata', () => {
    class Base {}
    class CreateUser extends Base {}
    Reflect.defineMetadata('swagger/apiModelPropertiesArray', [':name', ':age'], CreateUser.prototype);
    Reflect.defineMetadata('swagger/apiModelProperties', { type: 'integer' }, CreateUser.prototype, 'age');
    Reflect.defineMetadata('design:type', String, CreateUser.prototype, 'name');
    const previous = globalThis.classValidatorMetadataStorage;
    globalThis.classValidatorMetadataStorage = {
      getTargetValidationMetadatas: cls => (cls === CreateUser ? [
        { propertyName: 'owner', name: 'isUUID' },
        { propertyName: 'active', name: 'customValidation', constraints: ['isBoolean'] },
        { propertyName: 'name', name: 'isString' },
        { propertyName: 'nick' },
        {}
      ] : null)
    };
    try {
      const reasons = [];
      expect(framework.classFields(CreateUser, 'body', reasons)).toEqual({ name: 'text', age: 'int', owner: 'uuid', active: 'bool', nick: 'text' });
      expect(reasons).toEqual([]);
      expect(framework.classFields(Base, 'body', reasons)).toEqual({});
      expect(codes({ reasons })).toEqual(['external_class']);
      expect(framework.classFields(function () {}, 'query', reasons)).toEqual({});
      expect(reasons[1].detail).toMatch(/^the query class/);
    } finally {
      globalThis.classValidatorMetadataStorage = previous;
    }
  });

  it('combines a route schema into a declaration', () => {
    expect(framework.fastifyInputs(null)).toBeNull();
    expect(framework.fastifyInputs({ response: {} })).toBeNull();
    const declared = framework.fastifyInputs({ querystring: { type: 'object', properties: { q: { type: 'string' } } }, body: { type: 'object', properties: { n: { type: 'integer' } } } });
    expect(declared).toEqual({ inputs: { q: 'text', n: 'int' }, covers: expect.arrayContaining(['container_whole']), reasons: [], standalone: false });
    expect(framework.hapiInputs(undefined)).toBeNull();
    expect(framework.hapiInputs({ query: false, payload: true })).toEqual(expect.objectContaining({ inputs: {} }));

    expect(framework.combine(null, declared)).toBe(declared);
    expect(framework.combine(declared, null)).toBe(declared);
    const other = { inputs: { x: 'bool' }, covers: ['dependency'], reasons: [{ code: 'helper' }], standalone: true };
    expect(framework.combine({ ...declared, standalone: true }, other)).toEqual({
      inputs: { q: 'text', n: 'int', x: 'bool' }, covers: ['dependency'], reasons: [{ code: 'helper' }], standalone: true
    });
  });

  it('collects Fastify routes and their schemas through the plugin', async () => {
    let plugin;
    jest.isolateModules(() => {
      jest.doMock('../lib/wafMiddleware', () => () => (_req, _res, next) => next());
      plugin = require('../lib/fastifyPlugin');
    });
    jest.dontMock('../lib/wafMiddleware');
    const fastify = fastifyFactory();
    await fastify.register(plugin);
    fastify.get('/search', { schema: { querystring: { type: 'object', properties: { q: { type: 'string' }, page: { type: 'integer' } }, additionalProperties: false } } }, async () => ({}));
    fastify.post('/items', { schema: { body: { type: 'object', properties: { name: { type: 'string' } } } } }, async () => ({}));
    await fastify.ready();
    const routes = extractFastifyRoutes(fastify);
    expect(routes['/search'].declared_inputs.inputs).toEqual({ q: 'text', page: 'int' });
    expect(routes['/search'].methods).toContain('GET');
    expect(routes['/items'].declared_inputs.inputs).toEqual({ name: 'text' });
    await fastify.close();

    const bare = { addHook: jest.fn(), hasDecorator: () => true };
    plugin(bare, {}, () => {});
    bare.addHook.mock.calls.find(([name]) => name === 'onRoute')[1]({ url: '/x' });
  });

  it('reads Hapi validate.query and validate.payload', async () => {
    const server = Hapi.server();
    server.validator(Validate);
    server.route({ method: 'GET', path: '/search', options: { handler: () => 'ok', validate: { query: { q: Validate.string(), page: Validate.number().integer() } } } });
    server.route({ method: 'POST', path: '/items', options: { handler: () => 'ok', validate: { payload: Validate.object({ name: Validate.string() }).unknown() } } });
    server.route({ method: 'GET', path: '/plain', handler: () => 'ok' });
    const routes = extractHapiRoutes(server);
    expect(routes['/search'].declared_inputs.inputs).toEqual({ q: 'text', page: 'int' });
    expect(codes(routes['/items'].declared_inputs)).toEqual(['container_whole']);
    expect(routes['/plain'].declared_inputs).toBeUndefined();
  });

  it('reads Nest controllers, their parameters and DTOs', async () => {
    class SearchQuery {}
    Reflect.defineMetadata('swagger/apiModelPropertiesArray', [':term'], SearchQuery.prototype);
    class CreateItem {}
    Reflect.defineMetadata('swagger/apiModelPropertiesArray', [':name', ':qty'], CreateItem.prototype);
    Reflect.defineMetadata('swagger/apiModelProperties', { type: Number }, CreateItem.prototype, 'qty');
    const User = createParamDecorator((_data, ctx) => ctx.switchToHttp().getRequest().user);

    class BaseController {
      list(page) { return { page }; }
    }
    class ItemsController extends BaseController {
      show(id, verbose) { return { id, verbose }; }
      find(query) { return query; }
      create(body) { return body; }
      raw(req, body) { return [req.query, body]; }
      mine(user) { return user; }
      helper() { return null; }
    }
    decorate(BaseController, 'list', [Get('list')]);
    Query('page', ParseIntPipe)(BaseController.prototype, 'list', 0);
    decorate(ItemsController, 'show', [Get(['show', 'view'])]);
    Query('id', ParseIntPipe)(ItemsController.prototype, 'show', 0);
    Query('verbose')(ItemsController.prototype, 'show', 1);
    Reflect.defineMetadata('design:paramtypes', [Number, Boolean], ItemsController.prototype, 'show');
    decorate(ItemsController, 'find', [Get('find')]);
    Query()(ItemsController.prototype, 'find', 0);
    Reflect.defineMetadata('design:paramtypes', [SearchQuery], ItemsController.prototype, 'find');
    decorate(ItemsController, 'create', [Post()]);
    Body()(ItemsController.prototype, 'create', 0);
    Reflect.defineMetadata('design:paramtypes', [CreateItem], ItemsController.prototype, 'create');
    decorate(ItemsController, 'raw', [Post('raw')]);
    Req()(ItemsController.prototype, 'raw', 0);
    Body()(ItemsController.prototype, 'raw', 1);
    decorate(ItemsController, 'mine', [Get('mine')]);
    User()(ItemsController.prototype, 'mine', 0);
    Controller('items')(ItemsController);

    class AppModule {}
    Module({ controllers: [ItemsController, ItemsController] })(AppModule);
    const app = await NestFactory.create(AppModule, { logger: false });
    app.setGlobalPrefix('api');
    await app.init();
    try {
      const routes = extractNestRoutes(app);
      expect(Object.keys(routes).sort()).toEqual([
        '/api/items', '/api/items/find', '/api/items/list', '/api/items/mine', '/api/items/raw', '/api/items/show', '/api/items/view'
      ]);
      expect(routes['/api/items/list'].declared_inputs.inputs).toEqual({ page: 'int' });
      expect(routes['/api/items/show'].declared_inputs.inputs).toEqual({ id: 'int', verbose: 'bool' });
      expect(routes['/api/items/find'].declared_inputs.inputs).toEqual({ term: 'text' });
      expect(routes['/api/items'].declared_inputs).toEqual(expect.objectContaining({ inputs: { name: 'text', qty: 'float' }, standalone: true }));
      expect(routes['/api/items'].methods).toEqual(['POST']);
      const raw = routes['/api/items/raw'].declared_inputs;
      expect(raw.standalone).toBe(false);
      expect(raw.covers).not.toContain('container_whole');
      expect(codes(raw)).toEqual(['container_whole']);
      expect(codes(routes['/api/items/mine'].declared_inputs)).toEqual(['dependency']);
    } finally {
      await app.close();
    }
  });

  it('marks Nest routes JSON unless they render, redirect or answer through @Res()', async () => {
    class OrdersController {
      constructor() { this.orders = { all: () => [] }; }
      list() { return this.orders.all(); }
      page() { return {}; }
      away() { return undefined; }
      manual(res) { res.send('x'); }
      pass(res) { res.status(201); return { ok: true }; }
      chained(next) { next(); }
    }
    decorate(OrdersController, 'list', [Get()]);
    decorate(OrdersController, 'page', [Get('page'), Render('orders')]);
    decorate(OrdersController, 'away', [Get('away'), Redirect('/orders')]);
    decorate(OrdersController, 'manual', [Get('manual')]);
    Res()(OrdersController.prototype, 'manual', 0);
    decorate(OrdersController, 'pass', [Post('pass')]);
    Res({ passthrough: true })(OrdersController.prototype, 'pass', 0);
    decorate(OrdersController, 'chained', [Get('chained')]);
    Next()(OrdersController.prototype, 'chained', 0);
    Controller('orders')(OrdersController);
    class OrdersModule {}
    Module({ controllers: [OrdersController] })(OrdersModule);
    const app = await NestFactory.create(OrdersModule, { logger: false });
    await app.init();
    try {
      const routes = extractNestRoutes(app);
      const types = Object.fromEntries(Object.entries(routes).map(([path, entry]) => [path, entry.response_type || null]));
      expect(types).toEqual({
        '/orders': 'json', '/orders/page': 'html', '/orders/away': 'html', '/orders/manual': 'html', '/orders/pass': 'json', '/orders/chained': 'html'
      });
      expect(routes['/orders'].category).toBe('api');
    } finally {
      await app.close();
    }
  });

  it('handles Nest edge cases and non-Nest apps', () => {
    expect(framework.nestRoutes(null)).toEqual([]);
    expect(framework.nestRoutes({ container: {} })).toEqual([]);
    class AnyController {
      any() {}
    }
    decorate(AnyController, 'any', [require('@nestjs/common').All()]);
    Query('x')(AnyController.prototype, 'any', 0);
    const container = { getModules: () => new Map([['m', { controllers: new Map([['a', { metatype: AnyController }], ['b', {}]]) }], ['n', {}]]) };
    const [route] = framework.nestRoutes({ container });
    expect(route).toEqual(expect.objectContaining({ routePath: '/', methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'] }));
    expect(route.declared.inputs).toEqual({ x: 'text' });

    class Odd {}
    Reflect.defineMetadata('__routeArguments__', { '3:x': null }, Odd, 'm');
    expect(codes(framework.nestInputs(Odd, 'm'))).toEqual(['container_whole']);

    const getMetadata = Reflect.getMetadata;
    Reflect.getMetadata = undefined;
    try {
      expect(framework.nestRoutes({ container })).toEqual([]);
      expect(framework.classFields(Odd, 'body', [])).toEqual({});
    } finally {
      Reflect.getMetadata = getMetadata;
    }
  });

  it('reads Express 4 and 5 routers', () => {
    function handler(req, res) { res.json(req.query.q); }
    expect(extractExpressRoutes(require('express')())).toEqual({});
    const routes = extractExpressRoutes({ router: { stack: [{ route: { path: '/five', methods: { get: true }, stack: [{ handle: handler }] } }] } });
    expect(routes['/five'].methods).toEqual(['GET']);
  });

  it('combines framework declarations with the code reading', () => {
    const routes = {
      '/covered': {
        inputs: { a: 'text' }, inputs_complete: false, inputs_incomplete: [{ code: 'external_class' }, { code: 'helper' }],
        declared_inputs: { inputs: { b: 'int' }, covers: ['external_class'], reasons: [{ code: 'container_whole' }], standalone: false }
      },
      '/standalone': { declared_inputs: { inputs: { q: 'text' }, covers: [], reasons: [], standalone: true } },
      '/needs-code': { declared_inputs: { inputs: { q: 'text' }, standalone: false } },
      '/bare': { declared_inputs: {} , inputs_complete: true }
    };
    const counts = finalize(routes);
    expect(routes['/covered'].inputs).toEqual({ a: 'text', b: 'int' });
    expect(routes['/covered'].inputs_incomplete.map(item => item.code)).toEqual(['helper', 'container_whole']);
    expect(routes['/standalone']).toEqual({ inputs: { q: 'text' }, inputs_complete: true });
    expect(routes['/needs-code']).toEqual({});
    expect(routes['/bare'].inputs_complete).toBe(true);
    expect(counts.complete).toBe(2);

    const manifest = { routes: { '/x': { declared_inputs: {}, input_declaration: {} }, '/y': null } };
    auditIntoManifest(manifest, null);
    // Without the audit, the declaration still counts (an empty one says the route reads nothing),
    // but not as complete: the route's risks weren't looked for.
    expect(manifest.routes['/x']).toMatchObject({ inputs: {}, inputs_complete: false });
    expect(manifest.routes['/x'].inputs_incomplete.map(item => item.code)).toEqual(['not_audited']);
  });
});

describe('framework inputs for Express validators and Nest paths', () => {
  const chain = (fields, locations, validators = []) => {
    const middleware = (_req, _res, next) => next();
    return Object.assign(middleware, { builder: { fields, locations, stack: validators.map(name => ({ validator: { name } })) } });
  };

  it('reads celebrate and express-validator middleware on Express routes', () => {
    const express = require('express');
    const app = express();
    const celebrated = Object.assign((_req, _res, next) => next(), {
      _schema: { query: { q: Validate.string() }, body: Validate.object({ n: Validate.number().integer() }).unknown(), params: { id: Validate.string() } }
    });
    function search(req, res) { res.json(req.query); }
    app.get('/celebrate', celebrated, search);
    app.post('/validated', [chain(['name'], ['body']), chain(['page'], ['query'], ['isInt']), chain(['user.email', 'tags[0]', '*', ''], ['body', 'query'])], chain(['id'], ['params']), search);
    app.post('/mixed', chain(['a'], ['body']), function auth(_req, _res, next) { next(); }, search);
    app.get('/plain', search);
    const routes = extractExpressRoutes(app);
    expect(routes['/celebrate'].declared_inputs).toEqual({ inputs: { q: 'text', n: 'int' }, covers: ['middleware'], reasons: [expect.objectContaining({ code: 'container_whole' })], standalone: false });
    expect(routes['/celebrate'].view).toBe('search');
    expect(routes['/validated'].declared_inputs.inputs).toEqual({ name: 'text', page: 'int', user: 'text', tags: 'text' });
    expect(routes['/validated'].declared_inputs.covers).toEqual(['middleware']);
    expect(routes['/mixed'].declared_inputs).toEqual({ inputs: { a: 'text' }, covers: [], reasons: [], standalone: false });
    expect(routes['/plain'].declared_inputs).toBeUndefined();
    expect(framework.expressInputs(undefined)).toBeNull();
  });

  it('builds Nest paths the way Nest does: versions, prefix exclusions and RouterModule', async () => {
    const { Version, VersioningType, VERSION_NEUTRAL } = require('@nestjs/common');
    const { RouterModule } = require('@nestjs/core');
    class HealthController { check() { return 'ok'; } }
    decorate(HealthController, 'check', [Get()]);
    Controller('health')(HealthController);
    class UsersController {
      list(q) { return q; }
      latest() { return []; }
    }
    decorate(UsersController, 'list', [Get()]);
    Query('q')(UsersController.prototype, 'list', 0);
    decorate(UsersController, 'latest', [Get('latest'), Version(['2', VERSION_NEUTRAL])]);
    Controller({ path: 'users', version: '1' })(UsersController);
    class AdminModule {}
    Module({ controllers: [UsersController] })(AdminModule);
    class AppModule {}
    Module({ imports: [AdminModule, RouterModule.register([{ path: 'admin', module: AdminModule }])], controllers: [HealthController] })(AppModule);

    const app = await NestFactory.create(AppModule, { logger: false });
    app.setGlobalPrefix('api', { exclude: ['health'] });
    app.enableVersioning({ type: VersioningType.URI });
    await app.init();
    try {
      const routes = extractNestRoutes(app);
      expect(Object.keys(routes).sort()).toEqual(['/api/admin/users/latest', '/api/v1/admin/users', '/api/v2/admin/users/latest', '/health']);
      expect(routes['/api/v1/admin/users'].declared_inputs.inputs).toEqual({ q: 'text' });

      const plain = framework.nestRoutes(app, null).map(route => route.routePath).sort();
      expect(plain).toEqual(['/api/admin/users', '/api/admin/users/latest', '/api/health']);
    } finally {
      await app.close();
    }
  });
});

describe('zod schemas', () => {
  // Stand-ins shaped like zod 3 (schema._def.typeName) and zod 4 (schema._zod.def.type, with
  // classic's _def alias), as their sources define them.
  const z3 = (typeName, extra = {}) => ({ _def: { typeName, ...extra } });
  const z4 = (type, extra = {}) => {
    const def = { type, ...extra };
    return { _zod: { def }, get _def() { return def; } };
  };
  const format = value => ({ _zod: { def: { check: 'format', format: value } } });

  it('reads zod 3 and zod 4 objects, wrappers and types', () => {
    const v3 = z3('ZodObject', {
      unknownKeys: 'strip',
      catchall: z3('ZodNever'),
      shape: () => ({
        n: z3('ZodOptional', { innerType: z3('ZodNumber', { checks: [{ kind: 'int' }] }) }),
        f: z3('ZodEffects', { schema: z3('ZodNumber', { checks: [] }) }),
        b: z3('ZodDefault', { innerType: z3('ZodBoolean') }),
        id: z3('ZodPipeline', { in: z3('ZodString', { checks: [{ kind: 'uuid' }] }) }),
        s: z3('ZodBranded', { type: z3('ZodString') }),
        big: z3('ZodBigInt'),
        list: z3('ZodArray', { type: z3('ZodNumber', { checks: [{ kind: 'int' }] }) }),
        other: z3('ZodDate')
      })
    });
    const reasons = [];
    expect(framework.fieldsOf(v3, 'body', reasons)).toEqual({ n: 'int', f: 'float', b: 'bool', id: 'uuid', s: 'text', big: 'int', list: 'int', other: 'text' });
    expect(reasons).toEqual([]);
    expect(framework.fieldsOf(z3('ZodObject', { unknownKeys: 'passthrough', shape: {} }), 'body', reasons)).toEqual({});
    expect(framework.fieldsOf(z3('ZodObject', { unknownKeys: 'strip', shape: () => ({}), catchall: z3('ZodString') }), 'body', reasons)).toEqual({});
    expect(framework.fieldsOf(z3('ZodString'), 'body', reasons)).toEqual({});
    expect(codes({ reasons })).toEqual(['container_whole', 'container_whole', 'container_whole']);

    const v4 = z4('object', {
      shape: {
        n: z4('optional', { innerType: z4('number', { format: 'safeint' }) }),
        m: z4('number', { checks: [format('int32'), {}] }),
        f: z4('nullable', { innerType: z4('number') }),
        big: z4('bigint'),
        b: z4('default', { innerType: z4('boolean') }),
        id: z4('pipe', { in: z4('string', { format: 'uuid' }) }),
        g: z4('string', { checks: [format('guid')] }),
        s: z4('string', { checks: [format('email')] }),
        list: z4('array', { element: z4('number', { format: 'safeint' }) }),
        other: z4('date')
      }
    });
    const four = [];
    expect(framework.fieldsOf(v4, 'query', four)).toEqual({ n: 'int', m: 'int', f: 'float', big: 'int', b: 'bool', id: 'uuid', g: 'uuid', s: 'text', list: 'int', other: 'text' });
    expect(framework.fieldsOf(z4('object', { shape: { a: z4('string') }, catchall: z4('never') }), 'query', four)).toEqual({ a: 'text' });
    expect(four).toEqual([]);
    expect(framework.fieldsOf(z4('object', { catchall: z4('unknown') }), 'query', four)).toEqual({});
    expect(codes({ reasons: four })).toEqual(['container_whole']);

    let deep = z4('string');
    for (let index = 0; index < 20; index += 1) deep = z4('optional', { innerType: deep });
    expect(framework.fieldsOf({ q: deep, n: z3('ZodNumber') }, 'query', [])).toEqual({ q: 'text', n: 'float' });
  });

  it('reads zod in Fastify schemas, nestjs-zod DTOs and aiwaf.inputs', () => {
    const body = z4('object', { shape: { name: z4('string') } });
    expect(framework.fastifyInputs({ body }).inputs).toEqual({ name: 'text' });
    class ItemDto {}
    ItemDto.isZodDto = true;
    ItemDto.schema = body;
    expect(framework.classFields(ItemDto, 'body', [])).toEqual({ name: 'text' });

    const { inputs, declaredOn } = require('../lib/inputSources');
    const handler = () => null;
    expect(declaredOn(inputs(body)(handler))).toEqual({ inputs: { name: 'text' }, complete: true });
    const open = z4('object', { shape: { a: z4('string') }, catchall: z4('unknown') });
    expect(declaredOn(inputs(open)(() => null))).toEqual({ inputs: { a: 'text' }, complete: false });
    expect(declaredOn(inputs(open, { complete: true })(() => null)).complete).toBe(true);
    expect(declaredOn(inputs(Validate.object({ q: Validate.string() }), { complete: false })(() => null))).toEqual({ inputs: { q: 'text' }, complete: false });
    expect(declaredOn(inputs(ItemDto)(() => null)).inputs).toEqual({ name: 'text' });
    expect(declaredOn(inputs({ type: 'int' })(() => null)).inputs).toEqual({ type: 'int' });
  });
});

describe('real validation libraries', () => {
  it('reads real zod 4 and zod 3 schemas', () => {
    const { z } = require('zod');
    const z3 = require('zod/v3').z;
    const four = z.object({
      n: z.number().int(), i: z.int().optional(), f: z.number().nullable(), b: z.boolean().default(false),
      id: z.string().uuid(), u: z.uuid(), s: z.string().email(), list: z.array(z.int()), t: z.string().transform(Number), big: z.bigint()
    });
    const reasons = [];
    expect(framework.fieldsOf(four, 'body', reasons)).toEqual({ n: 'int', i: 'int', f: 'float', b: 'bool', id: 'uuid', u: 'uuid', s: 'text', list: 'int', t: 'text', big: 'int' });
    expect(reasons).toEqual([]);
    expect(framework.fieldsOf(z.object({ a: z.string() }).strict(), 'body', reasons)).toEqual({ a: 'text' });
    expect(reasons).toEqual([]);
    expect(framework.fieldsOf(z.object({ a: z.string() }).loose(), 'body', reasons)).toEqual({ a: 'text' });
    expect(framework.fieldsOf(z.object({}).catchall(z.string()), 'body', reasons)).toEqual({});
    expect(codes({ reasons })).toEqual(['container_whole', 'container_whole']);

    const three = z3.object({
      n: z3.number().int(), f: z3.number().optional(), b: z3.boolean().default(true), id: z3.string().uuid(),
      r: z3.string().refine(Boolean), list: z3.array(z3.number().int()), p: z3.string().pipe(z3.string().uuid()), br: z3.string().brand('x')
    });
    const old = [];
    expect(framework.fieldsOf(three, 'query', old)).toEqual({ n: 'int', f: 'float', b: 'bool', id: 'uuid', r: 'text', list: 'int', p: 'text', br: 'text' });
    expect(old).toEqual([]);
    expect(framework.fieldsOf(z3.object({ a: z3.string() }).passthrough(), 'query', old)).toEqual({ a: 'text' });
    expect(framework.fieldsOf(z3.object({ a: z3.string() }).strict(), 'query', old)).toEqual({ a: 'text' });
    expect(codes({ reasons: old })).toEqual(['container_whole']);
  });

  it('reads real nestjs-zod DTOs and aiwaf.inputs with real schemas', () => {
    const { z } = require('zod');
    const { createZodDto } = require('nestjs-zod');
    class CreateItem extends createZodDto(z.object({ name: z.string(), qty: z.int() })) {}
    expect(framework.classFields(CreateItem, 'body', [])).toEqual({ name: 'text', qty: 'int' });
    const { inputs, declaredOn } = require('../lib/inputSources');
    const Joi = require('joi');
    expect(declaredOn(inputs(Joi.object({ q: Joi.string(), page: Joi.number().integer() }))(() => null))).toEqual({ inputs: { q: 'text', page: 'int' }, complete: true });
    expect(declaredOn(inputs(CreateItem)(() => null))).toEqual({ inputs: { name: 'text', qty: 'int' }, complete: true });
  });

  it('reads real celebrate and express-validator middleware on Express routes', () => {
    const express = require('express');
    const { celebrate, Joi, Segments } = require('celebrate');
    const { body, query, param, checkSchema, checkExact } = require('express-validator');
    const app = express();
    function handler(req, res) { res.json(req.query); }
    app.get('/celebrate', celebrate({ [Segments.QUERY]: Joi.object({ q: Joi.string(), page: Joi.number().integer() }), [Segments.BODY]: Joi.object({ id: Joi.string().guid() }).unknown() }), handler);
    app.post('/ev', query('page').isInt(), body('name').isString(), body(['user.email', 'tags[0]']), param('id').isUUID(), handler);
    app.post('/schema', checkSchema({ email: { in: ['body'], isEmail: true }, page: { in: ['query'], isInt: true }, id: { in: ['params'] } }), handler);
    app.post('/exact', checkExact([body('a')]), handler);
    const routes = extractExpressRoutes(app);
    expect(routes['/celebrate'].declared_inputs).toEqual({ inputs: { q: 'text', page: 'int', id: 'uuid' }, covers: ['middleware'], reasons: [expect.objectContaining({ code: 'container_whole' })], standalone: false });
    expect(routes['/ev'].declared_inputs.inputs).toEqual({ page: 'int', name: 'text', user: 'text', tags: 'text' });
    expect(routes['/ev'].declared_inputs.covers).toEqual(['middleware']);
    expect(routes['/schema'].declared_inputs.inputs).toEqual({ email: 'text', page: 'int' });
    expect(routes['/exact'].declared_inputs).toBeUndefined();
  });
});

describe('validation middleware with its schema in its closure', () => {
  it('reads zod and Joi schemas a middleware closed over', () => {
    const { z } = require('zod');
    const Joi = require('joi');
    const express = require('express');
    function validate(schema, where = 'body') { return (req, _res, next) => { schema.parse(req[where]); next(); }; }
    function validateParts(parts) { return (req, _res, next) => { Object.keys(parts).forEach(key => parts[key].validate(req[key])); next(); }; }
    function plain(options) { return (_req, _res, next) => { next(options); }; }
    const Shared = z.object({ shared: z.string() });
    const moduleLevel = (req, _res, next) => { Shared.parse(req.body); next(); };
    const app = express();
    function handler(req, res) { res.json(req.query); }
    app.get('/zod', validate(z.object({ q: z.string(), page: z.number().int() }), 'query'), handler);
    app.post('/joi', validateParts({ body: Joi.object({ name: Joi.string() }), query: Joi.object({ v: Joi.boolean() }), params: Joi.object() }), handler);
    app.post('/plain', plain({ limit: 1 }), handler);
    app.post('/module', moduleLevel, handler);
    const routes = extractExpressRoutes(app);
    expect(routes['/zod'].declared_inputs).toEqual({ inputs: { q: 'text', page: 'int' }, covers: ['middleware'], reasons: [], standalone: false });
    expect(routes['/joi'].declared_inputs.inputs).toEqual({ name: 'text', v: 'bool' });
    expect(routes['/plain'].declared_inputs).toBeUndefined();
    expect(routes['/module'].declared_inputs).toBeUndefined();
  });

  it('names nothing when the inspector is busy or missing', () => {
    const inspector = require('inspector');
    const session = jest.spyOn(inspector.Session.prototype, 'connect').mockImplementation(() => { throw new Error('busy'); });
    try {
      expect(framework.closureValues(() => null)).toEqual([]);
    } finally {
      session.mockRestore();
    }
    jest.isolateModules(() => {
      jest.doMock('inspector', () => { throw new Error('no inspector'); });
      expect(require('../lib/frameworkInputs').closureValues(() => null)).toEqual([]);
    });
    jest.dontMock('inspector');
  });
});

describe('routes on mounted routers', () => {
  const mountedApp = express => {
    const app = express();
    const account = express.Router();
    account.post('/settings', (req, res) => res.json(req.body.theme));
    const users = express.Router({ mergeParams: true });
    const nested = express.Router();
    nested.get('/deep', (_req, res) => res.end());
    users.get('/x', (_req, res) => res.end());
    users.use('/nested', nested);
    app.use('/account', account);
    app.use('/users/:id', users);
    app.use(account);
    app.use(/^\/re(gex)?/, express.Router().get('/z', (_req, res) => res.end()));
    app.get('/top', (_req, res) => res.end());
    return app;
  };

  it('finds them under their mount paths in Express 4 and 5', () => {
    const expected = ['/account/settings', '/settings', '/top', '/users/:id/nested/deep', '/users/:id/x'];
    expect(Object.keys(extractExpressRoutes(mountedApp(require('express')))).sort()).toEqual(expected);
    expect(Object.keys(extractExpressRoutes(mountedApp(require('@nestjs/platform-express/node_modules/express')))).sort()).toEqual(expected);
  });

  it('reads Express 4 patterns with unnamed and unloosened groups', () => {
    const { mountPath } = require('../lib/pathManifest');
    expect(mountPath({ regexp: /^\/a\/(?:([^\/]+?))\/?(?=\/|$)/i, keys: [] })).toBe('/a/:0');
    expect(mountPath({ regexp: /^\/a(?=\/|$)/i, keys: [{ name: 'x' }] })).toBe('/a');
    expect(mountPath({ regexp: { fast_slash: true } })).toBe('');
    expect(mountPath({})).toBeNull();
    expect(mountPath({ matchers: [() => null] })).toBeNull();
  });
});
