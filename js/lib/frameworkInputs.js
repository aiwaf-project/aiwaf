'use strict';

// What a route accepts, from what the framework itself knows at runtime, when the manifest is
// generated from the live app (generateFrameworkManifest(framework, app)). Reading the handler's
// code (inputSchema.js) can't see a schema or DTO that's only attached through the framework; this
// reads it:
//
// - Fastify: each route's schema.querystring and schema.body (JSON Schema, which TypeBox builds).
// - Hapi: each route's validate.query and validate.payload (Joi, or @hapi/validate).
// - NestJS: each handler's @Query() and @Body() parameters, and the DTO classes they're typed with,
//   from class-validator or @nestjs/swagger property metadata, at the paths Nest itself builds
//   (global prefix and its exclusions, RouterModule paths, URI versioning).
// - Express: validation middleware on the route: celebrate (Joi) and express-validator chains.
//
// Each result is { inputs, covers, reasons, standalone }, the shape Python's framework sources
// give: covers are the code reading's reason codes the framework answers, and standalone means the
// list needs no code reading. inputSources.finalize combines it with everything else.

const { kindOf, schemaKind } = require('./inputSources');

const reason = (code, detail) => ({ code, detail });
const COVERS = ['dependency', 'external_class', 'model_fields', 'container_whole'];

/** A JSON Schema object's named properties; one that allows others is a container_whole reason. */
/**
 * A JSON Schema with its $refs replaced by what they name, from registry ({ id: schema }, as
 * Fastify's getSchemas() gives) or the schema's own definitions: "id#", "id", "id#/properties/x"
 * and "#/definitions/x". A $ref that can't be resolved is left, and reads as unknown.
 */
function resolveRefs(schema, registry = {}, root = schema, depth = 0) {
  if (!schema || typeof schema !== 'object' || depth > 16) return schema;
  if (Array.isArray(schema)) return schema.map(item => resolveRefs(item, registry, root, depth + 1));
  if (typeof schema.$ref === 'string') {
    const [id, pointer = ''] = schema.$ref.split('#');
    const base = id ? (registry[id] || registry[`${id}#`] || Object.values(registry).find(item => item && item.$id === id)) : root;
    let target = base;
    for (const part of pointer.split('/').filter(Boolean)) target = target && typeof target === 'object' ? target[part.replace(/~1/g, '/').replace(/~0/g, '~')] : undefined;
    if (target && typeof target === 'object' && target !== schema) return resolveRefs(target, registry, id ? base : root, depth + 1);
    return schema;
  }
  const out = {};
  for (const [key, value] of Object.entries(schema)) out[key] = resolveRefs(value, registry, root, depth + 1);
  return out;
}

const closed = schema => [false, null, undefined].includes(schema.additionalProperties) && !schema.patternProperties;

function jsonSchemaFields(schema, label, reasons, nested = null) {
  if (schema.$ref !== undefined) {
    reasons.push(reason('external_class', `the ${label} schema is a reference (${schema.$ref})`));
    return {};
  }
  const found = {};
  const combined = ['allOf', 'oneOf', 'anyOf'].filter(key => Array.isArray(schema[key]));
  for (const key of combined) {
    for (const part of schema[key]) Object.assign(found, fieldsOf(part, label, reasons) || {});
  }
  if (combined.length && !schema.properties) return found;
  if ((schema.type || 'object') !== 'object') {
    reasons.push(reason('container_whole', `the ${label} isn't an object with named properties`));
    return found;
  }
  if (![false, null, undefined].includes(schema.additionalProperties) || schema.patternProperties) {
    reasons.push(reason('container_whole', `the ${label} schema allows properties it doesn't name`));
  }
  for (const [name, value] of Object.entries(schema.properties || {})) {
    found[name] = value && value.$ref === undefined ? schemaKind(value) : 'text';
    // An object property listing its own properties, which allows no others: the keys inside it.
    if (nested && value && typeof value === 'object' && (value.type === 'object' || (!value.type && value.properties))
      && value.properties && Object.keys(value.properties).length && closed(value)) {
      nested[name] = Object.keys(value.properties).sort();
    }
  }
  return found;
}

const isJoi = value => Boolean(value && typeof value === 'object' && value.$_terms && typeof value.type === 'string');

function joiKind(schema) {
  const rules = new Set((schema._rules || []).map(rule => rule.name));
  if (schema.type === 'number') return rules.has('integer') ? 'int' : 'float';
  if (schema.type === 'boolean') return 'bool';
  if (schema.type === 'string' && (rules.has('guid') || rules.has('uuid'))) return 'uuid';
  const items = schema.type === 'array' ? (schema.$_terms.items || []) : [];
  return items.length === 1 && isJoi(items[0]) ? joiKind(items[0]) : 'text';
}

/** A Joi (or @hapi/validate) object schema's keys; unknown(), pattern() or no keys allow others. */
function joiFields(schema, label, reasons) {
  if (schema.type !== 'object') {
    reasons.push(reason('container_whole', `the ${label} isn't an object with named keys`));
    return {};
  }
  const keys = schema.$_terms.keys;
  if (!keys) {
    reasons.push(reason('container_whole', `the ${label} schema allows any keys`));
    return {};
  }
  if ((schema._flags && schema._flags.unknown) || (schema.$_terms.patterns || []).length) {
    reasons.push(reason('container_whole', `the ${label} schema allows keys it doesn't name`));
  }
  return Object.fromEntries(keys.map(item => [item.key, joiKind(item.schema)]));
}

// zod 3 keeps its definition on schema._def (with a typeName), zod 4 on schema._zod.def (with a type).
const zod4 = value => (value && typeof value === 'object' && value._zod && value._zod.def && typeof value._zod.def.type === 'string' ? value._zod.def : null);
const zod3 = value => (value && typeof value === 'object' && !value._zod && value._def && typeof value._def.typeName === 'string' ? value._def : null);
const isZod = value => Boolean(zod4(value) || zod3(value));
const WRAPPERS_4 = new Set(['optional', 'nullable', 'default', 'prefault', 'nonoptional', 'readonly', 'catch']);
const WRAPPERS_3 = new Set(['ZodOptional', 'ZodNullable', 'ZodDefault', 'ZodReadonly', 'ZodCatch']);

/** The schema under optional, nullable, default, refine/transform and pipe wrappers. */
function zodUnwrap(schema) {
  for (let depth = 0; depth < 16; depth += 1) {
    const four = zod4(schema);
    const three = zod3(schema);
    let inner = null;
    if (four) inner = WRAPPERS_4.has(four.type) ? four.innerType : (four.type === 'pipe' ? four.in : null);
    else if (three) {
      if (WRAPPERS_3.has(three.typeName)) inner = three.innerType;
      else if (three.typeName === 'ZodEffects') inner = three.schema;
      else if (three.typeName === 'ZodPipeline') inner = three.in;
      else if (three.typeName === 'ZodBranded') inner = three.type;
    }
    if (!inner) return schema;
    schema = inner;
  }
  return schema;
}

/** z.number().int() and z.int() are int, z.string().uuid() and z.uuid() uuid. */
function zodKind(input) {
  const schema = zodUnwrap(input);
  const four = zod4(schema);
  if (four) {
    const formats = [four.format, ...(four.checks || []).map(check => check && check._zod && check._zod.def && check._zod.def.format)]
      .filter(format => typeof format === 'string');
    if (four.type === 'number' || four.type === 'bigint') return four.type === 'bigint' || formats.some(format => format.includes('int')) ? 'int' : 'float';
    if (four.type === 'boolean') return 'bool';
    if (four.type === 'string') return formats.some(format => format === 'uuid' || format === 'guid') ? 'uuid' : 'text';
    return four.type === 'array' && four.element ? zodKind(four.element) : 'text';
  }
  const three = zod3(schema) || {};
  const kinds = (three.checks || []).map(check => check && check.kind);
  if (three.typeName === 'ZodNumber') return kinds.includes('int') ? 'int' : 'float';
  if (three.typeName === 'ZodBigInt') return 'int';
  if (three.typeName === 'ZodBoolean') return 'bool';
  if (three.typeName === 'ZodString') return kinds.includes('uuid') ? 'uuid' : 'text';
  return three.typeName === 'ZodArray' && three.type ? zodKind(three.type) : 'text';
}

/** A zod object schema's keys; passthrough/loose or a catchall allow others. */
function zodFields(input, label, reasons) {
  const schema = zodUnwrap(input);
  const four = zod4(schema);
  const three = zod3(schema);
  let shape = null;
  let open = false;
  if (four && four.type === 'object') {
    shape = four.shape || {};
    const catchall = zod4(four.catchall);
    open = Boolean(catchall && catchall.type !== 'never');
  } else if (three && three.typeName === 'ZodObject') {
    shape = typeof three.shape === 'function' ? three.shape() : (three.shape || {});
    const catchall = zod3(three.catchall);
    open = three.unknownKeys === 'passthrough' || Boolean(catchall && catchall.typeName !== 'ZodNever');
  }
  if (!shape) {
    reasons.push(reason('container_whole', `the ${label} isn't a zod object with named keys`));
    return {};
  }
  if (open) reasons.push(reason('container_whole', `the ${label} schema allows keys it doesn't name`));
  return Object.fromEntries(Object.entries(shape).map(([name, value]) => [name, zodKind(value)]));
}

const BUILT_IN = new Set([Object, String, Number, Boolean, Array, Promise, Map, Set, Date, Buffer]);
const isDto = value => typeof value === 'function' && !BUILT_IN.has(value) && Boolean(value.prototype);
const metadata = (...args) => (typeof Reflect.getMetadata === 'function' ? Reflect.getMetadata(...args) : undefined);

const VALIDATOR_KINDS = {
  isInt: 'int', isNumber: 'float', isNumberString: 'float', isBoolean: 'bool', isBooleanString: 'bool', isUUID: 'uuid'
};

/**
 * A DTO class's properties, from class-validator's decorators (its global metadata storage, only
 * when the app loaded class-validator) or @nestjs/swagger's @ApiProperty, through inheritance.
 */
function classFields(cls, label, reasons) {
  // nestjs-zod's createZodDto(schema): a class marked isZodDto, with the schema as a static.
  if (cls.isZodDto && isZod(cls.schema)) return zodFields(cls.schema, label, reasons);
  const found = {};
  const proto = cls.prototype;
  const typed = name => {
    const swagger = metadata('swagger/apiModelProperties', proto, name);
    const type = (swagger && swagger.type) || metadata('design:type', proto, name);
    return kindOf(type);
  };
  for (const entry of metadata('swagger/apiModelPropertiesArray', proto) || []) {
    const name = String(entry).replace(/^:/, '');
    found[name] = typed(name);
  }
  const storage = globalThis.classValidatorMetadataStorage;
  if (storage && typeof storage.getTargetValidationMetadatas === 'function') {
    for (const item of storage.getTargetValidationMetadatas(cls, '', true, false) || []) {
      const name = item && item.propertyName;
      if (!name) continue;
      const kind = VALIDATOR_KINDS[item.name] || VALIDATOR_KINDS[item.constraints && item.constraints[0]];
      if (kind) found[name] = kind;
      else if (!(name in found)) found[name] = typed(name);
    }
  }
  if (!Object.keys(found).length) {
    reasons.push(reason('external_class', `${cls.name || 'the'} ${label} class has no class-validator or Swagger property metadata`));
  }
  return found;
}

/**
 * The named inputs a schema accepts: JSON Schema, zod (3 or 4), Joi, a DTO class, or a plain
 * object of Joi schemas (Hapi's shorthand). Null when it isn't one of those.
 */
function fieldsOf(schema, label, reasons, nested = null) {
  if (isDto(schema)) return classFields(schema, label, reasons);
  if (!schema || typeof schema !== 'object' || Array.isArray(schema)) return null;
  if (isZod(schema)) return zodFields(schema, label, reasons);
  if (isJoi(schema)) return joiFields(schema, label, reasons);
  if (schema.type !== undefined || schema.properties || schema.$ref !== undefined
    || ['allOf', 'oneOf', 'anyOf'].some(key => Array.isArray(schema[key]))) {
    return jsonSchemaFields(schema, label, reasons, nested);
  }
  const values = Object.values(schema);
  if (values.length && values.every(isJoi)) {
    return Object.fromEntries(Object.entries(schema).map(([name, value]) => [name, joiKind(value)]));
  }
  if (values.length && values.every(isZod)) {
    return Object.fromEntries(Object.entries(schema).map(([name, value]) => [name, zodKind(value)]));
  }
  return null;
}

/** Combine schemas for a route's query and body into its framework declaration, or null. */
function fromSchemas(parts) {
  const found = {};
  const reasons = [];
  const nested = {};
  let any = false;
  for (const [label, schema] of parts) {
    if (schema === false) {
      any = true; // Hapi: this input isn't allowed at all.
      continue;
    }
    const fields = fieldsOf(schema, label, reasons, nested);
    if (!fields) continue;
    any = true;
    for (const [name, kind] of Object.entries(fields)) found[name] = kind;
  }
  if (!any) return null;
  // The schema only checks what's sent; the handler may still read the raw request itself.
  const result = { inputs: found, covers: COVERS, reasons, standalone: false };
  if (Object.keys(nested).length) result.nested = nested;
  return result;
}

/** A Fastify route's schema.querystring and schema.body. */
function fastifyInputs(schema, registry = {}) {
  if (!schema || typeof schema !== 'object') return null;
  // Each part is its own JSON Schema: a local $ref (#/definitions/x) is relative to it.
  const part = value => (value && typeof value === 'object' ? resolveRefs(value, registry, value) : value);
  return fromSchemas([['query', part(schema.querystring || schema.query)], ['body', part(schema.body)]]);
}

/** A Hapi route's validate.query and validate.payload. */
function hapiInputs(validate) {
  if (!validate || typeof validate !== 'object') return null;
  return fromSchemas([['query', validate.query], ['body', validate.payload]]);
}

// Nest's RouteParamtypes and RequestMethod enums.
const REQUEST = 0;
const BODY = 3;
const QUERY = 4;
const METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', null, 'OPTIONS', 'HEAD', 'SEARCH'];
const EVERY_METHOD = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'];
const PIPE_KINDS = { ParseIntPipe: 'int', ParseFloatPipe: 'float', ParseBoolPipe: 'bool', ParseUUIDPipe: 'uuid' };

function pipeKind(pipes) {
  for (const pipe of pipes || []) {
    const name = typeof pipe === 'function' ? pipe.name : pipe && pipe.constructor && pipe.constructor.name;
    if (PIPE_KINDS[name]) return PIPE_KINDS[name];
  }
  return null;
}

/** A Nest handler's @Query() and @Body() parameters, and the DTO classes they're typed with. */
function nestInputs(controller, method) {
  const args = metadata('__routeArguments__', controller, method) || {};
  const types = metadata('design:paramtypes', controller.prototype, method) || [];
  const found = {};
  const reasons = [];
  let standalone = true;
  for (const [key, arg] of Object.entries(args)) {
    const index = arg && Number.isInteger(arg.index) ? arg.index : -1;
    if (key.includes('__customRouteArgs__')) {
      standalone = false;
      reasons.push(reason('dependency', `parameter ${index} of ${method} uses a custom decorator that reads the request`));
      continue;
    }
    const paramtype = Number(key.split(':')[0]);
    if (paramtype === REQUEST) {
      standalone = false; // @Req(): the handler's code says what it reads.
      continue;
    }
    if (paramtype !== BODY && paramtype !== QUERY) continue;
    const label = paramtype === BODY ? 'body' : 'query';
    const type = types[index];
    const data = arg && arg.data;
    if (typeof data === 'string' && data) {
      found[data] = pipeKind(arg.pipes) || kindOf(type);
    } else if (isDto(type)) {
      Object.assign(found, classFields(type, label, reasons));
    } else {
      reasons.push(reason('container_whole', `@${paramtype === BODY ? 'Body' : 'Query'}() in ${method} takes the whole ${label} without a DTO class`));
    }
  }
  // With @Req() the code reading still decides about whole reads of the request.
  const covers = standalone ? COVERS : COVERS.filter(code => code !== 'container_whole');
  return { inputs: found, covers, reasons, standalone };
}

const MODULE_PATH = '__module_path__';
const VERSION_METADATA = '__version__';

/** Nest's own RoutePathFactory, from the app's @nestjs/core install, or null. */
function nestPathFactory(app) {
  for (const options of [{ paths: [process.cwd()] }, undefined]) {
    try {
      const { RoutePathFactory } = require(require.resolve('@nestjs/core/router/route-path-factory', options));
      return new RoutePathFactory(app.config);
    } catch (_error) {
      // Not installed where the app runs from; try aiwaf's own resolution next.
    }
  }
  return null;
}

const joinPath = (...parts) => `/${parts.map(part => String(part || '').replace(/^\/+|\/+$/g, '')).filter(Boolean).join('/')}`;

function handlerNames(proto) {
  const names = new Set();
  for (let current = proto; current && current !== Object.prototype; current = Object.getPrototypeOf(current)) {
    for (const name of Object.getOwnPropertyNames(current)) {
      const descriptor = Object.getOwnPropertyDescriptor(current, name);
      if (name !== 'constructor' && descriptor && typeof descriptor.value === 'function') names.add(name);
    }
  }
  return [...names];
}

/**
 * Every route of a Nest application, from its controllers' metadata: { routePath, methods,
 * handler, declared }, at the paths Nest's own RoutePathFactory gives (global prefix and its
 * exclusions, RouterModule paths, URI versioning). Empty when app isn't a Nest application (or
 * reflect-metadata isn't loaded).
 */
// Nest's RouteParamtypes.RESPONSE and NEXT: a handler taking @Res() or @Next() answers itself.
const RESPONSE = 1;
const NEXT = 2;

/**
 * How Nest answers for the handler: 'html' when it renders a template (@Render) or redirects
 * (@Redirect), null for a server-sent event stream (@Sse) or a handler taking @Res() or @Next()
 * without passthrough, which shapes its own response, and otherwise 'json': Nest sends the
 * return value as JSON.
 */
function nestResponseType(controller, name, handler) {
  if (metadata('__renderTemplate__', handler) !== undefined || metadata('__redirect__', handler) !== undefined) return 'html';
  if (metadata('__sse__', handler)) return null;
  const args = metadata('__routeArguments__', controller, name) || {};
  for (const [key, value] of Object.entries(args)) {
    const type = Number(String(key).split(':')[0]);
    if ((type === RESPONSE || type === NEXT) && !(value && value.data && value.data.passthrough)) return null;
  }
  return 'json';
}

function nestRoutes(app, factory = undefined) {
  const container = app && app.container;
  if (!container || typeof container.getModules !== 'function' || typeof Reflect.getMetadata !== 'function') return [];
  const config = app.config || {};
  const globalPrefix = typeof config.getGlobalPrefix === 'function' ? config.getGlobalPrefix() : '';
  const versioningOptions = typeof config.getVersioning === 'function' ? config.getVersioning() : undefined;
  const paths = factory === undefined ? nestPathFactory(app) : factory;
  const modules = container.getModules();
  const routes = [];
  const seen = new Set();
  for (const module of modules.values()) {
    const owner = module && module.metatype;
    const modulePath = owner ? (metadata(MODULE_PATH + modules.applicationId, owner) ?? metadata(MODULE_PATH, owner)) : undefined;
    for (const wrapper of ((module && module.controllers) || new Map()).values()) {
      const controller = wrapper && wrapper.metatype;
      if (typeof controller !== 'function' || seen.has(controller)) continue;
      seen.add(controller);
      const controllerVersion = versioningOptions ? (metadata(VERSION_METADATA, controller) ?? versioningOptions.defaultVersion) : undefined;
      for (const name of handlerNames(controller.prototype)) {
        const handler = controller.prototype[name];
        const methodPaths = metadata('path', handler);
        if (methodPaths === undefined) continue;
        const code = metadata('method', handler) ?? 0;
        const verb = METHODS[code];
        const declared = nestInputs(controller, name);
        const methodVersion = metadata(VERSION_METADATA, handler);
        for (const ctrlPath of [].concat(metadata('path', controller) ?? '/')) {
          for (const methodPath of [].concat(methodPaths)) {
            const full = paths
              ? paths.create({ ctrlPath, modulePath, globalPrefix, controllerVersion, versioningOptions, methodVersion, methodPath }, code)
              : [joinPath(globalPrefix, modulePath, ctrlPath, methodPath)];
            for (const routePath of full) {
              routes.push({
                routePath, methods: verb ? [verb] : EVERY_METHOD, handler, name: `${controller.name}.${name}`, declared,
                responseType: nestResponseType(controller, name, handler)
              });
            }
          }
        }
      }
    }
  }
  return routes;
}

/** A field express-validator checks, as the top-level input name: user.name and items[0] are user, items. */
function fieldName(field) {
  const name = String(field || '').split(/[.[]/)[0];
  return name && name !== '*' ? name : null;
}

const SEGMENTS = new Set(['query', 'body']);
const isReadableSchema = value => isZod(value) || isJoi(value) || (isDto(value) && value.isZodDto === true);

/**
 * The variables a function closes over in the scopes of the factories that made it, as
 * [name, value] pairs, read through the inspector, synchronously. The module's own scope (an
 * anonymous closure) is left out: its variables can belong to any function in the file. Empty
 * when the inspector isn't available.
 */
function closureValues(fn) {
  let inspector;
  try {
    inspector = require('inspector');
  } catch (_error) {
    return [];
  }
  const key = Symbol.for(`aiwaf.closure.${process.pid}.${Date.now()}`);
  const found = [];
  const session = new inspector.Session();
  try {
    session.connect();
    globalThis[key] = { target: fn, found };
    const call = (method, params) => {
      let result = null;
      session.post(method, params, (error, value) => { result = error ? null : value; });
      return result;
    };
    // The function can live in another context than the main one (a vm context, as Jest's), so
    // each is asked for it.
    const contexts = [];
    session.on('Runtime.executionContextCreated', event => contexts.push(event.params.context.id));
    call('Runtime.enable', {});
    const expression = `globalThis[Symbol.for(${JSON.stringify(key.description)})]?.target`;
    let target = null;
    for (const contextId of [undefined, ...contexts]) {
      const result = call('Runtime.evaluate', contextId === undefined ? { expression } : { expression, contextId });
      if (result && result.result && result.result.type === 'function') {
        target = result;
        break;
      }
    }
    const own = target && call('Runtime.getProperties', { objectId: target.result.objectId, ownProperties: true });
    const scopes = own && (own.internalProperties || []).find(item => item.name === '[[Scopes]]');
    const list = scopes && call('Runtime.getProperties', { objectId: scopes.value.objectId });
    // Named factories' scopes, as Closure (validate), and block scopes in them; a bare Closure is
    // the module's own.
    const closures = ((list && list.result) || []).filter(item => /^(?:Closure \(|Block)/.test(String(item.value && item.value.description)));
    for (const scope of closures) {
      const variables = call('Runtime.getProperties', { objectId: scope.value.objectId });
      for (const variable of (variables && variables.result) || []) {
        const value = variable.value || {};
        if (value.objectId) {
          call('Runtime.callFunctionOn', {
            objectId: value.objectId,
            functionDeclaration: `function () { globalThis[Symbol.for(${JSON.stringify(key.description)})].found.push([${JSON.stringify(variable.name)}, this]); }`
          });
        } else if (value.type === 'string') {
          found.push([variable.name, value.value]);
        }
      }
    }
  } catch (_error) {
    // The inspector can be busy or closed; a function it can't read just names nothing.
  } finally {
    delete globalThis[key];
    try { session.disconnect(); } catch (_error) { /* Already closed. */ }
  }
  return found;
}

/**
 * What a validation middleware that keeps its schema in its own closure names, as in
 * validate(schema, 'query') or validate({ body: schema }): a zod or Joi schema (or a nestjs-zod
 * DTO) it closed over, at the location a 'query' or 'body' string beside it names, or under query
 * and body keys of an object. Null when its closure holds none.
 */
function closureInputs(fn, reasons) {
  const values = closureValues(fn);
  const found = {};
  let any = false;
  for (const [, value] of values) {
    if (isReadableSchema(value)) {
      any = true;
      Object.assign(found, fieldsOf(value, 'validated', reasons) || {});
    } else if (value && typeof value === 'object' && !Array.isArray(value)
      && Object.keys(value).some(name => SEGMENTS.has(name) && isReadableSchema(value[name]))) {
      any = true;
      for (const segment of SEGMENTS) {
        if (isReadableSchema(value[segment])) Object.assign(found, fieldsOf(value[segment], segment, reasons) || {});
      }
    }
  }
  return any ? found : null;
}

/**
 * What an Express route's validation middleware names: celebrate's schema (middleware._schema,
 * by segment) and express-validator chains (middleware.builder: fields, locations and
 * validators). Null when none of the route's middleware is one. The validators check only the
 * names they list, so the handler's own code still says what it reads (standalone is false); when
 * every middleware before the handler is a validator, they answer the code reading's middleware
 * reasons.
 */
function expressInputs(handlers) {
  const stack = (handlers || []).filter(handler => typeof handler === 'function');
  const found = {};
  const reasons = [];
  let recognized = 0;
  for (const handler of stack.slice(0, -1)) {
    const celebrate = handler._schema;
    const builder = handler.builder;
    if (celebrate && typeof celebrate === 'object') {
      recognized += 1;
      for (const [label, segment] of [['query', 'query'], ['body', 'body']]) {
        if (celebrate[segment] === undefined) continue;
        Object.assign(found, fieldsOf(celebrate[segment], label, reasons) || {});
      }
    } else if (builder && Array.isArray(builder.fields) && Array.isArray(builder.locations)) {
      recognized += 1;
      if (!builder.locations.some(location => location === 'query' || location === 'body')) continue;
      const kinds = (builder.stack || []).map(item => VALIDATOR_KINDS[item && item.validator && item.validator.name]).filter(Boolean);
      for (const field of builder.fields) {
        const name = fieldName(field);
        if (name) found[name] = (field === name && kinds[0]) || found[name] || 'text';
      }
    } else {
      // A validation middleware of the app's own, with its schema in its closure.
      const hidden = closureInputs(handler, reasons);
      if (hidden) {
        recognized += 1;
        Object.assign(found, hidden);
      }
    }
  }
  if (!recognized) return null;
  const covers = recognized === stack.length - 1 ? ['middleware'] : [];
  return { inputs: found, covers, reasons, standalone: false };
}

/** Combine two framework declarations for one path (say, its GET and POST handlers). */
function combine(first, second) {
  if (!first) return second;
  if (!second) return first;
  // A name both declare with different types takes the weaker, as the engine keeps one type for
  // every method: POST's integer can't hold for a GET that sends any text.
  const inputs = { ...first.inputs };
  for (const [name, kind] of Object.entries(second.inputs)) inputs[name] = inputs[name] === undefined || inputs[name] === kind ? kind : 'text';
  const combined = {
    inputs,
    covers: first.covers.filter(code => second.covers.includes(code)),
    reasons: [...first.reasons, ...second.reasons],
    standalone: first.standalone && second.standalone
  };
  // A field's inner keys hold only when neither takes the field whole.
  const nested = {};
  for (const [own, other] of [[first, second], [second, first]]) {
    for (const [name, keys] of Object.entries(own.nested || {})) {
      if (name in other.inputs && !(other.nested && name in other.nested)) continue;
      nested[name] = [...new Set([...(nested[name] || []), ...keys])].sort();
    }
  }
  if (Object.keys(nested).length) combined.nested = nested;
  return combined;
}

/** Whether value is a schema object or class fieldsOf reads, rather than a plain { name: type } map. */
const isSchema = value => isZod(value) || isJoi(value) || isDto(value);

module.exports = { isSchema, classFields, resolveRefs, closureInputs, closureValues, combine, expressInputs, fastifyInputs, fieldsOf, hapiInputs, nestInputs, nestRoutes };
