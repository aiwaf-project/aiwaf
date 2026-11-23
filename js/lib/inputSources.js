'use strict';

// Where a route's inputs come from besides reading its code, and how they're combined. init reads
// each handler's code for what it reads (see inputSchema.js); this completes what that can't see,
// then decides each route's inputs_complete:
//
// 0. What the framework knows at runtime: Fastify and Hapi route schemas, Nest DTOs (see
//    frameworkInputs.js), saved on the route as declared_inputs while the manifest is built.
// 1. Declarations: aiwaf.inputs({ q: 'text', page: 'int' })(handler), or .aiwaf/inputs.json.
// 2. An OpenAPI spec (AIWAF_OPENAPI_SPEC).
// 3. Inputs recorded while tests or staging traffic ran (see inputRecorder.js). These only add
//    names, and a name the code read that the analysis didn't find makes the route incomplete.
//
// A declaration or a closed OpenAPI operation is authoritative: the route is complete with what it
// lists plus anything the code was seen reading. The same rules as Python's input_sources.

const fs = require('fs');
const path = require('path');
const { merge } = require('./inputSchema');

const KINDS = new Set(['int', 'float', 'bool', 'uuid', 'text']);
const MARK = Symbol.for('aiwaf.inputs');
const WHOLE = '*';
const MAX_REASONS = 20;

function kindOf(value) {
  if (value === Number) return 'float';
  if (value === Boolean) return 'bool';
  if (value === String) return 'text';
  const name = String(value || '').trim().toLowerCase();
  if (KINDS.has(name)) return name;
  if (name === 'integer' || name === 'bigint') return 'int';
  if (name === 'number') return 'float';
  if (name === 'boolean') return 'bool';
  return 'text';
}

/**
 * Declare a handler's query and body inputs, for when reading its code can't find them all:
 * app.get('/search', aiwaf.inputs({ q: 'text', page: 'int' })(handler)). Types are int, float,
 * bool, uuid or text (or Number, Boolean, String). complete: false only adds the names. Or pass
 * the schema the handler validates with (zod, Joi, or a DTO class): aiwaf.inputs(Search)(handler);
 * one that allows names it doesn't list only adds them.
 */
function inputs(fields = {}, options = {}) {
  const framework = require('./frameworkInputs');
  let declared;
  if (framework.isSchema(fields)) {
    // A zod or Joi schema, or a DTO class: its names, complete unless it allows others.
    const reasons = [];
    const found = framework.fieldsOf(fields, 'declared', reasons) || {};
    declared = { inputs: found, complete: options.complete === undefined ? reasons.length === 0 : options.complete !== false };
  } else {
    declared = {
      inputs: Object.fromEntries(Object.entries(fields).map(([name, kind]) => [name, kindOf(kind)])),
      complete: options.complete !== false
    };
  }
  return handler => {
    if (typeof handler === 'function') handler[MARK] = declared;
    return handler;
  };
}

/** The aiwaf.inputs declaration on any of a route's handlers, or null. */
function declaredOn(...handlers) {
  for (const handler of handlers.flat()) {
    const declared = handler && handler[MARK];
    if (declared) return { inputs: { ...declared.inputs }, complete: declared.complete };
  }
  return null;
}

const PARAMETER = /^(?:<[^>]+>|\{[^}]+\}|:[A-Za-z_]\w*\??|\*\w*|\[[^\]]+\])$/;

/** A route template with every parameter written {}, so /a/:id, /a/{id} and /a/<int:id> match. */
function canonical(route) {
  const parts = String(route || '').split('?')[0].trim().split('/').filter(Boolean);
  return `/${parts.map(part => (PARAMETER.test(part) ? '{}' : part)).join('/')}`;
}

function openapiKind(schema) {
  let kind = schema && schema.type;
  if (Array.isArray(kind)) {
    const kinds = kind.filter(item => item !== 'null');
    kind = kinds.length === 1 ? kinds[0] : null;
  }
  if (kind === 'integer') return 'int';
  if (kind === 'number') return 'float';
  if (kind === 'boolean') return 'bool';
  if (kind === 'string' && schema.format === 'uuid') return 'uuid';
  if (kind === 'array' && schema.items && typeof schema.items === 'object') return openapiKind(schema.items);
  return 'text';
}

const reason = (code, detail) => ({ code, detail });

/** Each operation's query and body inputs in an OpenAPI 3 spec, by canonical path and method. */
function openapiInputs(spec) {
  const resolve = (node, depth = 0) => {
    while (node && typeof node === 'object' && node.$ref !== undefined && depth < 16) {
      const ref = String(node.$ref);
      if (!ref.startsWith('#/')) return null;
      let target = spec;
      for (const part of ref.slice(2).split('/')) {
        target = target && typeof target === 'object' ? target[part.replace(/~1/g, '/').replace(/~0/g, '~')] : undefined;
      }
      node = target;
      depth += 1;
    }
    return node && typeof node === 'object' ? node : null;
  };
  const properties = (input, reasons, depth = 0) => {
    const schema = resolve(input);
    if (!schema || depth > 8) {
      reasons.push(reason('external_class', "the body schema can't be resolved"));
      return {};
    }
    const found = {};
    const combined = ['allOf', 'oneOf', 'anyOf'].filter(key => Array.isArray(schema[key]));
    for (const key of combined) for (const part of schema[key]) Object.assign(found, properties(part, reasons, depth + 1));
    if (combined.length && !schema.properties) return found;
    if ((schema.type || 'object') !== 'object') {
      reasons.push(reason('container_whole', "the body isn't an object with named properties"));
      return found;
    }
    if (![false, null, undefined].includes(schema.additionalProperties)) {
      reasons.push(reason('container_whole', "the body schema allows properties it doesn't name"));
    }
    for (const [name, value] of Object.entries(schema.properties || {})) found[name] = openapiKind(resolve(value) || {});
    return found;
  };
  const result = {};
  for (const [route, rawItem] of Object.entries(spec.paths || {})) {
    const item = resolve(rawItem) || {};
    const common = item.parameters || [];
    for (const [method, operation] of Object.entries(item)) {
      if (!['GET', 'POST', 'PUT', 'PATCH', 'DELETE'].includes(method.toUpperCase()) || !operation || typeof operation !== 'object') continue;
      const found = {};
      const reasons = [];
      for (const rawParameter of [...common, ...(operation.parameters || [])]) {
        const parameter = resolve(rawParameter);
        if (!parameter) {
          reasons.push(reason('external_class', "a parameter reference can't be resolved"));
          continue;
        }
        if (parameter.in !== 'query') continue;
        const schema = resolve(parameter.schema) || {};
        if (schema.type === 'object' && parameter.explode !== false && !schema.properties) {
          reasons.push(reason('container_whole', `query parameter ${parameter.name} is a free-form object`));
          continue;
        }
        found[String(parameter.name)] = openapiKind(schema);
      }
      const body = resolve(operation.requestBody);
      for (const [media, content] of Object.entries((body && body.content) || {})) {
        const type = media.split(';')[0].trim();
        if (['application/json', 'application/x-www-form-urlencoded', 'multipart/form-data'].includes(type) || type.endsWith('+json')) {
          for (const [name, kind] of Object.entries(properties((content || {}).schema, reasons))) merge(found, name, kind);
        } else {
          reasons.push(reason('raw_body', `the body is ${media}`));
        }
      }
      const key = canonical(route);
      result[key] = result[key] || {};
      result[key][method.toUpperCase()] = { inputs: found, complete: reasons.length === 0, reasons };
    }
  }
  return result;
}

/** An OpenAPI spec from a JSON file, or YAML when the yaml package is installed. */
function loadSpec(file) {
  const text = fs.readFileSync(file, 'utf8');
  let loaded;
  if (/\.ya?ml$/i.test(file)) {
    let yaml;
    try {
      yaml = require('yaml');
    } catch (_error) {
      throw new Error(`${file} is YAML; install the yaml package, or give a JSON spec`);
    }
    loaded = yaml.parse(text);
  } else {
    loaded = JSON.parse(text);
  }
  if (!loaded || typeof loaded !== 'object' || !loaded.paths) throw new Error(`${file} isn't an OpenAPI spec: it has no paths`);
  return loaded;
}

/** Recorded reads (see inputRecorder.js), by canonical route to { name: type }; * is a whole read. */
function loadObservations(file) {
  const found = {};
  for (const line of fs.readFileSync(file, 'utf8').split(/\r?\n/)) {
    let item;
    try {
      item = JSON.parse(line);
    } catch (_error) {
      continue;
    }
    if (!item || typeof item !== 'object' || Array.isArray(item) || !item.route || !item.key) continue;
    const route = canonical(item.route);
    found[route] = found[route] || {};
    const key = String(item.key);
    if (key === WHOLE) found[route][WHOLE] = WHOLE;
    else merge(found[route], key, kindOf(item.type || 'text'));
  }
  return found;
}

/** .aiwaf/inputs.json: { "/route": { "inputs": {...}, "complete": true } }, optionally under "routes". */
function loadDeclarations(file) {
  const loaded = JSON.parse(fs.readFileSync(file, 'utf8'));
  const routes = loaded && typeof loaded === 'object' ? (loaded.routes || loaded) : {};
  const declared = {};
  for (const [route, spec] of Object.entries(routes)) {
    if (!spec || typeof spec !== 'object' || Array.isArray(spec)) continue;
    declared[canonical(route)] = {
      inputs: Object.fromEntries(Object.entries(spec.inputs || {}).map(([name, kind]) => [name, kindOf(kind)])),
      complete: spec.complete !== false
    };
  }
  return declared;
}

/** Combine every source into each route's inputs, inputs_complete and inputs_incomplete. */
function finalize(routes, sources = {}) {
  const counts = { complete: 0, incomplete: 0, declared: 0, openapi: 0, observed: 0, reasons: {} };
  for (const [route, data] of Object.entries(routes)) {
    if (!data || typeof data !== 'object' || Array.isArray(data)) continue;
    const framework = data.declared_inputs;
    const declaration = data.input_declaration;
    delete data.declared_inputs;
    delete data.input_declaration;
    let read = 'inputs_complete' in data;
    const found = { ...(data.inputs || {}) };
    let reasons = [...(data.inputs_incomplete || [])];
    if (framework && typeof framework === 'object') {
      const covered = new Set(framework.covers || []);
      reasons = reasons.filter(item => !covered.has(item.code)).concat(framework.reasons || []);
      for (const [name, kind] of Object.entries(framework.inputs || {})) merge(found, name, kind);
      read = read || Boolean(framework.standalone);
    }
    const key = canonical(route);
    let authoritative = false;
    for (const declared of [declaration, (sources.declarations || {})[key]]) {
      if (!declared || typeof declared !== 'object') continue;
      for (const [name, kind] of Object.entries(declared.inputs || {})) found[name] = kind;
      authoritative = declared.complete !== false || authoritative;
      read = true;
      counts.declared += 1;
    }
    const operations = (sources.openapi || {})[key];
    if (operations) {
      const methods = data.methods && data.methods.length ? data.methods : Object.keys(operations);
      const listed = methods.filter(method => operations[method]).map(method => operations[method]);
      for (const operation of listed) for (const [name, kind] of Object.entries(operation.inputs)) merge(found, name, kind);
      if (listed.length && listed.length === methods.length && listed.every(operation => operation.complete)) authoritative = true;
      read = true;
      counts.openapi += 1;
    }
    if (authoritative) reasons = [];
    const seen = (sources.observations || {})[key] || {};
    if (Object.keys(seen).length) counts.observed += 1;
    for (const [name, kind] of Object.entries(seen)) {
      if (name === WHOLE) {
        reasons.push(reason('container_whole', 'the code was seen reading a query or body container whole'));
        continue;
      }
      if (!(name in found) && read) reasons.push(reason('observed', `${name} was read while recording, but the analysis didn't find it`));
      merge(found, name, kind);
      read = true;
    }
    if (!read) continue;
    data.inputs = found;
    data.inputs_complete = reasons.length === 0;
    if (reasons.length) {
      data.inputs_incomplete = reasons.slice(0, MAX_REASONS);
      counts.incomplete += 1;
      for (const item of reasons) counts.reasons[item.code] = (counts.reasons[item.code] || 0) + 1;
    } else {
      delete data.inputs_incomplete;
      counts.complete += 1;
    }
  }
  return counts;
}

/**
 * The declaration file, OpenAPI spec and recorded inputs init should read, where present:
 * AIWAF_INPUTS_FILE (default .aiwaf/inputs.json), AIWAF_OPENAPI_SPEC (no default) and
 * AIWAF_OBSERVED_INPUTS (default .aiwaf/observed_inputs.jsonl, which the recorder writes).
 */
function sourcesFromEnvironment(root = '.', env = process.env) {
  const sources = {};
  const declarations = env.AIWAF_INPUTS_FILE || path.join(root, '.aiwaf', 'inputs.json');
  if (fs.existsSync(declarations)) sources.declarations = loadDeclarations(declarations);
  if (env.AIWAF_OPENAPI_SPEC) sources.openapi = openapiInputs(loadSpec(env.AIWAF_OPENAPI_SPEC));
  const observed = env.AIWAF_OBSERVED_INPUTS || path.join(root, '.aiwaf', 'observed_inputs.jsonl');
  if (fs.existsSync(observed)) sources.observations = loadObservations(observed);
  return sources;
}

module.exports = {
  WHOLE,
  canonical,
  declaredOn,
  finalize,
  inputs,
  kindOf,
  loadDeclarations,
  loadObservations,
  loadSpec,
  openapiInputs,
  schemaKind: openapiKind,
  sourcesFromEnvironment
};
