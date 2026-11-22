'use strict';

// What each handler reads from the query and body, with types, for the route manifest. init
// saves these as each route's inputs (name to int, float, bool, uuid or text) and
// inputs_complete: true only when every read was understood. Requests are then reviewed against
// them in aiwaf-wasm: see AIWAF_ROUTE_SCHEMA_MODE.
//
// Anything not understood makes the list incomplete rather than guessed: the request passed to a
// function, req.query or req.body used whole, a key that isn't a constant, the raw body or URL
// parsed by hand, or route middleware that isn't a body parser.

const REQUEST_NAMES = new Set(['req', 'request']);
const CONTEXT_NAMES = new Set(['ctx', 'context']);
const CONTAINERS = new Set(['query', 'body']);
const CONVERTERS = new Map([
  ['parseInt', 'int'], ['Number.parseInt', 'int'], ['Number', 'float'], ['parseFloat', 'float'],
  ['Number.parseFloat', 'float'], ['Boolean', 'bool']
]);
// Request members that read the query string or body some other way.
const RAW_MEMBERS = new Set(['rawBody', 'read', 'pipe', 'on', 'once', 'body_raw', 'files', 'file']);
const URL_PARSERS = /^(?:URL|URLSearchParams|url\.parse|querystring\.parse|qs\.parse|parseUrl)$/;
// Route middleware that only parses or guards: anything else may read the body itself.
const SAFE_MIDDLEWARE = /^(?:express\.(?:json|urlencoded|text|raw)|bodyParser\.\w+|cors|helmet|compression|multer)\b/;
const ROUTE_METHODS = new Set(['all', 'delete', 'get', 'head', 'options', 'patch', 'post', 'put']);
const SKIPPED = new Set(['loc', 'start', 'end', 'extra', 'leadingComments', 'trailingComments', 'innerComments', 'typeAnnotation', 'returnType', 'typeParameters']);

function dotted(node) {
  if (!node) return '';
  if (node.type === 'Identifier') return node.name;
  if (node.type === 'ThisExpression') return 'this';
  if ((node.type === 'MemberExpression' || node.type === 'OptionalMemberExpression') && !node.computed) {
    const base = dotted(node.object);
    return base ? `${base}.${node.property.name}` : '';
  }
  if (node.type === 'CallExpression' || node.type === 'OptionalCallExpression') return dotted(node.callee);
  if (node.type === 'NewExpression') return dotted(node.callee);
  return '';
}

function isMember(node) {
  return node && (node.type === 'MemberExpression' || node.type === 'OptionalMemberExpression');
}

function isCall(node) {
  return node && (node.type === 'CallExpression' || node.type === 'OptionalCallExpression' || node.type === 'NewExpression');
}

function isFunction(node) {
  return node && ['ArrowFunctionExpression', 'FunctionDeclaration', 'FunctionExpression', 'ObjectMethod', 'ClassMethod'].includes(node.type);
}

function memberKey(node, constants = new Map()) {
  if (!node.computed) return node.property.name;
  if (node.property.type === 'StringLiteral') return node.property.value;
  // A key held in a module-level string constant, as in req.query[PAGE].
  return node.property.type === 'Identifier' && constants.has(node.property.name) ? constants.get(node.property.name) : null;
}

function children(node) {
  const out = [];
  for (const key of Object.keys(node)) {
    if (SKIPPED.has(key)) continue;
    const value = node[key];
    if (Array.isArray(value)) {
      for (const item of value) if (item && typeof item.type === 'string') out.push(item);
    } else if (value && typeof value.type === 'string') {
      out.push(value);
    }
  }
  return out;
}

function merge(inputs, name, kind) {
  // A name read as two types, or as text anywhere, is text: the weaker reading wins.
  inputs[name] = inputs[name] === undefined || inputs[name] === kind ? kind : 'text';
}

// The type a zod field's chain says, as in z.number().int() or z.string().uuid().
function zodType(source) {
  if (/\.uuid\(/.test(source)) return 'uuid';
  if (/\bboolean\(/.test(source)) return 'bool';
  if (/\bnumber\(/.test(source) || /\bbigint\(/.test(source)) return /\.int\(/.test(source) ? 'int' : 'float';
  return 'text';
}

// Module-level zod object schemas, as const Body = z.object({ name: z.string() }), by name.
function zodSchemas(ast, source) {
  const schemas = new Map();
  const visit = node => {
    if (node.type === 'VariableDeclarator' && node.id.type === 'Identifier' && node.init) {
      let call = node.init;
      const text = source.slice(node.init.start, node.init.end);
      while (isCall(call) && isMember(call.callee) && dotted(call.callee) !== 'z.object') call = call.callee.object;
      const object = isCall(call) && dotted(call.callee) === 'z.object' ? call.arguments[0] : null;
      if (object && object.type === 'ObjectExpression') {
        const fields = {};
        let complete = !/\.(?:passthrough|catchall)\(/.test(text.slice(call.end - node.init.start));
        for (const property of object.properties) {
          const key = property.type === 'ObjectProperty' ? (property.key.name || property.key.value) : null;
          if (!key) { complete = false; continue; }
          merge(fields, key, zodType(source.slice(property.value.start, property.value.end)));
        }
        schemas.set(node.id.name, complete ? fields : null);
      }
    }
    for (const child of children(node)) visit(child);
  };
  visit(ast);
  return schemas;
}

/**
 * What one handler reads: { inputs, complete }. scope holds the file's functions, which the
 * request is followed into when passed to one, and its string constants. named, when given, is
 * the parameter that holds the request in a followed function.
 */
function handlerInputs(node, schemas, scope = { functions: new Map(), constants: new Map(), active: new Set() }, named = null) {
  const inputs = {};
  let complete = true;
  const requests = new Set();
  const contexts = new Set();
  for (const parameter of node.params || []) {
    const target = parameter.type === 'AssignmentPattern' ? parameter.left : parameter;
    if (named !== null) {
      if (target.type === 'Identifier' && target.name === named) requests.add(target.name);
    } else if (target.type === 'Identifier' && REQUEST_NAMES.has(target.name)) requests.add(target.name);
    else if (target.type === 'Identifier' && CONTEXT_NAMES.has(target.name)) contexts.add(target.name);
    else if (target.type === 'ObjectPattern') complete = false; // ({ body, query }) destructured from the request.
  }
  const follow = (call, item) => {
    // The request passed to a function in this file, as in return render(req, res): its reads count.
    const target = call.callee.type === 'Identifier' ? scope.functions.get(call.callee.name) : null;
    if (!target || scope.active.has(target) || scope.active.size >= 4) return false;
    const index = call.arguments.indexOf(item);
    const parameter = (target.params || [])[index];
    const name = parameter && (parameter.type === 'AssignmentPattern' ? parameter.left : parameter);
    if (!name || name.type !== 'Identifier') return false;
    scope.active.add(target);
    try {
      const found = handlerInputs(target, schemas, scope, name.name);
      for (const [key, kind] of Object.entries(found.inputs)) merge(inputs, key, kind);
      if (!found.complete) complete = false;
    } finally {
      scope.active.delete(target);
    }
    return true;
  };
  const parents = new Map();
  const nodes = [];
  const index = (item, parent) => {
    parents.set(item, parent);
    nodes.push(item);
    for (const child of children(item)) index(child, item);
  };
  // The body only: the parameters themselves aren't uses of the request.
  if (node.body) index(node.body, node);
  const aliases = new Set();

  const converted = read => {
    let current = read;
    let parent = parents.get(current);
    // Through a default, as in parseInt(req.query.page || '1').
    while (parent && parent.type === 'LogicalExpression' && parent.left === current) {
      current = parent;
      parent = parents.get(current);
    }
    if (isCall(parent) && parent.arguments[0] === current) return CONVERTERS.get(dotted(parent.callee)) || 'text';
    if (parent && parent.type === 'UnaryExpression' && parent.operator === '+') return 'float';
    return 'text';
  };

  const container = expr => {
    const parent = parents.get(expr);
    if (isMember(parent) && parent.object === expr) {
      const key = memberKey(parent, scope.constants);
      if (key === null) complete = false;
      else merge(inputs, key, converted(parent));
      return;
    }
    if (parent && parent.type === 'VariableDeclarator' && parent.init === expr) {
      if (parent.id.type === 'Identifier') {
        aliases.add(parent.id.name);
        return;
      }
      if (parent.id.type === 'ObjectPattern') {
        for (const property of parent.id.properties) {
          const key = property.type === 'ObjectProperty' ? (property.key.name || property.key.value) : null;
          if (!key || property.computed) complete = false;
          else merge(inputs, key, 'text');
        }
        return;
      }
    }
    if (parent && parent.type === 'BinaryExpression' && parent.operator === 'in' && parent.right === expr) return;
    if (isCall(parent) && parent.arguments.includes(expr) && isMember(parent.callee)) {
      // Body.parse(req.body) with a zod object schema from this file.
      const method = parent.callee.property.name;
      const fields = ['parse', 'safeParse', 'parseAsync', 'safeParseAsync'].includes(method)
        ? schemas.get(dotted(parent.callee.object)) : undefined;
      if (fields) {
        for (const [key, kind] of Object.entries(fields)) merge(inputs, key, kind);
        return;
      }
    }
    complete = false;
  };

  const isRequest = item => item && item.type === 'Identifier' && requests.has(item.name);
  const isContext = item => item && item.type === 'Identifier' && contexts.has(item.name);
  for (const item of nodes) {
    if (isMember(item) && !item.computed && (isRequest(item.object) || isContext(item.object)
        || (isMember(item.object) && isContext(item.object.object) && memberKey(item.object) === 'request'))) {
      const name = item.property.name;
      // Koa's ctx.body is the response; its request body is ctx.request.body.
      const response = isContext(item.object) && name === 'body';
      if (response) continue;
      if (CONTAINERS.has(name)) container(item);
      else if (RAW_MEMBERS.has(name)) complete = false;
      else if (name === 'param') {
        const call = parents.get(item);
        const key = isCall(call) && call.arguments[0];
        if (key && key.type === 'StringLiteral') merge(inputs, key.value, converted(call));
        else complete = false;
      } else if (['url', 'originalUrl', 'querystring', 'search', 'href'].includes(name)) {
        // The URL is fine to log or redirect to, but not to read the query from by hand.
        const call = parents.get(item);
        if (isCall(call) && URL_PARSERS.test(dotted(call.callee))) complete = false;
      }
    } else if (item.type === 'Identifier' && (requests.has(item.name) || contexts.has(item.name))) {
      const parent = parents.get(item);
      const declared = parent && (parent.type === 'VariableDeclarator' && parent.id === item);
      if (!(isMember(parent) && parent.object === item) && !declared && !(parent && parent.type === 'ObjectProperty' && parent.key === item)
          && !(isCall(parent) && parent.arguments.includes(item) && follow(parent, item))) {
        complete = false; // The request goes somewhere that may read it.
      }
    }
  }
  // What's read from an alias, as const body = req.body, then body.name.
  for (const item of nodes) {
    if (item.type === 'Identifier' && aliases.has(item.name)) {
      const parent = parents.get(item);
      if (parent && parent.type === 'VariableDeclarator' && parent.id === item) continue;
      if (isMember(parent) && parent.property === item && !parent.computed) continue;
      if (parent && parent.type === 'ObjectProperty' && parent.key === item && !parent.computed) continue;
      container(item);
    }
  }
  return { inputs, complete };
}

/**
 * Each handler's inputs in one parsed file: { function, file, inputs, complete }. label(node,
 * parent) names a handler as the injection audit does, so both link to the same routes.
 */
function fileInputs(ast, filename, source, label) {
  const schemas = zodSchemas(ast, source || '');
  // The file's top-level functions and string constants.
  const scope = { functions: new Map(), constants: new Map(), active: new Set() };
  for (const statement of (ast.program || ast).body || []) {
    const declaration = statement.type === 'ExportNamedDeclaration' ? statement.declaration : statement;
    if (!declaration) continue;
    if (declaration.type === 'FunctionDeclaration' && declaration.id) scope.functions.set(declaration.id.name, declaration);
    if (declaration.type === 'VariableDeclaration') {
      for (const item of declaration.declarations) {
        if (item.id.type !== 'Identifier' || !item.init) continue;
        if (isFunction(item.init)) scope.functions.set(item.id.name, item.init);
        if (declaration.kind === 'const' && item.init.type === 'StringLiteral') scope.constants.set(item.id.name, item.init.value);
      }
    }
  }
  const results = [];
  const visit = (node, parent) => {
    if (isFunction(node) && (node.params || []).some(parameter => {
      const target = parameter.type === 'AssignmentPattern' ? parameter.left : parameter;
      return target.type === 'Identifier' && (REQUEST_NAMES.has(target.name) || CONTEXT_NAMES.has(target.name));
    })) {
      scope.active.add(node);
      const found = handlerInputs(node, schemas, scope);
      scope.active.delete(node);
      // app.post('/x', auth, handler): middleware before the handler may read the body itself.
      if (isCall(parent) && isMember(parent.callee) && ROUTE_METHODS.has(parent.callee.property.name)) {
        const others = parent.arguments.slice(1).filter(argument => argument !== node);
        if (others.some(argument => !(isCall(argument) && SAFE_MIDDLEWARE.test(dotted(argument.callee))))) found.complete = false;
      }
      results.push({ function: label(node, parent), file: filename, inputs: found.inputs, complete: found.complete });
    }
    for (const child of children(node)) visit(child, node);
  };
  visit(ast.program || ast, null);
  return results;
}

/** Adds one handler's inputs to a route; a route served by several takes them all. */
function mergeInto(routes, path, entry) {
  const data = routes[path];
  data.inputs = data.inputs || {};
  for (const [name, kind] of Object.entries(entry.inputs)) merge(data.inputs, name, kind);
  data.inputs_complete = (data.inputs_complete ?? true) && Boolean(entry.complete);
}

module.exports = { fileInputs, handlerInputs, mergeInto };
