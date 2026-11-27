'use strict';

// What each handler reads from the query and body, with types, for the route manifest. init
// saves these as each route's inputs (name to int, float, bool, uuid or text) and
// inputs_complete: true only when every read was understood. Requests are then reviewed against
// them in aiwaf-wasm: see AIWAF_ROUTE_SCHEMA_MODE.
//
// Anything not understood makes the list incomplete rather than guessed, and says why in
// inputs_incomplete: the request passed to a function that isn't followed (helper), req.query or
// req.body used whole (container_whole), a key that isn't a constant (dynamic_key), the raw body
// or URL parsed by hand (raw_body), a schema that isn't found (external_class), or route
// middleware that isn't a body parser (middleware). Zod schemas, helpers and string constants
// are followed into the project's other files through import and require.

const REQUEST_NAMES = new Set(['req', 'request']);
const CONTEXT_NAMES = new Set(['ctx', 'context']);
const CONTAINERS = new Set(['query', 'body']);
// Array methods: called on an input, they use it whole.
const ARRAY_METHODS = new Set(['map', 'forEach', 'filter', 'reduce', 'some', 'every', 'find', 'findIndex', 'flatMap', 'join', 'sort', 'reverse', 'push', 'entries', 'keys', 'values']);

// Fastify's hooks that run before a route's handler, with the request.
const REQUEST_HOOKS = new Set(['onRequest', 'preParsing', 'preValidation', 'preHandler']);

// String and array methods: called on a container, they read a raw text body whole.
const TEXT_METHODS = new Set([
  'trim', 'trimStart', 'trimEnd', 'split', 'slice', 'substring', 'substr', 'toLowerCase', 'toUpperCase', 'includes',
  'startsWith', 'endsWith', 'indexOf', 'lastIndexOf', 'match', 'matchAll', 'replace', 'replaceAll', 'charAt', 'charCodeAt',
  'padStart', 'padEnd', 'concat', 'toString', 'at', 'search', 'normalize', 'localeCompare'
]);

const CONVERTERS = new Map([
  ['parseInt', 'int'], ['Number.parseInt', 'int'], ['Number', 'float'], ['parseFloat', 'float'],
  ['Number.parseFloat', 'float'], ['Boolean', 'bool']
]);
// Request members that read the query string or body some other way.
const RAW_MEMBERS = new Set(['rawBody', 'read', 'pipe', 'on', 'once', 'body_raw', 'files', 'file']);
const URL_PARSERS = /^(?:URL|URLSearchParams|url\.parse|querystring\.parse|qs\.parse|parseUrl)$/;
// Route middleware that only parses or guards: anything else may read the body itself.
const ROUTE_HOOKS = new Set(['onRequest', 'preParsing', 'preValidation', 'preHandler', 'preSerialization']);
const RAW_PARSERS = /^(?:express|bodyParser)\.(?:raw|text)$/;
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
function handlerInputs(node, schemas, scope = { functions: new Map(), constants: new Map(), active: new Set() }, named = null, given = null) {
  const inputs = {};
  const reasons = [];
  // Keys read inside an input, as name in req.body.user.name; and inputs used any other way.
  // Prototype-free, so an input named constructor or toString is a key like any other.
  const nested = Object.create(null);
  const flat = new Set();
  // Where each name was read: from a body container, or something else.
  const fromBody = new Set();
  const elsewhere = new Set();
  const inner = (read, key) => {
    const parent = parents.get(read);
    const sub = isMember(parent) && parent.object === read ? memberKey(parent, scope.constants) : null;
    // lines.length, or a string or array method called on it: the value used whole, not a key.
    const call = parent ? parents.get(parent) : null;
    const builtIn = sub !== null && sub !== undefined && !parent.computed
      && (sub === 'length' || (TEXT_METHODS.has(sub) && isCall(call) && call.callee === parent)
        || (ARRAY_METHODS.has(sub) && isCall(call) && call.callee === parent));
    if (sub === null || sub === undefined || builtIn) flat.add(key);
    else (nested[key] = nested[key] || new Set()).add(sub);
  };
  const incomplete = (code, detail, at) => {
    const reason = { code, detail };
    const line = at && at.loc && at.loc.start && at.loc.start.line;
    if (line) reason.line = line;
    if (scope.file) reason.file = scope.file;
    if (!reasons.some(item => item.code === code && item.detail === detail && item.line === reason.line)) reasons.push(reason);
  };
  const requests = new Set();
  const contexts = new Set();
  for (const parameter of node.params || []) {
    const target = parameter.type === 'AssignmentPattern' ? parameter.left : parameter;
    if (named !== null) {
      if (target.type === 'Identifier' && target.name === named) requests.add(target.name);
    } else if (target.type === 'Identifier' && REQUEST_NAMES.has(target.name)) requests.add(target.name);
    else if (target.type === 'Identifier' && CONTEXT_NAMES.has(target.name)) contexts.add(target.name);
    else if (target.type === 'ObjectPattern') incomplete('container_whole', 'the request is destructured in the parameters', target);
  }
  // The request, or a query or body container (label), passed to a function in this file or one
  // it imports, as in return render(req, res) or readAddress(req.body): its reads count.
  const follow = (call, item, label = null) => {
    const found = lookup(scope, 'function', dotted(call.callee));
    if (!found || scope.active.has(found.node) || scope.active.size >= 4) return false;
    const target = found.node;
    const index = call.arguments.indexOf(item);
    const parameter = (target.params || [])[index];
    const name = parameter && (parameter.type === 'AssignmentPattern' ? parameter.left : parameter);
    if (!name || name.type !== 'Identifier') return false;
    scope.active.add(target);
    try {
      const other = found.scope === scope ? scope : { ...found.scope, active: scope.active };
      const result = label === null
        ? handlerInputs(target, found.scope.schemas || schemas, other, name.name)
        : handlerInputs(target, found.scope.schemas || schemas, other, '', { [name.name]: label });
      for (const [key, kind] of Object.entries(result.inputs)) {
        merge(inputs, key, kind);
        if (result.nested && Object.prototype.hasOwnProperty.call(result.nested, key)) {
          for (const sub of result.nested[key]) (nested[key] = nested[key] || new Set()).add(sub);
        }
        else flat.add(key);
      }
      for (const reason of result.reasons) {
        if (!reasons.some(item => item.code === reason.code && item.detail === reason.detail && item.line === reason.line)) reasons.push(reason);
      }
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
  // A container this function was given, as a helper's body parameter, is read like an alias.
  const aliases = new Map(Object.entries(given || {}));

  const converted = read => {
    let current = read;
    let parent = parents.get(current);
    // Through a default, as in parseInt(req.query.page || '1').
    while (parent && parent.type === 'LogicalExpression' && ['||', '??'].includes(parent.operator) && parent.left === current) {
      current = parent;
      parent = parents.get(current);
    }
    if (isCall(parent) && parent.arguments[0] === current) return CONVERTERS.get(dotted(parent.callee)) || 'text';
    if (parent && parent.type === 'UnaryExpression' && parent.operator === '+') return 'float';
    // const page = req.query.page, cast later: parseInt(page, 10).
    if (parent && parent.type === 'VariableDeclarator' && parent.init === current && parent.id.type === 'Identifier') {
      return castLater(parent.id.name);
    }
    if (parent && parent.type === 'AssignmentExpression' && parent.operator === '=' && parent.right === current
      && parent.left.type === 'Identifier') return castLater(parent.left.name);
    return 'text';
  };

  // The cast a use of a variable is the argument of (through a default, as in
  // parseInt(page || '1')), or null.
  const castOf = use => {
    let current = use;
    let parent = parents.get(current);
    while (parent && parent.type === 'LogicalExpression' && ['||', '??'].includes(parent.operator) && parent.left === current) {
      current = parent;
      parent = parents.get(current);
    }
    if (isCall(parent) && parent.arguments[0] === current && CONVERTERS.has(dotted(parent.callee))) return CONVERTERS.get(dotted(parent.callee));
    if (parent && parent.type === 'UnaryExpression' && parent.operator === '+') return 'float';
    return null;
  };
  // A use that only tests the value: if (page), !page, page === undefined, page == null, page !== ''.
  const checksOnly = use => {
    const parent = parents.get(use);
    if (!parent) return false;
    if (['IfStatement', 'WhileStatement', 'ConditionalExpression'].includes(parent.type) && parent.test === use) return true;
    if (parent.type === 'UnaryExpression' && parent.operator === '!') return true;
    if (parent.type === 'BinaryExpression' && ['===', '!==', '==', '!='].includes(parent.operator)) {
      const other = parent.left === use ? parent.right : parent.left;
      return other.type === 'NullLiteral' || (other.type === 'Identifier' && other.name === 'undefined')
        || (other.type === 'StringLiteral' && other.value === '');
    }
    return false;
  };
  // A reference to the variable, not its declaration, an assignment to it, or a property name.
  const references = (item, name) => {
    if (item.type !== 'Identifier' || item.name !== name) return false;
    const parent = parents.get(item);
    if (!parent) return false;
    if (parent.type === 'VariableDeclarator' && parent.id === item) return false;
    if (parent.type === 'AssignmentExpression' && parent.left === item) return false;
    if (isMember(parent) && parent.property === item && !parent.computed) return false;
    if (parent.type === 'ObjectProperty' && parent.key === item && !parent.computed && !parent.shorthand) return false;
    // The name declared by a pattern, as limit in const { limit } = req.query.
    if (parent.type === 'ObjectProperty' && parents.get(parent) && parents.get(parent).type === 'ObjectPattern') return false;
    if (parent.type === 'ArrayPattern' || parent.type === 'AssignmentPattern' && parent.left === item) return false;
    return true;
  };
  // Where name = Number(name) replaces the read for the rest of the function: the first such
  // assignment that always runs (in the body, or a try block, not under an if).
  const recastAt = name => {
    for (const item of nodes) {
      if (!(item.type === 'AssignmentExpression' && item.operator === '=' && item.left.type === 'Identifier' && item.left.name === name)) continue;
      const value = item.right;
      if (!(isCall(value) && CONVERTERS.has(dotted(value.callee)) && value.arguments[0]
        && value.arguments[0].type === 'Identifier' && value.arguments[0].name === name)) continue;
      let current = parents.get(item);
      if (!current || current.type !== 'ExpressionStatement') continue;
      current = parents.get(current);
      while (current && current.type === 'BlockStatement') {
        const owner = parents.get(current);
        if (owner === node) return item.end;
        if (!(owner && owner.type === 'TryStatement' && owner.block === current)) break;
        current = parents.get(owner);
      }
    }
    return null;
  };
  const castLater = name => {
    const kinds = new Set();
    const after = recastAt(name);
    for (const item of nodes) {
      if (!references(item, name)) continue;
      const kind = castOf(item);
      if (kind) kinds.add(kind);
      else if (after !== null && item.start > after) continue;
      else if (!checksOnly(item)) return 'text';
    }
    return kinds.size === 1 ? [...kinds][0] : 'text';
  };

  const container = (expr, label) => {
    const parent = parents.get(expr);
    if (isMember(parent) && parent.object === expr) {
      const key = memberKey(parent, scope.constants);
      // body.length, or body.trim() and other string methods called on it: a text body read
      // whole, not a field named length.
      const call = parents.get(parent);
      if (!parent.computed && key !== null && (key === 'length'
        || (TEXT_METHODS.has(key) && isCall(call) && call.callee === parent))) {
        incomplete('raw_body', `${label}.${key} reads the body as text`, parent);
        return;
      }
      const imported = key === null && parent.computed && parent.property.type === 'Identifier'
        ? lookup(scope, 'constant', parent.property.name) : null;
      if (key !== null) {
        merge(inputs, key, converted(parent));
        inner(parent, key);
        (/\bbody\b/.test(label) ? fromBody : elsewhere).add(key);
      } else if (imported) {
        merge(inputs, imported.node, converted(parent));
        flat.add(imported.node);
      }
      else incomplete('dynamic_key', `${label} is read with a key that isn't a constant`, parent);
      return;
    }
    if (parent && parent.type === 'VariableDeclarator' && parent.init === expr) {
      if (parent.id.type === 'Identifier') {
        aliases.set(parent.id.name, label);
        return;
      }
      if (parent.id.type === 'ObjectPattern') {
        for (const property of parent.id.properties) {
          const key = property.type === 'ObjectProperty' ? (property.key.name || property.key.value) : null;
          if (!key || property.computed) incomplete('container_whole', `the rest of ${label} is kept whole`, property);
          // const { limit } = req.query, then Number(limit): the local name's casts give the type.
          else merge(inputs, key, property.value && property.value.type === 'Identifier' ? castLater(property.value.name) : 'text');
        }
        return;
      }
    }
    if (parent && parent.type === 'BinaryExpression' && parent.operator === 'in' && parent.right === expr) return;
    if (isCall(parent) && parent.arguments.includes(expr) && isMember(parent.callee)) {
      // Body.parse(req.body) with a zod object schema from this file or one it imports.
      const method = parent.callee.property.name;
      if (['parse', 'safeParse', 'parseAsync', 'safeParseAsync'].includes(method)) {
        const name = dotted(parent.callee.object);
        const local = schemas.get(name);
        const found = local !== undefined ? { node: local } : lookup(scope, 'schema', name);
        if (found && found.node) {
          for (const [key, kind] of Object.entries(found.node)) merge(inputs, key, kind);
          return;
        }
        incomplete('external_class', found ? `${name} accepts keys it doesn't declare` : `schema ${name || 'here'} isn't found`, parent);
        return;
      }
    }
    if (isCall(parent) && parent.arguments.includes(expr)) {
      if (follow(parent, expr, label)) return;
      incomplete('container_whole', `${label} is passed whole to ${dotted(parent.callee) || 'a call'}`, parent);
      return;
    }
    incomplete('container_whole', `${label} is used whole`, parent || expr);
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
      if (CONTAINERS.has(name)) container(item, dotted(item) || name);
      else if (RAW_MEMBERS.has(name)) incomplete('raw_body', `${dotted(item) || name} reads the raw body`, item);
      else if (name === 'param') {
        const call = parents.get(item);
        const key = isCall(call) && call.arguments[0];
        if (key && key.type === 'StringLiteral') merge(inputs, key.value, converted(call));
        else incomplete('dynamic_key', `${dotted(item)}() is called with a key that isn't a constant`, item);
      } else if (['url', 'originalUrl', 'querystring', 'search', 'href'].includes(name)) {
        // The URL is fine to log or redirect to, but not to read the query from by hand.
        const call = parents.get(item);
        if (isCall(call) && URL_PARSERS.test(dotted(call.callee))) incomplete('raw_body', `${dotted(item)} is parsed by hand`, call);
      }
    } else if (item.type === 'Identifier' && (requests.has(item.name) || contexts.has(item.name))) {
      const parent = parents.get(item);
      const declared = parent && (parent.type === 'VariableDeclarator' && parent.id === item);
      if (!(isMember(parent) && parent.object === item) && !declared && !(parent && parent.type === 'ObjectProperty' && parent.key === item)
          && !(isCall(parent) && parent.arguments.includes(item) && follow(parent, item))) {
        // The request goes somewhere that may read it.
        if (isCall(parent)) incomplete('helper', `the request is passed to ${dotted(parent.callee) || 'a call'}, which isn't followed`, parent);
        else incomplete('helper', 'the request is stored or returned where it may be read', item);
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
      container(item, aliases.get(item.name));
    }
  }
  const known = Object.fromEntries(Object.entries(nested).filter(([key]) => !flat.has(key) && key in inputs)
    .map(([key, subs]) => [key, [...subs].sort()]));
  const bodyOnly = [...fromBody].filter(key => !elsewhere.has(key));
  return { inputs, complete: reasons.length === 0, reasons, nested: known, bodyOnly };
}

const MAX_DEPTH = 6;

/**
 * The function, string constant or zod schema a name refers to in a file's scope, through its
 * imports into the project's other files: { node, scope }, or null.
 */
function lookup(scope, kind, name, depth = 0) {
  if (!name || !scope || depth > MAX_DEPTH) return null;
  const table = kind === 'function' ? scope.functions : kind === 'constant' ? scope.constants : scope.schemas;
  const [head, ...rest] = name.split('.');
  if (!rest.length && table && table.has(head)) return { node: table.get(head), scope };
  const imported = scope.imports && scope.imports.get(head);
  const target = imported && imported.file && scope.project ? scope.project.get(imported.file) : null;
  if (!target) return null;
  let exported = imported.name === '*' ? rest.join('.') : [imported.name, ...rest].join('.');
  if (!exported && imported.name === '*') exported = 'default'; // const fn = require('./fn'), module.exports = fn.
  const [first, ...more] = exported.split('.');
  const local = target.exports.get(first);
  return local === undefined ? null : lookup(target, kind, [local, ...more].join('.'), depth + 1);
}

function requireSpec(node) {
  return isCall(node) && node.callee.type === 'Identifier' && node.callee.name === 'require'
    && node.arguments[0] && node.arguments[0].type === 'StringLiteral' ? node.arguments[0].value : null;
}

/**
 * A file's scope for reading handlers: its top-level functions, string constants, zod schemas,
 * imports and exports. resolve(spec) gives the absolute path of an imported project file.
 */
function moduleScope(ast, source, filename, resolve = () => null) {
  const scope = {
    file: filename, functions: new Map(), constants: new Map(), schemas: zodSchemas(ast, source || ''),
    imports: new Map(), exports: new Map(), project: null, active: new Set()
  };
  const define = (name, value) => {
    if (isFunction(value)) scope.functions.set(name, value);
    else if (value && value.type === 'StringLiteral') scope.constants.set(name, value.value);
  };
  const exportObject = object => {
    for (const property of object.properties || []) {
      if (property.type !== 'ObjectProperty' && property.type !== 'ObjectMethod') continue;
      const key = property.key.name || property.key.value;
      if (!key) continue;
      if (property.type === 'ObjectMethod') {
        scope.functions.set(key, property);
        scope.exports.set(key, key);
      } else if (property.value.type === 'Identifier') scope.exports.set(key, property.value.name);
      else {
        define(key, property.value);
        scope.exports.set(key, key);
      }
    }
  };
  for (const statement of (ast.program || ast).body || []) {
    if (statement.type === 'ImportDeclaration') {
      const file = resolve(statement.source.value);
      for (const specifier of statement.specifiers) {
        const name = specifier.type === 'ImportDefaultSpecifier' ? 'default'
          : specifier.type === 'ImportNamespaceSpecifier' ? '*' : (specifier.imported.name || specifier.imported.value);
        scope.imports.set(specifier.local.name, { file, name });
      }
      continue;
    }
    if (statement.type === 'ExportNamedDeclaration' && statement.source) {
      // export { a as b } from './x': b re-exports x's a.
      const file = resolve(statement.source.value);
      for (const specifier of statement.specifiers) {
        const local = `\0${specifier.exported.name || specifier.exported.value}`;
        scope.imports.set(local, { file, name: specifier.local.name });
        scope.exports.set(specifier.exported.name || specifier.exported.value, local);
      }
      continue;
    }
    if (statement.type === 'ExportNamedDeclaration' && !statement.declaration) {
      for (const specifier of statement.specifiers) scope.exports.set(specifier.exported.name || specifier.exported.value, specifier.local.name);
      continue;
    }
    if (statement.type === 'ExportDefaultDeclaration') {
      const declaration = statement.declaration;
      if (declaration.type === 'Identifier') scope.exports.set('default', declaration.name);
      else if (isFunction(declaration)) {
        const name = (declaration.id && declaration.id.name) || '\0default';
        scope.functions.set(name, declaration);
        scope.exports.set('default', name);
      }
      continue;
    }
    const exported = statement.type === 'ExportNamedDeclaration';
    const declaration = exported ? statement.declaration : statement;
    if (declaration.type === 'FunctionDeclaration' && declaration.id) {
      scope.functions.set(declaration.id.name, declaration);
      if (exported) scope.exports.set(declaration.id.name, declaration.id.name);
    }
    if (declaration.type === 'VariableDeclaration') {
      for (const item of declaration.declarations) {
        if (!item.init) continue;
        let spec = requireSpec(item.init);
        let member = null;
        if (!spec && isMember(item.init) && !item.init.computed && requireSpec(item.init.object)) {
          spec = requireSpec(item.init.object);
          member = item.init.property.name;
        }
        if (spec) {
          const file = resolve(spec);
          if (item.id.type === 'Identifier') scope.imports.set(item.id.name, { file, name: member || '*' });
          else if (item.id.type === 'ObjectPattern') {
            for (const property of item.id.properties) {
              if (property.type === 'ObjectProperty' && property.value.type === 'Identifier') {
                scope.imports.set(property.value.name, { file, name: property.key.name || property.key.value });
              }
            }
          }
          continue;
        }
        if (item.id.type !== 'Identifier') continue;
        if (isFunction(item.init) || declaration.kind === 'const') define(item.id.name, item.init);
        if (exported) scope.exports.set(item.id.name, item.id.name);
      }
    }
    if (declaration.type === 'ExpressionStatement' && declaration.expression.type === 'AssignmentExpression') {
      // module.exports = { a, b }, module.exports = fn, exports.a = ..., module.exports.a = ...
      const { left, right } = declaration.expression;
      const target = dotted(left);
      if (target === 'module.exports') {
        if (right.type === 'ObjectExpression') exportObject(right);
        else if (right.type === 'Identifier') scope.exports.set('default', right.name);
        else if (isFunction(right)) {
          scope.functions.set('\0default', right);
          scope.exports.set('default', '\0default');
        }
      } else if (/^(?:module\.)?exports\.\w+$/.test(target)) {
        const name = target.split('.').pop();
        if (right.type === 'Identifier') scope.exports.set(name, right.name);
        else {
          define(name, right);
          scope.exports.set(name, name);
        }
      }
    }
  }
  return scope;
}

/**
 * Each handler's inputs in one parsed file: { function, file, inputs, complete, reasons }, with
 * reasons only on incomplete ones. label(node, parent) names a handler as the injection audit
 * does, so both link to the same routes. scope, from moduleScope with its project set, follows
 * imports into the project's other files; without, only this file is read.
 */
function fileInputs(ast, filename, source, label, scope = null) {
  if (!scope) scope = moduleScope(ast, source, filename);
  const schemas = scope.schemas;
  const results = [];
  // The handler of a route call, as app.get('/health', () => ...), even when it takes nothing.
  const routeHandler = (node, parent) => isCall(parent) && isMember(parent.callee) && ROUTE_METHODS.has(parent.callee.property.name)
    && parent.arguments.length > 1 && parent.arguments[parent.arguments.length - 1] === node
    && parent.arguments[0] && ['StringLiteral', 'TemplateLiteral'].includes(parent.arguments[0].type);
  const visit = (node, parent) => {
    if (isFunction(node) && (routeHandler(node, parent) || (node.params || []).some(parameter => {
      const target = parameter.type === 'AssignmentPattern' ? parameter.left : parameter;
      return target.type === 'Identifier' && (REQUEST_NAMES.has(target.name) || CONTEXT_NAMES.has(target.name));
    }))) {
      scope.active.add(node);
      const found = handlerInputs(node, schemas, scope);
      scope.active.delete(node);
      // Names read by validators or route middleware, whose inner keys aren't known.
      let middlewareNames = {};
      // app.post('/x', auth, handler): middleware before the handler may read the body itself.
      if (isCall(parent) && isMember(parent.callee) && ROUTE_METHODS.has(parent.callee.property.name)) {
        const others = parent.arguments.slice(1).filter(argument => argument !== node);
        const validated = { inputs: {}, exact: false, segments: {} };
        middlewareNames = validated.inputs;
        for (const argument of others) {
          // Fastify's route options, as { schema }: only their hooks run before the handler.
          if (argument.type === 'ObjectExpression' && !argument.properties.some(property => property.type !== 'ObjectProperty'
            || ROUTE_HOOKS.has(property.key.name || property.key.value))) continue;
          if (isCall(argument) && RAW_PARSERS.test(dotted(argument.callee))) {
            // express.raw() and express.text(): the handler gets the body as bytes or text, not
            // names, so what it reads off req.body (as .length) aren't inputs.
            const reason = { code: 'raw_body', detail: `${dotted(argument.callee)}() gives the handler the raw body`, file: filename };
            if (argument.loc) reason.line = argument.loc.start.line;
            found.reasons.push(reason);
            for (const name of found.bodyOnly || []) delete found.inputs[name];
            continue;
          }
          if (isCall(argument) && SAFE_MIDDLEWARE.test(dotted(argument.callee))) continue;
          if (routeValidator(argument, validated)) continue;
          // Middleware of the project's own is read like the handler: what it reads is the route's.
          const own = (argument.type === 'Identifier' || isMember(argument)) ? lookup(scope, 'function', dotted(argument)) : null;
          if (own && isFunction(own.node) && takesRequest(own.node) && !own.scope.active.has(own.node)) {
            own.scope.active.add(own.node);
            const read = handlerInputs(own.node, own.scope.schemas, own.scope);
            own.scope.active.delete(own.node);
            for (const [name, kind] of Object.entries(read.inputs)) {
              merge(found.inputs, name, kind);
              validated.inputs[name] = validated.inputs[name] || kind; // Its inner keys aren't merged, so unknown.
            }
            found.reasons.push(...read.reasons);
            continue;
          }
          const reason = { code: 'middleware', detail: `route middleware ${dotted(argument) || 'here'} may read the body`, file: filename };
          if (argument.loc) reason.line = argument.loc.start.line;
          found.reasons.push(reason);
        }
        for (const [name, kind] of Object.entries(validated.inputs)) merge(found.inputs, name, kind);
        // Requests with any other name are rejected first, so only a raw body read stays unknown.
        if (validated.exact || (validated.segments.query && validated.segments.body)) {
          found.reasons = found.reasons.filter(reason => reason.code === 'raw_body');
        }
      }
      const nested = Object.fromEntries(Object.entries(found.nested || {}).filter(([key]) => !(key in middlewareNames)));
      const entry = { function: label(node, parent), file: filename, inputs: found.inputs, complete: found.reasons.length === 0 };
      // fastify.addHook('preHandler', fn): it runs before the handlers of the routes in its
      // scope, so what it reads is theirs.
      if (isCall(parent) && isMember(parent.callee) && (parent.callee.property.name || parent.callee.property.value) === 'addHook'
        && parent.arguments[parent.arguments.length - 1] === node && parent.arguments[0]
        && parent.arguments[0].type === 'StringLiteral' && REQUEST_HOOKS.has(parent.arguments[0].value)) entry.hook = parent.arguments[0].value;
      if (Object.keys(nested).length) entry.nested = nested;
      if (found.reasons.length) entry.reasons = found.reasons;
      results.push(entry);
    }
    for (const child of children(node)) visit(child, node);
  };
  visit(ast.program || ast, null);
  return results;
}

// Route validation middleware read from source: express-validator's query('q'), body('x'),
// check(...), param/header/cookie, checkSchema({...}) and checkExact([...]), and celebrate({...}).
const VALIDATOR_LOCATIONS = new Map([
  ['query', ['query']], ['body', ['body']], ['check', ['query', 'body']], ['param', []], ['header', []], ['cookie', []]
]);
const VALIDATOR_NAMES = new Set(['query', 'body', 'check', 'param', 'header', 'cookie', 'checkSchema', 'checkExact', 'celebrate']);
const CHAIN_KINDS = { isInt: 'int', toInt: 'int', isFloat: 'float', toFloat: 'float', isNumeric: 'float', isBoolean: 'bool', toBoolean: 'bool', isUUID: 'uuid' };

/**
 * body('a').isInt().toInt() is { name: 'body', call, methods: ['toInt', 'isInt'] }, and so are
 * ev.body('a'), this.validators.body('a') and makeValidators().body('a'): a validator reached
 * through a namespace, an object on this, or a function's result. Null for anything else.
 */
function chainRoot(node) {
  const calls = [];
  for (let current = node; isCall(current) && current.type === 'CallExpression'; current = isMember(current.callee) ? current.callee.object : null) {
    calls.push(current);
    if (!isMember(current.callee) || current.callee.computed) break;
  }
  // The innermost call whose name is a validator's, counting from the start of the chain.
  for (let index = calls.length - 1; index >= 0; index -= 1) {
    const call = calls[index];
    const callee = call.callee;
    const name = callee.type === 'Identifier' ? callee.name : (isMember(callee) && !callee.computed ? callee.property.name : null);
    if (!name || !VALIDATOR_NAMES.has(name)) continue;
    if (isMember(callee) && !namespaceLike(callee.object) && !isCall(callee.object)) continue;
    const methods = calls.slice(0, index).map(item => item.callee.property.name);
    return { name, call, methods };
  }
  return null;
}

// A namespace import or an object reached from one or from this, as in ev or this.validators.
function namespaceLike(node) {
  if (!node) return false;
  if (node.type === 'Identifier' || node.type === 'ThisExpression') return true;
  return isMember(node) && !node.computed && namespaceLike(node.object);
}

function strings(node) {
  if (!node) return [];
  if (node.type === 'StringLiteral') return [node.value];
  if (node.type === 'ArrayExpression') return node.elements.flatMap(strings);
  return [];
}

function propertyName(property) {
  const key = property.key;
  if (!property.computed && key.type === 'Identifier') return key.name;
  if (key.type === 'StringLiteral') return key.value;
  // celebrate's { [Segments.QUERY]: ... }
  return property.computed && isMember(key) && !key.computed ? key.property.name.toLowerCase() : null;
}

/** A field express-validator checks, as the top-level input name: user.name and items[0] are user, items. */
function topName(field) {
  const name = String(field).split(/[.[]/)[0];
  return name && name !== '*' ? name : null;
}

function joiKind(methods) {
  if (methods.includes('integer')) return 'int';
  if (methods.includes('number')) return 'float';
  if (methods.includes('boolean')) return 'bool';
  return methods.includes('guid') || methods.includes('uuid') ? 'uuid' : 'text';
}

/** A celebrate segment: Joi.object({...}) (closed unless .unknown()), or a plain object of keys (closed). */
function joiObject(node) {
  if (node && node.type === 'ObjectExpression') return { keys: node, closed: true };
  let closed = true;
  for (let current = node; isCall(current) && isMember(current.callee); current = current.callee.object) {
    const method = current.callee.property.name;
    if (method === 'unknown' || method === 'pattern') closed = false;
    if ((method === 'object' || method === 'keys') && current.arguments[0] && current.arguments[0].type === 'ObjectExpression') {
      return { keys: current.arguments[0], closed };
    }
  }
  return null;
}

function chainMethods(node) {
  const methods = [];
  for (let current = node; isCall(current) && isMember(current.callee); current = current.callee.object) methods.push(current.callee.property.name);
  return methods;
}

/**
 * Reads one route middleware argument as validators into out ({ inputs, exact, segments });
 * false when it isn't one. checkExact, or celebrate closing both query and body, means requests
 * carrying any other name are rejected before the handler runs.
 */
function routeValidator(node, out) {
  if (node && node.type === 'ArrayExpression') return node.elements.every(element => routeValidator(element, out));
  const root = chainRoot(node);
  if (!root) return false;
  const [first] = root.call.arguments;
  if (root.name === 'checkExact') {
    out.exact = true;
    return !first || routeValidator(first, out);
  }
  if (root.name === 'checkSchema') {
    for (const property of (first && first.type === 'ObjectExpression' ? first.properties : [])) {
      const name = property.type === 'ObjectProperty' ? topName(propertyName(property) || '') : null;
      if (!name) continue;
      const spec = property.value.type === 'ObjectExpression' ? property.value.properties : [];
      const located = spec.find(item => item.type === 'ObjectProperty' && propertyName(item) === 'in');
      const locations = located ? strings(located.value) : ['query', 'body'];
      if (!locations.some(location => location === 'query' || location === 'body')) continue;
      const kind = spec.map(item => item.type === 'ObjectProperty' && CHAIN_KINDS[propertyName(item)]).find(Boolean);
      merge(out.inputs, name, kind || 'text');
    }
    return true;
  }
  if (root.name === 'celebrate') {
    for (const property of (first && first.type === 'ObjectExpression' ? first.properties : [])) {
      const segment = property.type === 'ObjectProperty' ? propertyName(property) : null;
      if (segment !== 'query' && segment !== 'body') continue;
      const schema = joiObject(property.value);
      if (!schema) continue;
      out.segments[segment] = schema.closed;
      for (const key of schema.keys.properties) {
        const name = key.type === 'ObjectProperty' ? propertyName(key) : null;
        if (name) merge(out.inputs, name, joiKind(chainMethods(key.value)));
      }
    }
    return true;
  }
  if (!VALIDATOR_LOCATIONS.has(root.name)) return false;
  if (VALIDATOR_LOCATIONS.get(root.name).length) {
    const kind = root.methods.map(method => CHAIN_KINDS[method]).find(Boolean) || 'text';
    for (const field of strings(first)) {
      const name = topName(field);
      if (name) merge(out.inputs, name, name === field ? kind : 'text');
    }
  }
  return true;
}

// Whether a function takes the request first, as a handler or middleware does.
function takesRequest(node) {
  const [first] = node.params || [];
  const target = first && first.type === 'AssignmentPattern' ? first.left : first;
  return Boolean(target && target.type === 'Identifier' && (REQUEST_NAMES.has(target.name) || CONTEXT_NAMES.has(target.name)));
}

const MAX_REASONS = 20;

/** Adds one handler's inputs to a route; a route served by several takes them all, and their reasons. */
function mergeInto(routes, path, entry) {
  const data = routes[path];
  data.inputs = data.inputs || {};
  for (const [name, kind] of Object.entries(entry.inputs)) {
    merge(data.inputs, name, kind);
    // An input's inner keys are only known when every handler reading it reads them.
    const own = (table, key) => Boolean(table) && Object.prototype.hasOwnProperty.call(table, key);
    if (own(entry.nested, name)) {
      data._nested = data._nested || Object.create(null);
      data._nested[name] = [...new Set([...(own(data._nested, name) ? data._nested[name] : []), ...entry.nested[name]])].sort();
    } else {
      data._flat = [...new Set([...(data._flat || []), name])];
    }
  }
  data.inputs_complete = (data.inputs_complete ?? true) && Boolean(entry.complete);
  const reasons = data.inputs_incomplete || [];
  for (const reason of entry.reasons || []) {
    const item = { ...reason, function: entry.function || '' };
    if (reasons.length < MAX_REASONS && !reasons.some(other => JSON.stringify(other) === JSON.stringify(item))) reasons.push(item);
  }
  if (reasons.length) data.inputs_incomplete = reasons;
  else delete data.inputs_incomplete;
}

module.exports = { fileInputs, handlerInputs, lookup, merge, mergeInto, moduleScope };
