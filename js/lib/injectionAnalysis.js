'use strict';

// Static analysis that follows request input to dangerous operations.
// It reads source files only, and never runs application code or affects request handling.
// Calls into other project files are followed through relative paths, tsconfig or jsconfig paths
// and baseUrl, and workspace packages, including methods on instances of the project's classes.
// A clean report doesn't prove code is safe: dynamic calls and packages outside the project can be missed.

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const parser = require('@babel/parser');
const { fileInputs, moduleScope } = require('./inputSchema');

const SCHEMA_VERSION = '1.0';
const SKIPPED_DIRECTORIES = new Set(['.git', '.next', '.nuxt', 'build', 'coverage', 'dist', 'node_modules', 'out', 'vendor']);
const TEST_DIRECTORIES = new Set(['__tests__', 'test', 'tests']);
const SOURCE_EXTENSIONS = new Set(['.cjs', '.cts', '.js', '.jsx', '.mjs', '.mts', '.ts', '.tsx']);
const MAX_FILE_BYTES = 2 * 1024 * 1024;
const MAX_HELPER_DEPTH = 3;

// Express, Next.js, Sails, Fastify, Hapi and NestJS use req or request; Koa and Adonis use ctx.
const REQUEST_NAMES = new Set(['req', 'request']);
const CONTEXT_NAMES = new Set(['ctx', 'context']);
const REQUEST_PROPERTIES = new Set([
  'body', 'cookies', 'files', 'file', 'headers', 'hostname', 'originalUrl', 'params', 'path', 'payload', 'query',
  'querystring', 'raw', 'rawBody', 'state', 'url'
]);
const CONTEXT_PROPERTIES = new Set(['body', 'cookies', 'headers', 'params', 'path', 'query', 'querystring', 'request', 'url']);
// formData(), json() and text() read the body of a Fetch API Request, as in Next.js and Remix.
const REQUEST_METHODS = new Set([
  'all', 'allParams', 'arrayBuffer', 'blob', 'body', 'cookie', 'file', 'formData', 'get', 'header', 'input', 'json', 'only',
  'param', 'params', 'qs', 'raw', 'text'
]);
// NestJS (REST, GraphQL and WebSocket) and TypeGraphQL decorators for request input.
const NEST_INPUT_DECORATORS = new Set(['Arg', 'Args', 'Body', 'Cookies', 'Headers', 'MessageBody', 'Param', 'Query', 'UploadedFile', 'UploadedFiles']);
const GRAPHQL_ROOTS = new Set(['Mutation', 'Query', 'Subscription']);
const TRPC_PROCEDURES = new Set(['mutation', 'query', 'subscription']);
const SOCKET_EVENTS_IGNORED = new Set(['close', 'connect', 'connection', 'disconnect', 'error', 'listening', 'open']);
// Methods that call back with each value of what they are called on, as in items.forEach(item => ...).
const CALLBACK_METHODS = new Set(['every', 'filter', 'find', 'flatMap', 'forEach', 'map', 'some', 'then']);
// NestJS pipes that reject anything but a number, boolean or UUID, so the value can't carry an injection.
const NEST_SAFE_PIPES = new Set(['ParseBoolPipe', 'ParseFloatPipe', 'ParseIntPipe', 'ParseUUIDPipe']);
// Methods that put a value into the collection they are called on.
const COLLECTION_ADDS = new Set(['add', 'push', 'set', 'splice', 'unshift']);
const SANITIZERS = new Set(['Boolean', 'Number', 'Number.parseFloat', 'Number.parseInt', 'encodeURIComponent', 'isNaN', 'parseFloat', 'parseInt']);
// Encoders whose result is safe only for some operations, as escapeHtml is for HTML.
const KIND_SANITIZERS = new Map([
  ...['DOMPurify.sanitize', 'escape', 'escapeHtml', 'filterXSS', 'he.encode', 'he.escape', 'sanitizeHtml', 'validator.escape', 'xss']
    .map(name => [name, ['xss']]),
  ['path.basename', ['path']],
  ...['shellescape', 'shellQuote.quote', 'quote'].map(name => [name, ['command']]),
  ...['mysql.escape', 'mysql.escapeId', 'pgFormat', 'sqlstring.escape', 'sqlstring.escapeId', 'format.literal'].map(name => [name, ['sql']]),
  ...['_.escapeRegExp', 'escapeRegExp', 'escapeStringRegexp', 'lodash.escapeRegExp'].map(name => [name, ['redos']]),
  ...['ldapEscape.filter', 'ldapEscape'].map(name => [name, ['ldap']]),
  ['encodeURI', ['redirect']]
]);
// Marks what a source was made safe for: source + SAFE + 'xss,ldap', or '*' for everything.
const SAFE = '\u0001';
// Schema checks whose result is a number, boolean, UUID or one of fixed values, as in z.number().parse(x).
const SAFE_SCHEMA = /\b(?:bigint|boolean|date|datetime|email|enum|int|literal|nativeEnum|number|uuid)\(/;
const VALIDATOR_CHECKS = /^(?:validator\.)?is(?:Alpha|Alphanumeric|Base32|Boolean|Date|Decimal|Email|Float|Hexadecimal|IP|ISO8601|Int|MongoId|Numeric|Port|UUID)$/;
const RESPONSE_NAMES = new Set(['res', 'response']);
const XSS_METHODS = new Set(['end', 'send', 'write']);
const MERGE_FUNCTIONS = new Set([
  '_.defaultsDeep', '_.merge', '_.mergeWith', '_.set', '_.setWith', '_.zipObjectDeep', 'deepmerge', 'defaultsDeep', 'lodash.merge',
  'lodash.set', 'merge', 'mergeDeep', 'set', 'setWith'
]);
const XPATH_FUNCTIONS = new Set(['select', 'select1', 'xpath.evaluate', 'xpath.parse', 'xpath.select', 'xpath.select1', 'xpath.useNamespaces']);
const XML_FUNCTIONS = new Set(['libxml.parseXml', 'libxml.parseXmlString', 'libxmljs.parseXml', 'libxmljs.parseXmlString', 'parseXml', 'parseXmlString']);

function safeSource(source, kinds) {
  if (!source || !kinds || !kinds.length) return source;
  const all = new Set([...safeKinds(source), ...kinds]);
  return `${baseSource(source)}${SAFE}${[...all].sort().join(',')}`;
}

function baseSource(source) {
  const marker = source.indexOf(SAFE);
  return marker < 0 ? source : source.slice(0, marker);
}

function safeKinds(source) {
  const marker = source ? source.indexOf(SAFE) : -1;
  return marker < 0 ? [] : source.slice(marker + 1).split(',');
}

// Two inputs in one value: it's safe only for what both are safe for.
function combine(left, right) {
  if (!left || !right) return left || right || null;
  const kinds = safeKinds(left).includes('*') ? safeKinds(right)
    : safeKinds(right).includes('*') ? safeKinds(left) : safeKinds(left).filter(kind => safeKinds(right).includes(kind));
  return kinds.length ? safeSource(baseSource(left), kinds) : baseSource(safeKinds(left).length ? right : left);
}

const UNKNOWN = Symbol('unknown');

// True when a block always leaves: return, throw, continue or break.
function exits(node) {
  if (!node) return false;
  if (['BreakStatement', 'ContinueStatement', 'ReturnStatement', 'ThrowStatement'].includes(node.type)) return true;
  return node.type === 'BlockStatement' && node.body.length > 0 && exits(node.body[node.body.length - 1]);
}

function assignedNames(node, names = new Set()) {
  if (!node || typeof node.type !== 'string') return names;
  if (node.type === 'AssignmentExpression' && node.left.type === 'Identifier') names.add(node.left.name);
  if (node.type === 'UpdateExpression' && node.argument.type === 'Identifier') names.add(node.argument.name);
  eachChild(node, child => assignedNames(child, names));
  return names;
}
const PASSTHROUGH_METHODS = new Set([
  'concat', 'join', 'padEnd', 'padStart', 'replace', 'replaceAll', 'slice', 'split', 'substring', 'toLowerCase',
  'toString', 'toUpperCase', 'trim', 'trimEnd', 'trimStart'
]);
const PASSTHROUGH_FUNCTIONS = new Set([
  'Array.from', 'Buffer.from', 'JSON.parse', 'JSON.stringify', 'Object.assign', 'Object.entries', 'Object.keys', 'Object.values',
  'Promise.resolve', 'String', 'decodeURIComponent', 'path.join', 'path.normalize', 'path.resolve', 'qs.parse', 'querystring.parse'
]);
const SQL_METHODS = new Set(['$executeRawUnsafe', '$queryRawUnsafe', 'execute', 'havingRaw', 'orderByRaw', 'query', 'raw', 'whereRaw']);
const SQLITE_METHODS = new Set(['all', 'each', 'exec', 'get', 'prepare', 'run']);
const SQLITE_RECEIVER = /(^|\.)(db|sqlite|database)$/i;
const NOSQL_METHODS = new Set(['aggregate', 'count', 'countDocuments', 'deleteMany', 'deleteOne', 'find', 'findOne', 'findOneAndDelete', 'findOneAndUpdate', 'updateMany', 'updateOne']);
const NOSQL_RECEIVER = /(coll|db|mongo|model|users?$|^[A-Z])/;
const COMMAND_FUNCTIONS = new Set([
  'child_process.exec', 'child_process.execFile', 'child_process.execFileSync', 'child_process.execSync', 'child_process.fork',
  'cp.exec', 'cp.execFile', 'cp.execFileSync', 'cp.execSync', 'cp.fork', 'exec', 'execFile', 'execFileSync', 'execSync', 'fork'
]);
const SHELL_FUNCTIONS = new Set(['child_process.spawn', 'child_process.spawnSync', 'cp.spawn', 'cp.spawnSync', 'spawn', 'spawnSync']);
const PATH_METHODS = new Set([
  'appendFile', 'appendFileSync', 'createReadStream', 'createWriteStream', 'readFile', 'readFileSync', 'readdir',
  'readdirSync', 'rm', 'rmSync', 'unlink', 'unlinkSync', 'writeFile', 'writeFileSync'
]);
const FS_RECEIVER = /(^|\.)(fs|fsp|promises)$/;
const RESPONSE_PATH_METHODS = new Set(['download', 'sendFile']);
const SSRF_FUNCTIONS = new Set([
  'axios', 'axios.delete', 'axios.get', 'axios.head', 'axios.patch', 'axios.post', 'axios.put', 'axios.request', 'fetch', 'got',
  'got.get', 'got.post', 'http.get', 'http.request', 'https.get', 'https.request', 'needle', 'superagent.get', 'superagent.post',
  'undici.fetch', 'undici.request'
]);
// A string given to setTimeout or setInterval runs as code; require() and import() of a module
// named by input run that module.
const CODE_FUNCTIONS = new Set([
  'Function', 'eval', 'import', 'require', 'setImmediate.call', 'setInterval', 'setTimeout', 'vm.compileFunction',
  'vm.runInContext', 'vm.runInNewContext', 'vm.runInThisContext', 'window.eval'
]);
const TEMPLATE_FUNCTIONS = new Set(['Handlebars.compile', 'dot.template', 'ejs.compile', 'ejs.render', 'handlebars.compile', 'lodash.template', 'nunjucks.renderString', 'pug.compile', 'pug.render', '_.template']);
const DESERIALIZE_FUNCTIONS = new Set(['serialize.unserialize', 'nodeSerialize.unserialize', 'funcster.deepDeserialize', 'v8.deserialize']);

function dottedName(node) {
  if (!node) return '';
  if (node.type === 'Identifier') return node.name;
  if (node.type === 'ThisExpression') return 'this';
  if (node.type === 'Super') return 'super';
  if (node.type === 'MemberExpression' || node.type === 'OptionalMemberExpression') {
    const object = dottedName(node.object);
    const property = !node.computed && node.property.type === 'Identifier' ? node.property.name : '';
    return object && property ? `${object}.${property}` : '';
  }
  if (node.type === 'CallExpression' || node.type === 'OptionalCallExpression') return dottedName(node.callee);
  if (node.type === 'AwaitExpression') return dottedName(node.argument);
  return '';
}

function propertyName(node) {
  if (!node) return '';
  if (!node.computed && node.property && node.property.type === 'Identifier') return node.property.name;
  return '';
}

// Adds a parameter name to a whole-container source, as in req.query.id.
function withKey(source, key) {
  if (!source || !key || !/(?:\.|^request )(?:body|params|payload|query)$/.test(source)) return source;
  return `${source}.${key}`;
}

function isFunction(node) {
  return Boolean(node) && [
    'ArrowFunctionExpression', 'ClassMethod', 'ClassPrivateMethod', 'FunctionDeclaration', 'FunctionExpression', 'ObjectMethod'
  ].includes(node.type);
}

// The class a TypeScript annotation names, as in users: UsersService.
function typeName(annotation) {
  const type = annotation && annotation.type === 'TSTypeAnnotation' ? annotation.typeAnnotation : annotation;
  if (!type || type.type !== 'TSTypeReference') return '';
  const name = type.typeName;
  if (name.type === 'Identifier') return name.name;
  return name.type === 'TSQualifiedName' && name.left.type === 'Identifier' ? `${name.left.name}.${name.right.name}` : '';
}

// The class a value is an instance of, when it is built with new, as in new UsersService().
function constructed(node) {
  let value = node;
  while (value && (value.type === 'AwaitExpression' || value.type === 'TSAsExpression' || value.type === 'ParenthesizedExpression')) {
    value = value.argument || value.expression;
  }
  return value && value.type === 'NewExpression' ? dottedName(value.callee) : '';
}

function eachChild(node, visit) {
  for (const key of Object.keys(node)) {
    if (['loc', 'start', 'end', 'extra', 'leadingComments', 'trailingComments', 'innerComments', 'typeAnnotation', 'returnType', 'typeParameters'].includes(key)) continue;
    const value = node[key];
    if (Array.isArray(value)) {
      for (const item of value) if (item && typeof item.type === 'string') visit(item);
    } else if (value && typeof value.type === 'string') {
      visit(value);
    }
  }
}

function functionName(node, parent) {
  if (node.id && node.id.name) return node.id.name;
  if (node.key) return node.key.name || node.key.value || '<anonymous>';
  if (parent && parent.type === 'VariableDeclarator' && parent.id.name) return parent.id.name;
  if (parent && (parent.type === 'CallExpression' || parent.type === 'OptionalCallExpression')) {
    const route = parent.arguments[0];
    const label = dottedName(parent.callee);
    return route && route.type === 'StringLiteral' ? `${label} ${route.value}` : (label || '<anonymous>');
  }
  if (parent && parent.type === 'ExportDefaultDeclaration') return 'default';
  if (parent && parent.type === 'AssignmentExpression' && parent.right === node) {
    const target = dottedName(parent.left);
    if (target === 'module.exports' || target === 'exports') return 'default';
    if (/^(module\.)?exports\.\w+$/.test(target)) return target.split('.').pop();
  }
  return '<anonymous>';
}

class FileAnalyzer {
  constructor(ast, filename, fullPath = '', project = null, source = '') {
    this.ast = ast;
    this.source = source;
    // Values middleware stores on the request for later handlers, as in req.search = req.query.q.
    this.requestExtras = project ? project.requestExtras : new Map();
    this.serverActions = Boolean(ast.program && ast.program.directives.some(item => item.value.value === 'use server'));
    this.filename = filename;
    this.fullPath = fullPath;
    this.project = project;
    this.findings = new Map();
    this.functions = new Map();
    this.classes = new Map();
    this.methodOwners = new WeakMap();
    this.defaultExport = '';
    this.imports = new Map();
    this.routeLabels = new WeakMap();
    this.collectFunctions(ast);
    if (project) this.collectImports(ast);
  }

  // Records local names bound to other project files, as in import { build } from './sql'
  // or const sql = require('./sql').
  collectImports(root) {
    const add = (local, spec, name) => {
      const file = this.project.resolver.resolve(this.fullPath, spec);
      if (file && local) this.imports.set(local, { file, name });
    };
    const visit = node => {
      if (node.type === 'ImportDeclaration' && node.importKind !== 'type') {
        for (const item of node.specifiers) {
          if (item.type === 'ImportSpecifier') add(item.local.name, node.source.value, item.imported.name || item.imported.value);
          else if (item.type === 'ImportDefaultSpecifier') add(item.local.name, node.source.value, 'default');
          else if (item.type === 'ImportNamespaceSpecifier') add(item.local.name, node.source.value, '*');
        }
      } else if (node.type === 'VariableDeclarator' && node.init) {
        let init = node.init;
        let member = '';
        if (init.type === 'MemberExpression' && propertyName(init)) {
          member = propertyName(init);
          init = init.object;
        }
        if (init.type === 'CallExpression' && init.callee.type === 'Identifier' && init.callee.name === 'require'
            && init.arguments[0] && init.arguments[0].type === 'StringLiteral') {
          const spec = init.arguments[0].value;
          if (node.id.type === 'Identifier') add(node.id.name, spec, member || '*');
          else if (node.id.type === 'ObjectPattern' && !member) {
            for (const property of node.id.properties) {
              if (property.type === 'ObjectProperty' && property.value.type === 'Identifier') {
                add(property.value.name, spec, property.key.name || property.key.value);
              }
            }
          }
        }
      }
      eachChild(node, visit);
    };
    visit(root);
  }

  // Finds the function a call runs, in this file or in a project file it imports. A method call on a
  // value of a known class, as in this.users.find(id) or service.find(id), runs that class's method.
  findFunction(callee, walker = null) {
    return this.findMethod(callee, walker) || this.findPlain(callee);
  }

  findMethod(callee, walker) {
    if (walker && (callee.type === 'MemberExpression' || callee.type === 'OptionalMemberExpression')) {
      const method = propertyName(callee);
      const object = callee.object;
      let type = '';
      if (object.type === 'ThisExpression' && walker.owner) {
        const node = walker.owner.methods.get(method);
        if (node) return { file: this, node };
      } else if (object.type === 'MemberExpression' && object.object.type === 'ThisExpression' && walker.owner) {
        type = walker.owner.fields.get(propertyName(object)) || '';
      } else if (object.type === 'Identifier') {
        type = walker.types.get(object.name) || (/^[A-Z]/.test(object.name) ? object.name : '');
      } else {
        type = constructed(object);
      }
      const found = method && this.classFor(type);
      const node = found && found.info.methods.get(method);
      if (node) return { file: found.file, node };
    }
    return null;
  }

  findPlain(callee) {
    const local = name => (this.functions.has(name) ? { file: this, node: this.functions.get(name) } : null);
    const imported = (entry, name) => {
      const target = this.project && this.project.get(entry.file);
      const node = target && target.functions.get(name);
      return node ? { file: target, node } : null;
    };
    if (callee.type === 'Identifier') {
      const entry = this.imports.get(callee.name);
      if (entry) return entry.name === '*' ? imported(entry, 'default') : imported(entry, entry.name);
      return local(callee.name);
    }
    const name = dottedName(callee);
    const parts = name.split('.');
    if (parts.length === 2) {
      const entry = this.imports.get(parts[0]);
      if (entry && entry.name === '*') return imported(entry, parts[1]);
    }
    return name ? local(parts[parts.length - 1]) : null;
  }

  collectFunctions(root) {
    const visit = (node, parent) => {
      if (node.type === 'ObjectExpression') this.collectRouteLabel(node);
      if (node.type === 'ClassDeclaration' || node.type === 'ClassExpression') this.collectClass(node, parent);
      if (parent && parent.type === 'ExportDefaultDeclaration' && node.type === 'Identifier') this.defaultExport = node.name;
      if (node.type === 'AssignmentExpression' && node.right.type === 'Identifier'
          && ['module.exports', 'exports'].includes(dottedName(node.left))) this.defaultExport = node.right.name;
      if (isFunction(node)) {
        const name = functionName(node, parent);
        if (name && name !== '<anonymous>' && !this.functions.has(name)) this.functions.set(name, node);
      }
      eachChild(node, child => visit(child, node));
    };
    visit(root, null);
  }

  // Records a class's methods, and the classes its fields hold, as in
  // constructor(private readonly users: UsersService) or this.users = new UsersService().
  collectClass(node, parent) {
    const name = (node.id && node.id.name)
      || (parent && parent.type === 'VariableDeclarator' && parent.id.type === 'Identifier' ? parent.id.name : '');
    const info = { name, methods: new Map(), fields: new Map() };
    for (const member of node.body.body) {
      if ((member.type === 'ClassMethod' || member.type === 'ClassPrivateMethod') && member.key) {
        const key = member.key.name || member.key.value || (member.key.id && member.key.id.name);
        if (key) info.methods.set(key, member);
        this.methodOwners.set(member, info);
      } else if ((member.type === 'ClassProperty' || member.type === 'ClassPrivateProperty') && member.key) {
        const type = typeName(member.typeAnnotation) || constructed(member.value);
        const key = member.key.name || (member.key.id && member.key.id.name);
        if (type && key) info.fields.set(key, type);
      }
      if (member.type === 'ClassMethod' && member.kind === 'constructor') {
        const types = new Map();
        for (const parameter of member.params) {
          const target = parameter.type === 'TSParameterProperty' ? parameter.parameter : parameter;
          const inner = target.type === 'AssignmentPattern' ? target.left : target;
          const type = inner.type === 'Identifier' ? typeName(inner.typeAnnotation) : '';
          if (!type) continue;
          types.set(inner.name, type);
          if (parameter.type === 'TSParameterProperty') info.fields.set(inner.name, type);
        }
        for (const statement of member.body.body) {
          const expression = statement.type === 'ExpressionStatement' ? statement.expression : null;
          if (!expression || expression.type !== 'AssignmentExpression' || expression.left.type !== 'MemberExpression'
              || expression.left.object.type !== 'ThisExpression') continue;
          const key = propertyName(expression.left);
          const type = constructed(expression.right) || (expression.right.type === 'Identifier' ? types.get(expression.right.name) : '');
          if (key && type) info.fields.set(key, type);
        }
      }
    }
    if (name && !this.classes.has(name)) this.classes.set(name, info);
    if (parent && parent.type === 'ExportDefaultDeclaration') this.classes.set('default', info);
  }

  // The class a name refers to here: declared in this file, or imported from a project file.
  classFor(type) {
    if (!type) return null;
    const own = (file, name) => {
      const info = file.classes.get(name) || (name === 'default' && file.defaultExport ? file.classes.get(file.defaultExport) : null);
      return info ? { file, info } : null;
    };
    const imported = (entry, name) => {
      const target = this.project && this.project.get(entry.file);
      return target ? own(target, name) : null;
    };
    const parts = type.split('.');
    if (parts.length === 2) {
      const entry = this.imports.get(parts[0]);
      return entry && entry.name === '*' ? imported(entry, parts[1]) : null;
    }
    const entry = this.imports.get(type);
    if (entry) return imported(entry, entry.name === '*' ? 'default' : entry.name);
    return own(this, type);
  }

  collectRouteLabel(node) {
    const value = name => node.properties.find(item => item.key && (item.key.name || item.key.value) === name);
    // Hapi's { path, handler } and Fastify's route({ url, handler }).
    const route = value('path') || value('url');
    const handler = value('handler');
    if (route && handler && route.value.type === 'StringLiteral' && handler.value && isFunction(handler.value)) {
      this.routeLabels.set(handler.value, `route ${route.value.value}`);
    }
  }

  run() {
    const parents = [];
    const visit = (node, parent, inherited) => {
      if (isFunction(node)) {
        const scope = this.initialScope(node, inherited);
        this.addFrameworkInput(node, parents, scope);
        if (scope.tainted.size || scope.requests.size || scope.contexts.size) {
          this.analyze(node, scope, this.routeLabels.get(node) || functionName(node, parent), 0, new Set());
        }
        // Nested functions are handled while analyzing the function that contains them.
        if (scope.tainted.size || scope.requests.size || scope.contexts.size) return;
      }
      parents.push(node);
      eachChild(node, child => visit(child, node, inherited));
      parents.pop();
    };
    visit(this.ast, null, null);
    return [...this.findings.values()];
  }

  // Input that frameworks pass as plain arguments: GraphQL resolver arguments, tRPC input, Next.js
  // server action arguments and WebSocket messages.
  addFrameworkInput(node, parents, scope) {
    const parent = parents[parents.length - 1];
    const grand = parents[parents.length - 2];
    const greatGrand = parents[parents.length - 3];
    const params = node.params || [];
    const taintParameter = (parameter, label) => {
      const target = parameter && parameter.type === 'AssignmentPattern' ? parameter.left : parameter;
      if (!target) return;
      if (target.type === 'Identifier') scope.tainted.set(target.name, label);
      else if (target.type === 'ObjectPattern') {
        for (const property of target.properties) {
          const inner = property.type === 'ObjectProperty' ? (property.value.type === 'AssignmentPattern' ? property.value.left : property.value) : property.argument;
          const key = property.type === 'ObjectProperty' ? (property.key.name || property.key.value) : '';
          if (inner && inner.type === 'Identifier') scope.tainted.set(inner.name, key ? `${label}.${key}` : label);
        }
      }
    };
    // { Query: { user: (parent, args) => ... } }
    const property = parent && parent.type === 'ObjectProperty' && parent.value === node ? parent : (node.type === 'ObjectMethod' ? node : null);
    const owner = property && (node.type === 'ObjectMethod' ? parent : grand);
    const ownerProperty = node.type === 'ObjectMethod' ? grand : greatGrand;
    if (owner && owner.type === 'ObjectExpression' && ownerProperty && ownerProperty.type === 'ObjectProperty'
        && GRAPHQL_ROOTS.has(ownerProperty.key.name || ownerProperty.key.value)) {
      taintParameter(params[1], 'graphql args');
    }
    // t.procedure.input(schema).query(({ input }) => ...)
    if (parent && (parent.type === 'CallExpression' || parent.type === 'OptionalCallExpression') && parent.arguments.includes(node)) {
      const callee = parent.callee;
      const method = callee.type === 'MemberExpression' ? propertyName(callee) : '';
      const chain = this.source ? this.source.slice(callee.start, callee.end) : dottedName(callee);
      if (TRPC_PROCEDURES.has(method) && /procedure|\.input\(/i.test(chain)) {
        const first = params[0];
        if (first && first.type === 'ObjectPattern') {
          for (const item of first.properties) {
            if (item.type === 'ObjectProperty' && (item.key.name || item.key.value) === 'input') {
              const inner = item.value.type === 'AssignmentPattern' ? item.value.left : item.value;
              if (inner.type === 'Identifier') scope.tainted.set(inner.name, 'trpc input');
              else taintParameter(inner, 'trpc input');
            }
          }
        } else if (first && first.type === 'Identifier') {
          scope.tainted.set(`${first.name}.input`, 'trpc input');
        }
      }
      // ws.on('message', data => ...), socket.on('chat', payload => ...)
      const receiver = method ? dottedName(callee.object) : '';
      const event = parent.arguments[0];
      if (['on', 'once', 'addListener'].includes(method) && /(^|\.)(ws|wss|socket|io|client|conn|connection)$/i.test(receiver)
          && event && event.type === 'StringLiteral' && !SOCKET_EVENTS_IGNORED.has(event.value)) {
        for (const parameter of params) taintParameter(parameter, `websocket ${event.value}`);
      }
    }
    // ws.onmessage = event => ...
    if (parent && parent.type === 'AssignmentExpression' && parent.right === node && propertyName(parent.left) === 'onmessage') {
      taintParameter(params[0], 'websocket message');
    }
    // 'use server' functions, whose arguments come from the browser.
    const directives = node.body && node.body.type === 'BlockStatement' ? node.body.directives || [] : [];
    const serverAction = directives.some(item => item.value.value === 'use server')
      || (this.serverActions && parents.some(item => item.type === 'ExportNamedDeclaration' || item.type === 'ExportDefaultDeclaration')
          && !parents.some(item => isFunction(item)));
    if (serverAction) {
      for (const parameter of params) {
        const target = parameter.type === 'AssignmentPattern' ? parameter.left : parameter;
        if (target.type === 'Identifier' && (target.name === 'formData' || typeName(target.typeAnnotation) === 'FormData')) {
          scope.requests.add(target.name);
        } else {
          taintParameter(parameter, 'server action argument');
        }
      }
    }
  }

  initialScope(node, inherited) {
    const scope = {
      tainted: new Map(inherited ? inherited.tainted : []),
      requests: new Set(inherited ? inherited.requests : []),
      contexts: new Set(inherited ? inherited.contexts : [])
    };
    for (const parameter of node.params || []) this.addParameter(parameter, scope);
    return scope;
  }

  addParameter(parameter, scope) {
    let target = parameter;
    if (target.type === 'TSParameterProperty') target = target.parameter;
    if (target.type === 'AssignmentPattern') target = target.left;
    const decorators = parameter.decorators || target.decorators || [];
    const decorator = decorators.find(item => NEST_INPUT_DECORATORS.has(dottedName(item.expression)));
    if (decorator && target.type === 'Identifier') {
      const pipes = (decorator.expression.arguments || []).map(item => dottedName(item.type === 'NewExpression' ? item.callee : item));
      if (pipes.some(pipe => NEST_SAFE_PIPES.has(pipe))) return;
      const key = decorator.expression.arguments && decorator.expression.arguments[0];
      const nest = dottedName(decorator.expression);
      scope.tainted.set(target.name, key && key.type === 'StringLiteral' ? `@${nest} ${key.value}` : `@${nest}`);
      return;
    }
    if (target.type === 'Identifier') {
      if (REQUEST_NAMES.has(target.name)) scope.requests.add(target.name);
      if (CONTEXT_NAMES.has(target.name)) scope.contexts.add(target.name);
      return;
    }
    if (target.type === 'ObjectPattern') {
      // Adonis and some Next.js handlers destructure, as in ({ request, params }).
      for (const property of target.properties) {
        if (property.type !== 'ObjectProperty') continue;
        const key = property.key.name || property.key.value;
        const value = property.value.type === 'AssignmentPattern' ? property.value.left : property.value;
        if (value.type !== 'Identifier') continue;
        if (key === 'request' || key === 'req') scope.requests.add(value.name);
        else if (['body', 'params', 'query', 'headers', 'cookies', 'payload'].includes(key)) {
          scope.tainted.set(value.name, `request ${key}`);
        }
      }
    }
  }

  // Walks a function and returns the input source its return value carries, if any. A helper
  // walked before with the same input is not walked again: its findings are recorded again under
  // the new caller, so large projects with shared helpers stay fast.
  analyze(node, scope, label, depth, active, entry = '') {
    const key = `${this.filename}:${node.start}`;
    if (active.has(key) || depth > MAX_HELPER_DEPTH) return null;
    const summaryKey = depth > 0 ? JSON.stringify([key, depth, [...scope.tainted].sort(), [...scope.requests].sort(),
      [...scope.contexts].sort()]) : null;
    const summaries = this.project ? this.project.summaries : (this.summaries = this.summaries || new Map());
    if (summaryKey && summaries.has(summaryKey)) {
      const summary = summaries.get(summaryKey);
      for (const item of summary.recorded) {
        item.file.record(item.node, label + item.label.slice(summary.label.length), item.sink, item.call, item.source, item.confidence, entry);
      }
      return summary.returned;
    }
    const recorded = [];
    const recordings = this.project ? this.project.recordings : (this.recordings = this.recordings || []);
    if (summaryKey) recordings.push(recorded);
    const next = new Set(active);
    next.add(key);
    const walker = new FunctionWalker(this, scope, label, depth, next, entry);
    walker.owner = this.methodOwners.get(node) || null;
    walker.addParameterTypes(node);
    try {
      walker.visit(node.body, 0);
      if (node.body.type !== 'BlockStatement') walker.returned = walker.returned || walker.taint(node.body);
    } finally {
      if (summaryKey) recordings.splice(recordings.indexOf(recorded), 1);
    }
    if (summaryKey) summaries.set(summaryKey, { label, recorded, returned: walker.returned });
    return walker.returned;
  }

  // entry is the file of the handler that called into this one, when it is another file.
  record(node, label, sink, call, source, confidence, entry = '') {
    const recordings = this.project ? this.project.recordings : (this.recordings || []);
    for (const recorded of recordings) recorded.push({ file: this, node, label, sink, call, source, confidence });
    const kinds = safeKinds(source);
    if (kinds.includes(sink) || kinds.includes('*')) return;
    source = baseSource(source);
    const line = node.loc ? node.loc.start.line : 0;
    const key = `${line}:${sink}:${call}:${label.split(' -> ')[0]}:${entry}`;
    if (!this.findings.has(key)) {
      const finding = { file: this.filename, line, function: label, sink, call, source, confidence };
      if (entry) finding.entry_file = entry;
      this.findings.set(key, finding);
    }
  }
}

class FunctionWalker {
  constructor(file, scope, label, depth, active, entry = '') {
    this.file = file;
    this.scope = scope;
    this.label = label;
    this.depth = depth;
    this.active = active;
    this.entry = entry;
    this.returned = null;
    this.helperReturns = new WeakMap();
    // The class of the method being walked, and the classes of local values, for method calls.
    this.owner = null;
    this.types = new Map();
    // Values known before running, objects followed by literal key and arrays by position.
    this.constants = new Map();
    this.keyed = new Map();
    this.lists = new Map();
    // What objects built here keep from their constructor: this.<field> input and the request.
    this.fieldInput = new Map();
    this.fieldRequests = new Map();
    // Callbacks called with each value of a tainted receiver, as in items.forEach(item => ...).
    this.callbackInput = new WeakMap();
    // Objects made with Object.create(null), which have no prototype to pollute.
    this.bare = new Set();
  }

  constant(node) {
    if (!node) return UNKNOWN;
    switch (node.type) {
      case 'StringLiteral':
      case 'NumericLiteral':
      case 'BooleanLiteral':
        return node.value;
      case 'NullLiteral':
        return null;
      case 'Identifier':
        return this.constants.has(node.name) ? this.constants.get(node.name) : UNKNOWN;
      case 'ParenthesizedExpression':
        return this.constant(node.expression);
      case 'UnaryExpression': {
        const value = this.constant(node.argument);
        if (value === UNKNOWN) return UNKNOWN;
        if (node.operator === '!') return !value;
        if (node.operator === '-' && typeof value === 'number') return -value;
        return UNKNOWN;
      }
      case 'LogicalExpression': {
        const left = this.constant(node.left);
        const right = this.constant(node.right);
        if (left === UNKNOWN || right === UNKNOWN) return UNKNOWN;
        return node.operator === '&&' ? left && right : node.operator === '||' ? left || right : left ?? right;
      }
      case 'ConditionalExpression': {
        const test = this.constant(node.test);
        return test === UNKNOWN ? UNKNOWN : this.constant(test ? node.consequent : node.alternate);
      }
      case 'MemberExpression': {
        const object = this.constant(node.object);
        const index = node.computed ? this.constant(node.property) : UNKNOWN;
        return typeof object === 'string' && typeof index === 'number' && index >= 0 && index < object.length ? object[index] : UNKNOWN;
      }
      case 'CallExpression': {
        if (node.callee.type === 'MemberExpression' && propertyName(node.callee) === 'charAt') {
          const object = this.constant(node.callee.object);
          const index = this.constant(node.arguments[0]);
          return typeof object === 'string' && typeof index === 'number' ? object.charAt(index) : UNKNOWN;
        }
        return UNKNOWN;
      }
      case 'BinaryExpression': {
        const left = this.constant(node.left);
        const right = this.constant(node.right);
        if (left === UNKNOWN || right === UNKNOWN) return UNKNOWN;
        const operations = {
          '+': () => left + right, '-': () => left - right, '*': () => left * right, '/': () => (right === 0 ? UNKNOWN : left / right),
          '%': () => (right === 0 ? UNKNOWN : left % right), '<': () => left < right, '<=': () => left <= right, '>': () => left > right,
          '>=': () => left >= right, '===': () => left === right, '!==': () => left !== right, '==': () => left == right, // eslint-disable-line eqeqeq
          '!=': () => left != right // eslint-disable-line eqeqeq
        };
        return operations[node.operator] ? operations[node.operator]() : UNKNOWN;
      }
      default:
        return UNKNOWN;
    }
  }

  // The tainted names a condition shows are safe when it is outcome, and what for. A check that
  // returns when it fails is read with outcome false after it; if (ALLOWED.includes(x)) with true inside.
  validated(test, outcome) {
    const names = node => {
      const found = new Set();
      const walk = item => {
        if (!item || typeof item.type !== 'string') return;
        if (item.type === 'Identifier' && this.scope.tainted.has(item.name)) found.add(item.name);
        eachChild(item, walk);
      };
      walk(node);
      return found;
    };
    const all = (node, kinds) => new Map([...names(node)].map(name => [name, kinds]));
    const merge = parts => {
      const found = new Map();
      for (const part of parts) for (const [name, kinds] of part) found.set(name, [...new Set([...(found.get(name) || []), ...kinds])]);
      return found;
    };
    if (!test) return new Map();
    if (test.type === 'UnaryExpression' && test.operator === '!') return this.validated(test.argument, !outcome);
    if (test.type === 'ParenthesizedExpression') return this.validated(test.expression, outcome);
    if (test.type === 'LogicalExpression' && test.operator !== '??') {
      return (test.operator === '&&') === outcome ? merge([this.validated(test.left, outcome), this.validated(test.right, outcome)]) : new Map();
    }
    if (test.type === 'BinaryExpression' && ['===', '==', '!==', '!='].includes(test.operator)) {
      const equal = test.operator.startsWith('=') === outcome;
      for (const [subject, other] of [[test.left, test.right], [test.right, test.left]]) {
        if (subject.type === 'UnaryExpression' && subject.operator === 'typeof' && other.type === 'StringLiteral'
            && equal && ['number', 'boolean'].includes(other.value)) return all(subject.argument, ['*']);
        if (equal && ['StringLiteral', 'NumericLiteral', 'BooleanLiteral'].includes(other.type)) return all(subject, ['*']);
      }
      return new Map();
    }
    if (test.type !== 'CallExpression' && test.type !== 'OptionalCallExpression') return new Map();
    const name = dottedName(test.callee);
    const method = test.callee.type === 'MemberExpression' ? propertyName(test.callee) : '';
    const receiver = method ? test.callee.object : null;
    const argument = test.arguments[0];
    if (method === 'test' && outcome && receiver && (receiver.type === 'RegExpLiteral' || /(re|regex|pattern)$/i.test(dottedName(receiver)))) {
      return all(argument, ['*']);
    }
    if ((method === 'includes' || method === 'has') && receiver && !names(receiver).size && outcome) {
      return all(argument, ['*']);  // One of a fixed set of allowed values.
    }
    if (method === 'includes' && receiver && names(receiver).size && argument && argument.type === 'StringLiteral' && !outcome) {
      // A character or substring that isn't there: '..' and slashes only make a path safe.
      return all(receiver, /\.\.|\/|\\/.test(argument.value) ? ['path'] : ['*']);
    }
    if (outcome && (VALIDATOR_CHECKS.test(name) || ['Number.isInteger', 'Number.isSafeInteger', 'isUUID'].includes(name))) {
      return all(argument, ['*']);
    }
    return new Map();
  }

  // Marks names as checked, as their taint carries which operations they're safe for.
  check(checks) {
    const before = new Map();
    for (const [name, kinds] of checks) {
      const source = this.scope.tainted.get(name);
      if (!source) continue;
      before.set(name, source);
      this.scope.tainted.set(name, safeSource(source, kinds));
    }
    return before;
  }

  // Records what an object built from one of the project's classes keeps from its constructor.
  remember(name, value) {
    this.fieldInput.delete(name);
    this.fieldRequests.delete(name);
    if (!value || value.type !== 'NewExpression') return;
    const found = this.file.classFor(dottedName(value.callee));
    const constructor = found && found.info.methods.get('constructor');
    if (!constructor) return;
    const inputs = new Map();
    const requests = new Set();
    constructor.params.forEach((parameter, index) => {
      const target = parameter.type === 'TSParameterProperty' ? parameter.parameter : parameter;
      const inner = target.type === 'AssignmentPattern' ? target.left : target;
      if (inner.type !== 'Identifier') return;
      const argument = value.arguments[index];
      const source = this.taint(argument);
      if (source) inputs.set(inner.name, source);
      else if (argument && this.scope.requests.has(dottedName(argument))) requests.add(inner.name);
      if (parameter.type === 'TSParameterProperty' && source) inputs.set(`=${inner.name}`, source);
      if (parameter.type === 'TSParameterProperty' && requests.has(inner.name)) requests.add(`=${inner.name}`);
    });
    const fields = new Map();
    const requestFields = new Set();
    for (const [key, source] of inputs) if (key.startsWith('=')) fields.set(key.slice(1), source);
    for (const key of requests) if (key.startsWith('=')) requestFields.add(key.slice(1));
    for (const statement of constructor.body.body) {
      const expression = statement.type === 'ExpressionStatement' ? statement.expression : null;
      if (!expression || expression.type !== 'AssignmentExpression' || expression.left.type !== 'MemberExpression'
          || expression.left.object.type !== 'ThisExpression' || expression.right.type !== 'Identifier') continue;
      const field = propertyName(expression.left);
      if (inputs.has(expression.right.name)) fields.set(field, inputs.get(expression.right.name));
      else if (requests.has(expression.right.name)) requestFields.add(field);
    }
    if (fields.size) this.fieldInput.set(name, fields);
    if (requestFields.size) this.fieldRequests.set(name, requestFields);
  }

  // Follows a known object or array through one method call; false when it isn't one.
  trackCollection(node) {
    if (node.callee.type !== 'MemberExpression' || node.callee.object.type !== 'Identifier') return false;
    const name = node.callee.object.name;
    const method = propertyName(node.callee);
    const items = this.lists.get(name);
    if (items) {
      if (method === 'push') items.push(...node.arguments.map(argument => this.taint(argument)));
      else if (method === 'unshift') items.unshift(...node.arguments.map(argument => this.taint(argument)));
      else if (method === 'shift') items.shift();
      else if (method === 'pop') items.pop();
      else if (['splice', 'fill', 'copyWithin', 'reverse', 'sort'].includes(method)) this.degrade(name);
      return true;
    }
    const values = this.keyed.get(name);
    if (values && values.isMap) {
      const key = this.constant(node.arguments[0]);
      if (method === 'set' && key !== UNKNOWN) values.set(key, this.taint(node.arguments[1]));
      else if (method === 'delete' && key !== UNKNOWN) values.delete(key);
      else if (method === 'set' || method === 'clear') this.degrade(name);
      return true;
    }
    return false;
  }

  degrade(name) {
    const values = [...(this.keyed.get(name) || new Map()).values(), ...(this.lists.get(name) || [])];
    this.keyed.delete(name);
    this.lists.delete(name);
    const source = values.reduce((found, value) => combine(found, value), null);
    if (source) this.scope.tainted.set(name, source);
  }

  read(name, key) {
    if (key === UNKNOWN) return UNKNOWN;
    const values = this.keyed.get(name);
    if (values) return values.has(key) ? values.get(key) : null;
    const items = this.lists.get(name);
    if (items && typeof key === 'number') return key >= 0 && key < items.length ? items[key] : null;
    return UNKNOWN;
  }

  addParameterTypes(node) {
    for (const parameter of node.params || []) {
      const target = parameter.type === 'TSParameterProperty' ? parameter.parameter : parameter;
      const inner = target.type === 'AssignmentPattern' ? target.left : target;
      const type = inner.type === 'Identifier' ? typeName(inner.typeAnnotation) : '';
      if (type) this.types.set(inner.name, type);
    }
  }

  visit(node, nesting) {
    if (!node || typeof node.type !== 'string') return;
    switch (node.type) {
      case 'VariableDeclarator':
        if (node.id.type === 'Identifier') {
          const type = constructed(node.init) || typeName(node.id.typeAnnotation);
          if (type) this.types.set(node.id.name, type);
        }
        if (node.init) this.visit(node.init, nesting);
        if (node.id.type === 'Identifier') this.track(node.id.name, node.init, nesting);
        this.bind(node.id, node.init ? this.taint(node.init) : null, nesting, node.init);
        return;
      case 'AssignmentExpression':
        this.visit(node.right, nesting);
        if (node.left.type === 'Identifier') {
          this.constants.delete(node.left.name);
          if (node.operator === '=') this.track(node.left.name, node.right, nesting);
        }
        if (this.assignItem(node)) return;
        this.checkAssignment(node);
        if (node.operator === '=') this.bind(node.left, this.taint(node.right), nesting, node.right);
        else {
          const source = this.taint(node.right);
          if (source) this.bind(node.left, combine(this.taint(node.left), source), nesting, node.right);
        }
        return;
      case 'UpdateExpression':
        if (node.argument.type === 'Identifier') this.constants.delete(node.argument.name);
        return;
      case 'CallExpression':
      case 'OptionalCallExpression':
      case 'NewExpression':
        this.checkCall(node);
        this.followHelper(node);
        if (node.type !== 'NewExpression' && this.trackCollection(node)) {
          eachChild(node, child => this.visit(child, nesting));
          return;
        }
        this.collect(node, nesting);
        // A callback called with each value of a tainted receiver gets that value.
        if (node.type !== 'NewExpression' && node.callee.type === 'MemberExpression' && CALLBACK_METHODS.has(propertyName(node.callee))) {
          const source = this.taint(node.callee.object);
          const callback = node.arguments[0];
          if (source && callback && isFunction(callback)) this.callbackInput.set(callback, source);
        }
        eachChild(node, child => this.visit(child, nesting));
        return;
      case 'IfStatement': {
        this.visit(node.test, nesting);
        const known = this.constant(node.test);
        if (known !== UNKNOWN) {
          // Only one branch can run, and it always does.
          this.visit(known ? node.consequent : node.alternate, nesting);
          return;
        }
        const failed = this.validated(node.test, false);
        if (failed.size && !node.alternate && exits(node.consequent)) {
          // A check that leaves when it fails: what follows only sees values that passed.
          this.visit(node.consequent, nesting + 1);
          this.check(failed);
          return;
        }
        const passed = this.validated(node.test, true);
        if (passed.size && !exits(node.consequent)) {
          const before = this.check(passed);
          this.visit(node.consequent, nesting + 1);
          for (const [name, source] of before) {
            if (this.scope.tainted.get(name) === safeSource(source, passed.get(name))) this.scope.tainted.set(name, source);
          }
          this.visit(node.alternate, nesting + 1);
          return;
        }
        this.visit(node.consequent, nesting + 1);
        this.visit(node.alternate, nesting + 1);
        return;
      }
      case 'SwitchStatement': {
        this.visit(node.discriminant, nesting);
        const selector = this.constant(node.discriminant);
        const labels = node.cases.map(item => (item.test ? this.constant(item.test) : 'default'));
        if (selector === UNKNOWN || labels.includes(UNKNOWN)) {
          for (const item of node.cases) this.visit(item, nesting + 1);
          return;
        }
        let start = labels.findIndex(label => label !== 'default' && label === selector);
        if (start < 0) start = node.cases.findIndex(item => !item.test);
        // Only the matching case runs, through to its break.
        for (let index = start; start >= 0 && index < node.cases.length; index += 1) {
          for (const statement of node.cases[index].consequent) this.visit(statement, nesting);
          if (node.cases[index].consequent.some(statement => statement.type === 'BreakStatement' || exits(statement))) break;
        }
        return;
      }
      case 'ReturnStatement':
        if (node.argument) {
          this.visit(node.argument, nesting);
          this.returned = this.returned || this.taint(node.argument);
        }
        return;
      case 'TaggedTemplateExpression':
        eachChild(node, child => this.visit(child, nesting));
        return;
      case 'ForOfStatement':
      case 'ForInStatement': {
        for (const name of assignedNames(node.body)) this.constants.delete(name);
        const source = this.taint(node.right);
        const target = node.left.type === 'VariableDeclaration' ? node.left.declarations[0].id : node.left;
        if (source) this.bind(target, source, nesting + 1, node.right);
        this.visit(node.right, nesting);
        this.visit(node.body, nesting + 1);
        return;
      }
      case 'WhileStatement':
      case 'DoWhileStatement':
      case 'ForStatement':
        // A loop may change what it assigns, so none of it stays known inside or after it.
        for (const name of assignedNames(node)) this.constants.delete(name);
        eachChild(node, child => this.visit(child, nesting + 1));
        return;
      case 'TryStatement':
      case 'ConditionalExpression':
        eachChild(node, child => this.visit(child, nesting + 1));
        return;
      default:
        if (isFunction(node)) {
          // Callbacks keep the request and values from the enclosing handler.
          const nested = this.file.initialScope(node, this.scope);
          const input = this.callbackInput.get(node);
          const first = node.params[0];
          if (input && first) {
            if (first.type === 'Identifier') nested.tainted.set(first.name, input);
            else this.bindInto(nested, first, input);
          }
          const walker = new FunctionWalker(this.file, nested, this.label, this.depth, this.active, this.entry);
          walker.owner = this.owner;
          walker.types = new Map(this.types);
          walker.constants = new Map(this.constants);
          walker.keyed = this.keyed;
          walker.lists = this.lists;
          walker.addParameterTypes(node);
          walker.visit(node.body, 0);
          return;
        }
        eachChild(node, child => this.visit(child, nesting));
    }
  }

  bind(target, source, nesting, value) {
    if (!target) return;
    if (target.type === 'Identifier') {
      if (source) this.scope.tainted.set(target.name, source);
      else if (nesting === 0) this.scope.tainted.delete(target.name);
      if (value && CONTEXT_NAMES.size && dottedName(value)) {
        const name = dottedName(value);
        if (this.scope.contexts.has(name.split('.')[0]) && name.endsWith('.request')) this.scope.requests.add(target.name);
      }
      return;
    }
    if (target.type === 'ObjectPattern') {
      for (const property of target.properties) {
        if (property.type === 'RestElement') this.bind(property.argument, source, nesting);
        else if (property.type === 'ObjectProperty') {
          const inner = property.value.type === 'AssignmentPattern' ? property.value.left : property.value;
          if (source) this.bind(inner, withKey(source, property.key.name || property.key.value), nesting);
          else if (value && value.type === 'Identifier' && (this.scope.contexts.has(value.name) || this.scope.requests.has(value.name))) {
            const key = property.key.name || property.key.value;
            if (key === 'request' && inner.type === 'Identifier') this.scope.requests.add(inner.name);
          }
        }
      }
      return;
    }
    if (target.type === 'ArrayPattern') {
      for (const element of target.elements) if (element) this.bind(element, source, nesting);
      return;
    }
    if (target.type === 'AssignmentPattern') {
      this.bind(target.left, source, nesting);
      return;
    }
    if ((target.type === 'MemberExpression' || target.type === 'OptionalMemberExpression') && source) {
      const name = dottedName(target.object);
      const property = propertyName(target);
      if (name && this.scope.requests.has(name) && property && !REQUEST_PROPERTIES.has(property)) {
        // Middleware keeping input on the request for later handlers.
        if (!this.file.requestExtras.has(property)) this.file.requestExtras.set(property, `${name}.${property} (${baseSource(source)})`);
        return;
      }
      if (name) this.scope.tainted.set(name.split('.')[0], source);
    }
  }

  taint(node) {
    if (!node) return null;
    switch (node.type) {
      case 'Identifier': {
        const source = this.scope.tainted.get(node.name);
        if (source) return source;
        const values = [...(this.keyed.get(node.name) || new Map()).values(), ...(this.lists.get(node.name) || [])];
        return values.reduce((found, value) => combine(found, value), null);
      }
      case 'MemberExpression':
      case 'OptionalMemberExpression': {
        const whole = dottedName(node);
        if (whole && this.scope.tainted.has(whole)) return this.scope.tainted.get(whole);
        const source = this.requestProperty(node);
        if (source) return source;
        if (node.object.type === 'Identifier') {
          const item = this.read(node.object.name, node.computed ? this.constant(node.property) : propertyName(node));
          if (item !== UNKNOWN) return item;
        }
        return withKey(this.taint(node.object), propertyName(node));
      }
      case 'TemplateLiteral':
        return this.first(node.expressions);
      case 'BinaryExpression':
        return node.operator === '+' ? combine(this.taint(node.left), this.taint(node.right)) : null;
      case 'LogicalExpression':
        return combine(this.taint(node.left), this.taint(node.right));
      case 'ConditionalExpression': {
        const known = this.constant(node.test);
        if (known !== UNKNOWN) return this.taint(known ? node.consequent : node.alternate);
        return combine(this.taint(node.consequent), this.taint(node.alternate));
      }
      case 'AwaitExpression':
        return this.taint(node.argument);
      case 'TSAsExpression':
      case 'TSNonNullExpression':
      case 'TSSatisfiesExpression':
      case 'ParenthesizedExpression':
        return this.taint(node.expression);
      case 'SpreadElement':
        return this.taint(node.argument);
      case 'ArrayExpression':
        return this.first(node.elements);
      case 'ObjectExpression':
        return this.first(node.properties.map(property => property.value || property.argument));
      case 'CallExpression':
      case 'OptionalCallExpression':
        return this.callTaint(node);
      case 'NewExpression':
        return dottedName(node.callee) === 'String' || dottedName(node.callee) === 'URL' ? this.first(node.arguments) : null;
      default:
        return null;
    }
  }

  requestProperty(node) {
    const name = dottedName(node.object);
    const property = propertyName(node);
    if (!name || !property) return null;
    if (this.scope.requests.has(name) && REQUEST_PROPERTIES.has(property)) return `${name}.${property}`;
    if (this.scope.requests.has(name) && this.file.requestExtras.has(property)) {
      return this.file.requestExtras.get(property);
    }
    if (this.scope.contexts.has(name) && CONTEXT_PROPERTIES.has(property) && property !== 'request') return `${name}.${property}`;
    const parts = name.split('.');
    if (parts.length === 2 && this.scope.contexts.has(parts[0]) && parts[1] === 'request' && REQUEST_PROPERTIES.has(property)) {
      return `${name}.${property}`;
    }
    return null;
  }

  first(nodes) {
    let found = null;
    for (const node of nodes) found = combine(found, this.taint(node));
    return found;
  }

  // Records known values, and objects and arrays to follow by key and position.
  track(name, value, nesting) {
    const known = this.constant(value);
    if (known !== UNKNOWN && nesting === 0 && known !== null && typeof known !== 'object') this.constants.set(name, known);
    else this.constants.delete(name);
    this.keyed.delete(name);
    this.lists.delete(name);
    this.bare.delete(name);
    this.remember(name, value);
    if (!value) return;
    if (value.type === 'ObjectExpression' && value.properties.every(item => item.type === 'ObjectProperty' && !item.computed)) {
      this.keyed.set(name, new Map(value.properties.map(item => [item.key.name || item.key.value, this.taint(item.value)])));
    } else if (value.type === 'ArrayExpression' && value.elements.every(item => item && item.type !== 'SpreadElement')) {
      this.lists.set(name, value.elements.map(item => this.taint(item)));
    } else if (value.type === 'NewExpression' && dottedName(value.callee) === 'Map' && !value.arguments.length) {
      const values = new Map();
      values.isMap = true;
      this.keyed.set(name, values);
    } else if (value.type === 'CallExpression' && dottedName(value.callee) === 'Object.create'
               && value.arguments[0] && value.arguments[0].type === 'NullLiteral') {
      this.bare.add(name);
    }
  }

  // values.key = x or items[0] = x on a followed object or array.
  assignItem(node) {
    if (node.operator !== '=' || node.left.type !== 'MemberExpression' || node.left.object.type !== 'Identifier') return false;
    const name = node.left.object.name;
    const key = node.left.computed ? this.constant(node.left.property) : propertyName(node.left);
    const values = this.keyed.get(name);
    if (values && !values.isMap && key !== UNKNOWN) {
      values.set(key, this.taint(node.right));
      return true;
    }
    const items = this.lists.get(name);
    if (items && typeof key === 'number' && key >= 0 && key <= items.length) {
      items[key] = this.taint(node.right);
      return true;
    }
    if (values || items) this.degrade(name);
    return false;
  }

  bindInto(scope, pattern, source) {
    if (pattern.type === 'Identifier') scope.tainted.set(pattern.name, source);
    else if (pattern.type === 'ObjectPattern') {
      for (const property of pattern.properties) {
        const inner = property.type === 'ObjectProperty' ? (property.value.type === 'AssignmentPattern' ? property.value.left : property.value) : property.argument;
        if (inner) this.bindInto(scope, inner, property.type === 'ObjectProperty' ? withKey(source, property.key.name || property.key.value) : source);
      }
    } else if (pattern.type === 'ArrayPattern') {
      for (const element of pattern.elements) if (element) this.bindInto(scope, element, source);
    } else if (pattern.type === 'AssignmentPattern') this.bindInto(scope, pattern.left, source);
  }

  callTaint(node) {
    const name = dottedName(node.callee);
    if (SANITIZERS.has(name)) return null;
    if (KIND_SANITIZERS.has(name)) return safeSource(this.first(node.arguments), KIND_SANITIZERS.get(name));
    if (PASSTHROUGH_FUNCTIONS.has(name)) return this.first(node.arguments);
    const method = node.callee.type === 'MemberExpression' ? propertyName(node.callee) : '';
    if (['parse', 'parseAsync', 'validateSync'].includes(method) && this.file.source
        && SAFE_SCHEMA.test(this.file.source.slice(node.callee.object.start, node.callee.object.end))
        && !/\b(?:object|string|array|any|unknown|record)\(/.test(this.file.source.slice(node.callee.object.start, node.callee.object.end).replace(/string\(\)\.(?:uuid|email|datetime|date)\(/g, ''))) {
      return null;  // A schema that only lets numbers, booleans, UUIDs or fixed values through.
    }
    if (['parse', 'parseAsync', 'safeParse', 'validate', 'validateAsync', 'validateSync'].includes(method)
        && node.callee.object.type !== 'Identifier' || (['parse', 'parseAsync', 'validateSync'].includes(method)
        && /schema$/i.test(dottedName(node.callee.object)))) {
      const value = this.first(node.arguments);
      if (value) return value;  // Validated against a schema that still lets text through.
    }
    if (node.callee.type === 'MemberExpression' && node.callee.object.type === 'Identifier' && ['get', 'at'].includes(method)) {
      const item = this.read(node.callee.object.name, method === 'at' ? this.constant(node.arguments[0]) : this.constant(node.arguments[0]));
      if (item !== UNKNOWN) return item;
    }
    if (node.callee.type === 'MemberExpression' || node.callee.type === 'OptionalMemberExpression') {
      const receiver = dottedName(node.callee.object);
      const method = propertyName(node.callee);
      const parts = receiver.split('.');
      const isRequest = this.scope.requests.has(receiver)
        || (parts.length === 2 && this.scope.contexts.has(parts[0]) && parts[1] === 'request');
      if (isRequest && REQUEST_METHODS.has(method)) {
        const key = node.arguments[0];
        return key && key.type === 'StringLiteral' ? `${receiver}.${method}('${key.value}')` : `${receiver}.${method}()`;
      }
      if (PASSTHROUGH_METHODS.has(method)) {
        return this.taint(node.callee.object) || (method === 'concat' || method === 'replace' || method === 'join' ? this.first(node.arguments) : null);
      }
      // URLSearchParams, FormData and Headers read back what they hold.
      if (['get', 'getAll'].includes(method) && this.taint(node.callee.object)) {
        const key = node.arguments[0];
        return withKey(this.taint(node.callee.object), key && key.type === 'StringLiteral' ? key.value : '');
      }
    }
    return this.followHelper(node);
  }

  // A value added to an array, set or map taints the collection, as in parts.push(req.query.q).
  collect(node, nesting) {
    if (node.type === 'NewExpression' || (node.callee.type !== 'MemberExpression' && node.callee.type !== 'OptionalMemberExpression')) return;
    if (!COLLECTION_ADDS.has(propertyName(node.callee))) return;
    const source = this.first(node.arguments);
    if (source) this.bind(node.callee.object, source, nesting, null);
  }

  // Walks a project function that receives input, and returns the input its result carries.
  followHelper(node) {
    if (this.helperReturns.has(node)) return this.helperReturns.get(node);
    this.helperReturns.set(node, null);
    if (this.depth >= MAX_HELPER_DEPTH || node.type === 'NewExpression') return null;
    const helper = this.file.findFunction(node.callee, this);
    if (!helper) return null;
    const passed = new Map();
    (helper.node.params || []).forEach((parameter, index) => {
      const target = parameter.type === 'AssignmentPattern' ? parameter.left : parameter;
      const source = this.taint(node.arguments[index]);
      if (source && target.type === 'Identifier') passed.set(target.name, source);
    });
    const requests = new Set();
    // An object's method also reads what its constructor kept, as this.<field>.
    if (node.callee.type === 'MemberExpression' && node.callee.object.type === 'Identifier') {
      const variable = node.callee.object.name;
      for (const [field, source] of this.fieldInput.get(variable) || []) passed.set(`this.${field}`, source);
      for (const field of this.fieldRequests.get(variable) || []) requests.add(`this.${field}`);
    }
    if (!passed.size && !requests.size) return null;
    const scope = { tainted: passed, requests, contexts: new Set() };
    const name = propertyName(node.callee) || dottedName(node.callee).split('.').pop();
    const entry = helper.file === this.file ? this.entry : (this.entry || this.file.filename);
    const returned = helper.file.analyze(helper.node, scope, `${this.label} -> ${name}`, this.depth + 1, this.active, entry);
    this.helperReturns.set(node, returned);
    return returned;
  }

  checkCall(node) {
    const name = dottedName(node.callee);
    const method = node.callee.type === 'MemberExpression' || node.callee.type === 'OptionalMemberExpression'
      ? propertyName(node.callee) : '';
    const receiver = method ? dottedName(node.callee.object) : '';
    const first = node.arguments[0];
    const source = this.taint(first);

    // A method of one of the project's classes is followed into, not guessed at by its name.
    if (method && this.file.findMethod(node.callee, this)) return;
    if (name === 'RegExp' && source) {
      this.report(node, 'redos', node.type === 'NewExpression' ? 'new RegExp' : 'RegExp', source, 'medium');
      return;
    }
    if (node.type === 'NewExpression') {
      if (name === 'Function') {
        const code = this.first(node.arguments);
        if (code) this.report(node, 'code', 'new Function', code, 'high');
      } else if ((name === 'vm.Script' || name === 'Script') && source) {
        this.report(node, 'code', `new ${name}`, source, 'high');
      } else if (name === 'Worker' && source && node.arguments.some(argument => argument.type === 'ObjectExpression'
                 && argument.properties.some(item => item.key && item.key.name === 'eval' && item.value.value === true))) {
        this.report(node, 'code', 'new Worker', source, 'high');
      }
      return;
    }
    if (node.callee.type === 'Import' && source) {
      this.report(node, 'code', 'import()', source, 'high');
      return;
    }
    const head = receiver.split('.')[0];
    if (method && XSS_METHODS.has(method) && RESPONSE_NAMES.has(head) && source && this.isText(first, source)) {
      this.report(node, 'xss', name || `res.${method}`, source, 'medium');
      return;
    }
    if (method && (method === 'redirect' || method === 'location') && (RESPONSE_NAMES.has(head) || this.scope.contexts.has(head) || head === 'reply')) {
      const target = this.taint(node.arguments[node.arguments.length - 1]);
      if (target) this.report(node, 'redirect', name || method, target, 'medium');
      return;
    }
    if (MERGE_FUNCTIONS.has(name) || (method && ['merge', 'mergeWith', 'defaultsDeep', 'set', 'setWith'].includes(method) && /^(_|lodash)$/.test(receiver))) {
      const value = this.first(node.arguments.slice(1));
      if (value && (name !== 'set' && method !== 'set' || this.taint(node.arguments[1]))) {
        this.report(node, 'prototype', name || method, value, 'medium');
      }
      return;
    }
    if ((XPATH_FUNCTIONS.has(name) || (method === 'evaluate' && /(doc|document|xpath)$/i.test(receiver))) && source) {
      this.report(node, 'xpath', name, source, 'high');
      return;
    }
    if (method === 'search' && /(client|ldap)$/i.test(receiver) && node.arguments[1]) {
      const options = node.arguments[1];
      const filter = options.type === 'ObjectExpression'
        ? this.taint((options.properties.find(item => item.key && (item.key.name || item.key.value) === 'filter') || {}).value)
        : this.taint(options);
      if (filter) this.report(node, 'ldap', name, filter, 'high');
      return;
    }
    if (XML_FUNCTIONS.has(name) && source) {
      const options = node.arguments[1];
      const entities = options && options.type === 'ObjectExpression' && options.properties.some(item =>
        item.key && (item.key.name || item.key.value) === 'noent' && item.value.type === 'BooleanLiteral' && item.value.value);
      if (entities) this.report(node, 'xxe', name, source, 'high');
      return;
    }
    if (!source) {
      if (SHELL_FUNCTIONS.has(name) && this.spawnUsesShell(node)) {
        const value = this.first(node.arguments.slice(0, 2));
        if (value) this.report(node, 'command', name, value, 'high');
      }
      return;
    }
    if (method && SQL_METHODS.has(method) && !this.isRequestLike(receiver)) {
      this.report(node, 'sql', name || method, source, 'high');
    } else if (method && SQLITE_METHODS.has(method) && SQLITE_RECEIVER.test(receiver)) {
      this.report(node, 'sql', name, source, 'high');
    } else if (method && NOSQL_METHODS.has(method) && NOSQL_RECEIVER.test(receiver) && !this.isFunctionNode(first)) {
      this.report(node, 'nosql', name, source, 'medium');
    } else if (COMMAND_FUNCTIONS.has(name)) {
      this.report(node, 'command', name, source, 'high');
    } else if (SHELL_FUNCTIONS.has(name)) {
      if (this.spawnUsesShell(node)) this.report(node, 'command', name, source, 'high');
    } else if ((method && PATH_METHODS.has(method) && FS_RECEIVER.test(receiver)) || PATH_METHODS.has(name)) {
      this.report(node, 'path', name, source, 'medium');
    } else if (method && RESPONSE_PATH_METHODS.has(method)) {
      this.report(node, 'path', name, source, 'medium');
    } else if (SSRF_FUNCTIONS.has(name)) {
      this.report(node, 'ssrf', name, source, 'medium');
    } else if (CODE_FUNCTIONS.has(name)) {
      this.report(node, 'code', name, source, 'high');
    } else if (TEMPLATE_FUNCTIONS.has(name)) {
      this.report(node, 'template', name, source, 'high');
    } else if (DESERIALIZE_FUNCTIONS.has(name)) {
      this.report(node, 'deserialization', name, source, 'high');
    }
  }

  // True for text sent as HTML: built strings and single request values, not whole objects sent as JSON.
  isText(node, source) {
    if (!node) return false;
    if (['TemplateLiteral', 'BinaryExpression', 'StringLiteral'].includes(node.type)) return true;
    if (node.type === 'CallExpression' && node.callee.type === 'MemberExpression') {
      return ['concat', 'join', 'replace', 'toString', 'trim', 'toUpperCase', 'toLowerCase'].includes(propertyName(node.callee));
    }
    return /\.(?:body|cookies|headers|params|query)\.[^.]+$|\('[^']*'\)$|^websocket|^graphql args\.|^trpc input\./.test(baseSource(source));
  }

  // obj[key] = value with a key from input can set __proto__ and change every object.
  checkAssignment(node) {
    const target = node.left;
    if (target.type !== 'MemberExpression') return;
    // ctx.body = text in Koa is sent as HTML.
    const body = dottedName(target);
    if (body && this.scope.contexts.has(body.split('.')[0]) && body.endsWith('.body')) {
      const source = this.taint(node.right);
      if (source && this.isText(node.right, source)) this.report(node, 'xss', body, source, 'medium');
      return;
    }
    if (!target.computed) return;
    const object = target.object.type === 'MemberExpression' && target.object.computed ? target.object : null;
    const keys = [target.property, ...(object ? [object.property] : [])];
    const root = dottedName(object ? object.object : target.object);
    if (this.bare.has(root) || this.lists.has(root)) return;
    for (const key of keys) {
      const source = this.taint(key);
      if (source && key.type !== 'NumericLiteral' && this.constant(key) === UNKNOWN) {
        this.report(node, 'prototype', `${root || 'object'}[...] =`, source, 'medium');
        return;
      }
    }
  }

  isRequestLike(receiver) {
    const head = receiver.split('.')[0];
    return this.scope.requests.has(head) || this.scope.contexts.has(head);
  }

  isFunctionNode(node) {
    return isFunction(node);
  }

  spawnUsesShell(node) {
    return node.arguments.some(argument => argument.type === 'ObjectExpression' && argument.properties.some(property =>
      property.type === 'ObjectProperty' && (property.key.name === 'shell') && !(property.value.type === 'BooleanLiteral' && property.value.value === false)));
  }

  report(node, sink, call, source, confidence) {
    this.file.record(node, this.label, sink, call, source, confidence, this.entry);
  }
}

function parseSource(source) {
  return parser.parse(source, {
    sourceType: 'unambiguous',
    errorRecovery: false,
    allowReturnOutsideFunction: true,
    plugins: [
      'asyncGenerators', 'classProperties', 'classPrivateProperties', 'classPrivateMethods', 'decorators-legacy', 'dynamicImport',
      'importAttributes', 'jsx', 'objectRestSpread', 'optionalChaining', 'topLevelAwait', 'typescript'
    ]
  });
}

// Runs every analyzer, then again if middleware was found keeping input on the request, since
// the handlers that read it may have run first.
function runAll(analyzers, extras, summaries) {
  for (let pass = 0; pass < 3; pass += 1) {
    const before = extras.size;
    for (const analyzer of analyzers) analyzer.run();
    if (extras.size === before) return;
    summaries.clear();
  }
}

function analyzeSource(source, filename = '<source>') {
  const analyzer = new FileAnalyzer(parseSource(source), filename, '', null, source);
  analyzer.summaries = new Map();
  runAll([analyzer], analyzer.requestExtras, analyzer.summaries);
  const findings = [...analyzer.findings.values()];
  return findings.sort((a, b) => a.line - b.line || a.sink.localeCompare(b.sink));
}

// Reads JSON that may have comments and trailing commas, as tsconfig.json may.
function readLooseJson(file) {
  const text = fs.readFileSync(file, 'utf8');
  let out = '';
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (char === '"') {
      const end = /"(?:[^"\\\n]|\\.)*"/y;
      end.lastIndex = index;
      const match = end.exec(text);
      if (!match) return null;
      out += match[0];
      index += match[0].length - 1;
    } else if (char === '/' && text[index + 1] === '/') {
      while (index < text.length && text[index] !== '\n') index += 1;
    } else if (char === '/' && text[index + 1] === '*') {
      const close = text.indexOf('*/', index + 2);
      index = close < 0 ? text.length : close + 1;
    } else {
      out += char;
    }
  }
  try {
    return JSON.parse(out.replace(/,(\s*[}\]])/g, '$1'));
  } catch (_error) {
    return null;
  }
}

// Maps import specifiers to scanned files, as Node, TypeScript and bundlers do: relative paths,
// tsconfig.json or jsconfig.json paths and baseUrl, and packages in the scanned folder.
class ModuleResolver {
  constructor(root, files) {
    this.root = root;
    this.files = files;
    this.configs = new Map();
    this.packages = new Map();
    this.findPackages(root);
  }

  findPackages(directory) {
    let entries;
    try {
      entries = fs.readdirSync(directory, { withFileTypes: true });
    } catch (_error) {
      return;
    }
    for (const entry of entries) {
      const full = path.join(directory, entry.name);
      if (entry.isDirectory() && !SKIPPED_DIRECTORIES.has(entry.name) && !entry.name.startsWith('.')) this.findPackages(full);
      else if (entry.isFile() && entry.name === 'package.json') {
        const manifest = readLooseJson(full);
        if (!manifest || typeof manifest.name !== 'string' || this.packages.has(manifest.name)) continue;
        let main = manifest.exports;
        if (main && typeof main === 'object' && !Array.isArray(main)) main = main['.'] || main;
        if (main && typeof main === 'object' && !Array.isArray(main)) main = main.import || main.require || main.default;
        this.packages.set(manifest.name, { directory, main: [main, manifest.module, manifest.main].filter(item => typeof item === 'string') });
      }
    }
  }

  // The nearest tsconfig.json or jsconfig.json, with extends followed.
  configFor(directory) {
    if (this.configs.has(directory)) return this.configs.get(directory);
    let config = null;
    for (const name of ['tsconfig.json', 'jsconfig.json']) {
      const file = path.join(directory, name);
      if (fs.existsSync(file)) {
        config = this.loadConfig(file, 0);
        break;
      }
    }
    const parent = path.dirname(directory);
    if (!config && parent !== directory && (directory + path.sep).startsWith(this.root + path.sep) && directory !== this.root) {
      config = this.configFor(parent);
    }
    this.configs.set(directory, config);
    return config;
  }

  loadConfig(file, depth) {
    const data = readLooseJson(file);
    if (!data || typeof data !== 'object') return null;
    const directory = path.dirname(file);
    let inherited = { baseUrl: '', paths: [] };
    if (typeof data.extends === 'string' && data.extends.startsWith('.') && depth < 3) {
      const parent = path.resolve(directory, data.extends);
      inherited = this.loadConfig(parent.endsWith('.json') ? parent : `${parent}.json`, depth + 1) || inherited;
    }
    const options = data.compilerOptions || {};
    const baseUrl = typeof options.baseUrl === 'string' ? path.resolve(directory, options.baseUrl) : inherited.baseUrl;
    if (!options.paths || typeof options.paths !== 'object') return { baseUrl, paths: inherited.paths };
    const from = baseUrl || directory;
    const paths = Object.entries(options.paths)
      .filter(([, targets]) => Array.isArray(targets))
      .map(([pattern, targets]) => [pattern, targets.filter(item => typeof item === 'string').map(item => path.resolve(from, item))]);
    return { baseUrl, paths };
  }

  file(base) {
    const stem = base.replace(/\.[cm]?js$/, '');
    const candidates = [base];
    for (const extension of SOURCE_EXTENSIONS) candidates.push(`${stem}${extension}`);
    for (const extension of SOURCE_EXTENSIONS) candidates.push(path.join(base, `index${extension}`));
    return candidates.find(item => this.files.has(item)) || null;
  }

  resolve(fromFile, spec) {
    if (typeof spec !== 'string' || !spec) return null;
    if (spec.startsWith('.')) return this.file(path.resolve(path.dirname(fromFile), spec));
    const config = this.configFor(path.dirname(fromFile));
    if (config) {
      for (const [pattern, targets] of config.paths) {
        const star = pattern.indexOf('*');
        let rest = null;
        if (star < 0 && pattern === spec) rest = '';
        else if (star >= 0 && spec.startsWith(pattern.slice(0, star)) && spec.endsWith(pattern.slice(star + 1))
                 && spec.length >= pattern.length - 1) {
          rest = spec.slice(star, spec.length - (pattern.length - star - 1));
        }
        if (rest === null) continue;
        for (const target of targets) {
          const found = this.file(target.replace('*', rest));
          if (found) return found;
        }
      }
      if (config.baseUrl) {
        const found = this.file(path.join(config.baseUrl, spec));
        if (found) return found;
      }
    }
    const parts = spec.split('/');
    const name = spec.startsWith('@') ? parts.slice(0, 2).join('/') : parts[0];
    const workspace = this.packages.get(name);
    if (!workspace) return null;
    const subpath = spec.slice(name.length + 1);
    if (subpath) return this.file(path.join(workspace.directory, subpath)) || this.file(path.join(workspace.directory, 'src', subpath));
    for (const main of workspace.main) {
      const found = this.file(path.resolve(workspace.directory, main));
      if (found) return found;
    }
    return this.file(path.join(workspace.directory, 'index')) || this.file(path.join(workspace.directory, 'src', 'index'));
  }
}

function isTestFile(name) {
  return /\.(test|spec)\.[cm]?[jt]sx?$/.test(name);
}

function collectFiles(root, includeTests, files) {
  const stat = fs.statSync(root);
  if (stat.isFile()) {
    files.push(root);
    return files;
  }
  for (const entry of fs.readdirSync(root, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
    const full = path.join(root, entry.name);
    if (entry.isDirectory()) {
      if (SKIPPED_DIRECTORIES.has(entry.name) || (!includeTests && TEST_DIRECTORIES.has(entry.name))) continue;
      collectFiles(full, includeTests, files);
    } else if (entry.isFile() && SOURCE_EXTENSIONS.has(path.extname(entry.name)) && !entry.name.endsWith('.d.ts')) {
      if (includeTests || !isTestFile(entry.name)) files.push(full);
    }
  }
  return files;
}

function analyzePath(target = '.', options = {}) {
  if (!fs.existsSync(target)) throw new Error(`Path not found: ${target}`);
  const findings = [];
  const errors = [];
  // Every file is parsed first, so calls into other project files can be followed.
  const parsed = [];
  for (const file of collectFiles(target, Boolean(options.includeTests), [])) {
    const display = file.split(path.sep).join('/');
    try {
      if (fs.statSync(file).size > MAX_FILE_BYTES) {
        errors.push({ file: display, error: 'File is too large to analyze' });
        continue;
      }
      const text = fs.readFileSync(file, 'utf8');
      parsed.push({ full: path.resolve(file), display, ast: parseSource(text), text });
    } catch (error) {
      errors.push({ file: display, error: String(error.message || error) });
    }
  }
  const project = new Map(parsed.map(item => [item.full, null]));
  const root = path.resolve(fs.statSync(target).isDirectory() ? target : path.dirname(target));
  project.resolver = new ModuleResolver(root, new Set(project.keys()));
  project.requestExtras = new Map();
  project.summaries = new Map();
  project.recordings = [];
  for (const item of parsed) project.set(item.full, new FileAnalyzer(item.ast, item.display, item.full, project, item.text));
  const filesScanned = parsed.length;
  runAll([...project.values()], project.requestExtras, project.summaries);
  // One at a time: spreading a large project's findings into push() overflows the call stack.
  const lines = new Map(parsed.map(item => [item.display, item.text.split(/\r?\n/)]));
  let suppressed = 0;
  for (const analyzer of project.values()) {
    for (const finding of analyzer.findings.values()) {
      if (ignored(lines.get(finding.file) || [], finding.line, finding.sink)) suppressed += 1;
      else findings.push(finding);
    }
  }
  findings.sort((a, b) => a.file.localeCompare(b.file) || a.line - b.line || a.sink.localeCompare(b.sink)
    || a.function.localeCompare(b.function));
  // What each handler reads, with types, labelled as its findings are so both link to the same routes.
  const inputs = [];
  // Every file's scope first, so handlers can follow schemas and helpers into the files they import.
  const scopes = new Map();
  for (const item of parsed) {
    const scope = moduleScope(item.ast, item.text, item.display, spec => project.resolver.resolve(item.full, spec));
    scope.project = scopes;
    scopes.set(item.full, scope);
  }
  for (const item of parsed) {
    const analyzer = project.get(item.full);
    const label = (node, parent) => analyzer.routeLabels.get(node) || functionName(node, parent);
    for (const entry of fileInputs(item.ast, item.display, item.text, label, scopes.get(item.full))) inputs.push(entry);
  }
  // The functions and class methods each file defines, so a handler the audit read but that
  // reads nothing itself (a Nest controller method taking only a DTO) still counts as read.
  const definitions = [];
  for (const item of parsed) {
    const visit = (node, owner) => {
      if (!node || typeof node.type !== 'string') return;
      let inside = owner;
      if (node.type === 'FunctionDeclaration' && node.id) definitions.push({ function: node.id.name, file: item.display });
      else if (node.type === 'VariableDeclarator' && node.id && node.id.type === 'Identifier' && node.init
        && ['FunctionExpression', 'ArrowFunctionExpression'].includes(node.init.type)) definitions.push({ function: node.id.name, file: item.display });
      else if ((node.type === 'ClassDeclaration' || node.type === 'ClassExpression') && node.id) inside = node.id.name;
      else if ((node.type === 'ClassMethod' || node.type === 'ClassPrivateMethod') && owner && node.key && node.key.type === 'Identifier'
        && node.kind === 'method') definitions.push({ function: `${owner}.${node.key.name}`, file: item.display });
      for (const key of Object.keys(node)) {
        if (key === 'loc' || key === 'start' || key === 'end' || key === 'extra') continue;
        const value = node[key];
        if (Array.isArray(value)) value.forEach(child => visit(child, inside));
        else if (value && typeof value.type === 'string') visit(value, inside);
      }
    };
    visit(item.ast.program, null);
  }
  const report = { schema_version: SCHEMA_VERSION, language: 'javascript', files_scanned: filesScanned, findings, errors, inputs, definitions };
  if (suppressed) report.suppressed = suppressed;
  return report;
}

// aiwaf: ignore, or aiwaf: ignore[sql, xss], on a finding's line or the line before it.
const IGNORE = /aiwaf[:-]\s*ignore(?:\[([a-z_,\s]+)\])?/i;

function ignored(lines, line, sink) {
  for (const number of [line, line - 1]) {
    const match = number >= 1 && number <= lines.length ? IGNORE.exec(lines[number - 1]) : null;
    if (match && (!match[1] || match[1].split(',').map(item => item.trim().toLowerCase()).includes(sink))) return true;
  }
  return false;
}

/** Identifies a finding across runs, without its line, which moves as code changes. */
function fingerprint(finding) {
  const parts = ['file', 'sink', 'call', 'function', 'source'].map(key => String(finding[key] ?? ''));
  return crypto.createHash('sha256').update(parts.join('\0')).digest('hex').slice(0, 32);
}

/** Leaves out findings already in an earlier report, so CI fails only on new ones. */
function applyBaseline(report, baseline) {
  const known = new Set((baseline.findings || []).map(fingerprint));
  const findings = report.findings.filter(finding => !known.has(fingerprint(finding)));
  const result = { ...report, findings };
  if (findings.length !== report.findings.length) result.baselined = report.findings.length - findings.length;
  return result;
}

const SINK_TITLES = {
  code: 'Code injection', command: 'Command injection', deserialization: 'Unsafe deserialization', ldap: 'LDAP injection',
  nosql: 'NoSQL injection', path: 'Path traversal', prototype: 'Prototype pollution', redirect: 'Open redirect',
  redos: 'Regular expression denial of service', sql: 'SQL injection', ssrf: 'Server-side request forgery',
  template: 'Template injection', xpath: 'XPath injection', xss: 'Cross-site scripting', xxe: 'XML external entity expansion'
};

/** The report as SARIF 2.1.0, for GitHub code scanning and other tools that read it. */
function toSarif(report) {
  const sinks = [...new Set(report.findings.map(finding => finding.sink))].sort();
  const title = sink => SINK_TITLES[sink] || sink;
  const rules = sinks.map(sink => ({
    id: `aiwaf.${sink}`,
    name: title(sink).replace(/(^|[\s-])(\w)/g, (_match, _space, letter) => letter.toUpperCase()).replace(/[\s-]/g, ''),
    shortDescription: { text: title(sink) },
    fullDescription: { text: `Request input can reach an operation where it allows ${title(sink).toLowerCase()}.` },
    defaultConfiguration: { level: 'warning' },
    properties: { tags: ['security'] }
  }));
  const results = report.findings.map(finding => ({
    ruleId: `aiwaf.${finding.sink}`,
    ruleIndex: sinks.indexOf(finding.sink),
    level: finding.confidence === 'high' ? 'error' : 'warning',
    message: { text: `${finding.source} reaches ${finding.call} in ${finding.function}` },
    locations: [{ physicalLocation: {
      artifactLocation: { uri: String(finding.file).replace(/\\/g, '/').replace(/^\.\//, '') },
      region: { startLine: Math.max(Number(finding.line) || 1, 1) }
    } }],
    partialFingerprints: { 'aiwafFinding/v1': fingerprint(finding) }
  }));
  return JSON.stringify({
    $schema: 'https://json.schemastore.org/sarif-2.1.0.json',
    version: '2.1.0',
    runs: [{ tool: { driver: { name: 'AIWAF audit', informationUri: 'https://github.com/aiwaf-project/aiwaf', rules } }, results }]
  }, null, 2);
}

function formatReport(report) {
  const lines = report.findings.map(finding =>
    `${finding.file}:${finding.line}: ${finding.sink} (${finding.confidence} confidence) ${finding.source} reaches ${finding.call} in ${finding.function}`);
  for (const error of report.errors) lines.push(`${error.file}: not analyzed: ${error.error}`);
  lines.push(`Scanned ${report.files_scanned} files and found ${report.findings.length} possible injection paths.`);
  if (report.suppressed) lines.push(`Left out ${report.suppressed} marked with an aiwaf: ignore comment.`);
  if (report.baselined) lines.push(`Left out ${report.baselined} already in the baseline.`);
  if (!report.findings.length) lines.push("This doesn't prove the code is safe, because static analysis can miss paths.");
  return lines.join('\n');
}

function runAuditCommand(args, write = text => console.log(text)) {
  const positional = [];
  const options = { format: 'text', includeTests: false, failOnFindings: false, output: null, manifest: null, baseline: null };
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === '--format') options.format = args[++index];
    else if (argument === '--output') options.output = args[++index];
    else if (argument === '--manifest') options.manifest = args[++index];
    else if (argument === '--baseline') options.baseline = args[++index];
    else if (argument === '--include-tests') options.includeTests = true;
    else if (argument === '--fail-on-findings') options.failOnFindings = true;
    else positional.push(argument);
  }
  if (!['text', 'json', 'sarif'].includes(options.format)) {
    write('Usage: aiwaf audit [path] [--format text|json|sarif] [--output file] [--include-tests] [--fail-on-findings] [--manifest file] [--baseline file]');
    return 2;
  }
  let report;
  try {
    report = analyzePath(positional[0] || '.', options);
    if (options.baseline) report = applyBaseline(report, JSON.parse(fs.readFileSync(options.baseline, 'utf8')));
  } catch (error) {
    write(String(error.message || error));
    return 2;
  }
  // definitions only tell init which handlers were read; the printed report leaves them out.
  const { definitions, ...printed } = report;
  const content = options.format === 'json' ? JSON.stringify(printed, null, 2)
    : options.format === 'sarif' ? toSarif(report) : formatReport(report);
  if (options.output) fs.writeFileSync(options.output, `${content}\n`);
  else write(content);
  if (options.manifest) {
    const { formatCounts, updateManifest } = require('./routeRisks');
    try {
      write(formatCounts(updateManifest(report, options.manifest), options.manifest));
    } catch (error) {
      write(String(error.message || error));
      return 2;
    }
  }
  return options.failOnFindings && report.findings.length ? 1 : 0;
}

module.exports = { analyzePath, analyzeSource, applyBaseline, collectFiles, fingerprint, formatReport, parseSource, runAuditCommand, toSarif };
