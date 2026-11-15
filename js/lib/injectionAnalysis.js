'use strict';

// Static analysis that follows request input to dangerous operations.
// It reads source files only, and never runs application code or affects request handling.
// A clean report doesn't prove code is safe: input that crosses modules or dynamic calls can be missed.

const fs = require('fs');
const path = require('path');
const parser = require('@babel/parser');

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
const REQUEST_METHODS = new Set(['all', 'allParams', 'body', 'cookie', 'file', 'get', 'header', 'input', 'only', 'param', 'params', 'qs', 'raw']);
const NEST_INPUT_DECORATORS = new Set(['Body', 'Cookies', 'Headers', 'Param', 'Query', 'UploadedFile', 'UploadedFiles']);
const SANITIZERS = new Set(['Boolean', 'Number', 'encodeURIComponent', 'escape', 'isNaN', 'parseFloat', 'parseInt', 'path.basename', 'shellescape', 'validator.escape', 'sqlstring.escape', 'mysql.escape']);
const PASSTHROUGH_METHODS = new Set([
  'concat', 'join', 'padEnd', 'padStart', 'replace', 'replaceAll', 'slice', 'split', 'substring', 'toLowerCase',
  'toString', 'toUpperCase', 'trim', 'trimEnd', 'trimStart'
]);
const PASSTHROUGH_FUNCTIONS = new Set(['Array.from', 'Buffer.from', 'JSON.parse', 'JSON.stringify', 'Object.assign', 'String', 'decodeURIComponent', 'path.join', 'path.normalize', 'path.resolve']);
const SQL_METHODS = new Set(['$executeRawUnsafe', '$queryRawUnsafe', 'execute', 'havingRaw', 'orderByRaw', 'query', 'raw', 'whereRaw']);
const SQLITE_METHODS = new Set(['all', 'each', 'exec', 'get', 'prepare', 'run']);
const SQLITE_RECEIVER = /(^|\.)(db|sqlite|database)$/i;
const NOSQL_METHODS = new Set(['aggregate', 'count', 'countDocuments', 'deleteMany', 'deleteOne', 'find', 'findOne', 'findOneAndDelete', 'findOneAndUpdate', 'updateMany', 'updateOne']);
const NOSQL_RECEIVER = /(coll|db|mongo|model|users?$|^[A-Z])/;
const COMMAND_FUNCTIONS = new Set(['child_process.exec', 'child_process.execSync', 'cp.exec', 'cp.execSync', 'exec', 'execSync']);
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
const CODE_FUNCTIONS = new Set(['eval', 'setImmediate.call', 'vm.compileFunction', 'vm.runInContext', 'vm.runInNewContext', 'vm.runInThisContext']);
const TEMPLATE_FUNCTIONS = new Set(['Handlebars.compile', 'dot.template', 'ejs.compile', 'ejs.render', 'handlebars.compile', 'lodash.template', 'nunjucks.renderString', 'pug.compile', 'pug.render', '_.template']);
const DESERIALIZE_FUNCTIONS = new Set(['serialize.unserialize', 'nodeSerialize.unserialize', 'funcster.deepDeserialize']);

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
  return '<anonymous>';
}

class FileAnalyzer {
  constructor(ast, filename) {
    this.ast = ast;
    this.filename = filename;
    this.findings = new Map();
    this.functions = new Map();
    this.routeLabels = new WeakMap();
    this.collectFunctions(ast);
  }

  collectFunctions(root) {
    const visit = (node, parent) => {
      if (node.type === 'ObjectExpression') this.collectRouteLabel(node);
      if (isFunction(node)) {
        const name = functionName(node, parent);
        if (name && name !== '<anonymous>' && !this.functions.has(name)) this.functions.set(name, node);
      }
      eachChild(node, child => visit(child, node));
    };
    visit(root, null);
  }

  collectRouteLabel(node) {
    const value = name => node.properties.find(item => item.key && (item.key.name || item.key.value) === name);
    const route = value('path');
    const handler = value('handler');
    if (route && handler && route.value.type === 'StringLiteral' && handler.value && isFunction(handler.value)) {
      this.routeLabels.set(handler.value, `route ${route.value.value}`);
    }
  }

  run() {
    const visit = (node, parent, inherited) => {
      if (isFunction(node)) {
        const scope = this.initialScope(node, inherited);
        if (scope.tainted.size || scope.requests.size || scope.contexts.size) {
          this.analyze(node, scope, this.routeLabels.get(node) || functionName(node, parent), 0, new Set());
        }
        // Nested functions are handled while analyzing the function that contains them.
        if (scope.tainted.size || scope.requests.size || scope.contexts.size) return;
      }
      eachChild(node, child => visit(child, node, inherited));
    };
    visit(this.ast, null, null);
    return [...this.findings.values()];
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

  analyze(node, scope, label, depth, active) {
    const key = node.start;
    if (active.has(key) || depth > MAX_HELPER_DEPTH) return;
    const next = new Set(active);
    next.add(key);
    const walker = new FunctionWalker(this, scope, label, depth, next);
    if (node.body.type === 'BlockStatement') walker.visit(node.body, 0);
    else {
      walker.visit(node.body, 0);
    }
  }

  record(node, label, sink, call, source, confidence) {
    const line = node.loc ? node.loc.start.line : 0;
    const key = `${line}:${sink}:${call}`;
    if (!this.findings.has(key)) {
      this.findings.set(key, { file: this.filename, line, function: label, sink, call, source, confidence });
    }
  }
}

class FunctionWalker {
  constructor(file, scope, label, depth, active) {
    this.file = file;
    this.scope = scope;
    this.label = label;
    this.depth = depth;
    this.active = active;
  }

  visit(node, nesting) {
    if (!node || typeof node.type !== 'string') return;
    switch (node.type) {
      case 'VariableDeclarator':
        if (node.init) this.visit(node.init, nesting);
        this.bind(node.id, node.init ? this.taint(node.init) : null, nesting, node.init);
        return;
      case 'AssignmentExpression':
        this.visit(node.right, nesting);
        if (node.operator === '=') this.bind(node.left, this.taint(node.right), nesting, node.right);
        else {
          const source = this.taint(node.right);
          if (source) this.bind(node.left, source, nesting, node.right);
        }
        return;
      case 'CallExpression':
      case 'OptionalCallExpression':
      case 'NewExpression':
        this.checkCall(node);
        this.followHelper(node);
        eachChild(node, child => this.visit(child, nesting));
        return;
      case 'TaggedTemplateExpression':
        eachChild(node, child => this.visit(child, nesting));
        return;
      case 'ForOfStatement':
      case 'ForInStatement': {
        const source = this.taint(node.right);
        const target = node.left.type === 'VariableDeclaration' ? node.left.declarations[0].id : node.left;
        if (source) this.bind(target, source, nesting + 1, node.right);
        this.visit(node.right, nesting);
        this.visit(node.body, nesting + 1);
        return;
      }
      case 'IfStatement':
      case 'WhileStatement':
      case 'DoWhileStatement':
      case 'ForStatement':
      case 'SwitchStatement':
      case 'TryStatement':
      case 'ConditionalExpression':
        eachChild(node, child => this.visit(child, nesting + 1));
        return;
      default:
        if (isFunction(node)) {
          // Callbacks keep the request and values from the enclosing handler.
          const nested = this.file.initialScope(node, this.scope);
          const walker = new FunctionWalker(this.file, nested, this.label, this.depth, this.active);
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
      if (name) this.scope.tainted.set(name.split('.')[0], source);
    }
  }

  taint(node) {
    if (!node) return null;
    switch (node.type) {
      case 'Identifier':
        return this.scope.tainted.get(node.name) || null;
      case 'MemberExpression':
      case 'OptionalMemberExpression': {
        const source = this.requestProperty(node);
        if (source) return source;
        return withKey(this.taint(node.object), propertyName(node));
      }
      case 'TemplateLiteral':
        return this.first(node.expressions);
      case 'BinaryExpression':
        return node.operator === '+' ? this.taint(node.left) || this.taint(node.right) : null;
      case 'LogicalExpression':
        return this.taint(node.left) || this.taint(node.right);
      case 'ConditionalExpression':
        return this.taint(node.consequent) || this.taint(node.alternate);
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
    if (this.scope.contexts.has(name) && CONTEXT_PROPERTIES.has(property) && property !== 'request') return `${name}.${property}`;
    const parts = name.split('.');
    if (parts.length === 2 && this.scope.contexts.has(parts[0]) && parts[1] === 'request' && REQUEST_PROPERTIES.has(property)) {
      return `${name}.${property}`;
    }
    return null;
  }

  first(nodes) {
    for (const node of nodes) {
      const source = this.taint(node);
      if (source) return source;
    }
    return null;
  }

  callTaint(node) {
    const name = dottedName(node.callee);
    if (SANITIZERS.has(name)) return null;
    if (PASSTHROUGH_FUNCTIONS.has(name)) return this.first(node.arguments);
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
    }
    return null;
  }

  followHelper(node) {
    if (this.depth >= MAX_HELPER_DEPTH || node.type === 'NewExpression') return;
    const name = dottedName(node.callee).split('.').pop();
    const helper = name && this.file.functions.get(name);
    if (!helper) return;
    const passed = new Map();
    (helper.params || []).forEach((parameter, index) => {
      const source = this.taint(node.arguments[index]);
      if (source && parameter.type === 'Identifier') passed.set(parameter.name, source);
    });
    if (!passed.size) return;
    const scope = { tainted: passed, requests: new Set(), contexts: new Set() };
    this.file.analyze(helper, scope, `${this.label} -> ${name}`, this.depth + 1, this.active);
  }

  checkCall(node) {
    const name = dottedName(node.callee);
    const method = node.callee.type === 'MemberExpression' || node.callee.type === 'OptionalMemberExpression'
      ? propertyName(node.callee) : '';
    const receiver = method ? dottedName(node.callee.object) : '';
    const first = node.arguments[0];
    const source = this.taint(first);

    if (node.type === 'NewExpression') {
      if (name === 'Function') {
        const code = this.first(node.arguments);
        if (code) this.report(node, 'code', 'new Function', code, 'high');
      }
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
    this.file.record(node, this.label, sink, call, source, confidence);
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

function analyzeSource(source, filename = '<source>') {
  const findings = new FileAnalyzer(parseSource(source), filename).run();
  return findings.sort((a, b) => a.line - b.line || a.sink.localeCompare(b.sink));
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
  let filesScanned = 0;
  for (const file of collectFiles(target, Boolean(options.includeTests), [])) {
    const display = file.split(path.sep).join('/');
    try {
      if (fs.statSync(file).size > MAX_FILE_BYTES) {
        errors.push({ file: display, error: 'File is too large to analyze' });
        continue;
      }
      findings.push(...analyzeSource(fs.readFileSync(file, 'utf8'), display));
      filesScanned += 1;
    } catch (error) {
      errors.push({ file: display, error: String(error.message || error) });
    }
  }
  findings.sort((a, b) => a.file.localeCompare(b.file) || a.line - b.line || a.sink.localeCompare(b.sink));
  return { schema_version: SCHEMA_VERSION, language: 'javascript', files_scanned: filesScanned, findings, errors };
}

function formatReport(report) {
  const lines = report.findings.map(finding =>
    `${finding.file}:${finding.line}: ${finding.sink} (${finding.confidence} confidence) ${finding.source} reaches ${finding.call} in ${finding.function}`);
  for (const error of report.errors) lines.push(`${error.file}: not analyzed: ${error.error}`);
  lines.push(`Scanned ${report.files_scanned} files and found ${report.findings.length} possible injection paths.`);
  if (!report.findings.length) lines.push("This doesn't prove the code is safe, because static analysis can miss paths.");
  return lines.join('\n');
}

function runAuditCommand(args, write = text => console.log(text)) {
  const positional = [];
  const options = { format: 'text', includeTests: false, failOnFindings: false, output: null, manifest: null };
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === '--format') options.format = args[++index];
    else if (argument === '--output') options.output = args[++index];
    else if (argument === '--manifest') options.manifest = args[++index];
    else if (argument === '--include-tests') options.includeTests = true;
    else if (argument === '--fail-on-findings') options.failOnFindings = true;
    else positional.push(argument);
  }
  if (!['text', 'json'].includes(options.format)) {
    write('Usage: aiwaf audit [path] [--format text|json] [--output file] [--include-tests] [--fail-on-findings] [--manifest file]');
    return 2;
  }
  let report;
  try {
    report = analyzePath(positional[0] || '.', options);
  } catch (error) {
    write(String(error.message || error));
    return 2;
  }
  const content = options.format === 'json' ? JSON.stringify(report, null, 2) : formatReport(report);
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

module.exports = { analyzePath, analyzeSource, formatReport, runAuditCommand };
