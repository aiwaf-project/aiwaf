'use strict';

// Finds route handlers in source code, for routes that reach the manifest as a JSON list (which
// can't carry functions). Each found handler's source text stands in for the function, so the
// route gets the same API, form and request-body signals a live app's handler gives.

const fs = require('fs');
const path = require('path');

const METHODS = new Set(['get', 'post', 'put', 'patch', 'delete', 'del', 'options', 'all']);
const MAX_BYTES = 1024 * 1024;

/** A handler known only by its source text. sourceAst and pathManifest read it as a function's. */
class HandlerSource {
  constructor(text, name = '') {
    this.text = String(text || '');
    this.name = name;
  }
}

/** A handler's source: a function's own, or a HandlerSource's text. Null for anything else. */
function handlerText(handler) {
  if (typeof handler === 'function') return Function.prototype.toString.call(handler);
  if (handler instanceof HandlerSource) return handler.text;
  return null;
}

function literalPath(node) {
  if (!node) return null;
  if (node.type === 'StringLiteral') return node.value;
  if (node.type === 'TemplateLiteral' && node.expressions.length === 0) return node.quasis.map(item => item.value.cooked).join('');
  return null;
}

function isFunctionNode(node) {
  return node && ['FunctionExpression', 'ArrowFunctionExpression', 'FunctionDeclaration'].includes(node.type);
}

/**
 * Handlers registered as x.get('/path', ..., handler) or x.route('/path').post(handler), keyed
 * 'METHOD path'. A handler named by identifier is found when that name is a function declared
 * or assigned in the same file.
 */
function routeHandlers(source = '.') {
  const { collectFiles, parseSource } = require('./injectionAnalysis');
  const found = new Map();
  const root = path.resolve(source);
  if (!fs.existsSync(root)) return found;
  for (const file of collectFiles(root, false, [])) {
    let text;
    let ast;
    try {
      if (fs.statSync(file).size > MAX_BYTES) continue;
      text = fs.readFileSync(file, 'utf8');
      ast = parseSource(text);
    } catch (_) {
      continue;
    }
    const functions = new Map();
    const calls = [];
    const visit = node => {
      if (!node || typeof node.type !== 'string') return;
      if (node.type === 'FunctionDeclaration' && node.id) functions.set(node.id.name, node);
      if (node.type === 'VariableDeclarator' && node.id && node.id.type === 'Identifier' && isFunctionNode(node.init)) {
        functions.set(node.id.name, node.init);
      }
      if (node.type === 'CallExpression' && node.callee && node.callee.type === 'MemberExpression') calls.push(node);
      for (const key of Object.keys(node)) {
        if (key === 'loc' || key === 'start' || key === 'end') continue;
        const value = node[key];
        if (Array.isArray(value)) value.forEach(visit);
        else if (value && typeof value.type === 'string') visit(value);
      }
    };
    visit(ast.program);
    for (const call of calls) {
      const property = call.callee.property;
      const method = property && (property.name || property.value);
      if (!METHODS.has(method)) continue;
      // x.route('/path').get(a).post(b) has its path on the route() call, any number of calls back.
      let target = call.callee.object;
      const nameOf = node => node && node.callee && node.callee.type === 'MemberExpression' && (node.callee.property.name || node.callee.property.value);
      while (target && target.type === 'CallExpression' && METHODS.has(nameOf(target))) target = target.callee.object;
      const chained = Boolean(target && target.type === 'CallExpression' && nameOf(target) === 'route');
      const routePath = chained ? literalPath(target.arguments[0]) : literalPath(call.arguments[0]);
      if (routePath === null || !routePath.startsWith('/')) continue;
      const handlerNodes = chained ? call.arguments : call.arguments.slice(1);
      const last = handlerNodes[handlerNodes.length - 1];
      const node = isFunctionNode(last) ? last : (last && last.type === 'Identifier' ? functions.get(last.name) : null);
      if (!node) continue;
      const verb = method === 'del' ? 'DELETE' : method.toUpperCase();
      const key = `${verb} ${routePath}`;
      if (!found.has(key)) found.set(key, new HandlerSource(text.slice(node.start, node.end), last.name || ''));
    }
  }
  return found;
}

/**
 * The source handler for a listed route: the same method and path, else (for a router mounted
 * under a prefix) the one route whose path the listed path ends with. ALL matches any method.
 */
function handlerFor(handlers, routePath, methods = []) {
  const verbs = methods.length ? methods.map(method => String(method).toUpperCase()) : ['GET'];
  for (const verb of [...verbs, 'ALL']) {
    const exact = handlers.get(`${verb} ${routePath}`);
    if (exact) return exact;
  }
  const suffix = [];
  for (const [key, handler] of handlers) {
    const [verb, sourcePath] = [key.slice(0, key.indexOf(' ')), key.slice(key.indexOf(' ') + 1)];
    if (sourcePath === '/' || !(verbs.includes(verb) || verb === 'ALL')) continue;
    if (routePath.endsWith(sourcePath) && routePath.length > sourcePath.length) suffix.push(handler);
  }
  return suffix.length === 1 ? suffix[0] : null;
}

module.exports = { HandlerSource, handlerText, routeHandlers, handlerFor };
