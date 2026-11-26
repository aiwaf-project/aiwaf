'use strict';

// Loads a project's app for `aiwaf init`, as Python's init imports module:app, so the manifest
// comes from the routes the app actually registers rather than from a hand-written list.
//
// Loading runs the app's own module code. Servers it starts are stopped from binding a port:
// listen() is recorded instead, which is also how an app that only calls app.listen() without
// exporting anything is found.

const fs = require('fs');
const net = require('net');
const path = require('path');
const { pathToFileURL } = require('url');

const CANDIDATES = [
  'app.js', 'server.js', 'index.js', 'main.js',
  'src/app.js', 'src/server.js', 'src/index.js', 'src/main.js',
  'app.mjs', 'server.mjs', 'index.mjs', 'src/app.mjs', 'src/server.mjs', 'src/index.mjs',
  'app.cjs', 'server.cjs', 'index.cjs',
  'dist/main.js', 'dist/app.js', 'dist/server.js', 'dist/index.js'
];

class AppLoadError extends Error {
  constructor(message) {
    super(message);
    this.name = 'AppLoadError';
  }
}

/** The framework an object is the app of, or null. */
function frameworkOf(value) {
  if (!value || (typeof value !== 'object' && typeof value !== 'function')) return null;
  if (typeof value.getHttpAdapter === 'function') return 'nest';
  if (typeof value === 'function' && typeof value.use === 'function' && typeof value.handle === 'function'
    && typeof value.listen === 'function') return 'express';
  if (typeof value.printRoutes === 'function' && typeof value.register === 'function') return 'fastify';
  if (typeof value.table === 'function' && typeof value.route === 'function') return 'hapi';
  if (Array.isArray(value.middleware) && typeof value.callback === 'function') return 'koa';
  return null;
}

/** The app file: the one given, else package.json's main, else a usual name. */
function resolveAppFile(target, cwd) {
  if (target) {
    const file = path.resolve(cwd, target);
    if (!fs.existsSync(file)) throw new AppLoadError(`App file not found: ${target}`);
    if (fs.statSync(file).isDirectory()) return resolveAppFile(null, file);
    return file;
  }
  try {
    const pkg = JSON.parse(fs.readFileSync(path.join(cwd, 'package.json'), 'utf8'));
    if (typeof pkg.main === 'string') {
      const main = path.resolve(cwd, pkg.main);
      for (const file of [main, `${main}.js`, path.join(main, 'index.js')]) {
        if (fs.existsSync(file) && fs.statSync(file).isFile()) return file;
      }
    }
  } catch (_) { /* no package.json, or no main */ }
  for (const name of CANDIDATES) {
    const file = path.join(cwd, name);
    if (fs.existsSync(file)) return file;
  }
  throw new AppLoadError(`No app file found in ${cwd}. Pass it with --app, as in: aiwaf init --app src/app.js`);
}

/** What a module exports that could be the app, the named export first. */
function exportCandidates(namespace, exportName) {
  const root = namespace && namespace.default !== undefined ? namespace.default : namespace;
  const read = (holder, name) => {
    try { return holder && (typeof holder === 'object' || typeof holder === 'function') ? holder[name] : undefined; } catch (_) { return undefined; }
  };
  if (exportName) return [read(namespace, exportName), read(root, exportName)].filter(item => item !== undefined);
  return [root, ...['app', 'server', 'default'].flatMap(name => [read(namespace, name), read(root, name)])]
    .filter(item => item !== undefined);
}

// The Express app a server started with app.listen() or http.createServer(app) serves.
function appFromServer(server) {
  for (const listener of server.listeners('request')) {
    if (frameworkOf(listener)) return listener;
  }
  return null;
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Loads the app, as {app, framework, file, exportName}.
 *
 * options.app is the file (default: package.json's main or a usual name), options.exportName
 * the export holding the app or a factory returning it, options.waitMs how long to wait for an
 * app that starts asynchronously (default 3000).
 */
async function loadApp(options = {}) {
  const cwd = path.resolve(options.cwd || process.cwd());
  const file = resolveAppFile(options.app, cwd);
  const waitMs = Number.isFinite(options.waitMs) ? options.waitMs : 3000;
  const servers = [];
  const listen = net.Server.prototype.listen;
  // Records the server instead of binding a port. Its callback isn't run: the app isn't serving.
  net.Server.prototype.listen = function recordListen() {
    servers.push(this);
    return this;
  };
  let namespace;
  try {
    try {
      namespace = await import(pathToFileURL(file).href);
    } catch (error) {
      const shown = path.relative(cwd, file) || file;
      // Node runs TypeScript only from version 22.6 on, and only syntax it can strip.
      const hint = /\.[cm]?tsx?$/.test(file) ? '. If Node can\'t run this TypeScript, compile it and pass the built file, as in: aiwaf init --app dist/main.js' : '';
      throw new AppLoadError(`Couldn't load ${shown}: ${error.message || error}${hint}`);
    }
    const pick = async () => {
      for (const candidate of exportCandidates(namespace, options.exportName)) {
        if (frameworkOf(candidate)) return candidate;
      }
      // A factory, as createApp(): called once, since running it builds the app.
      const factory = exportCandidates(namespace, options.exportName).find(item => typeof item === 'function' && !frameworkOf(item));
      if (factory) {
        const built = await factory();
        if (frameworkOf(built)) return built;
      }
      return null;
    };
    let app = await pick();
    const deadline = Date.now() + waitMs;
    while (!app) {
      for (const server of servers) {
        app = appFromServer(server);
        if (app) break;
      }
      if (app || Date.now() >= deadline) break;
      await sleep(25);
    }
    if (!app) {
      const hint = options.exportName
        ? `its export ${options.exportName} isn't an Express, Fastify, Hapi or Nest app, or a function returning one`
        : 'it exports no Express, Fastify, Hapi or Nest app and started no Express server';
      throw new AppLoadError(`No app found in ${path.relative(cwd, file) || file}: ${hint}. `
        + 'Export the app (module.exports = app) or name it with --export.');
    }
    const framework = frameworkOf(app);
    if (framework === 'fastify' && typeof app.ready === 'function') await app.ready();
    return { app, framework, file, exportName: options.exportName || null };
  } finally {
    net.Server.prototype.listen = listen;
  }
}

module.exports = { loadApp, frameworkOf, resolveAppFile, AppLoadError };
