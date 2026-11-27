const fs = require('fs');
const os = require('os');
const path = require('path');
const { loadApp, frameworkOf, resolveAppFile, AppLoadError } = require('../lib/appLoader');

const FIXTURES = path.join(__dirname, 'fixtures', 'init');
const fixture = (...parts) => path.join(FIXTURES, ...parts);

afterEach(() => {
  delete global.aiwafFixtureServer;
  delete global.aiwafFixtureListened;
});

describe('frameworkOf', () => {
  test('tells apps apart by their shape', () => {
    const express = require('express');
    expect(frameworkOf(express())).toBe('express');
    expect(frameworkOf(express.Router())).toBeNull();
    expect(frameworkOf({ getHttpAdapter() {} })).toBe('nest');
    expect(frameworkOf({ printRoutes() {}, register() {} })).toBe('fastify');
    expect(frameworkOf({ table() {}, route() {} })).toBe('hapi');
    expect(frameworkOf({ middleware: [], callback() {} })).toBe('koa');
    expect(frameworkOf({})).toBeNull();
    expect(frameworkOf(null)).toBeNull();
    expect(frameworkOf('app')).toBeNull();
  });
});

describe('resolveAppFile', () => {
  test('uses the file given, a folder in it, package.json main, then a usual name', () => {
    expect(resolveAppFile('app.js', fixture('express'))).toBe(fixture('express', 'app.js'));
    expect(resolveAppFile('express', FIXTURES)).toBe(fixture('express', 'app.js'));
    expect(resolveAppFile(null, fixture('main'))).toBe(fixture('main', 'lib', 'start.js'));
    expect(resolveAppFile(null, fixture('fastify'))).toBe(fixture('fastify', 'app.js'));
  });

  test('says so when there is no app file', () => {
    const empty = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-noapp-'));
    try {
      fs.writeFileSync(path.join(empty, 'package.json'), '{"main": "missing.js"}');
      expect(() => resolveAppFile(null, empty)).toThrow(/No app file found.*--app/);
      expect(() => resolveAppFile('nope.js', empty)).toThrow(AppLoadError);
    } finally {
      fs.rmSync(empty, { recursive: true, force: true });
    }
  });
});

describe('loadApp', () => {
  test('loads an exported app without letting it bind a port', async () => {
    const loaded = await loadApp({ app: fixture('express', 'app.js') });
    expect(loaded.framework).toBe('express');
    expect(loaded.file).toBe(fixture('express', 'app.js'));
    expect(global.aiwafFixtureServer.listening).toBe(false);
    expect(global.aiwafFixtureListened).toBeUndefined();
    // listen() works normally again afterwards.
    const net = require('net');
    const server = net.createServer();
    await new Promise(resolve => server.listen(0, resolve));
    expect(server.listening).toBe(true);
    await new Promise(resolve => server.close(resolve));
  });

  test('finds an app that only calls app.listen(), including after a delay', async () => {
    expect((await loadApp({ app: fixture('express', 'noexport.js') })).framework).toBe('express');
    const late = await loadApp({ app: fixture('express', 'late.js'), waitMs: 2000 });
    expect(late.framework).toBe('express');
  });

  test('calls a factory export, and reads a named export', async () => {
    const built = await loadApp({ app: fixture('express', 'factory.js'), exportName: 'createApp' });
    expect(built.framework).toBe('express');
    expect(built.exportName).toBe('createApp');
    const named = await loadApp({ app: fixture('main', 'lib', 'start.js') });
    expect(named.framework).toBe('express');
  });

  // ES modules load through the real CLI in initCommand.test.js: Jest's transform rewrites import().
  test('readies a Fastify app', async () => {
    const fastify = await loadApp({ app: fixture('fastify', 'app.js') });
    expect(fastify.framework).toBe('fastify');
    expect(fastify.app.printRoutes({ commonPrefix: false })).toContain('/account/profile');
    await fastify.app.close();
  });

  test('explains what went wrong', async () => {
    await expect(loadApp({ app: fixture('express', 'notapp.js'), waitMs: 0 })).rejects.toThrow(/exports no Express, Fastify, Hapi or Nest app/);
    await expect(loadApp({ app: fixture('express', 'notapp.js'), exportName: 'app', waitMs: 0 })).rejects.toThrow(/its export app isn't/);
    await expect(loadApp({ app: fixture('express', 'broken.js') })).rejects.toThrow(/Couldn't load .*database unreachable/);
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-ts-'));
    try {
      fs.writeFileSync(path.join(dir, 'app.ts'), 'throw new Error("no TypeScript loader");\n');
      await expect(loadApp({ app: path.join(dir, 'app.ts') })).rejects.toThrow(/no TypeScript loader\. If Node can't run this TypeScript, compile it/);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('captureFastifyRoutes', () => {
  test('wraps the app\'s Fastify so each instance records its routes from the start, and undoes it', () => {
    const { captureFastifyRoutes, ROUTES } = require('../lib/appLoader');
    const resolved = require.resolve('fastify', { paths: [fixture('fastify')] });
    const hooks = [];
    const original = function fakeFastify() { return { addHook: (name, fn) => hooks.push([name, fn]) }; };
    original.extra = 'kept';
    const cache = { [resolved]: { exports: original } };
    const restore = captureFastifyRoutes(fixture('fastify', 'app.js'), cache, () => original);
    const wrapped = cache[resolved].exports;
    expect(wrapped).not.toBe(original);
    expect([wrapped.default, wrapped.fastify, wrapped.extra]).toEqual([wrapped, wrapped, 'kept']);
    const instance = wrapped();
    expect(hooks.map(([name]) => name)).toEqual(['onRoute']);
    hooks[0][1]({ url: '/x', method: 'GET' });
    expect(instance[ROUTES]).toEqual([{ url: '/x', method: 'GET' }]);
    // An instance without hooks still loads.
    const bare = function bareFastify() { return {}; };
    captureFastifyRoutes(fixture('fastify', 'app.js'), { [resolved]: { exports: bare } }, () => bare)();
    restore();
    expect(cache[resolved].exports).toBe(original);
  });

  test('leaves things alone when Fastify isn\'t there or isn\'t a factory', () => {
    const { captureFastifyRoutes } = require('../lib/appLoader');
    const outside = path.join(os.tmpdir(), 'no-such-app', 'app.js');
    expect(() => captureFastifyRoutes(outside)()).not.toThrow();
    const resolved = require.resolve('fastify', { paths: [fixture('fastify')] });
    expect(() => captureFastifyRoutes(fixture('fastify', 'app.js'), {}, () => { throw new Error('broken'); })()).not.toThrow();
    const cache = { [resolved]: { exports: {} } };
    captureFastifyRoutes(fixture('fastify', 'app.js'), cache, () => ({}))();
    expect(cache[resolved].exports).toEqual({});
  });
});

