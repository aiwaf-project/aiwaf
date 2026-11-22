const { execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

// npm usually hoists aiwaf-wasm to the app's node_modules, next to aiwaf rather than inside it.
// The loader's fallbacks and the version check must find it there too.
const ROOT = path.join(__dirname, '..');
const INSTALLED = path.dirname(require.resolve('aiwaf-wasm/package.json'));

let app;

beforeAll(() => {
  app = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-hoisted-'));
  const modules = path.join(app, 'node_modules');
  fs.cpSync(INSTALLED, path.join(modules, 'aiwaf-wasm'), { recursive: true });
  fs.mkdirSync(path.join(modules, 'aiwaf'), { recursive: true });
  fs.cpSync(path.join(ROOT, 'lib'), path.join(modules, 'aiwaf', 'lib'), { recursive: true });
  fs.copyFileSync(path.join(ROOT, 'package.json'), path.join(modules, 'aiwaf', 'package.json'));
  // Makes the plain require('aiwaf-wasm') fail, so only the fallbacks can load it.
  fs.writeFileSync(path.join(app, 'block.js'), `
const Module = require('module');
const load = Module._load;
Module._load = function (request, ...rest) {
  if (request === 'aiwaf-wasm') throw new Error('blocked for the test');
  return load.call(this, request, ...rest);
};
`);
});

afterAll(() => fs.rmSync(app, { recursive: true, force: true }));

function run(script) {
  return execFileSync(process.execPath, ['--no-warnings', '-r', './block.js', '-e', script], {
    cwd: app, encoding: 'utf8', env: { ...process.env, AIWAF_DISABLE_WASM: '' }
  }).trim();
}

test('aiwaf is installed without its own copy of aiwaf-wasm', () => {
  expect(fs.existsSync(path.join(app, 'node_modules', 'aiwaf', 'node_modules'))).toBe(false);
});

test('the fallback loader finds a hoisted aiwaf-wasm', () => {
  const out = run(`
const adapter = require('aiwaf/lib/wasmAdapter');
adapter.loadWasm().then(mod => {
  console.log(JSON.stringify({
    dir: adapter.wasmPackageDir(),
    sql: Boolean(mod) && typeof mod.sql_inspect === 'function',
    routes: Boolean(mod) && typeof mod.RouteRiskChecker === 'function'
  }));
});`);
  const result = JSON.parse(out.split('\n').pop());
  expect(fs.realpathSync(result.dir)).toBe(fs.realpathSync(path.join(app, 'node_modules', 'aiwaf-wasm')));
  expect(result).toMatchObject({ sql: true, routes: true });
});

test('wasmPackageDir is null when aiwaf-wasm is not installed', () => {
  fs.renameSync(path.join(app, 'node_modules', 'aiwaf-wasm'), path.join(app, 'aiwaf-wasm-away'));
  try {
    expect(run(`console.log(String(require('aiwaf/lib/wasmAdapter').wasmPackageDir()))`)).toBe('null');
  } finally {
    fs.renameSync(path.join(app, 'aiwaf-wasm-away'), path.join(app, 'node_modules', 'aiwaf-wasm'));
  }
});
