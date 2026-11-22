const fs = require('fs');
const os = require('os');
const path = require('path');
const { localWasmWarning } = require('../scripts/check-local-wasm');

// A repository layout: js/ with node_modules, and rust/crates/aiwaf_wasm with its pkg build.
function layout({ crate = '0.2.10', installed = '0.2.10', built = 'local', used = 'local' } = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-local-wasm-'));
  const js = path.join(root, 'js');
  const wasm = path.join(root, 'rust', 'crates', 'aiwaf_wasm');
  fs.mkdirSync(path.join(wasm, 'pkg'), { recursive: true });
  if (crate) fs.writeFileSync(path.join(wasm, 'Cargo.toml'), `[package]\nname = "aiwaf-wasm"\nversion = "${crate}"\n`);
  if (built) fs.writeFileSync(path.join(wasm, 'pkg', 'aiwaf_wasm_bg.wasm'), built);
  if (installed) {
    fs.mkdirSync(path.join(js, 'node_modules', 'aiwaf-wasm'), { recursive: true });
    fs.writeFileSync(path.join(js, 'node_modules', 'aiwaf-wasm', 'package.json'), JSON.stringify({ version: installed }));
    fs.writeFileSync(path.join(js, 'node_modules', 'aiwaf-wasm', 'aiwaf_wasm_bg.wasm'), used);
  }
  return { root, js };
}

test('warns when node_modules has another aiwaf-wasm than the local build', () => {
  const made = [];
  const check = options => { const item = layout(options); made.push(item.root); return localWasmWarning(item.js); };
  try {
    expect(check()).toBeNull();
    expect(check({ installed: '0.2.9' })).toMatch(/has aiwaf-wasm 0\.2\.9, but this repository builds 0\.2\.10/);
    expect(check({ used: 'published' })).toMatch(/isn't the build in rust\/crates\/aiwaf_wasm\/pkg/);
    expect(check({ installed: null })).toMatch(/isn't installed/);
    expect(check({ built: null })).toBeNull();
    // Outside the repository, as in an installed package, it does nothing.
    expect(check({ crate: null })).toBeNull();
  } finally {
    for (const root of made) fs.rmSync(root, { recursive: true, force: true });
  }
});

test('this checkout has the local build installed', () => {
  expect(localWasmWarning()).toBeNull();
});
