'use strict';

// Warns before the tests run when node_modules has a published aiwaf-wasm rather than this
// repository's build, as after npm install: the tests would pass or fail against the wrong
// engine. Outside the repository, as in an installed package, it does nothing.
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const FIX = 'Run: python scripts/build_wasm_from_source.py (from the repository root)';

const digest = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');

/** The warning for this checkout, or null when node_modules has the local build. */
function localWasmWarning(root = path.join(__dirname, '..')) {
  const crate = path.join(root, '..', 'rust', 'crates', 'aiwaf_wasm');
  const cargo = path.join(crate, 'Cargo.toml');
  if (!fs.existsSync(cargo)) return null;
  const wanted = (/^version\s*=\s*"([^"]+)"/m.exec(fs.readFileSync(cargo, 'utf8')) || [])[1];
  const installed = path.join(root, 'node_modules', 'aiwaf-wasm');
  if (!fs.existsSync(path.join(installed, 'package.json'))) return `aiwaf-wasm isn't installed. ${FIX}`;
  const version = JSON.parse(fs.readFileSync(path.join(installed, 'package.json'), 'utf8')).version;
  if (wanted && version !== wanted) {
    return `node_modules has aiwaf-wasm ${version}, but this repository builds ${wanted}, so the tests `
      + `would run against the published engine. ${FIX}`;
  }
  const built = path.join(crate, 'pkg', 'aiwaf_wasm_bg.wasm');
  const used = path.join(installed, 'aiwaf_wasm_bg.wasm');
  if (fs.existsSync(built) && fs.existsSync(used) && digest(built) !== digest(used)) {
    return `node_modules/aiwaf-wasm isn't the build in rust/crates/aiwaf_wasm/pkg, so the tests `
      + `would run against another engine. ${FIX}`;
  }
  return null;
}

if (require.main === module) {
  const warning = localWasmWarning();
  if (warning) console.warn(`\nWARNING: ${warning}\n`);
}

module.exports = { localWasmWarning };
