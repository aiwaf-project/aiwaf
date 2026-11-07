'use strict';

// The SQL check and route checks run in aiwaf-wasm, which Node.js requires, as Python requires
// aiwaf-rust. There is no JavaScript copy. Call load() once before the first check.
const NEEDED = ['sql_inspect', 'sql_tokenizer', 'route_rule', 'RouteRiskChecker'];
const MESSAGE = "AIWAF's request checks need aiwaf-wasm 0.2.6 or newer. Run: npm install aiwaf-wasm@^0.2.6";

let pending = null;
let loaded = null;

function check(mod) {
  if (!mod || NEEDED.some(name => typeof mod[name] !== 'function')) throw new Error(MESSAGE);
  return mod;
}

/** Loads aiwaf-wasm once. Rejects with an install hint when it's missing or too old. */
function load() {
  if (!pending) {
    pending = require('./wasmAdapter').loadWasm().then(mod => {
      loaded = check(mod);
      return loaded;
    });
    // A failed load is reported to whoever awaits it, and retried on the next call.
    pending.catch(() => { pending = null; });
  }
  return pending;
}

/** Returns the loaded module. Throws when load() hasn't finished. */
function get() {
  if (!loaded) throw new Error(`aiwaf-wasm isn't loaded yet. Await its load() first. ${MESSAGE}`);
  return loaded;
}

module.exports = { load, get, MESSAGE };
