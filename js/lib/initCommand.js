'use strict';

// `aiwaf init` and `aiwaf manifest`: write .aiwaf/paths.json from the app's own routes.
//
// init loads the app, as Python's init does, so the manifest has every route the app registers,
// with its real handlers for the API and input checks. manifest takes a JSON route list for
// apps that can't be loaded that way. Both refuse to write a manifest with no routes: it would
// look configured while protecting no route, and would replace one that works.

const path = require('path');

const DEFAULT_OUTPUT = process.env.AIWAF_PATH_MANIFEST || '.aiwaf/paths.json';

function readJson(file) {
  return JSON.parse(require('fs').readFileSync(file, 'utf8'));
}

function parseArgs(args = []) {
  const options = {
    framework: null,
    output: DEFAULT_OUTPUT,
    routes: null,
    source: '.',
    audit: true,
    app: null,
    exportName: null,
    waitMs: 3000,
    help: false
  };
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    const next = () => {
      const value = args[index + 1];
      if (value === undefined || String(value).startsWith('--')) throw new Error(`${arg} needs a value`);
      index += 1;
      return value;
    };
    if (arg === '--framework' || arg === '-f') options.framework = String(next()).toLowerCase();
    else if (arg === '--output' || arg === '-o') options.output = next();
    else if (arg === '--source') options.source = next();
    else if (arg === '--no-audit') options.audit = false;
    else if (arg === '--routes' || arg === '-r') options.routes = readJson(next());
    else if (arg === '--app' || arg === '-a') options.app = next();
    else if (arg === '--export' || arg === '-e') options.exportName = next();
    else if (arg === '--wait') options.waitMs = Number(next());
    else if (arg === '--help' || arg === '-h') options.help = true;
    else if (!arg.startsWith('-') && !options.app) options.app = arg;
    else throw new Error(`Unknown option: ${arg}`);
  }
  // AdonisJS's list:routes --json groups routes by domain in an object.
  if (options.routes !== null && !Array.isArray(options.routes) && !(options.framework === 'adonis' && options.routes && typeof options.routes === 'object')) {
    throw new Error('--routes must be a JSON array of routes');
  }
  if (!Number.isFinite(options.waitMs) || options.waitMs < 0) throw new Error('--wait must be a number of milliseconds');
  return options;
}

/** init's summary of a manifest's saved audit, as Python's describe_audit words it. */
function describeAudit(manifest) {
  const audit = manifest && manifest.injection_audit;
  if (!audit || typeof audit !== 'object') return 'Injection checks: not run';
  const routes = Object.values(manifest.routes || {}).filter(entry => entry && entry.injection_risks).length;
  let text = `Injection checks: ${routes} routes from ${audit.files_scanned || 0} files`;
  if (audit.routes_with_inputs) {
    text += `; inputs and types for ${audit.routes_with_inputs} routes, complete for ${audit.routes_with_complete_inputs || 0}`;
    const reasons = Object.entries(audit.incomplete_input_reasons || {})
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, 4);
    if (reasons.length) text += ` (incomplete because of ${reasons.map(([code, count]) => `${code} ${count}`).join(', ')})`;
  }
  if (audit.unlinked_findings) text += ` (${audit.unlinked_findings} findings couldn't be tied to a route)`;
  return text;
}

class EmptyManifestError extends Error {
  constructor(where, output) {
    super(`No routes found in ${where}, so ${output} was not written. `
      + 'Check that the app is loaded after its routes and routers are registered, or list them with --routes.');
    this.name = 'EmptyManifestError';
  }
}

function writeFound(manifest, output, where) {
  if (!Object.keys(manifest.routes || {}).length) throw new EmptyManifestError(where, output);
  require('./pathManifest').writeManifest(manifest, output);
}

// What each newer manifest entry needs from the engine, by the engine's feature name.
const ENGINE_FEATURES = {
  method_types: 'per-method types (inputs_types_by_method)',
  location_types: 'query and body types (inputs_types_by_location)',
  nested_paths: 'nested keys below the first level (inputs_nested)'
};

/**
 * The checks a manifest has that the installed aiwaf-wasm can't make, described. The manifest
 * stays as written: an engine that can't read an entry checks the type every method agrees on.
 */
function engineGaps(manifest, supports) {
  const used = new Set();
  for (const entry of Object.values((manifest && manifest.routes) || {})) {
    if (!entry || typeof entry !== 'object') continue;
    if (entry.inputs_types_by_method && Object.keys(entry.inputs_types_by_method).length) used.add('method_types');
    if (entry.inputs_types_by_location && Object.keys(entry.inputs_types_by_location).length) used.add('location_types');
    if (Object.keys(entry.inputs_nested || {}).some(key => key.includes('.') || key.endsWith('[]'))) used.add('nested_paths');
  }
  return [...used].sort().filter(feature => !(supports && supports(feature))).map(feature => ENGINE_FEATURES[feature]);
}

async function engineProbe() {
  try {
    const wasm = await require('./wasmAdapter').loadWasm();
    return wasm && typeof wasm.supports_feature === 'function' ? feature => wasm.supports_feature(feature) : null;
  } catch (_) {
    return null;
  }
}

function summary(manifest, output, extra = []) {
  const routes = Object.keys(manifest.routes || {});
  const api = routes.filter(name => {
    const entry = manifest.routes[name];
    return entry && (entry.response_type === 'json' || (entry.protections && entry.protections.api_rate_limit));
  }).length;
  return [
    `Generated ${output}`,
    `Framework: ${manifest.framework}`,
    ...extra,
    `Routes: ${routes.length} (${api} API)`,
    `Context hash: ${manifest.context_hash}`,
    describeAudit(manifest)
  ].join('\n');
}

const INIT_HELP = `Usage: aiwaf init [--app <file>] [--export <name>] [--framework <name>]
                  [--output .aiwaf/paths.json] [--source .] [--no-audit] [--wait 3000]

Loads your app and writes the routes it registers to the manifest. Without --app it uses
package.json's main, else app.js, server.js or index.js (also under src/ and dist/).
--export names the export holding the app, or a function returning it (default: the module's
default export, app or server). Servers the app starts don't bind a port while it loads.`;

/** aiwaf init: loads the app and writes its manifest. Returns the exit code. */
async function runInit(args, io = { out: console.log, err: console.error }) {
  let options;
  try {
    options = parseArgs(args);
  } catch (error) {
    io.err(`${error.message}\n\n${INIT_HELP}`);
    return 2;
  }
  if (options.help) {
    io.out(INIT_HELP);
    return 0;
  }
  const pathManifest = require('./pathManifest');
  const auditSource = options.audit ? options.source : null;
  try {
    let manifest;
    let where;
    const extra = [];
    if (options.routes) {
      const framework = options.framework || 'express';
      manifest = pathManifest.buildFrameworkManifest(framework, null, { routes: options.routes, auditSource, handlerSource: options.source });
      where = 'the --routes list';
    } else {
      const loaded = await require('./appLoader').loadApp({ app: options.app, exportName: options.exportName, waitMs: options.waitMs });
      const framework = options.framework || loaded.framework;
      const file = path.relative(process.cwd(), loaded.file) || loaded.file;
      // Handlers and schemas Fastify doesn't hand back come from the source, audited or not.
      manifest = pathManifest.buildFrameworkManifest(framework, loaded.app, { auditSource, handlerSource: options.source });
      where = `${file} (${framework})`;
      extra.push(`App: ${file}${loaded.exportName ? ` (export ${loaded.exportName})` : ''}`);
      if (framework === 'koa' && !Object.keys(manifest.routes || {}).length) {
        where += '; Koa keeps routes only in @koa/router or koa-router middleware';
      }
    }
    writeFound(manifest, options.output, where);
    io.out(summary(manifest, options.output, extra));
    const gaps = engineGaps(manifest, await engineProbe());
    if (gaps.length) {
      io.err(`The installed aiwaf-wasm can't check ${gaps.join(', ')}; it checks the type every method agrees on instead. `
        + 'Run: npm install aiwaf-wasm@latest');
    }
    return 0;
  } catch (error) {
    io.err(error.message || String(error));
    return 1;
  }
}

/**
 * aiwaf manifest: with --routes, the listed routes; without, the same as init, so it never writes
 * an empty manifest from no input. Prints JSON, as before. Returns the exit code.
 */
async function runManifest(args, io = { out: console.log, err: console.error }) {
  let options;
  try {
    options = parseArgs(args);
  } catch (error) {
    io.err(error.message);
    return 2;
  }
  const pathManifest = require('./pathManifest');
  const framework = options.framework || 'express';
  const auditSource = options.audit ? options.source : null;
  try {
    let manifest;
    let where;
    if (options.routes) {
      manifest = pathManifest.buildFrameworkManifest(framework, null, { routes: options.routes, auditSource, handlerSource: options.source });
      where = 'the --routes list';
    } else {
      const loaded = await require('./appLoader').loadApp({ app: options.app, exportName: options.exportName, waitMs: options.waitMs });
      manifest = pathManifest.buildFrameworkManifest(options.framework || loaded.framework, loaded.app, { auditSource });
      where = path.relative(process.cwd(), loaded.file) || loaded.file;
    }
    writeFound(manifest, options.output, where);
    const audit = manifest.injection_audit;
    io.out(JSON.stringify({
      ok: true,
      output: options.output,
      framework: manifest.framework,
      routes: Object.keys(manifest.routes || {}).length,
      injection_check_routes: Object.values(manifest.routes || {}).filter(entry => entry && entry.injection_risks).length,
      unlinked_findings: audit ? audit.unlinked_findings : null
    }, null, 2));
    return 0;
  } catch (error) {
    io.err(JSON.stringify({ ok: false, error: error.message || String(error) }, null, 2));
    return 1;
  }
}

module.exports = { runInit, runManifest, parseArgs, describeAudit, engineGaps, EmptyManifestError, INIT_HELP };
