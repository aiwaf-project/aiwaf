'use strict';

// Uses audit findings to check risky routes' inputs on live requests. The checks run in
// aiwaf-wasm. Matches are logged and, in block mode (the default), blocked. Log events never
// include request values.
const { loadManifest, writeManifest } = require('./pathManifest');
const { bodyBytes } = require('./sqlInjection');
const wasmChecks = require('./wasmChecks');
const { mergeInto } = require('./inputSchema');

// The operations a saved finding can name. The rules for each run in aiwaf-wasm.
const SINKS = [
  'sql', 'nosql', 'command', 'path', 'ssrf', 'code', 'template', 'deserialization', 'xss', 'redirect', 'ldap', 'xpath', 'xxe',
  'redos', 'prototype'
];

/** Returns the first rule for this operation that matches the value. Needs aiwaf-wasm loaded. */
function ruleFor(sink, value) {
  return wasmChecks.get().route_rule(String(sink), String(value)) ?? null;
}

/** Compiles a manifest's saved risks in aiwaf-wasm. Needs aiwaf-wasm loaded. */
function compileRisks(manifest) {
  return new (wasmChecks.get().RouteRiskChecker)(JSON.stringify(manifest ?? null));
}

// Turns a parsed query object back into name=value pairs, as in user[$ne]=1.
function queryString(query, prefix = '', out = []) {
  if (query === null || query === undefined) return out;
  if (typeof query !== 'object') {
    out.push(`${encodeURIComponent(prefix)}=${encodeURIComponent(String(query))}`);
    return out;
  }
  for (const [key, value] of Object.entries(query)) queryString(value, prefix ? `${prefix}[${key}]` : key, out);
  return out;
}

/** Checks one request against a compiled checker and returns its matches. */
function inspectRisks(checker, path, query, body) {
  const text = typeof query === 'string' ? query : queryString(query).join('&');
  return JSON.parse(checker.inspect(String(path || '/'), text, bodyBytes(body) || Buffer.alloc(0)));
}

const hasRisks = manifest => Object.values((manifest && manifest.routes) || {})
  .some(entry => entry && Array.isArray(entry.injection_risks) && entry.injection_risks.length);
const hasInputs = manifest => Object.values((manifest && manifest.routes) || {})
  .some(entry => entry && entry.inputs && typeof entry.inputs === 'object' && !Array.isArray(entry.inputs));

/**
 * Reviews a request against the inputs init saved for its route, after the general check found
 * `found` (a rule, or null). Returns {action: 'mismatch', route, input, type},
 * {action: 'relax', route, rule}, or null to keep the general check's answer. See aiwaf-rust.
 */
function reviewRequest(checker, path, query, body, maxBytes, found) {
  const text = typeof query === 'string' ? query : queryString(query).join('&');
  return JSON.parse(checker.review(String(path || '/'), text, bodyBytes(body) || Buffer.alloc(0), maxBytes, found || ''));
}

/**
 * Returns a function that checks a request, or null when there's nothing to check. The function
 * returns a blocking finding in block mode, and null otherwise.
 */
function createObserver(options = {}) {
  const mode = String(options.AIWAF_INJECTION_RISK_MODE ?? process.env.AIWAF_INJECTION_RISK_MODE ?? 'block').trim().toLowerCase();
  if (!['block', 'log', 'off'].includes(mode)) throw new Error('AIWAF_INJECTION_RISK_MODE must be block, log, or off');
  let schema = String(options.AIWAF_ROUTE_SCHEMA_MODE ?? process.env.AIWAF_ROUTE_SCHEMA_MODE ?? 'monitor').trim().toLowerCase();
  if (!['enforce', 'monitor', 'off'].includes(schema)) throw new Error('AIWAF_ROUTE_SCHEMA_MODE must be enforce, monitor, or off');
  const manifest = loadManifest(options.AIWAF_PATH_MANIFEST || undefined);
  if (!hasInputs(manifest)) schema = 'off';
  if ((mode === 'off' || !hasRisks(manifest)) && schema === 'off') return null;
  let checker = null;
  const logger = options.logger || console;
  // Adapters can inspect twice: before and after the body is parsed.
  const logged = new WeakMap();
  const target = req => {
    const url = String(req.path || req.url || '/');
    const raw = String(req.originalUrl || req.url || '').split('?').slice(1).join('?');
    // Compiled on first use, after the middleware has loaded aiwaf-wasm.
    checker = checker || compileRisks(manifest);
    return [url.split('?')[0], raw || req.query];
  };
  const observe = req => {
    if (mode === 'off') return null;
    let events;
    try {
      const [path, query] = target(req);
      events = inspectRisks(checker, path, query, req.body);
    } catch (_) {
      // A failing check must not break the request. Other protections still apply.
      return null;
    }
    if (!events.length) return null;
    const seen = logged.get(req) || new Set();
    logged.set(req, seen);
    for (const event of events) {
      const key = JSON.stringify(event);
      if (seen.has(key)) continue;
      seen.add(key);
      try {
        logger.warn?.({ event: 'aiwaf.injection_risk', mode, method: String(req.method || ''), ...event });
      } catch (_) { /* Logging must not decide the outcome. */ }
    }
    return mode === 'block' ? { rule: events[0].rule, status: 403, mode: 'block' } : null;
  };
  if (schema !== 'off') {
    /**
     * The general check's finding (or null), after the route's saved inputs review it: enforce
     * blocks a typed input of another type and drops findings the schema clears; monitor logs.
     */
    const decide = (req, finding, run) => {
      let result;
      try {
        result = run();
      } catch (_) {
        return finding;
      }
      if (!result) return finding;
      const { action, route } = result;
      const event = { event: 'aiwaf.route_schema', mode: schema, method: String(req.method || ''), route, action,
        ...(action === 'mismatch' ? { input: result.input, type: result.type } : { rule: result.rule }) };
      try { logger.warn?.(event); } catch (_) { /* Logging must not decide the outcome. */ }
      if (schema !== 'enforce') return finding;
      return action === 'mismatch' ? { rule: 'schema_type_mismatch', status: 403, mode: 'block' } : null;
    };
    observe.review = (req, finding, maxBytes, body = req.body) => decide(req, finding, () => {
      const [path, query] = target(req);
      const found = finding && finding.status === 403 ? finding.rule : null;
      return reviewRequest(checker, path, query, body, maxBytes, found);
    });
    /** A finding from the URL path, reviewed against the route's parameters. */
    observe.reviewPath = (req, finding, maxBytes) => (finding.status !== 403 ? finding : decide(req, finding, () => {
      const [path] = target(req);
      return JSON.parse(checker.review_path(path, maxBytes, finding.rule));
    }));
  }
  return observe;
}

function inputFromSource(source) {
  const text = String(source || '');
  let match = /^@\w+ (.+)$/.exec(text);
  if (match) return match[1];
  match = /\('(.+?)'\)$/.exec(text);
  if (match) return match[1];
  match = /(?:\.|^request )(?:body|params|payload|query)\.([^.]+)$/.exec(text);
  return match ? match[1] : '';
}

const trimRoute = path => (path.length > 1 ? path.replace(/\/+$/, '') : path);

function fileRoute(filename) {
  const file = String(filename).replace(/\\/g, '/');
  let match = /(?:^|\/)pages(\/api\/.*?)(?:\/index)?\.[cm]?[jt]sx?$/.exec(file);
  if (match) return match[1].replace(/\[\.{3}([^\]]+)\]/g, ':$1*').replace(/\[([^\]]+)\]/g, ':$1');
  match = /(?:^|\/)app(\/.*?)\/route\.[cm]?[jt]sx?$/.exec(file);
  return match ? match[1].replace(/\[\.{3}([^\]]+)\]/g, ':$1*').replace(/\[([^\]]+)\]/g, ':$1') : '';
}

function routesFor(finding, routes) {
  const entry = String(finding.function || '').split(' -> ')[0];
  const paths = Object.keys(routes);
  const fromSource = (/^\S+ (\/\S*)$/.exec(entry) || [])[1] || fileRoute(finding.entry_file || finding.file);
  if (fromSource) {
    const wanted = trimRoute(fromSource);
    const exact = paths.filter(path => trimRoute(path) === wanted);
    if (exact.length) return exact;
    // A router mounted under a prefix. Only trust this when one route fits.
    const mounted = paths.filter(path => trimRoute(path).endsWith(wanted));
    if (mounted.length === 1) return mounted;
    if (entry.includes(' ')) return [];
  }
  // JavaScript views are bare function names, so a name used twice can't be trusted.
  const name = entry.split('.').pop();
  const named = paths.filter(path => {
    const view = String(routes[path]?.view || '').replace(/^bound /, '');
    return view && view.split('.').pop() === name;
  });
  return named.length === 1 ? named : [];
}

/** Replaces saved risks in a manifest with this report's findings, and counts the results. */
function applyReport(report, manifest) {
  const routes = manifest && manifest.routes;
  if (!routes || typeof routes !== 'object' || Array.isArray(routes)) throw new Error('The manifest has no routes');
  for (const entry of Object.values(routes)) {
    if (entry && typeof entry === 'object') {
      delete entry.injection_risks;
      delete entry.inputs;
      delete entry.inputs_complete;
    }
  }
  let linked = 0;
  let unlinked = 0;
  const collected = new Map();
  for (const finding of report.findings || []) {
    const matches = routesFor(finding, routes);
    if (matches.length) linked += 1;
    else unlinked += 1;
    for (const path of matches) {
      const risk = { sink: String(finding.sink), input: inputFromSource(finding.source), confidence: String(finding.confidence) };
      if (!collected.has(path)) collected.set(path, new Map());
      collected.get(path).set(`${risk.sink}\0${risk.input}`, risk);
    }
  }
  for (const [path, risks] of collected) {
    routes[path].injection_risks = [...risks.keys()].sort().map(key => risks.get(key));
  }
  // What each route reads, with types. A route whose handlers weren't all found gets none.
  for (const entry of report.inputs || []) {
    for (const path of routesFor(entry, routes)) mergeInto(routes, path, entry);
  }
  const typed = Object.values(routes).filter(entry => entry && typeof entry === 'object' && entry.inputs).length;
  manifest.injection_audit = {
    generated_at: new Date().toISOString().replace(/\.\d{3}Z$/, 'Z'),
    files_scanned: Number(report.files_scanned || 0),
    linked_findings: linked,
    unlinked_findings: unlinked,
    routes_with_inputs: typed
  };
  return { routes: collected.size, linked, unlinked, inputs: typed };
}

function updateManifest(report, manifestPath) {
  const manifest = loadManifest(manifestPath);
  if (!manifest) throw new Error(`Manifest not found or unreadable: ${manifestPath}. Generate it first with init.`);
  const counts = applyReport(report, manifest);
  writeManifest(manifest, manifestPath);
  return counts;
}

function formatCounts(counts, manifestPath) {
  const saved = `Saved ${counts.linked} possible injection paths to ${counts.routes} routes in ${manifestPath}.`;
  return counts.unlinked
    ? `${saved} ${counts.unlinked} couldn't be tied to a route, so they get no route-specific checks.`
    : saved;
}

module.exports = {
  SINKS,
  applyReport,
  compileRisks,
  createObserver,
  formatCounts,
  inputFromSource,
  inspectRisks,
  reviewRequest,
  ruleFor,
  updateManifest
};
