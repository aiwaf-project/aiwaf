'use strict';

// Uses audit findings to check risky routes' inputs on live requests. The checks run in
// aiwaf-wasm. Matches are logged and, in block mode (the default), blocked. Log events never
// include request values.
const { loadManifest, writeManifest } = require('./pathManifest');
const { bodyBytes } = require('./sqlInjection');
const wasmChecks = require('./wasmChecks');
const { mergeInto } = require('./inputSchema');
const inputSources = require('./inputSources');

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

/**
 * Strict inputs (AIWAF_STRICT_INPUTS, off by default): on a route whose saved input list is
 * complete, a query or body name the route never reads is an unexpected input, logged in the
 * route schema's monitor mode and blocked in enforce mode. AIWAF_STRICT_INPUTS_ALLOW adds names
 * to always allow (a trailing * matches any rest) to the engine's own: CSRF tokens, utm_*,
 * fbclid, gclid, msclkid and _. Returns { allow } when on, or null.
 */
function strictInputs(options = {}) {
  const raw = options.AIWAF_STRICT_INPUTS ?? process.env.AIWAF_STRICT_INPUTS ?? '';
  const on = raw === true || ['1', 'true', 'yes', 'on'].includes(String(raw).trim().toLowerCase());
  if (!on) return null;
  const names = (value, setting) => {
    const listed = value ?? process.env[setting] ?? [];
    return (Array.isArray(listed) ? listed : String(listed).split(',')).map(name => String(name).trim()).filter(Boolean);
  };
  // AIWAF_STRICT_INPUTS_EXEMPT: path prefixes strict inputs skips, as /webhooks/ or /legacy/*.
  return { allow: names(options.AIWAF_STRICT_INPUTS_ALLOW, 'AIWAF_STRICT_INPUTS_ALLOW'),
    exempt: names(options.AIWAF_STRICT_INPUTS_EXEMPT, 'AIWAF_STRICT_INPUTS_EXEMPT') };
}

/** The manifest the engine gets: with strict_inputs added when strict inputs is on. */
const withStrict = (manifest, strict) => (strict && manifest && typeof manifest === 'object' ? { ...manifest, strict_inputs: strict } : manifest);

const hasRisks = manifest => Object.values((manifest && manifest.routes) || {})
  .some(entry => entry && Array.isArray(entry.injection_risks) && entry.injection_risks.length);
const hasInputs = manifest => Object.values((manifest && manifest.routes) || {})
  .some(entry => entry && entry.inputs && typeof entry.inputs === 'object' && !Array.isArray(entry.inputs));

/**
 * Reviews a request against the inputs init saved for its route, after the general check found
 * `found` (a rule, or null). Returns {action: 'mismatch', route, input, type},
 * {action: 'relax', route, rule}, or null to keep the general check's answer. See aiwaf-rust.
 */
function reviewRequest(checker, path, query, body, maxBytes, found, method = '') {
  const text = typeof query === 'string' ? query : queryString(query).join('&');
  const bytes = bodyBytes(body) || Buffer.alloc(0);
  // With the method, strict inputs check that method's own list; older engines only have review.
  if (method && typeof checker.review_method === 'function') {
    return JSON.parse(checker.review_method(String(method), String(path || '/'), text, bytes, maxBytes, found || ''));
  }
  return JSON.parse(checker.review(String(path || '/'), text, bytes, maxBytes, found || ''));
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
  const strict = schema === 'off' ? null : strictInputs(options);
  const manifest = withStrict(loadManifest(options.AIWAF_PATH_MANIFEST || undefined), strict);
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
    if (!checker) {
      checker = compileRisks(manifest);
      if (strict && strict.exempt.length && checker.strict_exempt !== true) {
        // An older engine would check the exempt paths too: better none at all.
        const { strict_inputs: _unchecked, ...plain } = manifest;
        checker = compileRisks(plain);
        try {
          logger.warn?.('AIWAF: AIWAF_STRICT_INPUTS_EXEMPT needs aiwaf-wasm 0.2.13 or newer, so strict inputs aren\'t checked rather than checked on exempt paths. Run: npm install aiwaf-wasm@latest');
        } catch (_) { /* Logging must not decide the outcome. */ }
      } else if (strict && schema !== 'off' && checker.strict !== true) {
        try {
          logger.warn?.('AIWAF: strict inputs need aiwaf-wasm 0.2.12 or newer, so they aren\'t checked. Run: npm install aiwaf-wasm@latest');
        } catch (_) { /* Logging must not decide the outcome. */ }
      }
    }
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
      const detail = action === 'mismatch' ? { input: result.input, type: result.type }
        : action === 'unexpected' ? { input: result.input } : { rule: result.rule };
      const event = { event: 'aiwaf.route_schema', mode: schema, method: String(req.method || ''), route, action, ...detail };
      try { logger.warn?.(event); } catch (_) { /* Logging must not decide the outcome. */ }
      if (schema !== 'enforce') return finding;
      if (action === 'mismatch') return { rule: 'schema_type_mismatch', status: 403, mode: 'block' };
      return action === 'unexpected' ? { rule: 'schema_unexpected_input', status: 403, mode: 'block' } : null;
    };
    observe.review = (req, finding, maxBytes, body = req.body) => decide(req, finding, () => {
      const [path, query] = target(req);
      const found = finding && finding.status === 403 ? finding.rule : null;
      return reviewRequest(checker, path, query, body, maxBytes, found, String(req.method || ''));
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
    // The route's view is its last handler; the others it records by name, with their methods.
    return (view && view.split('.').pop() === name) || Object.hasOwn(routes[path]?.handler_methods || {}, name);
  });
  return named.length === 1 ? named : [];
}

/**
 * Replaces saved risks in a manifest with this report's findings, and counts the results. sources,
 * from inputSources.sourcesFromEnvironment, adds declared, OpenAPI and recorded inputs.
 */
function applyReport(report, manifest, sources = {}) {
  const routes = manifest && manifest.routes;
  if (!routes || typeof routes !== 'object' || Array.isArray(routes)) throw new Error('The manifest has no routes');
  for (const entry of Object.values(routes)) {
    if (entry && typeof entry === 'object') {
      delete entry.injection_risks;
      delete entry.inputs;
      delete entry.inputs_complete;
      delete entry.inputs_incomplete;
      delete entry.inputs_by_method;
      delete entry.inputs_nested;
      delete entry.inputs_types_by_method;
      delete entry.inputs_types_by_location;
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
  const read = new Set();
  // Names the source defines once: a handler a factory makes for several routes, as
  // makeExport(format) for each format, shares its name across them, so it links to each.
  const defined = {};
  for (const definition of report.definitions || []) defined[definition.function] = (defined[definition.function] || 0) + 1;
  const linkedTo = item => {
    const linked = routesFor(item, routes);
    const name = String(item.function || '');
    if (linked.length || defined[name] !== 1 || name.includes('.') || name.includes(' ')) return linked;
    return Object.keys(routes).filter(path => {
      const view = String((routes[path] && routes[path].view) || '').replace(/^bound /, '');
      return view && view === name;
    });
  };
  // A request hook's reads (fastify.addHook('preHandler', ...)) are every route's: the app
  // doesn't say which routes a hook's scope holds, so each route allows its names, with their
  // types. Its own handler still has to be read for the route to count as audited.
  for (const entry of (report.inputs || []).filter(item => item && item.hook)) {
    for (const path of Object.keys(routes)) {
      if (!routes[path] || typeof routes[path] !== 'object') continue;
      mergeInto(routes, path, entry);
      routes[path]._input_shared = [...new Set([...(routes[path]._input_shared || []), ...Object.keys(entry.inputs || {})])].sort();
    }
  }
  for (const entry of (report.inputs || []).filter(item => item && !item.hook)) {
    // The method a handler serves, from its label, as app.get /items; unknown for a named one.
    const labelled = (/^(?:\S+\.)?(get|post|put|patch|delete|head|options) \//i.exec(String(entry.function || '')) || [])[1];
    for (const path of linkedTo(entry)) {
      read.add(path);
      mergeInto(routes, path, entry);
      const route = routes[path];
      // Or, for a handler passed by name, the methods the app routed it for.
      const named = (route.handler_methods || {})[String(entry.function || '').split('.').pop()];
      const methods = labelled ? [labelled.toUpperCase()] : (named || []);
      if (methods.length) {
        route._input_methods = route._input_methods || {};
        for (const method of methods) {
          const names = new Set([...(route._input_methods[method] || []), ...Object.keys(entry.inputs || {})]);
          route._input_methods[method] = [...names].sort();
        }
      } else {
        route._input_shared = [...new Set([...(route._input_shared || []), ...Object.keys(entry.inputs || {})])].sort();
      }
    }
  }
  // A handler defined in the audited source was read even when it reads nothing itself.
  for (const definition of report.definitions || []) {
    for (const path of linkedTo(definition)) read.add(path);
  }
  // A framework list for a handler the audit didn't read still goes in, as incomplete
  // (finalize skips a route it can't tell was read otherwise).
  for (const [path, entry] of Object.entries(routes)) {
    const framework = entry && typeof entry === 'object' ? entry.declared_inputs : null;
    if (read.has(path) || !framework || typeof framework !== 'object' || framework.standalone) continue;
    entry.inputs_complete = false;
    entry.inputs_incomplete = [{ code: 'unread_view', detail: `the audit didn't read ${entry.view || 'its handler'}` }];
  }
  const completeness = inputSources.finalize(routes, sources || {});
  // A route none of whose handlers the audit read has inputs only from the framework or a
  // declaration: complete as a list, but its risks weren't looked for. So it isn't complete.
  for (const [path, entry] of Object.entries(routes)) {
    if (!entry || typeof entry !== 'object' || read.has(path) || !entry.inputs || entry.inputs_complete !== true) continue;
    entry.inputs_complete = false;
    entry.inputs_incomplete = [{ code: 'unread_view', detail: `the audit didn't read ${entry.view || 'its handler'}` }];
    delete entry.inputs_by_method;
    delete entry.inputs_nested;
    completeness.complete -= 1;
    completeness.incomplete += 1;
    completeness.reasons.unread_view = (completeness.reasons.unread_view || 0) + 1;
  }
  const typed = Object.values(routes).filter(entry => entry && typeof entry === 'object' && entry.inputs).length;
  manifest.injection_audit = {
    generated_at: new Date().toISOString().replace(/\.\d{3}Z$/, 'Z'),
    files_scanned: Number(report.files_scanned || 0),
    linked_findings: linked,
    unlinked_findings: unlinked,
    routes_with_inputs: typed,
    routes_with_complete_inputs: completeness.complete
  };
  if (Object.keys(completeness.reasons).length) {
    manifest.injection_audit.incomplete_input_reasons = Object.fromEntries(Object.entries(completeness.reasons).sort());
  }
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
  strictInputs,
  updateManifest,
  withStrict
};
