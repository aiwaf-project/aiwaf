'use strict';

// Uses audit findings on live requests, in log-only mode. See py/aiwaf/core/route_risks.py.
// This never blocks or changes a request, and log events never include request values.
const { RULES: SQL_RULES } = require('./sqlInjection');
const { loadManifest, writeManifest } = require('./pathManifest');

const MAX_VALUE_CHARS = 4096;
const MAX_ITEMS = 512;
const MAX_DEPTH = 8;
const NOSQL = 'where|ne|eq|gt|gte|lt|lte|in|nin|regex|exists|or|and|nor|not|expr';

// Keep these in step with route_risks.py and RouteRiskCore.java. The shared cases in
// tests/fixtures/route_risk_cases.json run against all three.
const SINK_RULES = {
  sql: SQL_RULES,
  command: [
    ['command_chain', /[;&|]\s*(?:cat|curl|wget|nc|ncat|bash|sh|zsh|cmd|powershell|rm|id|whoami|uname|ping)\b/i],
    ['command_substitution', /\$\([^)]{0,200}\)|`[^`]{1,200}`/i]
  ],
  path: [
    ['path_traversal', /(?:^|[/\\])\.\.(?:[/\\]|$)/i],
    ['path_sensitive_location', /^(?:\/etc\/|\/proc\/|\/root\/|\/var\/log\/|[a-z]:[/\\]|\\\\)/i],
    ['path_null_byte', /\x00/]
  ],
  ssrf: [
    ['ssrf_scheme', /^\s*(?:file|gopher|dict|ftp|ldap|tftp|netdoc):\/\//i],
    ['ssrf_internal_host', /^\s*(?:https?:)?\/\/(?:[^/@?#\s]{0,128}@)?(?:localhost|127\.\d|0\.0\.0\.0|10\.\d|192\.168\.|169\.254\.|172\.(?:1[6-9]|2\d|3[01])\.|\[::1\]|metadata\.google\.internal)/i]
  ],
  nosql: [
    ['nosql_operator', new RegExp(`^\\$(?:${NOSQL})$|\\[\\$(?:${NOSQL})\\]|\\{\\s*"?\\$(?:${NOSQL})\\b`, 'i')]
  ],
  code: [
    ['code_expression', /__import__|\b(?:eval|exec|compile)\s*\(|\brequire\s*\(|\bprocess\.(?:env|exit|mainModule)|java\.lang\.|Runtime\.getRuntime|\bT\s*\(\s*java/i]
  ],
  template: [
    ['template_expression', /\{\{.{0,200}\}\}|\{%.{0,200}%\}|\$\{.{0,200}\}|<%.{0,200}%>|#\{.{0,200}\}/i]
  ],
  deserialization: [
    ['deserialization_marker', /^rO0AB|^aced0005|^gASV|!!python\/|!!javax\.|^O:\d+:"|__reduce__|_\$\$ND_FUNC\$\$_/]
  ]
};

const ROUTE_SEGMENT = /^(?:<(?:[^:>]+:)?([^>]+)>|\{\*?([^:}]+)(?::[^}]*)?\}|:([^/?*]+)[?*]?)$/;
const WILDCARD_SEGMENT = /^(?:<path:[^>]+>|\{\*?[^:}]+:(?:path|\.\*|\.\+)\}|\{\*[^}]+\}|:[^/?*]+\*|\*\*)$/;

function decoded(text) {
  try {
    return decodeURIComponent(text.replace(/\+/g, ' '));
  } catch (_) {
    return text;
  }
}

function ruleFor(sink, value) {
  const text = String(value).slice(0, MAX_VALUE_CHARS);
  const forms = text.includes('%') ? [text, decoded(text)] : [text];
  const rules = Object.prototype.hasOwnProperty.call(SINK_RULES, sink) ? SINK_RULES[sink] : [];
  for (const form of forms) {
    for (const [name, pattern] of rules) if (pattern.test(form)) return name;
  }
  return null;
}

function escapeRegex(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function compileRoute(path) {
  let pattern = '';
  const names = [];
  for (const segment of String(path).split('/')) {
    if (!segment) continue;
    const match = ROUTE_SEGMENT.exec(segment);
    const name = match ? match.slice(1).find(Boolean) || '' : '';
    if (WILDCARD_SEGMENT.test(segment)) {
      names.push(name);
      pattern += '/(.*)';
    } else if (segment === '*') {
      names.push('');
      pattern += '/([^/]+)';
    } else if (match) {
      names.push(name);
      pattern += '/([^/]+)';
    } else {
      pattern += `/${escapeRegex(segment)}`;
    }
  }
  return { pattern: new RegExp(`^${pattern}/?$`), names };
}

const inputName = name => String(name).replace(/\[.*$/, '');

// Nested values also count as the top-level field that holds them, as in { user: { $ne: 1 } }.
function walk(value, name, root, items, depth = 0) {
  if (items.length >= MAX_ITEMS || depth > MAX_DEPTH || value === undefined || value === null) return;
  if (Array.isArray(value)) {
    for (const item of value) walk(item, name, root, items, depth + 1);
  } else if (typeof value === 'object' && !Buffer.isBuffer(value)) {
    if (typeof value.pipe === 'function') return;
    for (const [key, item] of Object.entries(value)) {
      const top = root || key;
      items.push([key, key]);
      if (top !== key) items.push([top, key]);
      walk(item, key, top, items, depth + 1);
    }
  } else if (typeof value !== 'function') {
    const text = Buffer.isBuffer(value) ? value.toString('utf8') : String(value);
    items.push([name, text]);
    if (root && root !== name) items.push([root, text]);
  }
}

function pairsFrom(text, items) {
  for (const [key, value] of new URLSearchParams(text)) {
    if (items.length >= MAX_ITEMS) return;
    items.push([inputName(key), key], [inputName(key), value]);
  }
}

function requestInputs(query, body) {
  const items = [];
  if (typeof query === 'string') pairsFrom(query, items);
  else walk(query, '', '', items);
  const content = Buffer.isBuffer(body) ? body.toString('utf8') : body;
  if (typeof content === 'string' && content) {
    const stripped = content.trimStart();
    if (stripped[0] === '{' || stripped[0] === '[') {
      try {
        walk(JSON.parse(stripped), '', '', items);
      } catch (_) {
        items.push(['', content]);
      }
    } else if (content.includes('=')) {
      pairsFrom(content, items);
    } else {
      items.push(['', content]);
    }
  } else if (content && typeof content === 'object') {
    walk(content, '', '', items);
  }
  return items.slice(0, MAX_ITEMS);
}

function compileRisks(manifest) {
  const routes = manifest && typeof manifest.routes === 'object' && manifest.routes ? manifest.routes : {};
  const compiled = [];
  for (const [path, entry] of Object.entries(routes)) {
    const risks = entry && entry.injection_risks;
    if (Array.isArray(risks) && risks.length) {
      compiled.push({ path, ...compileRoute(path), risks: risks.filter(item => item && typeof item === 'object') });
    }
  }
  // Exact routes win over routes with parameters.
  return compiled.sort((a, b) => a.names.length - b.names.length || b.path.length - a.path.length);
}

function inspectRisks(routes, path, query, body) {
  for (const route of routes) {
    const match = route.pattern.exec(String(path || '/'));
    if (!match) continue;
    const items = requestInputs(query, body);
    route.names.forEach((name, index) => {
      if (match[index + 1]) items.push([name, match[index + 1]]);
    });
    const events = [];
    const seen = new Set();
    for (const risk of route.risks) {
      const sink = String(risk.sink || '');
      const wanted = String(risk.input || '');
      for (const [name, value] of items) {
        if (wanted && name !== wanted) continue;
        const rule = ruleFor(sink, value);
        const key = `${sink}\0${wanted}\0${rule}`;
        if (rule && !seen.has(key)) {
          seen.add(key);
          // The input name comes from the manifest, because request names are untrusted.
          events.push({ route: route.path, sink, input: wanted, rule, confidence: String(risk.confidence || '') });
        }
      }
    }
    return events;
  }
  return [];
}

/** Returns a function that logs matches for a request, or null unless log mode applies. */
function createObserver(options = {}) {
  const mode = options.AIWAF_INJECTION_RISK_MODE ?? process.env.AIWAF_INJECTION_RISK_MODE ?? 'off';
  if (String(mode).trim().toLowerCase() !== 'log') return null;
  const routes = compileRisks(loadManifest(options.AIWAF_PATH_MANIFEST || undefined));
  if (!routes.length) return null;
  const logger = options.logger || console;
  // Adapters can inspect twice: before and after the body is parsed.
  const logged = new WeakMap();
  return req => {
    // Log-only checks must never affect the request, whatever they receive.
    try {
      const url = String(req.path || req.url || '/');
      const raw = String(req.originalUrl || req.url || '').split('?').slice(1).join('?');
      const events = inspectRisks(routes, url.split('?')[0], req.query ?? raw, req.body);
      if (!events.length) return;
      const seen = logged.get(req) || new Set();
      logged.set(req, seen);
      for (const event of events) {
        const key = JSON.stringify(event);
        if (seen.has(key)) continue;
        seen.add(key);
        logger.warn?.({ event: 'aiwaf.injection_risk', mode: 'log', method: String(req.method || ''), ...event });
      }
    } catch (_) {
      // Nothing to do: this is a diagnostic, not a protection.
    }
  };
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
  const fromSource = (/^\S+ (\/\S*)$/.exec(entry) || [])[1] || fileRoute(finding.file);
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
  for (const entry of Object.values(routes)) if (entry && typeof entry === 'object') delete entry.injection_risks;
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
  manifest.injection_audit = {
    generated_at: new Date().toISOString().replace(/\.\d{3}Z$/, 'Z'),
    files_scanned: Number(report.files_scanned || 0),
    linked_findings: linked,
    unlinked_findings: unlinked
  };
  return { routes: collected.size, linked, unlinked };
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
  applyReport,
  compileRisks,
  compileRoute,
  createObserver,
  formatCounts,
  inputFromSource,
  inspectRisks,
  requestInputs,
  ruleFor,
  updateManifest
};
