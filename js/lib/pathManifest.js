const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { normalizePath } = require('./pathRules');
const { analyzeHandlerAst } = require('./sourceAst');

const SCHEMA_VERSION = '1.0';
const DEFAULT_MANIFEST_PATH = path.join('.aiwaf', 'paths.json');
const MIDDLEWARE_NAMES = new Set([
  'geo_block',
  'ip_keyword_block',
  'rate_limit',
  'ai_anomaly',
  'honeypot',
  'uuid_tamper',
  'header_validation',
  'logging'
]);

function stableJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

function computeContextHash(value) {
  return crypto.createHash('sha256').update(stableJson(value)).digest('hex');
}

function nowUtcIso() {
  return new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
}

function sourceSignals(handler) {
  const source = require('./sourceRoutes').handlerText(handler) || '';
  const lower = source.toLowerCase();
  const signals = [];
  const formSignals = [];
  let requestBody = false;
  let payloadType = '';
  let score = 0;
  let formScore = 0;

  if (/\bres\.json\s*\(|\breply\.send\s*\(\s*[{[]|\bctx\.body\s*=\s*[{[]/.test(source)) {
    score += 60;
    signals.push('json_response');
  }
  if (/\breq\.body\b|\brequest\.body\b/.test(source)) {
    score += 20;
    requestBody = true;
    payloadType = 'json';
    signals.push('request.body');
  }
  if (/application\/json/.test(lower)) {
    score += 30;
    payloadType = 'json';
    signals.push('content-type:application/json');
  }
  if (/\breq\.file\b|\breq\.files\b|\bformdata\b|\bmultipart\b/.test(lower)) {
    formScore += 45;
    requestBody = true;
    payloadType = 'form';
    formSignals.push('request.form');
  }
  if (/\bres\.render\s*\(|\bctx\.render\s*\(|\bredirect\s*\(/.test(source)) {
    formScore += 40;
    formSignals.push('render_or_redirect');
    if (/\breq\.body\b|\brequest\.body\b/.test(source)) {
      formScore += 45;
      requestBody = true;
      payloadType = 'form';
      formSignals.push('request.body');
    }
  }

  return {
    score,
    signals: Array.from(new Set(signals)).sort(),
    requestBody,
    formScore,
    formSignals: Array.from(new Set(formSignals)).sort(),
    payloadType
  };
}

function detectApiEndpoint(handler, routePath = '', methods = []) {
  const pathLower = normalizePath(routePath).toLowerCase();
  const signals = [];
  let score = 0;

  if (pathLower.startsWith('/api/') || pathLower.includes('/api/')) {
    score += 30;
    signals.push('path:/api');
  }
  if (['/graphql', '/token'].includes(pathLower.replace(/\/$/, ''))) {
    score += 30;
    signals.push(`path:${pathLower.replace(/\/$/, '')}`);
  }
  if (pathLower.includes('/webhook') || pathLower.includes('/callback')) {
    score += 25;
    signals.push('path:webhook');
  }

  const ast = analyzeHandlerAst(handler);
  const source = ast.available
    ? {
        score: ast.apiScore,
        signals: ast.apiSignals,
        requestBody: ast.requestBody,
        formScore: ast.formScore,
        formSignals: ast.formSignals,
        payloadType: ast.payloadType
      }
    : sourceSignals(handler);
  score += source.score;
  signals.push(...source.signals);

  const isForm = source.formScore >= 50 && source.formScore >= score;
  const isApi = score >= 50 && !isForm;
  return {
    isApi,
    responseType: isApi ? 'json' : (isForm ? 'mixed' : ''),
    payloadType: source.payloadType || (isApi ? 'json' : ''),
    confidence: Math.min(0.99, score / 100),
    signals: Array.from(new Set(signals)).sort(),
    requestBody: source.requestBody || methods.some(method => method !== 'GET'),
    formConfidence: Math.min(0.99, source.formScore / 100),
    formSignals: source.formSignals
  };
}

function detectAuthEndpoint(handler, routePath = '') {
  const pathLower = normalizePath(routePath).toLowerCase();
  const ast = analyzeHandlerAst(handler);
  const source = (require('./sourceRoutes').handlerText(handler) || '').toLowerCase();
  const signals = [];
  let score = 0;
  let action = '';

  const priorities = { login: 50, token_login: 50, logout: 40, register: 30, token_auth: 20 };
  const getPriority = a => priorities[a] || 0;
  const mergeAction = (curr, cand) => (!curr ? cand : (getPriority(cand) > getPriority(curr) ? cand : curr));

  if (pathLower.includes('/logout') || pathLower.includes('/signout')) {
    score += 40;
    action = mergeAction(action, 'logout');
    signals.push('path:logout');
  } else if (pathLower.includes('/register') || pathLower.includes('/signup')) {
    score += 40;
    action = mergeAction(action, 'register');
    signals.push('path:register');
  } else if (pathLower.includes('/login') || pathLower.includes('/signin') || pathLower.includes('/token')) {
    score += 40;
    action = mergeAction(action, pathLower.includes('/token') ? 'token_login' : 'login');
    signals.push('path:auth');
  }

  if (/bcrypt|comparepassword|jsonwebtoken|jwt|passport|login|authenticate|createsession|createaccesstoken/.test(source)) {
    score += 50;
    action = mergeAction(action, 'login');
    signals.push('auth_source_signal');
  }
  
  if (ast.available && ast.authScore) {
    score += ast.authScore;
    action = mergeAction(action, ast.authAction);
    signals.push(...ast.authSignals);
  }

  return {
    isAuth: score >= 50,
    action: action || 'login',
    confidence: Math.min(0.99, score / 100),
    signals
  };
}

function classifyRoute(routePath, { methods = [], metadata = {} } = {}) {
  const normalized = normalizePath(routePath);
  const pathLower = normalized.toLowerCase();
  const methodSet = new Set((methods || []).map(method => String(method).toUpperCase()));
  let category = 'unknown';
  let responseType = 'html';
  let authRequired = !!metadata.auth_required || pathLower.startsWith('/portal/') || pathLower.startsWith('/dashboard/');
  const protections = {
    rate_limit: { requests: 60, window_seconds: 60 },
    header_validation: { enabled: true },
    ip_keyword_block: { enabled: true },
    ai_anomaly: { enabled: true }
  };

  if (
    pathLower.startsWith('/static/')
    || pathLower.startsWith('/media/')
    || pathLower.startsWith('/assets/')
    || /\.(css|js|png|jpe?g|gif|ico|svg|woff2?|ttf)$/.test(pathLower)
  ) {
    category = 'static';
    protections.header_validation = { enabled: false };
    protections.ai_anomaly = { enabled: false };
    protections.honeypot = { enabled: false };
  } else if (pathLower.includes('/admin')) {
    category = 'admin';
    authRequired = true;
    protections.rate_limit = { requests: 30, window_seconds: 60 };
  } else if (metadata.auth_action || metadata.auth_confidence || pathLower.includes('/login') || pathLower.includes('/signin')) {
    category = 'auth';
    protections.rate_limit = { requests: 30, window_seconds: 60 };
    protections.honeypot = { enabled: true };
  } else if (metadata.payload_type === 'form' || metadata.form_confidence) {
    category = 'form';
    responseType = metadata.response_type || 'mixed';
    protections.rate_limit = { requests: 30, window_seconds: 60 };
    protections.honeypot = { enabled: true };
  } else if (metadata.api_confidence || pathLower.startsWith('/api/') || metadata.response_type === 'json') {
    category = 'api';
    responseType = 'json';
    protections.rate_limit = { requests: 120, window_seconds: 60 };
    protections.api_rate_limit = { requests: 120, window_seconds: 60 };
    protections.honeypot = { enabled: false };
  } else if (pathLower.includes('/upload') || pathLower.includes('/files')) {
    category = 'upload';
    protections.rate_limit = { requests: 20, window_seconds: 60 };
  } else if (authRequired) {
    category = 'app';
    protections.rate_limit = { requests: 90, window_seconds: 60 };
  }

  if (methodSet.has('POST') && !['api', 'upload'].includes(category)) {
    protections.rate_limit = { requests: 30, window_seconds: 60 };
  }

  return {
    category,
    response_type: metadata.response_type || responseType,
    auth_required: authRequired,
    protections,
    ...(metadata.auth_action ? { auth_action: metadata.auth_action } : {}),
    ...(metadata.auth_confidence !== undefined ? { auth_confidence: metadata.auth_confidence } : {}),
    ...(metadata.auth_signals ? { auth_signals: metadata.auth_signals } : {}),
    ...(metadata.api_confidence !== undefined ? { api_confidence: metadata.api_confidence } : {}),
    ...(metadata.api_signals ? { api_signals: metadata.api_signals } : {}),
    ...(metadata.payload_type ? { payload_type: metadata.payload_type } : {}),
    ...(metadata.form_confidence !== undefined ? { form_confidence: metadata.form_confidence } : {}),
    ...(metadata.form_signals ? { form_signals: metadata.form_signals } : {}),
    ...(metadata.request_body !== undefined ? { request_body: !!metadata.request_body } : {})
  };
}

function buildRouteEntry({ routePath, methods = [], view = '', name = '', metadata = {} }) {
  const normalized = normalizePath(routePath);
  const methodsList = Array.from(new Set(
    (methods || [])
      .map(method => String(method).toUpperCase())
      .filter(method => method && !['HEAD', 'OPTIONS'].includes(method))
  )).sort();
  return [normalized, {
    methods: methodsList,
    view: String(view || ''),
    name: String(name || ''),
    ...classifyRoute(normalized, { methods: methodsList, metadata })
  }];
}

function buildManifest({ framework, routes, appContext = {} }) {
  const normalizedRoutes = Object.fromEntries(
    Object.entries(routes || {})
      .sort(([left], [right]) => normalizePath(left).localeCompare(normalizePath(right)))
      .map(([routePath, data]) => [normalizePath(routePath), { ...data }])
  );
  const context = { framework, routes: normalizedRoutes, app_context: { ...appContext } };
  return {
    schema_version: SCHEMA_VERSION,
    framework,
    context_hash: computeContextHash(context),
    generated_at: nowUtcIso(),
    routes: normalizedRoutes
  };
}

// Writes the manifest atomically: to a temporary file beside it, then renaming it over the old
// one, so a write that fails (a full disk, a value JSON can't hold) leaves the old one intact.
function writeManifest(manifest, outputPath = DEFAULT_MANIFEST_PATH) {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  const text = `${JSON.stringify(manifest, null, 2)}\n`;
  const temporary = path.join(path.dirname(outputPath), `.${path.basename(outputPath)}.${process.pid}.${Date.now()}.tmp`);
  try {
    const handle = fs.openSync(temporary, 'wx', 0o644);
    try {
      fs.writeFileSync(handle, text, 'utf8');
      fs.fsyncSync(handle);
    } finally {
      fs.closeSync(handle);
    }
    try { fs.chmodSync(temporary, fs.statSync(outputPath).mode & 0o777); } catch (_) { /* a new manifest */ }
    fs.renameSync(temporary, outputPath);
  } catch (error) {
    try { fs.unlinkSync(temporary); } catch (_) { /* already gone */ }
    throw error;
  }
  return outputPath;
}

function loadManifest(manifestPath = DEFAULT_MANIFEST_PATH) {
  try {
    if (!fs.existsSync(manifestPath)) return null;
    const parsed = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch (err) {
    return null;
  }
}

function rateLimitOverride(config) {
  const override = {};
  if (config.WINDOW !== undefined) override.WINDOW = config.WINDOW;
  if (config.MAX !== undefined) override.MAX = config.MAX;
  if (config.FLOOD !== undefined) override.FLOOD = config.FLOOD;
  if (config.window_seconds !== undefined) override.WINDOW = config.window_seconds;
  if (config.requests !== undefined) override.MAX = config.requests;
  if (config.flood !== undefined) override.FLOOD = config.flood;
  return override;
}

function compileManifestToPathRules(manifest) {
  if (!manifest || typeof manifest !== 'object' || !manifest.routes || typeof manifest.routes !== 'object') return [];
  return Object.entries(manifest.routes)
    .map(([routePath, entry]) => {
      const protections = entry && typeof entry === 'object' && entry.protections && typeof entry.protections === 'object'
        ? entry.protections
        : {};
      const rule = { PREFIX: normalizePath(routePath, true) };
      const disabled = [];
      Object.entries(protections).forEach(([name, config]) => {
        const normalizedName = String(name).trim().toLowerCase();
        if (!MIDDLEWARE_NAMES.has(normalizedName)) return;
        if (config === false || (config && typeof config === 'object' && config.enabled === false)) {
          disabled.push(normalizedName);
        }
      });
      const rateConfig = protections.rate_limit || protections.api_rate_limit;
      if (rateConfig && typeof rateConfig === 'object') {
        const override = rateLimitOverride(rateConfig);
        if (Object.keys(override).length) rule.RATE_LIMIT = override;
      }
      if (disabled.length) rule.DISABLE = Array.from(new Set(disabled)).sort();
      return rule.DISABLE || rule.RATE_LIMIT ? rule : null;
    })
    .filter(Boolean);
}

function getEffectivePathRules(explicitRules = [], { manifestPath = DEFAULT_MANIFEST_PATH } = {}) {
  return [...(explicitRules || []), ...compileManifestToPathRules(loadManifest(manifestPath))];
}

function methodsFromExpressRoute(route) {
  return Object.entries(route?.methods || {})
    .filter(([, enabled]) => enabled)
    .map(([method]) => method.toUpperCase());
}

function inferMethodsFromHandler(handler) {
  return analyzeHandlerAst(handler).methods || [];
}

function routeEntryFor({ routePath, methods = [], handler, name = '', view = '', responseType = null }) {
  const inferredMethods = methods.length ? methods : inferMethodsFromHandler(handler);
  let api = detectApiEndpoint(handler, routePath, inferredMethods);
  // The framework's own word that the route answers with JSON (Nest's default, or a route list's
  // response_type), where the handler's code doesn't show it.
  if (responseType === 'html' && api.isApi) {
    api = { ...api, isApi: false, responseType: '', payloadType: api.payloadType === 'json' ? '' : api.payloadType };
  } else if (responseType === 'json' && !api.isApi) {
    api = {
      ...api,
      isApi: true,
      responseType: 'json',
      payloadType: api.payloadType || 'json',
      confidence: Math.max(api.confidence, 0.9),
      signals: Array.from(new Set([...api.signals, 'framework:json_response'])).sort()
    };
  }
  const auth = detectAuthEndpoint(handler, routePath);
  const metadata = {
    response_type: api.responseType || undefined,
    payload_type: api.payloadType || undefined,
    api_confidence: api.isApi ? api.confidence : undefined,
    api_signals: api.isApi ? api.signals : undefined,
    form_confidence: api.formConfidence || undefined,
    form_signals: api.formSignals.length ? api.formSignals : undefined,
    request_body: api.requestBody || undefined,
    auth_action: auth.isAuth ? auth.action : undefined,
    auth_confidence: auth.isAuth ? auth.confidence : undefined,
    auth_signals: auth.isAuth ? auth.signals : undefined
  };
  return buildRouteEntry({
    routePath,
    methods: inferredMethods,
    view: view || handler?.name || '',
    name: name || routePath,
    metadata
  });
}

function addRoute(routes, routeInfo) {
  if (!routeInfo || !routeInfo.routePath) return;
  const [normalized, entry] = routeEntryFor(routeInfo);
  const declared = require('./inputSources').declaredOn(routeInfo.handlers || [], routeInfo.handler);
  if (declared) entry.input_declaration = declared;
  const framework = require('./frameworkInputs');
  let known = routeInfo.declared || framework.fastifyInputs(routeInfo.schema) || framework.hapiInputs(routeInfo.validate)
    || framework.expressInputs(routeInfo.handlers);
  // A schema the route has but that couldn't be read (imported, built at runtime): its inputs
  // are unknown, so the route's list isn't complete.
  if (!known && routeInfo.schemaUnreadable) {
    known = {
      inputs: {}, covers: [], standalone: false,
      reasons: [{ code: 'external_class', detail: "the route's schema isn't available: init didn't load it, and the source doesn't spell it out" }]
    };
  }
  for (const [name, methods] of routeInfo.handlerMethods || []) {
    entry.handler_methods = entry.handler_methods || {};
    entry.handler_methods[name] = methods.length && entry.handler_methods[name] !== null
      ? [...new Set([...(entry.handler_methods[name] || []), ...methods])].sort() : null;
  }
  if (known) {
    entry.declared_inputs = known;
    // Which methods the framework's names are for, so each method's list can be told apart.
    const names = Object.keys(known.inputs || {}).sort();
    entry.declared_methods = Object.fromEntries((entry.methods || []).map(method => [String(method).toUpperCase(), names]));
    // And each method's own types, kept apart from another method's on the same path.
    const typed = Object.fromEntries(Object.entries(known.inputs || {}).filter(([, kind]) => kind && kind !== 'text'));
    if (Object.keys(typed).length) {
      entry.declared_types = Object.fromEntries((entry.methods || []).map(method => [String(method).toUpperCase(), { ...typed }]));
    }
    // And each method's types per location.
    if (known.locations) {
      entry.declared_locations = Object.fromEntries((entry.methods || []).map(method => [String(method).toUpperCase(), known.locations]));
    }
  }
  if (routes[normalized]) {
    const mergedMethods = Array.from(new Set([...(routes[normalized].methods || []), ...(entry.methods || [])])).sort();
    const combined = framework.combine(routes[normalized].declared_inputs, entry.declared_inputs);
    const handlerMethods = { ...(routes[normalized].handler_methods || {}) };
    for (const [name, methods] of Object.entries(entry.handler_methods || {})) {
      handlerMethods[name] = methods && handlerMethods[name] !== null ? [...new Set([...(handlerMethods[name] || []), ...methods])].sort() : null;
    }
    const declaredMethods = { ...(routes[normalized].declared_methods || {}) };
    for (const [method, names] of Object.entries(entry.declared_methods || {})) {
      declaredMethods[method] = [...new Set([...(declaredMethods[method] || []), ...names])].sort();
    }
    // Two handlers for one method that type a name differently leave it untyped for that method.
    const declaredTypes = {};
    for (const side of [routes[normalized].declared_types, entry.declared_types]) {
      for (const [method, types] of Object.entries(side || {})) {
        const own = (declaredTypes[method] = declaredTypes[method] || {});
        for (const [name, kind] of Object.entries(types)) own[name] = own[name] === undefined || own[name] === kind ? kind : 'text';
      }
    }
    // Every view serving the path, when several do, so each one's code links to it.
    const views = [...new Set([...(routes[normalized].views || [routes[normalized].view]), entry.view].filter(Boolean))].sort();
    routes[normalized] = {
      ...routes[normalized],
      ...entry,
      methods: mergedMethods
    };
    if (views.length > 1) routes[normalized].views = views;
    if (combined) routes[normalized].declared_inputs = combined;
    if (Object.keys(declaredMethods).length) routes[normalized].declared_methods = declaredMethods;
    if (Object.keys(declaredTypes).length) routes[normalized].declared_types = declaredTypes;
    const declaredLocations = {};
    for (const side of [routes[normalized].declared_locations, entry.declared_locations]) {
      for (const [method, byWhere] of Object.entries(side || {})) {
        const intoMethod = (declaredLocations[method] = declaredLocations[method] || {});
        for (const [where, types] of Object.entries(byWhere)) {
          const into = (intoMethod[where] = intoMethod[where] || {});
          for (const [name, kind] of Object.entries(types)) into[name] = into[name] === undefined || into[name] === kind ? kind : 'text';
        }
      }
    }
    if (Object.keys(declaredLocations).length) routes[normalized].declared_locations = declaredLocations;
    if (Object.keys(handlerMethods).length) routes[normalized].handler_methods = handlerMethods;
  } else {
    routes[normalized] = entry;
  }
}

// Express 4 keeps its router at app._router (its app.router getter throws), Express 5 at app.router.
function routerOf(app) {
  if (app?._router) return app._router;
  try {
    return app?.router;
  } catch (_error) {
    return null;
  }
}

// One mount pattern's path, '' at the root, or null when it can't be read back as a path.
function patternPath(regexp, keys) {
  let source = regexp.source.replace(/^\^/, '');
  // The end of a mount pattern: Express 4's \/?(?=\/|$), Express 5's (?:\/$)?(?=\/|$).
  for (const tail of ['\\/?(?=\\/|$)', '(?:\\/$)?(?=\\/|$)', '(?=\\/|$)']) {
    if (source.endsWith(tail)) {
      source = source.slice(0, -tail.length);
      break;
    }
  }
  // Express 5 wraps the path in (?:...).
  if (source.startsWith('(?:') && source.endsWith(')')) source = source.slice(3, -1);
  let index = 0;
  const name = () => {
    const key = keys[index];
    index += 1;
    return `:${key && key.name !== undefined ? key.name : index - 1}`;
  };
  const path = source
    .replace(/\(\?:\\\/\(\[\^\\?\/\]\+\?\)\)/g, () => `/${name()}`) // Express 4: (?:\/([^\/]+?))
    .replace(/\(\?:\(\[\^\\?\/\]\+\?\)\)/g, () => name()) // Express 4: (?:([^\/]+?))
    .replace(/\(\[\^\\?\/\]\+\??\)/g, () => name()) // Express 5: ([^\/]+)
    .replace(/\\(.)/g, '$1');
  return /[()[\]*+?|^$]/.test(path) ? null : path;
}

// Splits a pattern at its top-level |, as Express 4 compiles app.use(['/a', '/b'], router).
function alternatives(source) {
  const parts = [];
  let depth = 0;
  let start = 0;
  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if (char === '\\') { index += 1; continue; }
    if (char === '(' || char === '[') depth += 1;
    else if (char === ')' || char === ']') depth -= 1;
    else if (char === '|' && depth === 0) {
      parts.push(source.slice(start, index));
      start = index + 1;
    }
  }
  parts.push(source.slice(start));
  return parts;
}

/**
 * Every path a router layer is mounted at, as in app.use(['/account', '/me'], router): '' at the
 * root. Empty when a pattern can't be read back as a path. Express 4 keeps the pattern on the
 * layer; Express 5 keeps one matcher per path, read through the inspector.
 */
function mountPaths(layer) {
  if (layer.slash || (layer.regexp && layer.regexp.fast_slash)) return [''];
  const patterns = [];
  if (layer.regexp instanceof RegExp) {
    let offset = 0;
    const keys = layer.keys || [];
    for (const part of alternatives(layer.regexp.source)) {
      // Express 4 numbers an alternation's keys across every path, in order.
      const count = (part.match(/\((?!\?)/g) || []).length;
      patterns.push([new RegExp(part, layer.regexp.flags), keys.slice(offset, offset + count)]);
      offset += count;
    }
  } else if (Array.isArray(layer.matchers)) {
    for (const matcher of layer.matchers) {
      const values = require('./frameworkInputs').closureValues(matcher);
      const regexp = (values.find(([, value]) => value instanceof RegExp) || [])[1];
      const keys = (values.find(([name]) => name === 'keys') || [])[1] || [];
      if (regexp instanceof RegExp) patterns.push([regexp, keys]);
    }
  }
  const paths = patterns.map(([regexp, keys]) => patternPath(regexp, keys));
  return paths.length && paths.every(item => item !== null) ? paths : [];
}

/** Where a router layer is mounted: its first path (see mountPaths), or null. */
function mountPath(layer) {
  const paths = mountPaths(layer);
  return paths.length ? paths[0] : null;
}

function extractExpressRoutes(app) {
  const routes = {};
  // Routes on the app, and on routers it mounts, under their mount paths.
  const visit = (stack, prefix, depth) => (stack || []).forEach(layer => {
    if (!layer.route && layer.handle && Array.isArray(layer.handle.stack) && depth < 8) {
      for (const mounted of mountPaths(layer)) visit(layer.handle.stack, `${prefix}${mounted}`, depth + 1);
      return;
    }
    add(layer, prefix);
  });
  const add = (layer, prefix) => {
    const route = layer.route;
    if (!route || !route.path) return;
    const routePath = `${prefix}${String(route.path)}` || '/';
    // The last function is the handler; any before it are route middleware.
    const handler = (route.stack || []).filter(item => typeof item.handle === 'function').map(item => item.handle).pop();
    const methods = methodsFromExpressRoute(route);
    addRoute(routes, {
      routePath,
      methods,
      handler,
      handlers: (route.stack || []).map(item => item.handle),
      // The method each named handler serves (Express records it per layer), so a handler passed
      // by name, as app.get('/x', list), gets its own list; app.all's serve every method.
      handlerMethods: (route.stack || []).filter(item => item.handle && item.handle.name).map(item => [item.handle.name,
        item.method && item.method !== '_all' ? [String(item.method).toUpperCase()] : []]),
      name: routePath
    });
  };
  visit(routerOf(app)?.stack, '', 0);
  return routes;
}

function handlerMethodsOf(route) {
  const handler = route.handler || route.handle || route.endpoint;
  const name = handler && typeof handler.name === 'string' ? handler.name.replace(/^bound /, '') : '';
  const methods = [].concat(route.methods || route.method || []).map(method => String(method).toUpperCase()).filter(method => method !== 'HEAD');
  return name && methods.length ? [[name, methods]] : undefined;
}

/**
 * AdonisJS's routes as a route list: from `node ace list:routes --json`, or the router's own
 * toJSON(), which group routes by domain ({ root: [...] }, or [{ domain, routes }]). Each route's
 * controller handler, as #controllers/users_controller.index, becomes its view,
 * UsersController.index, so the controller method's code links to it.
 */
function adonisRouteList(input) {
  const routes = [];
  const add = list => {
    for (const route of Array.isArray(list) ? list : []) {
      if (!route || typeof route !== 'object') continue;
      if (Array.isArray(route.routes)) {
        add(route.routes); // { domain, routes }
        continue;
      }
      const methods = [].concat(route.methods || route.method || []).map(method => String(method).toUpperCase()).filter(method => method !== 'HEAD');
      const view = adonisView(route.handler);
      const method = view.includes('.') ? view.split('.').pop() : '';
      routes.push({
        path: route.pattern || route.path || route.route,
        methods,
        name: route.name || route.pattern || route.path || '',
        view,
        handlerMethods: method ? [[method, methods]] : undefined
      });
    }
  };
  if (Array.isArray(input)) add(input);
  else if (input && typeof input === 'object') for (const list of Object.values(input)) add(list);
  return routes;
}

/** A route handler's controller view: #controllers/users_controller.index is UsersController.index. */
function adonisView(handler) {
  let reference = '';
  let method = '';
  if (typeof handler === 'string') reference = handler;
  else if (Array.isArray(handler)) {
    reference = typeof handler[0] === 'function' ? handler[0].name : String(handler[0] || '');
    method = String(handler[1] || 'handle');
  } else if (handler && typeof handler === 'object') {
    reference = String(handler.reference || handler.moduleNameOrPath || handler.name || '');
    method = handler.method ? String(handler.method) : '';
  }
  if (!reference || reference === 'closure') return '';
  if (!method) {
    const dot = reference.lastIndexOf('.');
    // A module path's own dots (.js, .ts) aren't a method.
    if (dot > reference.lastIndexOf('/') && !/\.[cm]?[jt]s$/.test(reference)) {
      method = reference.slice(dot + 1);
      reference = reference.slice(0, dot);
    } else {
      method = 'handle';
    }
  }
  const file = reference.split('/').pop().replace(/\.[cm]?[jt]s$/, '');
  const name = /_|-/.test(file) || /^[a-z]/.test(file)
    ? file.split(/[_-]/).filter(Boolean).map(part => part[0].toUpperCase() + part.slice(1)).join('') : file;
  return `${name}.${method}`;
}

function normalizeRouteList(routesInput = []) {
  return (Array.isArray(routesInput) ? routesInput : [])
    .filter(route => route && typeof route === 'object')
    .map(route => ({
      routePath: route.path || route.url || route.routePath || route.pattern,
      methods: [].concat(route.methods || route.method || []).map(method => String(method).toUpperCase()),
      handler: route.handler || route.handle || route.endpoint,
      schema: route.schema,
      validate: route.validate,
      name: route.name || route.id || route.path || route.url || '',
      view: route.view || '',
      responseType: route.responseType || route.response_type || null,
      // The method a named handler serves, so the audit's reading of it (linked by name) is
      // that method's: fastify.delete('/plans', removePlan) reads force for DELETE alone.
      handlerMethods: route.handlerMethods || handlerMethodsOf(route),
      schemaUnreadable: Boolean(route.schemaUnreadable)
    }))
    .filter(route => route.routePath);
}

// A ready Fastify app's routes, as a route list, from printRoutes({ commonPrefix: false }). It
// prints a tree whose nodes share prefixes, each line one node's part of the path with the
// methods ending there: "└── /", "    ├── early (GET, HEAD)", "    │   ├── -order (POST)". A
// route's path is its line's part after its ancestors'. Fastify doesn't hand back its handlers,
// so the source supplies them (withSourceHandlers).
function fastifyRouteList(fastify) {
  if (typeof fastify?.printRoutes !== 'function') return [];
  let printed = '';
  try {
    printed = String(fastify.printRoutes({ commonPrefix: false }) || '');
  } catch (_) {
    return [];
  }
  const list = [];
  const parts = [];
  for (const line of printed.split('\n')) {
    const match = /^((?:[│ ] {3})*)[├└]── (.+?)(?: \(([A-Z, ]+)\))?\s*$/.exec(line);
    if (!match) continue;
    const depth = match[1].length / 4;
    parts.length = depth;
    parts.push(match[2]);
    const methods = (match[3] || '').split(',').map(method => method.trim()).filter(method => method && method !== 'HEAD');
    const routePath = parts.join('');
    // One entry per method: each has its own handler and schema.
    if (routePath.startsWith('/')) for (const method of methods) list.push({ path: routePath, methods: [method] });
  }
  return list;
}

/**
 * A Fastify app's routes, as a route list: the AIWAF plugin's records (with handlers and schemas)
 * and, for routes registered before the plugin, which its onRoute hook never saw, Fastify's own
 * route table.
 */
function fastifyRoutes(fastify) {
  // init's own capture, from before any route was added, when the app was loaded by init; else
  // the AIWAF plugin's, which misses routes added before it.
  const captured = fastify ? fastify[Symbol.for('aiwaf.init.fastifyRoutes')] : null;
  const source = Array.isArray(captured) && captured.length ? captured : fastify?.aiwafRoutes;
  const recorded = Array.isArray(source) ? source.filter(route => route && typeof route === 'object' && String(route.method).toUpperCase() !== 'HEAD'
    && !(Array.isArray(route.method) && route.method.every(method => String(method).toUpperCase() === 'HEAD'))) : [];
  const seen = new Set();
  for (const route of recorded) {
    const routePath = route.url || route.path;
    for (const method of [].concat(route.method || route.methods || [])) seen.add(`${String(method).toUpperCase()} ${routePath}`);
  }
  const missing = fastifyRouteList(fastify).filter(route => !seen.has(`${route.methods[0]} ${route.path}`));
  return [...recorded, ...missing];
}

function extractFastifyRoutes(fastify, routesInput = null) {
  const routes = {};
  normalizeRouteList(routesInput || (fastify?.aiwafRoutes ? fastifyRoutes(fastify) : null) || fastify?.routes || [])
    .forEach(route => addRoute(routes, route));
  if (Object.keys(routes).length) return routes;
  normalizeRouteList(fastifyRouteList(fastify)).forEach(route => addRoute(routes, route));
  if (Object.keys(routes).length) return routes;

  if (typeof fastify?.printRoutes === 'function') {
    const printed = String(fastify.printRoutes() || '');
    const routePattern = /(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+([^\s]+)/gi;
    let match;
    while ((match = routePattern.exec(printed)) !== null) {
      addRoute(routes, { routePath: match[2], methods: [match[1].toUpperCase()] });
    }
  }
  return routes;
}

function extractHapiRoutes(server, routesInput = null) {
  const routes = {};
  const sourceRoutes = routesInput || (typeof server?.table === 'function' ? server.table() : server?.routes);
  normalizeRouteList((sourceRoutes || []).map(route => ({
    path: route.path,
    method: route.method,
    handler: route.settings?.handler || route.handler,
    validate: route.settings?.validate,
    name: route.settings?.id || route.path
  }))).forEach(route => addRoute(routes, route));
  return routes;
}

function extractKoaRoutes(app, routesInput = []) {
  const routes = {};
  if (routesInput && routesInput.length) {
    normalizeRouteList(routesInput).forEach(route => addRoute(routes, route));
    return routes;
  }
  // A Koa app keeps no routes itself; @koa/router's and koa-router's routes() middleware keeps
  // its router, whose layers have the path, methods and handlers.
  for (const middleware of (app && Array.isArray(app.middleware) ? app.middleware : [])) {
    const router = middleware && middleware.router;
    for (const layer of (router && Array.isArray(router.stack) ? router.stack : [])) {
      if (!layer || typeof layer.path !== 'string' || !Array.isArray(layer.methods) || !layer.methods.length) continue;
      const handlers = (layer.stack || []).filter(item => typeof item === 'function');
      const methods = layer.methods.map(method => String(method).toUpperCase()).filter(method => method !== 'HEAD');
      addRoute(routes, {
        routePath: layer.path,
        methods,
        handler: handlers[handlers.length - 1],
        handlers,
        // The methods each named handler serves, so router.get('/x', list) reads for GET alone.
        handlerMethods: handlers.filter(item => item.name).map(item => [item.name.replace(/^bound /, ''), methods]),
        name: layer.name || layer.path
      });
    }
  }
  return routes;
}

const SAILS_METHODS = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'];
// What an actions2 action's declared inputs answer: its code gets only those.
const COVERS_ALL = ['dependency', 'external_class', 'model_fields', 'container_whole', 'helper'];
const SAILS_KINDS = { string: 'text', number: 'float', boolean: 'bool', json: 'text', ref: 'text' };

/**
 * A Sails app's routes, from its config.routes: 'GET /users': 'UserController.find', a
 * { controller, action } target, or an actions2 identity ('user/find', or { action }). Classic
 * targets' views are Controller.action, so the controller method's code links to them; an
 * actions2 action's view is its identity, and its declared inputs are the route's (Sails passes
 * the action only those). Anything else (a view, a redirect, a function) has no inputs to read.
 * A Sails app that isn't one is read as the Express app it is.
 */
function extractSailsRoutes(app) {
  const { isSails } = require('./appLoader');
  if (!isSails(app)) return extractExpressRoutes(app);
  const appPath = (app.config && app.config.appPath) || process.cwd();
  const controllers = path.join(appPath, 'api', 'controllers');
  const exists = file => { try { return fs.statSync(file).isFile(); } catch (_) { return false; } };
  const actionFile = identity => ['.js', '.cjs', '.mjs'].map(ext => path.join(controllers, `${identity}${ext}`)).find(exists) || null;
  const routes = [];
  for (const [address, target] of Object.entries((app.config && app.config.routes) || {})) {
    const match = /^\s*(?:(get|post|put|patch|delete|head|options|all)\s+)?(\/\S*)\s*$/i.exec(address);
    if (!match) continue;
    const methods = !match[1] || match[1].toLowerCase() === 'all' ? SAILS_METHODS : [match[1].toUpperCase()];
    let view = '';
    let declared = null;
    const named = typeof target === 'string' ? target : target && typeof target === 'object'
      ? (target.controller ? `${target.controller}.${target.action || 'index'}` : target.action || target.target || '') : '';
    if (/^[\w/-]+Controller\.\w+$/.test(named)) view = named;
    else if (/^[\w-]+(?:\/[\w-]+)+$/.test(named) || /^[\w-]+$/.test(named)) {
      // An action identity: an actions2 file, or a classic controller's action (user/find is
      // UserController.find).
      const file = actionFile(named);
      const parts = named.split('/');
      const classic = parts.length > 1 ? `${parts.slice(0, -1).join('/').replace(/(^|\/)(\w)/g, (_m, slash, c) => slash + c.toUpperCase())}Controller` : '';
      if (file) {
        view = named;
        declared = sailsActionInputs(file);
      } else if (classic && actionFile(classic)) {
        view = `${classic}.${parts[parts.length - 1]}`;
      }
    }
    routes.push({
      routePath: match[2],
      methods,
      name: address,
      view,
      declared: declared || undefined,
      handlerMethods: view ? [[view.split('.').pop(), methods]] : undefined
    });
  }
  const out = {};
  routes.forEach(route => addRoute(out, route));
  return out;
}

/** An actions2 action's declared inputs, as a framework declaration, or null. */
function sailsActionInputs(file) {
  let definition;
  try {
    definition = require(file);
  } catch (_) {
    return null;
  }
  definition = definition && definition.default ? definition.default : definition;
  if (!definition || typeof definition !== 'object' || !definition.inputs || typeof definition.fn !== 'function') return null;
  const inputs = {};
  for (const [name, spec] of Object.entries(definition.inputs)) {
    const kind = spec && typeof spec === 'object' ? SAILS_KINDS[spec.type] || 'text' : 'text';
    inputs[name] = kind === 'float' && spec.isInteger ? 'int' : kind;
  }
  // The action gets only its declared inputs, unless its code reaches for the request itself.
  const own = !/\bthis\.req\b/.test(String(definition.fn));
  return { inputs, covers: own ? COVERS_ALL : [], reasons: [], standalone: own };
}

function extractNextRoutes(handlers = []) {
  const routes = {};
  normalizeRouteList(handlers).forEach(route => addRoute(routes, route));
  return routes;
}

function extractNestRoutes(app, routesInput = null) {
  if (routesInput) {
    const routes = {};
    normalizeRouteList(routesInput).forEach(route => addRoute(routes, route));
    return routes;
  }
  // The controllers' own metadata: real handlers, and the @Query()/@Body() DTOs they take.
  const nest = require('./frameworkInputs').nestRoutes(app);
  if (nest.length) {
    const routes = {};
    nest.forEach(route => addRoute(routes, route));
    return routes;
  }
  try {
    const httpServer = typeof app?.getHttpServer === 'function' ? app.getHttpServer() : app;
    const expressApp = httpServer?._events?.request || httpServer?.app || app;
    return extractExpressRoutes(expressApp);
  } catch (err) {
    return {};
  }
}

/** init's injection audit couldn't run, so init must not write a manifest. */
class AuditError extends Error {
  constructor(message) {
    super(message);
    this.name = 'AuditError';
  }
}

// Saves the source's injection findings into a manifest that's about to be written. Without an
// audit (no source, or a scan that read no files) the framework's and declared inputs are still
// saved, but not as complete. Throws AuditError when the source is missing or the scan fails, so
// init stops without replacing a good manifest.
function auditIntoManifest(manifest, source) {
  let sources = {};
  const loadSources = () => {
    try {
      sources = require('./inputSources').sourcesFromEnvironment();
    } catch (error) {
      console.warn(`Declared, OpenAPI or recorded inputs skipped: ${error.message || error}`);
    }
  };
  let audited = false;
  try {
    loadSources();
    if (!source) return null;
    if (!require('fs').existsSync(source)) {
      throw new AuditError(`The audit source ${source} doesn't exist. Pass the app's source folder with --source, or --no-audit.`);
    }
    const { analyzePath } = require('./injectionAnalysis');
    let report;
    try {
      report = analyzePath(source);
    } catch (error) {
      throw new AuditError(`The injection audit of ${source} failed: ${error.message || error}`);
    }
    // No files read is no audit: nothing says these routes reach no risky call.
    if (!report.files_scanned) {
      console.warn(`The injection audit found no JavaScript or TypeScript files in ${source}, so no route counts as audited.`);
      return null;
    }
    const counts = require('./routeRisks').applyReport(report, manifest, sources);
    audited = true;
    return counts;
  } finally {
    // Without the code reading, the framework's and declared inputs are still saved.
    if (!audited && manifest && manifest.routes) {
      try {
        require('./inputSources').finalizeWithoutCode(manifest.routes, sources);
      } catch (error) {
        console.warn(`Framework inputs skipped: ${error.message || error}`);
      }
    }
    // A declaration's raw form is only for combining; never saved itself.
    for (const entry of Object.values((manifest && manifest.routes) || {})) {
      if (entry && typeof entry === 'object') {
        delete entry.input_declaration;
        delete entry.declared_inputs;
        delete entry.declared_methods;
        delete entry.declared_types;
        delete entry.declared_locations;
        delete entry._input_methods;
        delete entry._input_shared;
        delete entry._nested;
        delete entry._flat;
        delete entry.handler_methods;
      }
    }
  }
}

function generateExpressManifest(app, outputPath = DEFAULT_MANIFEST_PATH, options = {}) {
  const manifest = buildManifest({ framework: 'express', routes: extractExpressRoutes(app) });
  auditIntoManifest(manifest, options.auditSource);
  writeManifest(manifest, outputPath);
  return manifest;
}

function generateFrameworkManifest(framework, app, outputPath = DEFAULT_MANIFEST_PATH, options = {}) {
  const manifest = buildFrameworkManifest(framework, app, options);
  writeManifest(manifest, outputPath);
  return manifest;
}

// A JSON route list can't carry handlers; each listed route without one gets its handler's text
// from the source, when the source registers it, so its API and form signals aren't lost.
function withSourceHandlers(routesInput, source) {
  if (!Array.isArray(routesInput) || !routesInput.length || !source) return routesInput;
  let handlers;
  try {
    handlers = require('./sourceRoutes').routeHandlers(source);
  } catch (_) {
    return routesInput;
  }
  if (!handlers.size) return routesInput;
  const { handlerFor } = require('./sourceRoutes');
  return routesInput.map(route => {
    if (!route || ((route.handler || route.handle || route.endpoint) && route.schema)) return route;
    const routePath = route.path || route.url || route.routePath || route.pattern;
    if (!routePath) return route;
    const methods = [].concat(route.methods || route.method || []).map(method => String(method).toUpperCase());
    const found = handlerFor(handlers, routePath, methods);
    if (!found) return route;
    const filled = { ...route };
    if (!(route.handler || route.handle || route.endpoint)) filled.handler = found;
    // A schema written in the source, for a route Fastify didn't hand over with its options; or
    // the word that it has one that can't be read here.
    if (!route.schema && found.schema) filled.schema = found.schema;
    else if (!route.schema && found.schemaUnreadable) filled.schemaUnreadable = true;
    return filled;
  });
}

/** A framework app's manifest, with the source's injection findings, without writing it. */
function buildFrameworkManifest(framework, app, options = {}) {
  const normalized = String(framework || '').toLowerCase();
  const handlerSource = options.handlerSource || options.auditSource;
  if (options.routes) {
    options = { ...options, routes: withSourceHandlers(options.routes, handlerSource) };
  } else if (normalized === 'fastify' && app) {
    const listed = fastifyRoutes(app);
    if (listed.length) options = { ...options, routes: withSourceHandlers(listed, handlerSource) };
  }
  // Fastify schemas that $ref shared ones (fastify.addSchema): resolved from the live app's
  // getSchemas() and the source's literal addSchema calls.
  // A route's $refs name the schemas of the plugin scope it's registered in (fastify.addSchema in
  // that scope or a parent), which siblings may define differently under one $id. So each route
  // resolves against its own scope's getSchemas(); one whose scope isn't known only takes an $id
  // the source defines once, and an unresolved reference leaves its list incomplete.
  if (normalized === 'fastify' && Array.isArray(options.routes) && options.routes.some(route => route && route.schema)) {
    let fromSource = {};
    try {
      if (handlerSource) fromSource = require('./sourceRoutes').sourceSchemas(handlerSource);
    } catch (_) { /* source schemas are a bonus */ }
    const { resolveRefs } = require('./frameworkInputs');
    const owner = Symbol.for('aiwaf.fastify.routeOwner');
    options = {
      ...options,
      routes: options.routes.map(route => {
        if (!route || !route.schema) return route;
        let registry = fromSource;
        const scope = route[owner];
        if (scope && typeof scope.getSchemas === 'function') {
          try { registry = scope.getSchemas() || {}; } catch (_) { registry = {}; }
        }
        return { ...route, schema: resolveRefs(route.schema, registry) };
      })
    };
  }
  const extractors = {
    express: () => options.routes ? extractNextRoutes(options.routes) : extractExpressRoutes(app),
    sails: () => options.routes ? extractNextRoutes(options.routes) : extractSailsRoutes(app),
    fastify: () => extractFastifyRoutes(app, options.routes),
    hapi: () => extractHapiRoutes(app, options.routes),
    koa: () => extractKoaRoutes(app, options.routes),
    next: () => extractNextRoutes(options.routes || app),
    nest: () => extractNestRoutes(app, options.routes),
    // A route list, AdonisJS's list:routes JSON, or a router exported from the app.
    adonis: () => {
      const input = options.routes || (app && typeof app.toJSON === 'function' ? app.toJSON() : []);
      const listed = Array.isArray(input) && input.every(route => route && (route.path || route.url || route.routePath) && !route.pattern);
      return extractNextRoutes(listed ? input : adonisRouteList(input));
    }
  };
  const extractor = extractors[normalized];
  if (!extractor) {
    throw new Error(`Unsupported manifest framework: ${framework}`);
  }
  const manifest = buildManifest({
    framework: normalized,
    routes: extractor(),
    appContext: options.appContext || {}
  });
  auditIntoManifest(manifest, options.auditSource);
  return manifest;
}

module.exports = {
  AuditError,
  DEFAULT_MANIFEST_PATH,
  auditIntoManifest,
  SCHEMA_VERSION,
  computeContextHash,
  classifyRoute,
  buildRouteEntry,
  buildManifest,
  writeManifest,
  loadManifest,
  compileManifestToPathRules,
  getEffectivePathRules,
  routeEntryFor,
  extractExpressRoutes,
  mountPath,
  mountPaths,
  extractFastifyRoutes,
  fastifyRouteList,
  fastifyRoutes,
  extractHapiRoutes,
  extractSailsRoutes,
  adonisRouteList,
  extractKoaRoutes,
  extractNextRoutes,
  extractNestRoutes,
  generateExpressManifest,
  generateFrameworkManifest,
  buildFrameworkManifest,
  detectApiEndpoint,
  detectAuthEndpoint
};
