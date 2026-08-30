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
  const source = typeof handler === 'function' ? Function.prototype.toString.call(handler) : '';
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
  const source = typeof handler === 'function' ? Function.prototype.toString.call(handler).toLowerCase() : '';
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

function writeManifest(manifest, outputPath = DEFAULT_MANIFEST_PATH) {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
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

function routeEntryFor({ routePath, methods = [], handler, name = '', view = '' }) {
  const inferredMethods = methods.length ? methods : inferMethodsFromHandler(handler);
  const api = detectApiEndpoint(handler, routePath, inferredMethods);
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
  if (routes[normalized]) {
    const mergedMethods = Array.from(new Set([...(routes[normalized].methods || []), ...(entry.methods || [])])).sort();
    routes[normalized] = {
      ...routes[normalized],
      ...entry,
      methods: mergedMethods
    };
  } else {
    routes[normalized] = entry;
  }
}

function extractExpressRoutes(app) {
  const routes = {};
  const stack = app?._router?.stack || [];
  stack.forEach(layer => {
    const route = layer.route;
    if (!route || !route.path) return;
    const routePath = String(route.path);
    const handler = route.stack?.find(item => typeof item.handle === 'function')?.handle;
    const methods = methodsFromExpressRoute(route);
    addRoute(routes, {
      routePath,
      methods,
      handler,
      name: routePath
    });
  });
  return routes;
}

function normalizeRouteList(routesInput = []) {
  return (Array.isArray(routesInput) ? routesInput : [])
    .map(route => ({
      routePath: route.path || route.url || route.routePath || route.pattern,
      methods: Array.isArray(route.methods)
        ? route.methods
        : (route.method ? [route.method] : []),
      handler: route.handler || route.handle || route.endpoint,
      name: route.name || route.id || route.path || route.url || '',
      view: route.view || ''
    }))
    .filter(route => route.routePath);
}

function extractFastifyRoutes(fastify, routesInput = null) {
  const routes = {};
  normalizeRouteList(routesInput || fastify?.aiwafRoutes || fastify?.routes || [])
    .forEach(route => addRoute(routes, route));
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
    name: route.settings?.id || route.path
  }))).forEach(route => addRoute(routes, route));
  return routes;
}

function extractKoaRoutes(_app, routesInput = []) {
  const routes = {};
  normalizeRouteList(routesInput).forEach(route => addRoute(routes, route));
  return routes;
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
  try {
    const httpServer = typeof app?.getHttpServer === 'function' ? app.getHttpServer() : app;
    const expressApp = httpServer?._events?.request || httpServer?.app || app;
    return extractExpressRoutes(expressApp);
  } catch (err) {
    return {};
  }
}

function generateExpressManifest(app, outputPath = DEFAULT_MANIFEST_PATH) {
  const manifest = buildManifest({ framework: 'express', routes: extractExpressRoutes(app) });
  writeManifest(manifest, outputPath);
  return manifest;
}

function generateFrameworkManifest(framework, app, outputPath = DEFAULT_MANIFEST_PATH, options = {}) {
  const normalized = String(framework || '').toLowerCase();
  const extractors = {
    express: () => options.routes ? extractNextRoutes(options.routes) : extractExpressRoutes(app),
    sails: () => options.routes ? extractNextRoutes(options.routes) : extractExpressRoutes(app),
    fastify: () => extractFastifyRoutes(app, options.routes),
    hapi: () => extractHapiRoutes(app, options.routes),
    koa: () => extractKoaRoutes(app, options.routes),
    next: () => extractNextRoutes(options.routes || app),
    nest: () => extractNestRoutes(app, options.routes),
    adonis: () => extractNextRoutes(options.routes || [])
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
  writeManifest(manifest, outputPath);
  return manifest;
}

module.exports = {
  DEFAULT_MANIFEST_PATH,
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
  extractFastifyRoutes,
  extractHapiRoutes,
  extractKoaRoutes,
  extractNextRoutes,
  extractNestRoutes,
  generateExpressManifest,
  generateFrameworkManifest,
  detectApiEndpoint,
  detectAuthEndpoint
};
