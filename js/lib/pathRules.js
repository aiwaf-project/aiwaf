const MIDDLEWARE_NAME_MAP = {
  ipandkeywordblockmiddleware: 'ip_keyword_block',
  ratelimitmiddleware: 'rate_limit',
  ratelimiting: 'rate_limit',
  rate_limiting: 'rate_limit',
  honeypottimingmiddleware: 'honeypot',
  headervalidationmiddleware: 'header_validation',
  geoblockmiddleware: 'geo_block',
  aianomalymiddleware: 'ai_anomaly',
  uuidtampermiddleware: 'uuid_tamper',
  uuid: 'uuid_tamper',
  aiwafloggingmiddleware: 'logging',
  logging_middleware: 'logging'
};

const ALL_MIDDLEWARES = [
  'geo_block',
  'ip_keyword_block',
  'rate_limit',
  'ai_anomaly',
  'honeypot',
  'uuid_tamper',
  'header_validation',
  'logging'
];

function normalizePath(value, trailingSlash = null) {
  let cleaned = String(value || '').trim().replace(/\/{2,}/g, '/');
  if (!cleaned) cleaned = '/';
  if (!cleaned.startsWith('/')) cleaned = `/${cleaned}`;
  if (trailingSlash === true && !cleaned.endsWith('/')) cleaned += '/';
  if (trailingSlash === false && cleaned !== '/') cleaned = cleaned.replace(/\/+$/, '');
  return cleaned.toLowerCase();
}

function normalizePaths(items = []) {
  return (items || []).map(item => normalizePath(item, false));
}

function wildcardMatch(pathValue, pattern) {
  const escaped = String(pattern)
    .replace(/[.+?^${}()|[\]\\]/g, '\\$&')
    .replace(/\*/g, '.*');
  return new RegExp(`^${escaped}$`).test(pathValue);
}

function isPathExempt(pathValue, exemptPaths = [], { allowWildcards = true, allowPrefix = true } = {}) {
  const normalizedPath = normalizePath(pathValue, false);
  return (exemptPaths || []).some(entry => {
    const normalizedEntry = normalizePath(entry, false);
    if (normalizedPath === normalizedEntry) return true;
    if (allowWildcards && normalizedEntry.includes('*') && wildcardMatch(normalizedPath, normalizedEntry)) return true;
    if (allowPrefix && normalizedPath.startsWith(normalizedEntry.endsWith('/') ? normalizedEntry : `${normalizedEntry}/`)) return true;
    return false;
  });
}

function normalizeMiddlewareName(name) {
  if (!name) return '';
  let normalized = String(name).trim();
  if (normalized.includes('.')) {
    normalized = normalized.split('.').pop();
  }
  normalized = normalized.toLowerCase();
  return MIDDLEWARE_NAME_MAP[normalized] || normalized;
}

function getPathRule(path, rules = []) {
  if (!path || !Array.isArray(rules)) return null;
  const normalizedPath = normalizePath(path, false);
  let best = null;

  rules.forEach((rule, position) => {
    if (!rule || typeof rule !== 'object' || !rule.PREFIX) return;
    const prefix = normalizePath(rule.PREFIX, true);
    if (normalizedPath === prefix.replace(/\/+$/, '') || normalizedPath.startsWith(prefix)) {
      if (!best || prefix.length > best.prefix.length) {
        best = { prefix, rule, position };
      }
    }
  });

  return best ? best.rule : null;
}

function getPathRuleOverrides(path, rules, sectionKey) {
  const rule = getPathRule(path, rules);
  if (!rule || !sectionKey) return {};
  const upperKey = String(sectionKey).toUpperCase();
  const lowerKey = String(sectionKey).toLowerCase();
  const value = rule[upperKey] || rule[lowerKey];
  return value && typeof value === 'object' && !Array.isArray(value) ? { ...value } : {};
}

function isMiddlewareDisabledForPath(path, rules, middlewareName) {
  const rule = getPathRule(path, rules);
  if (!rule) return false;
  const disabled = rule.DISABLE || rule.disable || [];
  if (!Array.isArray(disabled)) return false;
  const target = normalizeMiddlewareName(middlewareName);
  return disabled.some(entry => normalizeMiddlewareName(entry) === target);
}

function createRoutePlan(path, rules = [], context = {}) {
  const required = new Set((context.requiredMiddlewares || []).map(normalizeMiddlewareName));
  const exempt = new Set((context.exemptMiddlewares || []).map(normalizeMiddlewareName));
  const fullyExempt = !!context.fullyExempt;
  const enabled = new Set();

  ALL_MIDDLEWARES.forEach(name => {
    if (required.has(name)) {
      enabled.add(name);
    } else if (isMiddlewareDisabledForPath(path, rules, name) || fullyExempt || exempt.has(name)) {
      return;
    } else {
      enabled.add(name);
    }
  });

  return {
    shouldApply(name) {
      return enabled.has(normalizeMiddlewareName(name));
    },
    getOverrides(sectionKey) {
      return getPathRuleOverrides(path, rules, sectionKey);
    },
    enabledMiddlewares: enabled
  };
}

function markRequest(req, options = {}) {
  req.aiwafRoute = {
    ...(req.aiwafRoute || {}),
    ...options,
    exemptMiddlewares: [
      ...((req.aiwafRoute && req.aiwafRoute.exemptMiddlewares) || []),
      ...((options && options.exemptMiddlewares) || [])
    ],
    requiredMiddlewares: [
      ...((req.aiwafRoute && req.aiwafRoute.requiredMiddlewares) || []),
      ...((options && options.requiredMiddlewares) || [])
    ]
  };
}

function exempt(req, _res, next) {
  markRequest(req, { fullyExempt: true });
  next();
}

function exemptFrom(...middlewareNames) {
  return (req, _res, next) => {
    markRequest(req, { exemptMiddlewares: middlewareNames });
    next();
  };
}

function only(...middlewareNames) {
  const selected = new Set(middlewareNames.map(normalizeMiddlewareName));
  return exemptFrom(...ALL_MIDDLEWARES.filter(name => !selected.has(name)));
}

function requireProtection(...middlewareNames) {
  return (req, _res, next) => {
    markRequest(req, { requiredMiddlewares: middlewareNames });
    next();
  };
}

module.exports = {
  ALL_MIDDLEWARES,
  normalizePath,
  normalizePaths,
  isPathExempt,
  normalizeMiddlewareName,
  getPathRule,
  getPathRuleOverrides,
  isMiddlewareDisabledForPath,
  createRoutePlan,
  exempt,
  exemptFrom,
  only,
  requireProtection
};
