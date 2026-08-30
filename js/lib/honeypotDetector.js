let config = {
  field: undefined,
  minFormTimeMs: 0,
  maxPageTimeMs: 3600 * 1000,
  loginPathPrefixes: [
    '/admin/login/',
    '/login/',
    '/accounts/login/',
    '/auth/login/',
    '/signin/'
  ],
  postOnlySuffixes: [
    '/create/',
    '/submit/',
    '/upload/',
    '/delete/',
    '/process/'
  ],
  allowedMethods: ['GET', 'POST', 'HEAD', 'OPTIONS'],
  methodPolicyEnabled: false
};

const lastGetByKey = new Map();

function getKey(ip, path) {
  return `${ip || 'unknown'}`;
}

function isLoginPath(path) {
  const pathLower = String(path || '').toLowerCase();
  return config.loginPathPrefixes.some(prefix => pathLower.startsWith(prefix));
}

function isObviousPostOnly(path) {
  const pathLower = String(path || '').toLowerCase();
  return config.postOnlySuffixes.some(suffix => pathLower.endsWith(suffix));
}

module.exports = {
  init(opts = {}) {
    config = {
      field: opts.HONEYPOT_FIELD,
      minFormTimeMs: Math.max(0, Number(opts.AIWAF_MIN_FORM_TIME || 0) * 1000),
      maxPageTimeMs: Math.max(0, Number(opts.AIWAF_MAX_PAGE_TIME || 3600) * 1000),
      loginPathPrefixes: (opts.AIWAF_LOGIN_PATH_PREFIXES || config.loginPathPrefixes).map(p => String(p)),
      postOnlySuffixes: (opts.AIWAF_POST_ONLY_SUFFIXES || config.postOnlySuffixes).map(p => String(p)),
      allowedMethods: (opts.AIWAF_ALLOWED_METHODS || config.allowedMethods).map(m => String(m).toUpperCase()),
      methodPolicyEnabled: opts.AIWAF_METHOD_POLICY_ENABLED === true
    };
  },

  evaluate(req, ip, path) {
    const method = String(req.method || 'GET').toUpperCase();
    const key = getKey(ip, path);

    if (config.methodPolicyEnabled && !config.allowedMethods.includes(method)) {
      return { triggered: true, reason: `method_not_allowed:${method}`, statusCode: 405, errorCode: 'blocked' };
    }

    if (method === 'GET') {
      if (config.methodPolicyEnabled && isObviousPostOnly(path)) {
        return { triggered: true, reason: `get_post_only:${path}`, statusCode: 405, errorCode: 'blocked' };
      }
      lastGetByKey.set(key, Date.now());
      return { triggered: false };
    }

    if (config.field && req.body && req.body[config.field]) {
      return { triggered: true, reason: 'honeypot_field' };
    }

    if (method === 'POST' && (config.minFormTimeMs > 0 || config.maxPageTimeMs > 0)) {
      const lastGet = lastGetByKey.get(key);
      if (lastGet) {
        const elapsed = Date.now() - lastGet;
        const minTime = isLoginPath(path) ? Math.min(config.minFormTimeMs, 100) : config.minFormTimeMs;

        if (minTime > 0 && elapsed < minTime) {
          return { triggered: true, reason: 'honeypot_too_fast' };
        }

        if (config.maxPageTimeMs > 0 && elapsed > config.maxPageTimeMs) {
          lastGetByKey.delete(key);
          return { triggered: true, reason: 'honeypot_too_slow', statusCode: 409, errorCode: 'page_expired' };
        }
      }
    }

    return { triggered: false };
  }
};
