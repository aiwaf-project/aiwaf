let config = {
  enabled: false,
  requiredHeaders: [],
  blockedUserAgents: [],
  minScore: 3,
  maxHeaderBytes: 32 * 1024,
  maxHeaderCount: 100,
  maxUserAgentLength: 500,
  maxAcceptLength: 4096,
  suspiciousUserAgents: [],
  legitimateBots: [],
  staticExtensions: ['.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.woff', '.woff2', '.ttf'],
  staticPaths: ['/static/', '/media/', '/assets/', '/favicon.ico']
};

const BROWSER_HEADERS = [
  'accept-language',
  'accept-encoding',
  'connection',
  'cache-control'
];

const DEFAULT_SUSPICIOUS_UA = [
  /bot/i,
  /crawler/i,
  /spider/i,
  /scraper/i,
  /curl/i,
  /wget/i,
  /python/i,
  /java/i,
  /node/i,
  /go-http/i,
  /axios/i,
  /okhttp/i,
  /libwww/i,
  /lwp-trivial/i,
  /mechanize/i,
  /requests/i,
  /urllib/i,
  /httpie/i,
  /postman/i,
  /insomnia/i,
  /^$/,
  /^mozilla\/4\.0$/i
];

const DEFAULT_LEGIT_BOTS = [
  /googlebot/i,
  /bingbot/i,
  /slurp/i,
  /duckduckbot/i,
  /baiduspider/i,
  /yandexbot/i,
  /facebookexternalhit/i,
  /twitterbot/i,
  /linkedinbot/i,
  /whatsapp/i,
  /telegrambot/i,
  /applebot/i,
  /pingdom/i,
  /uptimerobot/i,
  /statuscake/i,
  /site24x7/i
];

function isStaticRequest(pathname) {
  const pathLower = String(pathname || '').toLowerCase();
  if (config.staticExtensions.some(ext => pathLower.endsWith(ext))) return true;
  if (config.staticPaths.some(prefix => pathLower.startsWith(prefix))) return true;
  return false;
}

function enforceHeaderCaps(headers) {
  let totalBytes = 0;
  let headerCount = 0;

  for (const [key, value] of Object.entries(headers)) {
    headerCount += 1;
    const valueStr = value === undefined || value === null ? '' : String(value);
    totalBytes += String(key).length + valueStr.length;

    if (totalBytes > config.maxHeaderBytes) {
      return `header_bytes_exceeded:${config.maxHeaderBytes}`;
    }
  }

  if (headerCount > config.maxHeaderCount) {
    return `header_count_exceeded:${config.maxHeaderCount}`;
  }

  const ua = headers['user-agent'];
  if (ua && String(ua).length > config.maxUserAgentLength) {
    return `user_agent_too_long:${config.maxUserAgentLength}`;
  }

  const accept = headers['accept'];
  if (accept && String(accept).length > config.maxAcceptLength) {
    return `accept_too_long:${config.maxAcceptLength}`;
  }

  return null;
}

function matchesPattern(list, value) {
  return list.some(pattern => pattern.test(value));
}

function checkSuspiciousCombinations(headers) {
  const protocol = String(headers['server-protocol'] || '').toUpperCase();
  const ua = String(headers['user-agent'] || '').toLowerCase();
  const accept = headers['accept'];

  if (protocol.startsWith('HTTP/2') && ua.includes('mozilla/4.0')) {
    return 'http2_with_old_ua';
  }

  if (headers['user-agent'] && !accept) {
    return 'ua_without_accept';
  }

  if (accept === '*/*' && !headers['accept-language'] && !headers['accept-encoding']) {
    return 'generic_accept_no_locale';
  }

  if (headers['user-agent'] && !headers['accept-language'] && !headers['accept-encoding'] && !headers['connection']) {
    return 'missing_browser_headers';
  }

  if (protocol === 'HTTP/1.0' && ua.includes('chrome')) {
    return 'modern_browser_http10';
  }

  return null;
}

function calculateHeaderQuality(headers) {
  let score = 0;
  const ua = headers['user-agent'];
  const accept = headers['accept'];

  if (ua) score += 2;
  if (accept) score += 2;

  for (const header of BROWSER_HEADERS) {
    if (headers[header]) score += 1;
  }

  if (headers['accept-language'] && headers['accept-encoding']) score += 1;

  if (String(headers['connection'] || '').toLowerCase() === 'keep-alive') score += 1;

  const acceptValue = String(accept || '').toLowerCase();
  if (acceptValue.includes('text/html') && acceptValue.includes('application/xml')) score += 1;

  return score;
}

module.exports = {
  init(opts = {}) {
    const requiredHeaders = opts.AIWAF_REQUIRED_HEADERS || [];
    const toRegexList = (items, fallback) => {
      if (!items) return fallback;
      if (!Array.isArray(items)) return fallback;
      return items.map(item => {
        if (item instanceof RegExp) return item;
        return new RegExp(String(item), 'i');
      });
    };
    config = {
      enabled: !!opts.AIWAF_HEADER_VALIDATION,
      requiredHeaders: Array.isArray(requiredHeaders)
        ? requiredHeaders.map(h => String(h).toLowerCase())
        : (requiredHeaders && typeof requiredHeaders === 'object' ? requiredHeaders : []),
      blockedUserAgents: (opts.AIWAF_BLOCKED_USER_AGENTS || []).map(u => String(u).toLowerCase()),
      minScore: Number.isFinite(Number(opts.AIWAF_HEADER_QUALITY_MIN_SCORE))
        ? Number(opts.AIWAF_HEADER_QUALITY_MIN_SCORE)
        : 3,
      maxHeaderBytes: Number.isFinite(Number(opts.AIWAF_MAX_HEADER_BYTES))
        ? Number(opts.AIWAF_MAX_HEADER_BYTES)
        : 32 * 1024,
      maxHeaderCount: Number.isFinite(Number(opts.AIWAF_MAX_HEADER_COUNT))
        ? Number(opts.AIWAF_MAX_HEADER_COUNT)
        : 100,
      maxUserAgentLength: Number.isFinite(Number(opts.AIWAF_MAX_USER_AGENT_LENGTH))
        ? Number(opts.AIWAF_MAX_USER_AGENT_LENGTH)
        : 500,
      maxAcceptLength: Number.isFinite(Number(opts.AIWAF_MAX_ACCEPT_LENGTH))
        ? Number(opts.AIWAF_MAX_ACCEPT_LENGTH)
        : 4096,
      suspiciousUserAgents: toRegexList(opts.AIWAF_SUSPICIOUS_USER_AGENTS, DEFAULT_SUSPICIOUS_UA),
      legitimateBots: toRegexList(opts.AIWAF_LEGITIMATE_BOTS, DEFAULT_LEGIT_BOTS),
      staticExtensions: opts.AIWAF_STATIC_EXTENSIONS || config.staticExtensions,
      staticPaths: opts.AIWAF_STATIC_PATHS || config.staticPaths
    };
  },

  validate(req) {
    if (!config.enabled) return null;

    const headers = Object.fromEntries(
      Object.entries(req.headers || {}).map(([key, value]) => [String(key).toLowerCase(), value])
    );
    headers['server-protocol'] = req.httpVersion ? `HTTP/${req.httpVersion}` : headers['server-protocol'];

    if (process.env.AIWAF_DEBUG_HEADERS) {
      console.error(`[HEADER-VALIDATION] headers keys: ${Object.keys(headers).join(', ')}`);
      console.error(`[HEADER-VALIDATION] user-agent: ${headers['user-agent']}`);
      console.error(`[HEADER-VALIDATION] accept: ${headers['accept']}`);
      console.error(`[HEADER-VALIDATION] accept-language: ${headers['accept-language']}`);
      console.error(`[HEADER-VALIDATION] accept-encoding: ${headers['accept-encoding']}`);
      console.error(`[HEADER-VALIDATION] connection: ${headers['connection']}`);
    }

    if (isStaticRequest(req.path || req.url || '')) return null;

    const capReason = enforceHeaderCaps(headers);
    if (capReason) {
      return capReason;
    }

    let requiredHeaders = config.requiredHeaders;
    if (requiredHeaders && typeof requiredHeaders === 'object' && !Array.isArray(requiredHeaders)) {
      const method = String(req.method || '').toUpperCase();
      requiredHeaders = requiredHeaders[method] || requiredHeaders.DEFAULT || [];
    }
    for (const header of requiredHeaders || []) {
      if (!headers[header]) {
        return `missing_header:${header}`;
      }
    }

    const ua = String(headers['user-agent'] || '').toLowerCase();
    if (config.legitimateBots.length && matchesPattern(config.legitimateBots, ua)) {
      return null;
    }
    if (config.suspiciousUserAgents.length && matchesPattern(config.suspiciousUserAgents, ua)) {
      return 'suspicious_user_agent';
    }
    if (config.blockedUserAgents.some(pattern => ua.includes(pattern))) {
      return 'blocked_user_agent';
    }

    const comboReason = checkSuspiciousCombinations(headers);
    if (comboReason) {
      return `suspicious_headers:${comboReason}`;
    }

    const minScore = (Array.isArray(requiredHeaders) && requiredHeaders.length === 0) ? 0 : config.minScore;
    if (minScore > 0) {
      const qualityScore = calculateHeaderQuality(headers);
      if (qualityScore < minScore) {
        return 'header_quality_low';
      }
    }

    return null;
  }
  ,
  getWasmConfig(method) {
    if (!config.enabled) return null;
    let requiredHeaders = config.requiredHeaders;
    if (requiredHeaders && typeof requiredHeaders === 'object' && !Array.isArray(requiredHeaders)) {
      const normalizedMethod = String(method || '').toUpperCase();
      requiredHeaders = requiredHeaders[normalizedMethod] || requiredHeaders.DEFAULT || [];
    }
    return {
      requiredHeaders: Array.isArray(requiredHeaders) ? requiredHeaders : [],
      minScore: (Array.isArray(requiredHeaders) && requiredHeaders.length === 0) ? 0 : config.minScore
    };
  },
  isStaticRequest(pathname) {
    return isStaticRequest(pathname);
  }
};
