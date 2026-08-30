function firstDefined(...values) {
  for (const value of values) {
    if (value !== undefined && value !== null) return value;
  }
  return undefined;
}

function toBool(value, defaultValue = false) {
  if (value === undefined || value === null) return defaultValue;
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') return value !== 0;
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (['1', 'true', 'yes', 'on'].includes(normalized)) return true;
    if (['0', 'false', 'no', 'off'].includes(normalized)) return false;
  }
  return defaultValue;
}

function toNumber(value, defaultValue) {
  if (value === undefined || value === null || value === '') return defaultValue;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : defaultValue;
}

function toArray(value, defaultValue = []) {
  if (value === undefined || value === null) return defaultValue;
  if (Array.isArray(value)) return value;
  if (typeof value === 'string') {
    return value
      .split(',')
      .map(item => item.trim())
      .filter(Boolean);
  }
  return defaultValue;
}

function toOptionalArray(value) {
  if (value === undefined || value === null) return undefined;
  if (Array.isArray(value)) return value;
  if (typeof value === 'string') {
    return value
      .split(',')
      .map(item => item.trim())
      .filter(Boolean);
  }
  return undefined;
}

function toPathRules(value, defaultValue = []) {
  if (value === undefined || value === null || value === '') return defaultValue;
  if (Array.isArray(value)) return value;
  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value);
      return Array.isArray(parsed) ? parsed : defaultValue;
    } catch (err) {
      return defaultValue;
    }
  }
  return defaultValue;
}

function normalizeSettings(rawOpts = {}) {
  const legacy = rawOpts.AIWAF_SETTINGS || {};

  const normalized = {
    ...rawOpts,
    staticKeywords: firstDefined(rawOpts.staticKeywords, legacy.keywords?.static, []),
    dynamicTopN: toNumber(firstDefined(rawOpts.dynamicTopN, rawOpts.DYNAMIC_TOP_N, legacy.keywords?.dynamicTopN), 10),

    WINDOW_SEC: toNumber(firstDefined(rawOpts.WINDOW_SEC, rawOpts.windowSec, rawOpts.AIWAF_WINDOW_SECONDS, legacy.rate?.window, process.env.AIWAF_RATE_WINDOW), 60),
    MAX_REQ: toNumber(firstDefined(rawOpts.MAX_REQ, rawOpts.maxReq, legacy.rate?.max, process.env.AIWAF_RATE_MAX), 100),
    FLOOD_REQ: toNumber(firstDefined(rawOpts.FLOOD_REQ, rawOpts.floodReq, legacy.rate?.flood, process.env.AIWAF_RATE_FLOOD), 200),

    HONEYPOT_FIELD: firstDefined(rawOpts.HONEYPOT_FIELD, rawOpts.honeypotField, legacy.honeypot?.field),
    AIWAF_MIN_FORM_TIME: toNumber(firstDefined(rawOpts.AIWAF_MIN_FORM_TIME, rawOpts.minFormTime, legacy.honeypot?.minFormTime, process.env.AIWAF_MIN_FORM_TIME), 0),
    AIWAF_MAX_PAGE_TIME: toNumber(firstDefined(rawOpts.AIWAF_MAX_PAGE_TIME, rawOpts.maxPageTime, legacy.honeypot?.maxPageTime, process.env.AIWAF_MAX_PAGE_TIME), 3600),
    AIWAF_LOGIN_PATH_PREFIXES: toArray(firstDefined(rawOpts.AIWAF_LOGIN_PATH_PREFIXES, rawOpts.loginPathPrefixes, legacy.honeypot?.loginPathPrefixes, process.env.AIWAF_LOGIN_PATH_PREFIXES), undefined),
    AIWAF_POST_ONLY_SUFFIXES: toArray(firstDefined(rawOpts.AIWAF_POST_ONLY_SUFFIXES, rawOpts.postOnlySuffixes, legacy.honeypot?.postOnlySuffixes, process.env.AIWAF_POST_ONLY_SUFFIXES), undefined),
    AIWAF_ALLOWED_METHODS: toArray(firstDefined(rawOpts.AIWAF_ALLOWED_METHODS, rawOpts.allowedMethods, legacy.honeypot?.allowedMethods, process.env.AIWAF_ALLOWED_METHODS), undefined),
    AIWAF_METHOD_POLICY_ENABLED: toBool(firstDefined(rawOpts.AIWAF_METHOD_POLICY_ENABLED, rawOpts.methodPolicyEnabled, legacy.honeypot?.methodPolicyEnabled, process.env.AIWAF_METHOD_POLICY_ENABLED), false),

    AIWAF_EXEMPT_PATHS: toArray(firstDefined(rawOpts.AIWAF_EXEMPT_PATHS, rawOpts.exemptPaths, legacy.exemptions?.paths, process.env.AIWAF_EXEMPT_PATHS), []),
    AIWAF_EXEMPT_IPS: toArray(firstDefined(rawOpts.AIWAF_EXEMPT_IPS, rawOpts.exemptIps, legacy.exemptions?.ips, process.env.AIWAF_EXEMPT_IPS), []),
    AIWAF_ALLOWED_PATH_KEYWORDS: toArray(firstDefined(rawOpts.AIWAF_ALLOWED_PATH_KEYWORDS, rawOpts.allowedPathKeywords, legacy.exemptions?.allowedPathKeywords, process.env.AIWAF_ALLOWED_PATH_KEYWORDS), []),
    AIWAF_EXEMPT_KEYWORDS: toArray(firstDefined(rawOpts.AIWAF_EXEMPT_KEYWORDS, rawOpts.exemptKeywords, legacy.exemptions?.keywords, process.env.AIWAF_EXEMPT_KEYWORDS), []),
    AIWAF_EXEMPTIONS_DB: toBool(firstDefined(rawOpts.AIWAF_EXEMPTIONS_DB, rawOpts.exemptionsDb, legacy.exemptions?.dbEnabled, process.env.AIWAF_EXEMPTIONS_DB), true),

    AIWAF_HEADER_VALIDATION: toBool(firstDefined(rawOpts.AIWAF_HEADER_VALIDATION, rawOpts.enableHeaderValidation, legacy.headerValidation?.enabled, process.env.AIWAF_HEADER_VALIDATION), false),
    AIWAF_REQUIRED_HEADERS: firstDefined(rawOpts.AIWAF_REQUIRED_HEADERS, rawOpts.requiredHeaders, legacy.headerValidation?.requiredHeaders, process.env.AIWAF_REQUIRED_HEADERS) ?? [],
    AIWAF_HEADER_QUALITY_MIN_SCORE: toNumber(firstDefined(rawOpts.AIWAF_HEADER_QUALITY_MIN_SCORE, rawOpts.headerQualityMinScore, legacy.headerValidation?.minScore, process.env.AIWAF_HEADER_QUALITY_MIN_SCORE), 3),
    AIWAF_MAX_HEADER_BYTES: toNumber(firstDefined(rawOpts.AIWAF_MAX_HEADER_BYTES, rawOpts.maxHeaderBytes, legacy.headerValidation?.maxHeaderBytes, process.env.AIWAF_MAX_HEADER_BYTES), 32 * 1024),
    AIWAF_MAX_HEADER_COUNT: toNumber(firstDefined(rawOpts.AIWAF_MAX_HEADER_COUNT, rawOpts.maxHeaderCount, legacy.headerValidation?.maxHeaderCount, process.env.AIWAF_MAX_HEADER_COUNT), 100),
    AIWAF_MAX_USER_AGENT_LENGTH: toNumber(firstDefined(rawOpts.AIWAF_MAX_USER_AGENT_LENGTH, rawOpts.maxUserAgentLength, legacy.headerValidation?.maxUserAgentLength, process.env.AIWAF_MAX_USER_AGENT_LENGTH), 500),
    AIWAF_MAX_ACCEPT_LENGTH: toNumber(firstDefined(rawOpts.AIWAF_MAX_ACCEPT_LENGTH, rawOpts.maxAcceptLength, legacy.headerValidation?.maxAcceptLength, process.env.AIWAF_MAX_ACCEPT_LENGTH), 4096),
    AIWAF_BLOCKED_USER_AGENTS: toArray(firstDefined(rawOpts.AIWAF_BLOCKED_USER_AGENTS, rawOpts.blockedUserAgents, legacy.headerValidation?.blockedUserAgents, process.env.AIWAF_BLOCKED_USER_AGENTS), [
      'sqlmap',
      'nikto',
      'acunetix',
      'nmap',
      'masscan'
    ]),

    AIWAF_GEO_BLOCK_ENABLED: toBool(firstDefined(rawOpts.AIWAF_GEO_BLOCK_ENABLED, rawOpts.geoBlockEnabled, legacy.geo?.enabled, process.env.AIWAF_GEO_BLOCK_ENABLED), false),
    AIWAF_GEO_BLOCK_COUNTRIES: toArray(firstDefined(rawOpts.AIWAF_GEO_BLOCK_COUNTRIES, rawOpts.geoBlockCountries, legacy.geo?.blockedCountries, process.env.AIWAF_GEO_BLOCK_COUNTRIES), []),
    AIWAF_GEO_ALLOW_COUNTRIES: toArray(firstDefined(rawOpts.AIWAF_GEO_ALLOW_COUNTRIES, rawOpts.geoAllowCountries, legacy.geo?.allowCountries, process.env.AIWAF_GEO_ALLOW_COUNTRIES), []),
    AIWAF_GEO_MMDB_PATH: firstDefined(rawOpts.AIWAF_GEO_MMDB_PATH, rawOpts.geoDbPath, legacy.geo?.mmdbPath, process.env.AIWAF_GEO_MMDB_PATH),
    AIWAF_GEO_CACHE_SECONDS: toNumber(firstDefined(rawOpts.AIWAF_GEO_CACHE_SECONDS, rawOpts.geoCacheSeconds, legacy.geo?.cacheSeconds, process.env.AIWAF_GEO_CACHE_SECONDS), 3600),
    AIWAF_GEO_CACHE_PREFIX: firstDefined(rawOpts.AIWAF_GEO_CACHE_PREFIX, rawOpts.geoCachePrefix, legacy.geo?.cachePrefix, process.env.AIWAF_GEO_CACHE_PREFIX, 'aiwaf:geo:'),
    AIWAF_DYNAMIC_KEYWORDS_DB: toBool(firstDefined(rawOpts.AIWAF_DYNAMIC_KEYWORDS_DB, rawOpts.dynamicKeywordsDb, legacy.keywords?.dbEnabled, process.env.AIWAF_DYNAMIC_KEYWORDS_DB), true),
    AIWAF_MODEL_STORAGE: firstDefined(rawOpts.AIWAF_MODEL_STORAGE, rawOpts.modelStorage, legacy.model?.storage, process.env.AIWAF_MODEL_STORAGE, 'file'),
    AIWAF_MODEL_PATH: firstDefined(rawOpts.AIWAF_MODEL_PATH, rawOpts.modelPath, legacy.model?.path, process.env.AIWAF_MODEL_PATH),
    AIWAF_MODEL_STORAGE_FALLBACK: firstDefined(rawOpts.AIWAF_MODEL_STORAGE_FALLBACK, rawOpts.modelStorageFallback, legacy.model?.fallback, process.env.AIWAF_MODEL_STORAGE_FALLBACK, 'file'),
    AIWAF_MODEL_CACHE_KEY: firstDefined(rawOpts.AIWAF_MODEL_CACHE_KEY, rawOpts.modelCacheKey, legacy.model?.cacheKey, process.env.AIWAF_MODEL_CACHE_KEY),
    AIWAF_MODEL_CACHE_TTL: toNumber(firstDefined(rawOpts.AIWAF_MODEL_CACHE_TTL, rawOpts.modelCacheTtl, legacy.model?.cacheTtl, process.env.AIWAF_MODEL_CACHE_TTL), 0),

    AIWAF_MIDDLEWARE_LOGGING: toBool(firstDefined(rawOpts.AIWAF_MIDDLEWARE_LOGGING, rawOpts.middlewareLogging, legacy.logging?.enabled, process.env.AIWAF_MIDDLEWARE_LOGGING), false),
    AIWAF_MIDDLEWARE_LOG_PATH: firstDefined(rawOpts.AIWAF_MIDDLEWARE_LOG_PATH, rawOpts.middlewareLogPath, legacy.logging?.path, process.env.AIWAF_MIDDLEWARE_LOG_PATH, 'logs/aiwaf-requests.jsonl'),
    AIWAF_MIDDLEWARE_LOG_DB: toBool(firstDefined(rawOpts.AIWAF_MIDDLEWARE_LOG_DB, rawOpts.middlewareLogDb, legacy.logging?.dbEnabled, process.env.AIWAF_MIDDLEWARE_LOG_DB), false),
    AIWAF_MIDDLEWARE_LOG_CSV: toBool(firstDefined(rawOpts.AIWAF_MIDDLEWARE_LOG_CSV, rawOpts.middlewareLogCsv, legacy.logging?.csvEnabled, process.env.AIWAF_MIDDLEWARE_LOG_CSV), false),
    AIWAF_MIDDLEWARE_LOG_CSV_PATH: firstDefined(rawOpts.AIWAF_MIDDLEWARE_LOG_CSV_PATH, rawOpts.middlewareLogCsvPath, legacy.logging?.csvPath, process.env.AIWAF_MIDDLEWARE_LOG_CSV_PATH, 'logs/aiwaf-requests.csv'),

    AIWAF_MIN_AI_LOGS: toNumber(firstDefined(rawOpts.AIWAF_MIN_AI_LOGS, rawOpts.minAiLogs, legacy.training?.minAiLogs, process.env.AIWAF_MIN_AI_LOGS), 0),
    AIWAF_ENABLE_KEYWORD_LEARNING: toBool(firstDefined(rawOpts.AIWAF_ENABLE_KEYWORD_LEARNING, rawOpts.enableKeywordLearning, legacy.keywords?.enableLearning, process.env.AIWAF_ENABLE_KEYWORD_LEARNING), true),

    AIWAF_MIDDLEWARES: toOptionalArray(firstDefined(rawOpts.AIWAF_MIDDLEWARES, rawOpts.middlewares, legacy.middlewares?.enabled, process.env.AIWAF_MIDDLEWARES)),
    AIWAF_DISABLE_MIDDLEWARES: toArray(firstDefined(rawOpts.AIWAF_DISABLE_MIDDLEWARES, rawOpts.disableMiddlewares, rawOpts.disabledMiddlewares, legacy.middlewares?.disabled, process.env.AIWAF_DISABLE_MIDDLEWARES), []),
    AIWAF_PATH_RULES: toPathRules(firstDefined(rawOpts.AIWAF_PATH_RULES, rawOpts.pathRules, legacy.pathRules, process.env.AIWAF_PATH_RULES), []),
    AIWAF_PATH_MANIFEST: firstDefined(rawOpts.AIWAF_PATH_MANIFEST, rawOpts.pathManifest, legacy.pathManifest, process.env.AIWAF_PATH_MANIFEST, '.aiwaf/paths.json'),
    AIWAF_ROUTE_PLAN_VERSION: firstDefined(rawOpts.AIWAF_ROUTE_PLAN_VERSION, rawOpts.routePlanVersion, legacy.routePlanVersion, process.env.AIWAF_ROUTE_PLAN_VERSION, 0),
    AIWAF_ACCESS_LOG: firstDefined(rawOpts.AIWAF_ACCESS_LOG, rawOpts.accessLog, process.env.AIWAF_ACCESS_LOG, process.env.NODE_LOG_PATH),

    AIWAF_FORCE_JSON_ERRORS: toBool(firstDefined(rawOpts.AIWAF_FORCE_JSON_ERRORS, rawOpts.forceJsonErrors, legacy.errors?.forceJson), true)
  };

  return normalized;
}

module.exports = { normalizeSettings };
