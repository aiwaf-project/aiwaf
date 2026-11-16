'use strict';

// Builds native pipeline options from the options the regular aiwaf(options) middleware takes.
// Options are read through the same normalizeSettings(), with the same defaults and environment
// variables, and every option that doesn't carry over exactly is reported with the reason.
const { normalizeSettings } = require('./settingsCompat');
const { planEnabledMiddlewares, isAutoSelection } = require('./middlewarePlan');
const { normalizeMiddlewareName } = require('./pathRules');
// Loaded when first used, since nativePipeline loads this module's callers.
const regularStores = settings => require('./nativePipeline').regularStores(settings);

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

// The sample model shipped in resources/model.json. Its old format can't be loaded natively,
// and it's only a sample, so the native pipeline runs without it, as a new install does.
const SHIPPED_MODEL_SHA256 = '56254f9bdeba0f96bed3c72ce88ed9f1e5ad552d7510c84742fd6a349e72c091';
const SHIPPED_MODEL = path.join(__dirname, '..', 'resources', 'model.json');

/** What the regular middleware would load as its anomaly model, read without loading it. */
/** Path rules whose rate limits the native engine can apply; the others are reported. */
function nativePathRules(rules, note) {
  if (rules == null) return [];
  if (!Array.isArray(rules)) {
    note('AIWAF_PATH_RULES', 'not carried', 'It isn\'t a list of rules.');
    return [];
  }
  const { rateOverrides } = require('./nativePipeline');
  return rules.flatMap(rule => {
    if (!rule || typeof rule !== 'object' || Array.isArray(rule)) {
      note('AIWAF_PATH_RULES', 'not carried', `A path rule must be an object, not ${JSON.stringify(rule)}.`);
      return [];
    }
    let reason = null;
    try {
      const limits = rateOverrides(rule.RATE_LIMIT || rule.rate_limit || {});
      if ((limits.window !== undefined && !(limits.window > 0)) || (limits.max !== undefined && !(limits.max > 0))
          || (limits.flood !== undefined && !(limits.flood > 0))) reason = 'the native pipeline needs a positive window and limits';
    } catch (error) { reason = error.message; }
    if (!reason) return [rule];
    note('AIWAF_PATH_RULES', 'not carried', `The RATE_LIMIT for ${JSON.stringify(rule.PREFIX ?? rule.prefix)} isn't used natively: ${reason}.`);
    const { RATE_LIMIT: _a, rate_limit: _b, ...rest } = rule;
    return [rest];
  });
}

function settingsModel(opts) {
  if ((opts.AIWAF_MODEL_STORAGE || 'file') !== 'file') return { kind: 'storage' };
  const file = opts.AIWAF_MODEL_PATH || SHIPPED_MODEL;
  let text;
  try { text = fs.readFileSync(file); } catch (_) { return { kind: 'none', file }; }
  if (crypto.createHash('sha256').update(text).digest('hex') === SHIPPED_MODEL_SHA256) return { kind: 'shipped', file };
  try {
    const data = JSON.parse(text);
    return { kind: data?.model_type === 'aiwaf_wasm.IsolationForest' && data.model_state ? 'wasm' : 'js', file };
  } catch (_) { return { kind: 'unreadable', file }; }
}

const STAGES = ['geo_block', 'ip_keyword_block', 'rate_limit', 'ai_anomaly', 'honeypot', 'uuid_tamper', 'header_validation', 'logging'];
const PATTERN = /[*?[]/;
// Options with no native equivalent, reported when they differ from the regular defaults.
const NOT_CARRIED = {
  AIWAF_ALLOWED_PATH_KEYWORDS: 'The native pipeline has its own legitimate_keywords list.',
  AIWAF_EXEMPT_KEYWORDS: 'The native pipeline has no exempt keywords.',
  AIWAF_LOGIN_PATH_PREFIXES: 'The native pipeline treats paths under /login and /admin/login as login forms.',
  AIWAF_POST_ONLY_SUFFIXES: 'The native method policy is allowed_methods only.',
  AIWAF_MAX_HEADER_BYTES: 'The native header limits are fixed.',
  AIWAF_MAX_HEADER_COUNT: 'The native header limits are fixed.',
  AIWAF_MAX_USER_AGENT_LENGTH: 'The native header limits are fixed.',
  AIWAF_MAX_ACCEPT_LENGTH: 'The native header limits are fixed.',
  AIWAF_BLOCKED_USER_AGENTS: 'The native header check has its own list of scanner user agents.',
  AIWAF_GEO_MMDB_PATH: 'Pass a countryLookup(req) option to the native pipeline instead.',
  AIWAF_GEO_CACHE_SECONDS: 'Country lookups come from countryLookup(req).',
  AIWAF_GEO_CACHE_PREFIX: 'Country lookups come from countryLookup(req).',
  AIWAF_MODEL_STORAGE: 'The native pipeline loads a model from a file only; pass a model option instead.',
  AIWAF_MODEL_STORAGE_FALLBACK: 'The native pipeline loads a model from a file only; pass a model option instead.',
  AIWAF_MODEL_CACHE_KEY: 'Pass a model option to the native pipeline instead.',
  AIWAF_MODEL_CACHE_TTL: 'Pass a model option to the native pipeline instead.',
  AIWAF_MIDDLEWARE_LOG_PATH: 'Native logs go to logPath, by default native_requests.csv in dataDir.',
  AIWAF_MIDDLEWARE_LOG_DB: 'The native pipeline logs to a CSV file only.',
  AIWAF_MIDDLEWARE_LOG_CSV: 'The native pipeline logs to a CSV file only.',
  AIWAF_MIDDLEWARE_LOG_CSV_PATH: 'Native logs go to logPath, by default native_requests.csv in dataDir.',
  AIWAF_CAPTURE_EXTENDED_REQUEST_INFO: 'The native log has fixed columns.',
  AIWAF_EXTENDED_INFO_MAX_HEADERS: 'The native log has fixed columns.',
  AIWAF_EXTENDED_INFO_MAX_VALUE_LENGTH: 'The native log has fixed columns.',
  AIWAF_DYNAMIC_KEYWORDS_DB: 'The native pipeline reads learned keywords from the keyword store.',
  AIWAF_MIN_AI_LOGS: 'Training settings don\'t apply to request handling.',
  AIWAF_ROUTE_PLAN_VERSION: 'The native pipeline compiles path rules once at startup.',
  AIWAF_FORCE_JSON_ERRORS: 'Native responses are always JSON.'
};

// Options that carry over exactly, reported as carried when they differ from the regular defaults.
const CARRIED = ['WINDOW_SEC', 'MAX_REQ', 'FLOOD_REQ', 'AIWAF_MIN_FORM_TIME', 'AIWAF_MAX_PAGE_TIME', 'HONEYPOT_FIELD',
  'AIWAF_METHOD_POLICY_ENABLED', 'AIWAF_ALLOWED_METHODS', 'staticKeywords', 'AIWAF_ENABLE_KEYWORD_LEARNING',
  'AIWAF_GEO_BLOCK_ENABLED', 'AIWAF_GEO_ALLOW_COUNTRIES', 'AIWAF_GEO_BLOCK_COUNTRIES', 'AIWAF_HEADER_VALIDATION',
  'AIWAF_REQUIRED_HEADERS', 'AIWAF_HEADER_QUALITY_MIN_SCORE', 'AIWAF_EXEMPT_PATHS', 'AIWAF_EXEMPT_IPS',
  'AIWAF_MIDDLEWARES', 'AIWAF_DISABLE_MIDDLEWARES', 'AIWAF_MIDDLEWARE_LOGGING', 'AIWAF_PATH_RULES',
  'AIWAF_PATH_MANIFEST', 'AIWAF_INJECTION_RISK_MODE', 'dynamicTopN', 'AIWAF_EXEMPTIONS_DB', 'AIWAF_TRUSTED_PROXIES'];
// Where settingsCompat reads each number from, in the same order, so unreadable values can be reported.
const NUMBER_SOURCES = {
  WINDOW_SEC: ['WINDOW_SEC', 'windowSec', 'AIWAF_WINDOW_SECONDS', 'env:AIWAF_RATE_WINDOW'],
  MAX_REQ: ['MAX_REQ', 'maxReq', 'env:AIWAF_RATE_MAX'],
  FLOOD_REQ: ['FLOOD_REQ', 'floodReq', 'env:AIWAF_RATE_FLOOD'],
  AIWAF_MIN_FORM_TIME: ['AIWAF_MIN_FORM_TIME', 'minFormTime', 'env:AIWAF_MIN_FORM_TIME'],
  AIWAF_MAX_PAGE_TIME: ['AIWAF_MAX_PAGE_TIME', 'maxPageTime', 'env:AIWAF_MAX_PAGE_TIME'],
  AIWAF_HEADER_QUALITY_MIN_SCORE: ['AIWAF_HEADER_QUALITY_MIN_SCORE', 'headerQualityMinScore', 'env:AIWAF_HEADER_QUALITY_MIN_SCORE']
};

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const headerName = name => (String(name).toUpperCase().startsWith('HTTP_')
  ? String(name).toUpperCase() : `HTTP_${String(name).toUpperCase().replace(/-/g, '_')}`);

function nativeOptionsFromSettings(rawOpts = {}) {
  const opts = normalizeSettings(rawOpts);
  const defaults = normalizeSettings({});
  const notes = [];
  const note = (setting, outcome, detail = '') => notes.push({ setting, outcome, detail });
  const profile = {};

  // Regular defaults differ from the native ones, so effective values are always written.
  profile.rate_window = opts.WINDOW_SEC;
  profile.rate_max = opts.MAX_REQ;
  profile.rate_flood = opts.FLOOD_REQ;
  // Values the native engine can't take are reported and left at its defaults. A count over a
  // fractional limit is the same as over its whole part, as the regular comparisons work.
  const unusable = (key, setting, reason) => {
    delete profile[key];
    note(setting, 'not carried', `The native pipeline needs ${reason}, so its default applies.`);
  };
  if (!(Number(profile.rate_window) > 0)) unusable('rate_window', 'WINDOW_SEC', 'a positive window');
  for (const [key, setting] of [['rate_max', 'MAX_REQ'], ['rate_flood', 'FLOOD_REQ']]) {
    if (Number.isFinite(Number(profile[key])) && Math.floor(profile[key]) >= 1) profile[key] = Math.floor(profile[key]);
    else unusable(key, setting, 'a limit of at least 1');
  }
  if (profile.rate_max !== undefined && profile.rate_flood !== undefined && profile.rate_flood < profile.rate_max) {
    profile.rate_max = profile.rate_flood;
    note('MAX_REQ', 'carried', `It's above FLOOD_REQ (${profile.rate_flood}), so requests are flood-blocked first. The native limit is ${profile.rate_flood}, which blocks at the same point.`);
  }
  if (profile.rate_flood > 100000) {
    profile.rate_flood = 100000;
    note('FLOOD_REQ', 'changed', 'The native flood threshold is at most 100000.');
  }
  // A negative minimum form time never blocks, as 0 doesn't.
  profile.min_form_time = Number.isFinite(Number(opts.AIWAF_MIN_FORM_TIME)) ? Math.max(0, Number(opts.AIWAF_MIN_FORM_TIME)) : 1;
  if (opts.AIWAF_MAX_PAGE_TIME > 0) profile.max_page_time = opts.AIWAF_MAX_PAGE_TIME;
  else note('AIWAF_MAX_PAGE_TIME', 'not carried', 'The native page expiry must be positive, so its default applies.');
  profile.honeypot_field = opts.HONEYPOT_FIELD ? String(opts.HONEYPOT_FIELD) : '';
  profile.allowed_methods = opts.AIWAF_METHOD_POLICY_ENABLED === true
    ? (opts.AIWAF_ALLOWED_METHODS || []).map(method => String(method).toUpperCase()) : [];
  const list = (value, setting) => {
    if (value == null) return [];
    if (Array.isArray(value)) return value.map(String);
    if (typeof value === 'string') return value.split(',').map(item => item.trim()).filter(Boolean);
    note(setting, 'not carried', 'It isn\'t a list, so the native default applies.');
    return null;
  };
  const keywords = list(opts.staticKeywords, 'staticKeywords');
  if (keywords) profile.static_keywords = keywords;
  profile.learning = opts.AIWAF_ENABLE_KEYWORD_LEARNING !== false;
  for (const [key, setting] of [['geo_allow', 'AIWAF_GEO_ALLOW_COUNTRIES'], ['geo_block', 'AIWAF_GEO_BLOCK_COUNTRIES']]) {
    const countries = list(opts[setting], setting);
    if (countries) profile[key] = countries;
  }
  const required = opts.AIWAF_REQUIRED_HEADERS || [];
  if (required && typeof required === 'object' && !Array.isArray(required)) {
    profile.required_headers = (required.DEFAULT || []).map(headerName);
    note('AIWAF_REQUIRED_HEADERS', 'changed', 'Per-method lists aren\'t supported natively, so its DEFAULT list applies to every method.');
  } else {
    profile.required_headers = [].concat(required).map(headerName);
  }
  // With no required headers, the regular check also skips the quality score.
  // Scores are whole numbers, so a fractional minimum is the next whole one.
  const minScore = Number(opts.AIWAF_HEADER_QUALITY_MIN_SCORE);
  if (!profile.required_headers.length) profile.header_min_score = 0;
  else if (Number.isFinite(minScore) && Math.abs(minScore) < 2 ** 31) profile.header_min_score = Math.ceil(minScore);
  else note('AIWAF_HEADER_QUALITY_MIN_SCORE', 'not carried', 'It isn\'t a usable score, so the native default applies.');
  profile.sql_mode = String(rawOpts.AIWAF_SQL_INJECTION_MODE ?? process.env.AIWAF_SQL_INJECTION_MODE ?? 'block').trim().toLowerCase();
  if (!['block', 'monitor', 'off'].includes(profile.sql_mode)) unusable('sql_mode', 'AIWAF_SQL_INJECTION_MODE', 'block, monitor or off');
  profile.max_bytes = Number(rawOpts.AIWAF_PAYLOAD_MAX_BYTES ?? process.env.AIWAF_PAYLOAD_MAX_BYTES ?? 65536);
  if (!(Number.isInteger(profile.max_bytes) && profile.max_bytes >= 1 && profile.max_bytes <= 1048576)) {
    unusable('max_bytes', 'AIWAF_PAYLOAD_MAX_BYTES', 'a whole number of bytes from 1 to 1048576');
  }

  const paths = (opts.AIWAF_EXEMPT_PATHS || []).map(String);
  profile.exempt_paths = paths.filter(path => !PATTERN.test(path));
  const pathPatterns = paths.filter(path => PATTERN.test(path));
  if (pathPatterns.length) {
    note('AIWAF_EXEMPT_PATHS', 'changed', `The native pipeline matches literal prefixes only, so these patterns aren't exempt: ${pathPatterns.join(', ')}.`);
  }
  const ips = (opts.AIWAF_EXEMPT_IPS || []).map(String);
  profile.exempt_ips = ips.filter(ip => !ip.includes('/') && !ip.includes('*'));
  const ranges = ips.filter(ip => ip.includes('/') || ip.includes('*'));
  if (ranges.length) {
    note('AIWAF_EXEMPT_IPS', 'changed', `The native pipeline exempts exact addresses only, so these ranges aren't exempt: ${ranges.join(', ')}.`);
  }

  // The same stage plan as the regular middleware, then its per-stage switches.
  const requested = Array.isArray(opts.AIWAF_MIDDLEWARES) ? opts.AIWAF_MIDDLEWARES.map(normalizeMiddlewareName) : opts.AIWAF_MIDDLEWARES;
  const enabled = planEnabledMiddlewares({
    orderedAvailable: STAGES, requested, disabled: (opts.AIWAF_DISABLE_MIDDLEWARES || []).map(normalizeMiddlewareName),
    accessLog: opts.AIWAF_ACCESS_LOG, geoEnabledFlag: !!opts.AIWAF_GEO_BLOCK_ENABLED,
    staticBlockCountries: opts.AIWAF_GEO_BLOCK_COUNTRIES, dynamicBlockCountries: []
  });
  const auto = isAutoSelection(requested);
  const explicit = key => Object.hasOwn(rawOpts, key) || process.env[key] !== undefined;
  const logging = opts.AIWAF_MIDDLEWARE_LOGGING || (auto && enabled.has('logging') && !explicit('AIWAF_MIDDLEWARE_LOGGING'));
  if (!opts.AIWAF_HEADER_VALIDATION) enabled.delete('header_validation');
  if (!logging) enabled.delete('logging');
  profile.enabled = STAGES.filter(stage => enabled.has(stage));
  if (enabled.has('logging')) {
    note('AIWAF_MIDDLEWARE_LOGGING', 'changed', 'Requests are logged to native_requests.csv in dataDir, with the native log\'s columns.');
  }
  // The regular middleware's stores: the same blocks, exemptions, learned keywords and blocked
  // countries. An older aiwaf-wasm can't use them, and then the adapter keeps its own in dataDir.
  note('stores', 'carried');

  const options = {
    stores: regularStores({ dynamicTopN: opts.dynamicTopN, exemptionsDb: opts.AIWAF_EXEMPTIONS_DB }),
    storesFallback: 'csv',
    rateFailureMode: opts.AIWAF_RATE_CACHE_FAILURE_MODE,
    // Resolved as the regular middleware resolves it, with its default trusted proxies.
    trustedProxies: opts.AIWAF_TRUSTED_PROXIES ?? 'default',
    pathRules: nativePathRules(opts.AIWAF_PATH_RULES, note),
    manifestPath: opts.AIWAF_PATH_MANIFEST,
    injectionRiskMode: opts.AIWAF_INJECTION_RISK_MODE
  };
  if (options.injectionRiskMode != null) {
    const mode = String(options.injectionRiskMode).trim().toLowerCase();
    if (['block', 'log', 'off'].includes(mode)) options.injectionRiskMode = mode;
    else {
      delete options.injectionRiskMode;
      note('AIWAF_INJECTION_RISK_MODE', 'not carried', 'It must be block, log or off, so the native default applies.');
    }
  }
  // The regular UUID check matches one segment under uuidRoutePrefix, and asks uuidResolver.
  const uuidPrefix = rawOpts.uuidRoutePrefix || '/user';
  profile.uuid_prefix = `${String(uuidPrefix).replace(/\/+$/, '')}/`;
  if (rawOpts.uuidRoutePrefix) note('uuidRoutePrefix', 'carried');
  if (typeof rawOpts.uuidResolver === 'function') {
    options.uuidResolver = rawOpts.uuidResolver;
    note('uuidResolver', 'carried');
  }
  // The regular rate limiter counts in its cache option. A Redis client there is shared the
  // same way natively; an older aiwaf-wasm counts in each worker instead, and warns.
  const cache = rawOpts.cache;
  if (cache && (typeof cache.call === 'function' || typeof cache.sendCommand === 'function')) {
    options.rateCounter = require('./nativeRateCounter').redisRateCounter(cache);
    options.rateCounterFallback = 'memory';
    note('cache', 'carried');
  } else if (cache) {
    note('cache', 'not carried', 'Shared native rate limits need a node-redis or ioredis client, so rate limits are counted in each worker.');
  }
  if (opts.AIWAF_RATE_CACHE_FAILURE_MODE !== 'closed') note('AIWAF_RATE_CACHE_FAILURE_MODE', 'carried');
  const model = settingsModel(opts);
  if (model.kind === 'wasm') {
    options.modelFile = model.file;
    note('AIWAF_MODEL_PATH', 'carried');
  } else if (model.kind === 'js' || model.kind === 'unreadable') {
    note('AIWAF_MODEL_PATH', 'not carried', `${model.file} is in an older model format the native pipeline can't load. Retrain with: aiwaf train`);
  } else if (model.kind === 'none' && opts.AIWAF_MODEL_PATH) {
    note('AIWAF_MODEL_PATH', 'carried', 'No model file exists there yet, so anomaly detection runs without a model, as in the regular middleware.');
  }
  for (const [key, reason] of Object.entries(NOT_CARRIED)) {
    if (!same(opts[key], defaults[key])) note(key, 'not carried', reason);
  }
  // A number that couldn't be read falls back to the regular default, which is what's converted.
  for (const [key, sources] of Object.entries(NUMBER_SOURCES)) {
    const source = sources.find(name => (name.startsWith('env:') ? process.env[name.slice(4)] : rawOpts[name]) != null
      && (name.startsWith('env:') ? process.env[name.slice(4)] : rawOpts[name]) !== '');
    if (!source) continue;
    const raw = source.startsWith('env:') ? process.env[source.slice(4)] : rawOpts[source];
    if (!Number.isFinite(Number(raw))) {
      note(key, 'not carried', `Its value couldn't be read (expected a number), so the regular default of ${defaults[key]} applies, as in the regular middleware.`);
    }
  }
  const noted = new Set(notes.map(n => n.setting));
  for (const key of CARRIED) {
    if (!noted.has(key) && !same(opts[key], defaults[key])) note(key, 'carried');
  }
  for (const key of ['AIWAF_SQL_INJECTION_MODE', 'AIWAF_PAYLOAD_MAX_BYTES']) {
    if (rawOpts[key] !== undefined || process.env[key] !== undefined) note(key, 'carried');
  }
  const differences = () => notes.filter(n => n.outcome !== 'carried');
  const report = () => {
    const carried = notes.filter(n => n.outcome === 'carried').length;
    return [`Native pipeline settings: ${carried} carried over unchanged.`,
      ...differences().map(n => `- ${n.setting}: ${n.outcome}. ${n.detail}`.trimEnd())].join('\n');
  };
  return {
    // Ready for aiwaf.native(); profile and options are also available separately.
    config: { ...options, profile },
    profile,
    options,
    notes,
    differences,
    report,
    /** Logs the report: a warning when anything differs, otherwise one info line, as in Python. */
    log(logger = console) {
      const write = differences().length ? logger.warn : (logger.info || logger.log);
      write?.call(logger, report());
    }
  };
}

module.exports = { nativeOptionsFromSettings, settingsModel, SHIPPED_MODEL_SHA256 };
