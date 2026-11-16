// Explicit WASM-owned request engine. Node owns transport and immediate filesystem I/O.
const fs = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');
const { createRequestPipeline } = require('./wasmAdapter');
const wasmChecks = require('./wasmChecks');
const routeRisks = require('./routeRisks');
const { ALL_MIDDLEWARES, createRoutePlan } = require('./pathRules');
const { getEffectivePathRules } = require('./pathManifest');
const { getRequestIp } = require('./runtimeUtils');

// RATE_LIMIT names accepted in path rules, as in the regular rate limiter.
const RATE_KEYS = {
  window: ['WINDOW_SEC', 'WINDOW', 'window_seconds', 'windowSeconds'],
  max: ['MAX_REQ', 'MAX', 'max_requests', 'maxRequests'],
  flood: ['FLOOD_REQ', 'FLOOD', 'flood_threshold', 'floodThreshold']
};
function rateOverrides(rule = {}) {
  const limits = {};
  for (const [name, keys] of Object.entries(RATE_KEYS)) {
    const key = keys.find(k => rule[k] !== undefined && rule[k] !== null);
    if (key === undefined) continue;
    const value = Number(rule[key]);
    if (!Number.isFinite(value) || (name !== 'window' && !Number.isInteger(value))) throw new Error(`RATE_LIMIT ${key} must be a number`);
    limits[name] = value;
  }
  return limits;
}

const LOG_HEADER = 'timestamp,ip,method,path,status,response_time_ms,blocked,reason,country,user_agent\n';
const BLOCK_HEADER = 'ip,expires_at,reason\n';
const KEYWORD_HEADER = 'keyword\n';
const DEFER_BODY = Symbol('aiwaf.native.defer-body');
function writer(file, header, durable) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  if (fs.existsSync(file) && fs.statSync(file).size) {
    const fd = fs.openSync(file, 'r');
    try {
      const bytes = Buffer.alloc(Buffer.byteLength(header));
      fs.readSync(fd, bytes, 0, bytes.length, 0);
      if (bytes.toString() !== header) throw new Error(`Incompatible native CSV schema: ${file}`);
    } finally { fs.closeSync(fd); }
  }
  const fd = fs.openSync(file, 'a');
  const append = row => {
    const buffer = Buffer.from(row);
    let offset = 0;
    while (offset < buffer.length) {
      const written = fs.writeSync(fd, buffer, offset, buffer.length - offset);
      if (!written) throw new Error('Native CSV append made no progress');
      offset += written;
    }
    if (durable) fs.fsyncSync(fd);
  };
  try { if (!fs.statSync(file).size) append(header); } catch (error) { fs.closeSync(fd); throw error; }
  return { append, close() { fs.closeSync(fd); } };
}
// Startup only: quoted newlines and commas must round-trip persisted native CSV.
function readRows(file, expected) {
  const text = fs.readFileSync(file, 'utf8');
  const rows = []; let row = [], value = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '"') {
      if (quoted && text[i + 1] === '"') { value += '"'; i++; }
      else quoted = !quoted;
    } else if (!quoted && (ch === ',' || ch === '\n')) {
      row.push(value); value = '';
      if (ch === '\n') { rows.push(row); row = []; }
    } else value += ch;
  }
  if (quoted || row.length || value) throw new Error('Incomplete native policy CSV');
  if (rows.shift()?.join(',') !== expected.join(',')) throw new Error('Invalid native policy schema');
  return rows.map(cells => {
    if (cells.length !== expected.length) throw new Error('Invalid native policy CSV row');
    return Object.fromEntries(expected.map((key, index) => [key, cells[index]]));
  });
}
// The content types whose bodies are inspected, as in Python. Uploads and other bodies
// (multipart, binary) pass through unread, as in the regular middleware.
function inspectable(req) {
  const media = String(req.headers?.['content-type'] || '').split(';', 1)[0].trim().toLowerCase();
  return ['application/json', 'application/x-www-form-urlencoded', 'application/xml', 'text/xml'].includes(media)
    || media.endsWith('+json') || media.endsWith('+xml');
}
function declaredBody(req) {
  return Number(req.headers?.['content-length'] || 0) > 0 || !!req.headers?.['transfer-encoding'];
}
// An unparsed inspectable body this adapter can read itself: identity-encoded and unread.
function readable(req) {
  const encoding = String(req.headers?.['content-encoding'] || 'identity').trim().toLowerCase();
  return req.body === undefined && declaredBody(req) && inspectable(req) && encoding === 'identity'
    && typeof req.read === 'function' && typeof req.unshift === 'function' && !req.readableEnded;
}
const PEEKED = Symbol('aiwaf.native.peeked-body');
// Reads at most limit + 1 bytes of the body, then puts them back on the stream, so body
// parsers after this adapter still read the whole body. Python's adapters do the same.
function peekBody(req, limit) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    const expected = Number(req.headers?.['content-length']) || null;
    const finish = () => {
      req.removeListener('readable', onReadable);
      req.removeListener('error', onError);
      const prefix = Buffer.concat(chunks);
      if (prefix.length) req.unshift(prefix);
      resolve(prefix);
    };
    const onError = error => { req.removeListener('readable', onReadable); reject(error); };
    function onReadable() {
      let chunk;
      while ((chunk = req.read()) !== null) {
        chunks.push(chunk);
        size += chunk.length;
        if (size > limit || size === expected) return finish();
      }
      // The whole body is buffered: put it back before the stream emits 'end'.
      if (req._readableState?.ended) finish();
    }
    req.on('error', onError);
    req.on('readable', onReadable);
  });
}
function safeBody(req, limit) {
  const peeked = req[PEEKED];
  if (peeked) {
    if (peeked.length > limit) return { body: null, exhausted: true };
    const text = peeked.toString('utf8');
    // JSON is parsed here, as a body parser would, so it's checked exactly like a parsed body.
    if (/json/.test(String(req.headers?.['content-type']))) {
      try { return { body: JSON.parse(text), exhausted: false }; } catch (_) { /* checked as text */ }
    }
    return { body: text, exhausted: false };
  }
  const body = req.body;
  if (body === undefined || (body && typeof body.pipe === 'function')) {
    // An upload or other body that isn't inspected passes unread. An inspectable one this
    // adapter couldn't read (compressed, or already consumed) can't be checked.
    return { body: null, exhausted: declaredBody(req) && inspectable(req) };
  }
  try {
    const value = Buffer.isBuffer(body) ? body.toString('utf8') : body;
    const encoded = JSON.stringify(value);
    if (encoded === undefined || Buffer.byteLength(encoded) > limit * 8) return { body: null, exhausted: true };
    return { body: value, exhausted: false };
  } catch (_) { return { body: null, exhausted: true }; }
}
// Express trust proxy controls forwarded-IP handling. Never trust XFF directly.
function peerIp(req) {
  return String(req.ip || req.socket?.remoteAddress || 'unknown').replace(/^::ffff:/, '');
}
// A model saved by `aiwaf train`, as the regular middleware's file storage keeps it.
async function readModelFile(file) {
  const data = JSON.parse(await fs.promises.readFile(file, 'utf8'));
  if (data?.model_type !== 'aiwaf_wasm.IsolationForest' || !data.model_state) {
    throw new Error(`${file} isn't a model the native pipeline can use. Retrain with: aiwaf train`);
  }
  return require('./wasmAdapter').createIsolationForestFromJSON(data.model_state);
}
function headers(raw = {}) {
  return Object.fromEntries(Object.entries(raw).map(([key, value]) => [
    ['content-type', 'content-length'].includes(key.toLowerCase()) ? key.toUpperCase().replace(/-/g, '_') : `HTTP_${key.toUpperCase().replace(/-/g, '_')}`,
    Array.isArray(value) ? value.join(', ') : String(value ?? '')
  ]));
}
const STAGE_NAMES = ['geo_block', 'ip_keyword_block', 'rate_limit', 'ai_anomaly', 'honeypot', 'uuid_tamper', 'header_validation', 'logging'];
// Recent true/false answers per key, for `seconds`, keeping at most `limit` keys. A failed
// read isn't kept.
function recentAnswers(seconds, limit = 10000) {
  const items = new Map();
  const set = (key, value) => {
    if (!(seconds > 0)) return;
    items.delete(key);
    items.set(key, { until: performance.now() + seconds * 1000, value: !!value });
    while (items.size > limit) items.delete(items.keys().next().value);
  };
  return {
    items,
    set,
    async get(key, load) {
      const hit = items.get(key);
      if (hit && hit.until > performance.now()) {
        items.delete(key); items.set(key, hit);
        return hit.value;
      }
      const value = !!(await load());
      set(key, value);
      return value;
    }
  };
}
// The store methods a stores option provides. Each may return a promise.
const STORE_METHODS = ['isBlocked', 'isExempted', 'topKeywords', 'blockedCountries', 'block', 'learn'];
// Node's own blocklist, exemption, keyword and geo stores: a database when one is configured,
// otherwise their CSV files, as the regular middleware uses them.
function regularStores({ dynamicTopN = 10, exemptionsDb = true } = {}) {
  const blocklistManager = require('./blocklistManager');
  const exemptionStore = require('./exemptionStore');
  const dynamicKeywordStore = require('./dynamicKeywordStore');
  const geoStore = require('./geoStore');
  return {
    isBlocked: ip => blocklistManager.isBlocked(ip),
    async isExempted(ip, requestPath) {
      if (!exemptionsDb) return false;
      // Exemption storage failures fail open, as in the regular middleware.
      try {
        return !!(await exemptionStore.isIpExempt(String(ip).toLowerCase())
          || await exemptionStore.isPathExempt(String(requestPath).toLowerCase()));
      } catch (_) { return false; }
    },
    // A learned keyword applies once its count passes dynamicTopN, as in the regular middleware.
    topKeywords: async limit => (await dynamicKeywordStore.list(limit))
      .filter(row => Number(row.count || 0) > dynamicTopN).map(row => String(row.keyword)),
    blockedCountries: async () => (await geoStore.listBlockedCountries()).map(row => String(row.country_code)),
    block: (ip, reason) => blocklistManager.block(ip, reason),
    learn: keyword => dynamicKeywordStore.increment(keyword)
  };
}
function synchronous(callback, req, fallback) {
  const result = callback ? callback(req) : fallback;
  if (result && typeof result.then === 'function') throw new TypeError('Native fact callbacks must return synchronous values');
  return result;
}

module.exports = async function createNativeMiddleware(options = {}) {
  if (!options || typeof options !== 'object' || Array.isArray(options)) throw new TypeError('Native adapter options must be an object');
  if (options.profile !== undefined && (!options.profile || typeof options.profile !== 'object' || Array.isArray(options.profile))) throw new TypeError('Native profile must be an object');
  const allowed = new Set(['profile', 'dataDir', 'logPath', 'logFsync', 'persistBlocks', 'model', 'countryLookup', 'authenticated', 'routeExists', 'onError', 'pathRules', 'manifestPath', 'injectionRiskMode', 'logger', 'uuidResolver', 'stores', 'storeCacheSeconds', 'storesFallback',
    'rateCounter', 'rateCounterFallback', 'rateFailureMode', 'trustedProxies', 'modelFile']);
  for (const key of Object.keys(options)) if (!allowed.has(key)) throw new Error(`Unknown native adapter option: ${key}`);
  if (options.uuidResolver != null && typeof options.uuidResolver !== 'function') throw new TypeError('uuidResolver must be a function');
  for (const key of ['countryLookup','authenticated','routeExists','onError']) {
    if (options[key] != null && (typeof options[key] !== 'function' || options[key].constructor.name === 'AsyncFunction')) throw new TypeError(`${key} must be synchronous`);
  }
  if (options.pathRules != null && !Array.isArray(options.pathRules)) throw new TypeError('pathRules must be an array of rules');
  let stores = null;
  if (options.stores === 'regular') stores = regularStores();
  else if (options.stores != null) {
    if (typeof options.stores !== 'object') throw new TypeError('stores must be "regular" or an object with the store methods');
    const missing = STORE_METHODS.filter(name => typeof options.stores[name] !== 'function');
    if (missing.length) throw new TypeError(`stores is missing ${missing.join(', ')}`);
    stores = options.stores;
  }
  if (options.storesFallback != null && options.storesFallback !== 'csv') throw new TypeError('storesFallback must be "csv"');
  let rateCounter = options.rateCounter ?? null;
  if (rateCounter !== null && typeof rateCounter?.count !== 'function') throw new TypeError('rateCounter must have a count(key, now, windowSeconds) method');
  if (options.rateCounterFallback != null && options.rateCounterFallback !== 'memory') throw new TypeError('rateCounterFallback must be "memory"');
  const rateFailureMode = options.rateFailureMode ?? 'closed';
  if (!['closed', 'open'].includes(rateFailureMode)) throw new TypeError('rateFailureMode must be "closed" or "open"');
  // With trustedProxies, the client address is resolved as the regular middleware resolves
  // it (AIWAF_TRUSTED_PROXIES); without it, from req.ip, which Express trust proxy controls.
  const clientIp = options.trustedProxies === undefined ? peerIp
    : req => getRequestIp(req, options.trustedProxies).replace(/^::ffff:/, '');
  if (options.modelFile != null && (options.model != null || typeof options.modelFile !== 'string')) {
    throw new TypeError('modelFile must be a path, and can\'t be combined with model');
  }
  const storeCacheSeconds = Number(options.storeCacheSeconds ?? 5);
  if (!(storeCacheSeconds >= 0)) throw new TypeError('storeCacheSeconds must be a non-negative number');
  const profile = JSON.parse(JSON.stringify(options.profile || {}));
  const manifestPath = options.manifestPath ?? process.env.AIWAF_PATH_MANIFEST;
  const rules = getEffectivePathRules(options.pathRules || [], manifestPath ? { manifestPath } : undefined);
  // Checked once, so a bad rule fails at startup rather than on a request.
  const profileRate = { window: profile.rate_window ?? 10, max: profile.rate_max ?? 20, flood: profile.rate_flood ?? 40 };
  for (const rule of rules) {
    const limits = { ...profileRate, ...rateOverrides(rule.RATE_LIMIT || rule.rate_limit || {}) };
    if (!(limits.window > 0) || !(limits.max > 0) || limits.flood < limits.max || limits.flood > 100000) {
      throw new Error(`Invalid RATE_LIMIT for path rule ${JSON.stringify(rule.PREFIX ?? rule.prefix)}`);
    }
  }
  const routePlan = req => {
    const meta = req.aiwafRoute || {};
    const plan = createRoutePlan(String(req.path || req.url || '/').split('?')[0], rules, meta);
    return { plan, disabled: ALL_MIDDLEWARES.filter(name => !plan.shouldApply(name)), rate: rateOverrides(plan.getOverrides('RATE_LIMIT')) };
  };
  if (!Object.hasOwn(profile, 'blocklist') && Object.hasOwn(profile, 'blacklist')) profile.blocklist = profile.blacklist;
  delete profile.blacklist;
  const dataDir = path.resolve(options.dataDir || 'aiwaf_native_data');
  // State files are dedicated to this adapter. Legacy ORM/CSV schemas are not mixed.
  const blockPath = path.join(dataDir, 'native_blocks.csv');
  const keywordPath = path.join(dataDir, 'native_keywords.csv');
  if (rateCounter) {
    const probe = await createRequestPipeline({});
    const supported = typeof probe.native.supports === 'function' && probe.native.supports('rate_count');
    probe.native.free();
    if (!supported && options.rateCounterFallback === 'memory') {
      (options.logger || console).warn?.('AIWAF native: this aiwaf-wasm can\'t use a shared rate counter, so rate limits are counted in each worker. Run: npm install aiwaf-wasm@latest');
      rateCounter = null;
    } else if (!supported) throw new Error('rateCounter needs a newer aiwaf-wasm. Run: npm install aiwaf-wasm@latest');
  }
  if (stores) {
    // Checked before the profile is built, since older engines reject policy_store.
    const probe = await createRequestPipeline({});
    const supported = typeof probe.native.supports === 'function' && probe.native.supports('host_store');
    probe.native.free();
    if (!supported && options.storesFallback === 'csv') {
      (options.logger || console).warn?.('AIWAF native: this aiwaf-wasm can\'t use stores, so blocks and learned keywords are kept in dataDir. Run: npm install aiwaf-wasm@latest');
      stores = null;
    } else if (!supported) throw new Error('stores needs a newer aiwaf-wasm. Run: npm install aiwaf-wasm@latest');
  }
  // With stores, blocks and learned keywords live in the stores instead of dataDir.
  const persist = !stores && options.persistBlocks !== false;
  if (stores) profile.policy_store = 'host';
  const opened = [];
  let log, blocks, keywords, engine;
  try {
    if (persist) {
      blocks = writer(blockPath, BLOCK_HEADER, true); opened.push(blocks);
      keywords = writer(keywordPath, KEYWORD_HEADER, true); opened.push(keywords);
      const blocklist = { ...(profile.blocklist || {}) };
      for (const row of readRows(blockPath, ['ip','expires_at','reason'])) {
        if (!row.ip || !row.expires_at || !Number.isFinite(Number(row.expires_at))) throw new Error('Invalid persisted block row');
        blocklist[row.ip] = Number(row.expires_at);
      }
      profile.blocklist = blocklist;
      profile.dynamic_keywords = [...new Set([...(profile.dynamic_keywords || []), ...readRows(keywordPath, ['keyword']).map(r => r.keyword)])];
    }
    let model = options.model;
    if (options.modelFile != null) model = await readModelFile(options.modelFile);
    try {
      engine = await createRequestPipeline(profile, model);
    } catch (error) {
      // Older aiwaf-wasm builds reject monitor mode as an unsupported policy.
      if (profile.sql_mode === 'monitor' && /Unsupported native policy/.test(String(error?.message ?? error))) {
        throw new Error('sql_mode "monitor" needs a newer aiwaf-wasm. Run: npm install aiwaf-wasm@latest');
      }
      throw error;
    }
    if (!profile.enabled || profile.enabled.includes('logging')) {
      log = writer(path.resolve(options.logPath || path.join(dataDir, 'native_requests.csv')), LOG_HEADER, !!options.logFsync); opened.push(log);
    }
  } catch (error) { if (engine) engine.native.free(); for (const entry of opened) entry.close(); throw error; }
  const { native, setModel } = engine;
  const logger = options.logger || console;
  // Monitored SQL findings are logged as the regular middleware logs them, and never block.
  const monitored = finding => {
    if (finding?.mode !== 'monitor') return finding;
    logger.warn?.({ event: 'aiwaf.payload', rule: finding.rule, mode: 'monitor' });
    return null;
  };
  if (options.uuidResolver && !(typeof native.supports === 'function' && native.supports('uuid_resolver'))) {
    native.free(); for (const entry of opened) entry.close();
    throw new Error('uuidResolver needs a newer aiwaf-wasm. Run: npm install aiwaf-wasm@latest');
  }
  const uuidPrefix = String(profile.uuid_prefix ?? '/user/').toLowerCase();
  const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  // The UUID the engine checks, as it finds it: one path segment under uuid_prefix.
  const pathUuid = rawPath => {
    if (!rawPath.toLowerCase().startsWith(uuidPrefix)) return null;
    const uid = rawPath.slice(uuidPrefix.length).replace(/\/+$/, '');
    return !uid.includes('/') && UUID.test(uid) ? uid : null;
  };
  const needsRateOverrides = rules.some(rule => Object.keys(rateOverrides(rule.RATE_LIMIT || rule.rate_limit || {})).length);
  if (needsRateOverrides && !(typeof native.supports === 'function' && native.supports('rate_overrides'))) {
    native.free(); for (const entry of opened) entry.close();
    throw new Error('Path-rule RATE_LIMIT needs a newer aiwaf-wasm. Run: npm install aiwaf-wasm@latest');
  }
  const maxBytes = profile.max_bytes ?? 65536;
  let closed = false;
  const contextKey = Symbol('aiwaf.native.context');
  const pending = new Set();
  // Keywords and countries change rarely, so they're read at most every storeCacheSeconds.
  const cache = new Map();
  const cached = (name, load) => {
    const now = performance.now();
    const entry = cache.get(name);
    if (entry && now < entry.until) return entry.value;
    const value = Promise.resolve().then(load).then(items => [...(items || [])].map(String));
    cache.set(name, { until: now + storeCacheSeconds * 1000, value });
    // A failed read isn't kept, so the next request tries again.
    value.catch(() => { if (cache.get(name)?.value === value) cache.delete(name); });
    return value;
  };
  // Blocklist and exemption answers, per IP and per IP and path, for storeCacheSeconds. A block
  // or exemption made elsewhere can take that long to apply; this worker's own blocks apply at once.
  const blockedIps = recentAnswers(storeCacheSeconds);
  const exempted = recentAnswers(storeCacheSeconds);
  const stageOn = (route, name) => route.plan.shouldApply(name) && (!profile.enabled || profile.enabled.includes(name));
  // The host stores' facts for one request. A store read that fails is reported and treated
  // as not blocked, not exempt and nothing learned, so storage outages don't block traffic.
  const storeFacts = async (req, route) => {
    const ip = clientIp(req);
    const rawPath = String(req.path || req.url || '/').split('?')[0];
    const read = async (load, fallback) => { try { return await load(); } catch (error) { report(error); return fallback; } };
    // Only what this request's stages use: an exempt request runs logging alone.
    const active = STAGE_NAMES.filter(name => stageOn(route, name));
    if (!active.some(name => name !== 'logging')) return {};
    const exempt = await read(() => exempted.get(`${ip}\u0000${rawPath}`, () => stores.isExempted(ip, rawPath)), false);
    if (exempt) return { exempt: true };
    const [blocked, keywords, countries] = await Promise.all([
      active.includes('ip_keyword_block') || active.includes('ai_anomaly') ? read(() => blockedIps.get(ip, () => stores.isBlocked(ip)), false) : false,
      active.includes('ip_keyword_block') ? read(() => cached('keywords', () => stores.topKeywords(2000)), []) : [],
      active.includes('geo_block') ? read(() => cached('countries', () => stores.blockedCountries()), []) : []
    ]);
    return { blocked: !!blocked, exempt: false, dynamic_keywords: keywords, geo_countries: countries };
  };
  // The engine's literal, case-insensitive exemptions: exempt requests aren't rate limited,
  // so they aren't counted either.
  const exemptIps = new Set(profile.exempt_ips || []);
  const exemptPaths = (profile.exempt_paths ?? ['/health', '/robots.txt', '/sitemap.xml']).map(p => String(p).toLowerCase());
  // The shared count for this request's rate window, counted before the engine decides.
  const rateCount = async (req, route, exempt) => {
    const ip = clientIp(req);
    const rawPath = String(req.path || req.url || '/').split('?')[0];
    if (exempt || exemptIps.has(ip) || exemptPaths.some(p => rawPath.toLowerCase().startsWith(p))) return undefined;
    const key = profile.rate_key_mode === 'ip' ? `ratelimit:${ip}` : `ratelimit:${ip}:${rawPath}`;
    const window = Math.max(route.rate.window ?? profile.rate_window ?? 10, 1);
    return rateCounter.count(key, Date.now() / 1000, window);
  };
  const report = error => { if (options.onError) options.onError(error); else console.error('AIWAF native finalization failed:', error.message); };
  // This pipeline takes no general settings, so these two come from the environment. Route
  // checks run in aiwaf-wasm, loaded here so requests can use them straight away.
  const observeRisks = routeRisks.createObserver({ AIWAF_PATH_MANIFEST: manifestPath, AIWAF_INJECTION_RISK_MODE: options.injectionRiskMode });
  if (observeRisks) await wasmChecks.load();
  function payload(req) {
    const b = safeBody(req, maxBytes);
    const query = String(req.originalUrl || req.url || '').split('?').slice(1).join('?');
    if (b.exhausted && profile.sql_mode !== 'off') {
      return monitored({ status: 413, rule: 'payload_inspection_limit', mode: profile.sql_mode });
    }
    return monitored(JSON.parse(native.inspect_payload(JSON.stringify([query, b.body, req.query ?? null]))));
  }
  // A uuidResolver may be async, so only requests it applies to wait for it.
  const middleware = (req, res, next) => {
    if (closed) return next(new Error('Native pipeline is closed'));
    const start = performance.now();
    let route;
    try { route = routePlan(req); } catch (error) { return next(error); }
    const uid = options.uuidResolver && route.plan.shouldApply('uuid_tamper')
      && (!profile.enabled || profile.enabled.includes('uuid_tamper'))
      ? pathUuid(String(req.path || req.url || '/').split('?')[0]) : null;
    let found = null;
    if (uid) {
      // A failing resolver doesn't block, as in the regular middleware.
      try { found = options.uuidResolver(uid, req); } catch (_) { found = null; }
      if (found && typeof found.then === 'function') found = found.then(exists => !!exists, () => null);
      else if (found !== null) found = !!found;
    }
    const counted = rateCounter && stageOn(route, 'rate_limit');
    const peek = !req[DEFER_BODY] && !req[PEEKED] && profile.sql_mode !== 'off' && stageOn(route, 'ip_keyword_block') && readable(req)
      ? peekBody(req, maxBytes).then(prefix => { req[PEEKED] = prefix; }) : null;
    if (!stores && !counted && !peek && !(found && typeof found.then === 'function')) return handle(req, res, next, route, start, found, null);
    return Promise.all([found, stores ? storeFacts(req, route) : null, peek])
      .then(async ([exists, facts]) => {
        if (counted) {
          let count;
          try { count = await rateCount(req, route, facts?.exempt); } catch (error) {
            report(error);
            // As the regular rate limiter does: closed answers 503, open lets the request through.
            if (rateFailureMode === 'closed') return res.status(503).json({ error: 'temporarily_unavailable' });
          }
          if (count !== undefined) facts = { ...(facts || {}), rate_count: count };
        }
        return handle(req, res, next, route, start, exists, facts);
      }, next);
  };
  const handle = (req, res, next, route, start, uuidExists, facts) => {
    if (closed) return next(new Error('Native pipeline is closed'));
    let decision;
    try {
      const { plan, disabled, rate } = route;
      if (observeRisks && plan.shouldApply('ip_keyword_block')
          && (!profile.enabled || profile.enabled.includes('ip_keyword_block'))) {
        const riskFinding = observeRisks(req);
        if (riskFinding) return res.status(riskFinding.status).json({ error: riskFinding.rule });
      }
      const rawPath = String(req.path || req.url || '/').split('?')[0];
      const body = req[DEFER_BODY] ? {body:null,exhausted:false} : safeBody(req, maxBytes);
      decision = JSON.parse(native.begin(JSON.stringify({
        ip: clientIp(req),
        path: rawPath, method: String(req.method || 'GET'), headers: headers(req.headers),
        query: String(req.originalUrl || req.url || '').split('?').slice(1).join('?'),
        body: body.body, parsed_query: req.query ?? null,
        now: Date.now() / 1000, country: String(synchronous(options.countryLookup, req, '') || ''),
        authenticated: req[DEFER_BODY] ? false : !!synchronous(options.authenticated, req, false),
        exists: !!synchronous(options.routeExists, req, !!req.route), disabled,
        payload_exhausted: body.exhausted, defer_body: !!req[DEFER_BODY],
        ...(Object.keys(rate).length ? { rate } : {}),
        ...(uuidExists === null ? {} : { uuid_exists: uuidExists }),
        ...(facts || {})
      })));
    } catch (error) { return next(error); }
    if (decision.monitor) monitored({ rule: decision.monitor, mode: 'monitor' });
    pending.add(decision.context);
    req[contextKey] = decision.context;
    let done = false;
    const finalize = aborted => {
      if (done || closed) return;
      done = true; pending.delete(decision.context);
      try {
        const effects = JSON.parse(native.finish(decision.context, aborted ? 499 : res.statusCode || 200, performance.now() - start, Date.now() / 1000));
        if (blocks && effects.block_csv) blocks.append(effects.block_csv);
        if (keywords && effects.keyword_csv) keywords.append(effects.keyword_csv);
        if (log && effects.log_csv) log.append(effects.log_csv);
        if (stores) writeEffects(effects);
      } catch (error) { report(error); }
    };
    res.on('finish', () => finalize(false));
    res.on('close', () => finalize(true));
    if (decision.status) return res.status(decision.status).json({ error: decision.reason });
    return next();
  };
  // Store writes run after the response, so a slow or failing store never delays it.
  const writeEffects = effects => {
    if (effects.block) {
      blockedIps.set(effects.block.ip, true);
      Promise.resolve().then(() => stores.block(effects.block.ip, effects.block.reason, effects.block.expires_at)).catch(report);
    }
    if (effects.keyword) {
      Promise.resolve().then(() => stores.learn(effects.keyword))
        .then(() => cache.delete('keywords'), report);
    }
  };
  middleware.inspectPayload = req => {
    if (profile.enabled && !profile.enabled.includes('ip_keyword_block')) return null;
    if (!routePlan(req).plan.shouldApply('ip_keyword_block')) return null;
    const riskFinding = observeRisks ? observeRisks(req) : null;
    if (riskFinding) return riskFinding;
    return payload(req);
  };
  middleware.diagnostics = () => JSON.parse(native.diagnostics());
  middleware.inspectBody = (req, body) => {
    const captured = safeBody({body,headers:{}},maxBytes);
    return monitored(JSON.parse(native.inspect_body(req[contextKey], JSON.stringify(captured.body), captured.exhausted,
      !!synchronous(options.authenticated,req,false), Date.now()/1000)));
  };
  middleware.setModel = setModel;
  middleware.close = () => {
    if (closed) return;
    if (pending.size) throw new Error('Stop requests and finish in-flight responses before closing the native pipeline');
    closed = true;
    for (const entry of opened) entry.close();
    native.free();
  };
  return middleware;
};
module.exports.DEFER_BODY = DEFER_BODY;
module.exports.regularStores = regularStores;
module.exports.recentAnswers = recentAnswers;
module.exports.redisRateCounter = require('./nativeRateCounter').redisRateCounter;
