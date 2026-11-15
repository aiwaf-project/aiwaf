const crypto = require('crypto');
const net = require('net');

const STATIC_EXTENSIONS = new Set([
  '.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg',
  '.woff', '.woff2', '.ttf', '.eot', '.otf', '.map', '.json',
  '.xml', '.txt', '.pdf', '.zip', '.tar', '.gz', '.webp', '.avif'
]);

// Upstream peers allowed to supply X-Forwarded-For unless AIWAF_TRUSTED_PROXIES
// is set. A public peer is never trusted by default, so internet clients that
// reach the application directly can't choose their own address.
const DEFAULT_TRUSTED_PROXIES = [
  '127.0.0.0/8', '10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', '169.254.0.0/16',
  '::1/128', 'fc00::/7', 'fe80::/10'
];
// Published at https://www.cloudflare.com/ips/ (retrieved 2026-11-15). Keep in step with
// CLOUDFLARE_PROXIES in py/aiwaf/core/utils.py. Opt-in only: anyone can send requests
// through Cloudflare, so these ranges don't prove a request came from your own zone.
const CLOUDFLARE_PROXIES = [
  '173.245.48.0/20', '103.21.244.0/22', '103.22.200.0/22', '103.31.4.0/22',
  '141.101.64.0/18', '108.162.192.0/18', '190.93.240.0/20', '188.114.96.0/20',
  '197.234.240.0/22', '198.41.128.0/17', '162.158.0.0/15', '104.16.0.0/13',
  '104.24.0.0/14', '172.64.0.0/13', '131.0.72.0/22',
  '2400:cb00::/32', '2606:4700::/32', '2803:f800::/32', '2405:b500::/32',
  '2405:8100::/32', '2a06:98c0::/29', '2c0f:f248::/32'
];
// Names accepted in AIWAF_TRUSTED_PROXIES alongside addresses and ranges, for example
// 'default,cloudflare' to keep the built-in ranges and add Cloudflare's.
const TRUSTED_PROXY_PRESETS = { default: DEFAULT_TRUSTED_PROXIES, cloudflare: CLOUDFLARE_PROXIES };
const MAX_FORWARDED_FOR_LENGTH = 4096;
const MAX_FORWARDED_FOR_ENTRIES = 32;
const trustedProxyCache = new Map();

// Write an IPv6 address in canonical form (lowercase, zeros compressed) and an IPv4-mapped one as
// IPv4, as Python and aiwaf-wasm do, so the same client keeps the same address everywhere.
function canonicalIpv6(text) {
  const zone = text.indexOf('%');
  const address = zone === -1 ? text : text.slice(0, zone);
  let canonical;
  try {
    canonical = new URL(`http://[${address}]/`).hostname.slice(1, -1);
  } catch {
    return text;
  }
  const mapped = canonical.match(/^::ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$/);
  if (mapped) {
    const [high, low] = [parseInt(mapped[1], 16), parseInt(mapped[2], 16)];
    return `${high >> 8}.${high & 255}.${low >> 8}.${low & 255}`;
  }
  return zone === -1 ? canonical : canonical + text.slice(zone);
}

function normalizeIp(value) {
  let text = String(value ?? '').trim();
  if (text.startsWith('[') && text.includes(']')) {
    text = text.slice(1, text.indexOf(']'));
  } else if (text.split(':').length === 2) {
    // IPv4 with a port, as sent by some load balancers.
    text = text.split(':')[0];
  }
  text = text.replace(/^::ffff:(?=\d+\.)/i, '');
  const family = net.isIP(text);
  if (!family) return '';
  return family === 6 ? canonicalIpv6(text) : text;
}

function buildTrustedProxies(entries) {
  const list = new net.BlockList();
  let size = 0;
  for (const entry of entries) {
    const [address, prefix] = entry.split('/');
    const family = net.isIP(address);
    const bits = prefix === undefined ? (family === 6 ? 128 : 32) : Number(prefix);
    const max = family === 6 ? 128 : 32;
    if (!family || !Number.isInteger(bits) || bits < 0 || bits > max) {
      // An unparseable entry must never widen trust.
      console.warn(`[AIWAF] Ignoring invalid AIWAF_TRUSTED_PROXIES entry: ${entry}`);
      continue;
    }
    list.addSubnet(address, bits, family === 6 ? 'ipv6' : 'ipv4');
    size += 1;
  }
  return size ? list : null;
}

// Parsed once per setting, keyed by the raw value, so a request does no parsing.
function getTrustedProxyConfig(configured) {
  const source = configured ?? process.env.AIWAF_TRUSTED_PROXIES ?? DEFAULT_TRUSTED_PROXIES;
  const key = typeof source === 'string' ? `s:${source}` : `a:${Array.from(source).join('\u0000')}`;
  let config = trustedProxyCache.get(key);
  if (!config) {
    const entries = (typeof source === 'string' ? source.split(',') : Array.from(source))
      .map(entry => String(entry).trim())
      .filter(Boolean)
      .flatMap(entry => (Object.hasOwn(TRUSTED_PROXY_PRESETS, entry.toLowerCase())
        ? TRUSTED_PROXY_PRESETS[entry.toLowerCase()]
        : [entry]));
    if (trustedProxyCache.size >= 64) trustedProxyCache.clear();
    config = { entries, list: buildTrustedProxies(entries), native: undefined };
    trustedProxyCache.set(key, config);
  }
  return config;
}

function getTrustedProxies(configured) {
  return getTrustedProxyConfig(configured).list;
}

let wasmChecks = null;

// The aiwaf-wasm resolver for this setting, once aiwaf-wasm has loaded. Until then, or with an
// older aiwaf-wasm, requests use the JavaScript walk below, which gives the same answers.
function nativeTrustedProxies(config) {
  if (config.native !== undefined) return config.native;
  wasmChecks = wasmChecks || require('./wasmChecks');
  const mod = wasmChecks.loadedModule();
  if (!mod) return null;
  config.native = typeof mod.TrustedProxySet === 'function' ? new mod.TrustedProxySet(config.entries) : null;
  return config.native;
}

let untrustedPeerWarned = false;

// Point out, once per process, a proxy that may need adding to AIWAF_TRUSTED_PROXIES.
function warnUntrustedPeerOnce(peer, header) {
  if (untrustedPeerWarned) return;
  untrustedPeerWarned = true;
  console.warn(
    `[AIWAF] Ignoring ${header} from untrusted peer ${peer} and using the peer's address as the client. ` +
    `If ${peer} is your proxy or CDN, add it to AIWAF_TRUSTED_PROXIES (behind Cloudflare, use ` +
    'AIWAF_TRUSTED_PROXIES=default,cloudflare); otherwise no action is needed. This warning is logged once.'
  );
}

function resetUntrustedPeerWarning() {
  untrustedPeerWarned = false;
}

function isTrustedProxy(ip, list) {
  try {
    return list.check(ip, net.isIP(ip) === 6 ? 'ipv6' : 'ipv4');
  } catch {
    return false;
  }
}

function resolveForwardedIp(req, list) {
  const header = String(req.headers?.['x-forwarded-for'] ?? '');
  if (header && header.length <= MAX_FORWARDED_FOR_LENGTH) {
    // Proxies append, so walk from the right and stop at the first untrusted
    // address. Anything further left is client-supplied.
    const hops = header.split(',').slice(-MAX_FORWARDED_FOR_ENTRIES).reverse();
    let rightmost = '';
    for (const hop of hops) {
      const candidate = normalizeIp(hop);
      // Stop rather than skip, so a malformed hop can't expose values to its left.
      if (!candidate) break;
      if (!isTrustedProxy(candidate, list)) return candidate;
      rightmost = rightmost || candidate;
    }
    // Every hop is internal. Only the rightmost was written by our proxy.
    if (rightmost) return rightmost;
  }
  return normalizeIp(req.headers?.['x-real-ip']);
}

/**
 * Resolve the client address, honoring proxy headers only from a trusted peer.
 * A framework that already resolved req.ip (Express `trust proxy`, for example)
 * is respected. A missing peer means a local Unix socket.
 */
function getRequestIp(req = {}, trustedProxies) {
  if (req.aiwafClientIp) return req.aiwafClientIp;
  const socketPeer = req.socket?.remoteAddress || req.connection?.remoteAddress || '';
  const rawPeer = String(socketPeer || req.ip || '');
  const declared = normalizeIp(req.ip);
  if (socketPeer && declared && declared !== normalizeIp(rawPeer)) return declared;

  const config = getTrustedProxyConfig(trustedProxies);
  const native = nativeTrustedProxies(config);
  if (!native) return resolveRequestIpInJavaScript(req, rawPeer, config.list);
  const forwardedFor = req.headers?.['x-forwarded-for'];
  const realIp = req.headers?.['x-real-ip'];
  const ip = native.resolve(rawPeer, forwardedFor == null ? '' : String(forwardedFor),
    realIp == null ? '' : String(realIp));
  const ignored = native.ignored_header();
  if (ignored) warnUntrustedPeerOnce(ip, ignored);
  return ip;
}

function resolveRequestIpInJavaScript(req, rawPeer, list) {
  const peer = normalizeIp(rawPeer);
  const fallback = peer || rawPeer.trim();
  if (!list) return fallback;
  if (rawPeer.trim() && !(peer && isTrustedProxy(peer, list))) {
    const forwardedFor = req.headers?.['x-forwarded-for'];
    if (peer && (forwardedFor || req.headers?.['x-real-ip'])) {
      warnUntrustedPeerOnce(peer, forwardedFor ? 'X-Forwarded-For' : 'X-Real-IP');
    }
    return fallback;
  }
  return resolveForwardedIp(req, list) || fallback;
}

function isPrivateIp(ip) {
  const normalized = String(ip || '').trim().replace(/^::ffff:/, '');
  if (!normalized) return false;

  if (normalized === '::1' || normalized.toLowerCase().startsWith('fc') || normalized.toLowerCase().startsWith('fd')) {
    return true;
  }

  if (net.isIP(normalized) !== 4) return false;
  const parts = normalized.split('.').map(part => Number(part));
  if (parts.length !== 4 || parts.some(part => Number.isNaN(part))) return false;

  return parts[0] === 10
    || (parts[0] === 172 && parts[1] >= 16 && parts[1] <= 31)
    || (parts[0] === 192 && parts[1] === 168)
    || parts[0] === 127;
}

function isStaticFile(requestPath) {
  const pathLower = String(requestPath || '').toLowerCase();
  return [...STATIC_EXTENSIONS].some(ext => pathLower.endsWith(ext));
}

function sanitizeHeaderValue(value, maxLength = 500) {
  if (!value) return '';
  let sanitized = String(value);
  if (sanitized.length > maxLength) sanitized = `${sanitized.slice(0, maxLength)}...`;
  return sanitized
    .split('')
    .filter(char => {
      const code = char.charCodeAt(0);
      return code >= 32 || char === '\t' || char === '\n' || char === '\r';
    })
    .join('');
}

function parseUserAgent(userAgent) {
  if (!userAgent) return { browser: 'unknown', version: 'unknown', os: 'unknown' };

  const uaLower = String(userAgent).toLowerCase();
  const result = { browser: 'unknown', version: 'unknown', os: 'unknown' };

  if (uaLower.includes('chrome') && !uaLower.includes('edg')) result.browser = 'chrome';
  else if (uaLower.includes('firefox')) result.browser = 'firefox';
  else if (uaLower.includes('safari') && !uaLower.includes('chrome')) result.browser = 'safari';
  else if (uaLower.includes('edg')) result.browser = 'edge';
  else if (uaLower.includes('opera') || uaLower.includes('opr')) result.browser = 'opera';

  if (uaLower.includes('windows')) result.os = 'windows';
  else if (uaLower.includes('mac') || uaLower.includes('darwin')) result.os = 'macos';
  else if (uaLower.includes('linux')) result.os = 'linux';
  else if (uaLower.includes('android')) result.os = 'android';
  else if (uaLower.includes('iphone') || uaLower.includes('ipad')) result.os = 'ios';

  return result;
}

function getRequestFingerprint(req = {}) {
  const headers = req.headers || {};
  const keyHeaders = ['user-agent', 'accept', 'accept-language', 'accept-encoding', 'connection'];
  const parts = keyHeaders.map(header => `${header}:${headers[header] || ''}`);
  parts.push(`method:${req.method || ''}`);
  return crypto.createHash('md5').update(parts.join('|')).digest('hex').slice(0, 16);
}

function ipInAllowlist(ip, allowlist = []) {
  const normalizedIp = String(ip || '').trim();
  if (!normalizedIp) return false;
  return (allowlist || []).some(entry => {
    const normalizedEntry = String(entry || '').trim();
    if (!normalizedEntry) return false;
    if (normalizedEntry === normalizedIp) return true;
    if (normalizedEntry.endsWith('*')) return normalizedIp.startsWith(normalizedEntry.slice(0, -1));
    return false;
  });
}

function extractExtendedRequestInfo(req = {}, options = {}) {
  if (!options.enabled) return null;
  const headers = req.headers || {};
  const maxHeaders = Number(options.maxHeaders || 50);
  const maxValueLength = Number(options.maxValueLength || 512);
  const redact = new Set((options.redactHeaders || ['authorization', 'cookie', 'set-cookie']).map(item => String(item).toLowerCase()));
  const capturedHeaders = {};

  Object.keys(headers).slice(0, maxHeaders).forEach(name => {
    capturedHeaders[name] = redact.has(name.toLowerCase())
      ? '[redacted]'
      : sanitizeHeaderValue(headers[name], maxValueLength);
  });

  return {
    method: req.method || '',
    path: req.originalUrl || req.url || '',
    ip: getRequestIp(req),
    headers: capturedHeaders,
    fingerprint: getRequestFingerprint(req)
  };
}

class RateLimiter {
  constructor() {
    this.requests = new Map();
    this.cleanupIntervalMs = 300000;
    this.lastCleanup = 0;
  }

  isRateLimited(ip, requestPath, maxRequests = 100, windowSeconds = 300) {
    const now = Date.now();
    if (now - this.lastCleanup > this.cleanupIntervalMs) {
      this.cleanupOldEntries(now, windowSeconds * 2000);
      this.lastCleanup = now;
    }

    const key = String(ip || '');
    const cutoff = now - Number(windowSeconds) * 1000;
    const entries = (this.requests.get(key) || []).filter(entry => entry.timestamp > cutoff);
    if (entries.length >= Number(maxRequests)) {
      this.requests.set(key, entries);
      return true;
    }

    entries.push({ timestamp: now, path: requestPath });
    this.requests.set(key, entries);
    return false;
  }

  cleanupOldEntries(now = Date.now(), maxAgeMs = 600000) {
    const cutoff = now - maxAgeMs;
    for (const [ip, entries] of this.requests.entries()) {
      const kept = entries.filter(entry => entry.timestamp > cutoff);
      if (kept.length) this.requests.set(ip, kept);
      else this.requests.delete(ip);
    }
  }
}

module.exports = {
  STATIC_EXTENSIONS,
  CLOUDFLARE_PROXIES,
  getRequestIp,
  resetUntrustedPeerWarning,
  // For parity tests: the JavaScript walk used until aiwaf-wasm has loaded.
  _resolveRequestIpInJavaScript: (req, trustedProxies) => {
    const rawPeer = String(req.socket?.remoteAddress || req.connection?.remoteAddress || req.ip || '');
    return resolveRequestIpInJavaScript(req, rawPeer, getTrustedProxies(trustedProxies));
  },
  isPrivateIp,
  isStaticFile,
  sanitizeHeaderValue,
  parseUserAgent,
  getRequestFingerprint,
  ipInAllowlist,
  extractExtendedRequestInfo,
  RateLimiter
};
