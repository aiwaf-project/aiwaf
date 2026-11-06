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
const MAX_FORWARDED_FOR_LENGTH = 4096;
const MAX_FORWARDED_FOR_ENTRIES = 32;
const trustedProxyCache = new Map();

function normalizeIp(value) {
  let text = String(value ?? '').trim();
  if (text.startsWith('[') && text.includes(']')) {
    text = text.slice(1, text.indexOf(']'));
  } else if (text.split(':').length === 2) {
    // IPv4 with a port, as sent by some load balancers.
    text = text.split(':')[0];
  }
  text = text.replace(/^::ffff:(?=\d+\.)/i, '');
  return net.isIP(text) ? text : '';
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

function getTrustedProxies(configured) {
  const source = configured ?? process.env.AIWAF_TRUSTED_PROXIES ?? DEFAULT_TRUSTED_PROXIES;
  const entries = (typeof source === 'string' ? source.split(',') : Array.from(source))
    .map(entry => String(entry).trim())
    .filter(Boolean);
  const key = entries.join(',');
  if (!trustedProxyCache.has(key)) trustedProxyCache.set(key, buildTrustedProxies(entries));
  return trustedProxyCache.get(key);
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
  const peer = normalizeIp(rawPeer);
  const fallback = peer || rawPeer;
  const declared = normalizeIp(req.ip);
  if (socketPeer && declared && declared !== peer) return declared;

  const list = getTrustedProxies(trustedProxies);
  if (!list || (rawPeer && !(peer && isTrustedProxy(peer, list)))) return fallback;
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
  getRequestIp,
  isPrivateIp,
  isStaticFile,
  sanitizeHeaderValue,
  parseUserAgent,
  getRequestFingerprint,
  ipInAllowlist,
  extractExtendedRequestInfo,
  RateLimiter
};
