const crypto = require('crypto');
const net = require('net');

const STATIC_EXTENSIONS = new Set([
  '.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg',
  '.woff', '.woff2', '.ttf', '.eot', '.otf', '.map', '.json',
  '.xml', '.txt', '.pdf', '.zip', '.tar', '.gz', '.webp', '.avif'
]);

function getRequestIp(req = {}) {
  const forwarded = req.headers?.['x-forwarded-for'];
  const raw = forwarded
    ? String(forwarded).split(',')[0].trim()
    : (req.ip || req.connection?.remoteAddress || req.socket?.remoteAddress || '');
  return String(raw).replace(/^::ffff:/, '');
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
