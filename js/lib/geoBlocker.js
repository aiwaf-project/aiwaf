const fs = require('fs');
const path = require('path');
const geoStore = require('./geoStore');
const { evaluateGeoPolicy } = require('./geoPolicy');

let config = {
  enabled: false,
  blocked: [],
  allowed: [],
  resolver: null,
  mmdbPath: '',
  mmdbReader: null,
  cacheSeconds: 3600,
  cachePrefix: 'aiwaf:geo:',
  cache: new Map()
};

function normalizeCountry(value) {
  return String(value || '').trim().toUpperCase();
}

function getRequestIp(req) {
  const rawIp = req.headers?.['x-forwarded-for']
    ? String(req.headers['x-forwarded-for']).split(',')[0].trim()
    : (req.ip || req.connection?.remoteAddress || req.socket?.remoteAddress || '');
  return String(rawIp).replace(/^::ffff:/, '');
}

function getCachedCountry(ip) {
  if (!ip || config.cacheSeconds <= 0) return '';
  const key = `${config.cachePrefix}${ip}`;
  const entry = config.cache.get(key);
  if (!entry) return '';
  if (Date.now() > entry.expiresAt) {
    config.cache.delete(key);
    return '';
  }
  return entry.country || '';
}

function setCachedCountry(ip, country) {
  if (!ip || config.cacheSeconds <= 0) return;
  const key = `${config.cachePrefix}${ip}`;
  config.cache.set(key, {
    country,
    expiresAt: Date.now() + config.cacheSeconds * 1000
  });
}

async function resolveCountry(req) {
  const ip = getRequestIp(req);
  const cached = getCachedCountry(ip);
  if (cached) {
    return { country: cached, ip };
  }

  if (typeof config.resolver === 'function') {
    try {
      const result = normalizeCountry(await config.resolver(req));
      if (result) setCachedCountry(ip, result);
      return { country: result, ip };
    } catch (err) {
      return { country: '', ip };
    }
  }

  if (config.mmdbReader) {
    try {
      if (ip) {
        const result = config.mmdbReader.get(ip);
        const iso = result?.country?.iso_code || result?.registered_country?.iso_code;
        if (iso) {
          const normalized = normalizeCountry(iso);
          if (normalized) setCachedCountry(ip, normalized);
          return { country: normalized, ip };
        }
      }
    } catch (err) {
      // Fall through to header fallback.
    }
  }

  // Fallback when no MMDB reader is available.
  const fallback = normalizeCountry(req.headers?.['x-country-code']);
  if (fallback) setCachedCountry(ip, fallback);
  return { country: fallback, ip };
}

function tryLoadMmdbReader(mmdbPath) {
  if (!mmdbPath || !fs.existsSync(mmdbPath)) return null;
  try {
    // Optional dependency by design.
    // eslint-disable-next-line global-require, import/no-extraneous-dependencies
    const maxmind = require('maxmind');
    return maxmind.openSync(mmdbPath);
  } catch (err) {
    return null;
  }
}

module.exports = {
  init(opts = {}) {
    const packagedDefault = path.resolve(__dirname, '..', 'geolock', 'ipinfo_lite.mmdb');
    const monorepoDefault = path.resolve(
      __dirname,
      '..',
      '..',
      'py',
      'aiwaf',
      'core',
      'geolock',
      'ipinfo_lite.mmdb'
    );
    const resolvedPath = opts.AIWAF_GEO_MMDB_PATH
      ? path.resolve(opts.AIWAF_GEO_MMDB_PATH)
      : (fs.existsSync(packagedDefault) ? packagedDefault : monorepoDefault);
    config = {
      enabled: !!opts.AIWAF_GEO_BLOCK_ENABLED,
      blocked: (opts.AIWAF_GEO_BLOCK_COUNTRIES || []).map(normalizeCountry),
      allowed: (opts.AIWAF_GEO_ALLOW_COUNTRIES || []).map(normalizeCountry),
      resolver: opts.geoResolver || null,
      mmdbPath: resolvedPath,
      mmdbReader: tryLoadMmdbReader(resolvedPath),
      cacheSeconds: Number.isFinite(Number(opts.AIWAF_GEO_CACHE_SECONDS))
        ? Number(opts.AIWAF_GEO_CACHE_SECONDS)
        : 3600,
      cachePrefix: opts.AIWAF_GEO_CACHE_PREFIX || 'aiwaf:geo:',
      cache: new Map()
    };
    geoStore.initialize().catch(() => {});
  },

  async check(req) {
    if (!config.enabled) {
      return { blocked: false, country: '' };
    }

    // If both allow and block lists are empty, then geo-blocking is effectively disabled
    // for static rules. Return no blocking and no country.
    if (config.allowed.length === 0 && config.blocked.length === 0) {
      return { blocked: false, country: '' };
    }

    const result = await resolveCountry(req);
    const country = result.country;
    if (!country) {
      return { blocked: false, country: '' };
    }

    let dynamicBlocked = [];
    try {
      dynamicBlocked = await geoStore.isBlockedCountry(country) ? [country] : [];
    } catch (err) {
      dynamicBlocked = [];
    }

    const decision = evaluateGeoPolicy({
      country,
      allowCountries: config.allowed,
      blockCountries: config.blocked,
      dynamicBlocked
    });

    if (decision.shouldBlock) {
      return { blocked: true, country: decision.country, reason: decision.reason };
    }

    return { blocked: false, country: decision.country };
  }
};
