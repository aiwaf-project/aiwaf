const STATIC_KW = ['.php', '.xmlrpc', 'wp-', '.env', '.git', '.bak', 'shell'];
const STATUS_IDX = ['200', '403', '404', '500'];
const defaultCache = new Map();
const { getClient } = require('./redisClient');

let customCache = null;
let cleanupInterval = null;

// In-memory tracking for burst and 404 calculations
const ipRequestHistory = new Map(); // IP -> array of timestamps
const ip404Counts = new Map(); // IP -> count of 404s

function init(opts = {}) {
  customCache = opts.cache || null;

  if (cleanupInterval) return;

  // Clean up old entries every 5 minutes
  cleanupInterval = setInterval(() => {
    const fiveMinutesAgo = Date.now() - 5 * 60 * 1000;
    
    for (const [ip, timestamps] of ipRequestHistory.entries()) {
      const recentTimestamps = timestamps.filter(t => t > fiveMinutesAgo);
      if (recentTimestamps.length > 0) {
        ipRequestHistory.set(ip, recentTimestamps);
      } else {
        ipRequestHistory.delete(ip);
      }
    }
  }, 5 * 60 * 1000);
  if (typeof cleanupInterval.unref === 'function') {
    cleanupInterval.unref();
  }
}

function cleanup() {
  if (cleanupInterval) {
    clearInterval(cleanupInterval);
    cleanupInterval = null;
  }
  ipRequestHistory.clear();
  ip404Counts.clear();
}

function recordRequest(ip, statusCode) {
  const now = Date.now();
  
  // Record timestamp for burst calculation
  if (!ipRequestHistory.has(ip)) {
    ipRequestHistory.set(ip, []);
  }
  ipRequestHistory.get(ip).push(now);
  
  // Keep only last 100 requests per IP to prevent memory bloat
  const history = ipRequestHistory.get(ip);
  if (history.length > 100) {
    history.splice(0, history.length - 100);
  }
  
  // Track 404 counts
  if (statusCode === 404) {
    ip404Counts.set(ip, (ip404Counts.get(ip) || 0) + 1);
  }
}

function calculateBurstCount(ip) {
  if (!ipRequestHistory.has(ip)) return 0;
  
  const now = Date.now();
  const tenSecondsAgo = now - 10000; // 10 seconds
  const recentRequests = ipRequestHistory.get(ip).filter(t => t > tenSecondsAgo);
  
  return recentRequests.length;
}

function get404Count(ip) {
  return ip404Counts.get(ip) || 0;
}

// Helper to mark request start time for response time calculation
function markRequestStart(req) {
  req._startTime = Date.now();
}

// Helper to calculate response time from marked start
function getResponseTime(req) {
  if (req._startTime) {
    return Date.now() - req._startTime;
  }
  return 0;
}

async function extractFeatures(req, res = null) {
  const uri = req.path || req.url;
  const ip = req.ip || req.socket?.remoteAddress || req.connection?.remoteAddress || 'unknown';
  
  // Get status code from response if available, otherwise default to 200
  let statusCode = 200;
  if (res && res.statusCode) {
    statusCode = res.statusCode;
  } else if (req.res && req.res.statusCode) {
    statusCode = req.res.statusCode;
  }
  
  // Record this request for burst/404 tracking
  recordRequest(ip, statusCode);
  
  const redis = getClient();
  const cacheKey = `features:${uri.toLowerCase()}:${ip}`;

  // Custom cache logic
  if (customCache?.get && customCache?.set) {
    try {
      const cached = await customCache.get(cacheKey);
      if (cached) return cached;
    } catch (err) {
      console.warn('Custom cache read failed:', err.message);
    }
  }

  // Redis check
  if (redis) {
    try {
      const cached = await redis.get(cacheKey);
      if (cached) return JSON.parse(cached);
    } catch (err) {
      console.warn('⚠️ Redis read failed. Falling back.', err.message);
    }
  }

  // Compute features
  const pathLen = uri.length;
  const kwHits = STATIC_KW.reduce((count, kw) => count + (uri.toLowerCase().includes(kw) ? 1 : 0), 0);
  const statusIdx = STATUS_IDX.indexOf(String(statusCode));
  
  // Response time - try multiple sources
  let responseTime = 0;
  if (req.headers && req.headers['x-response-time']) {
    responseTime = parseFloat(req.headers['x-response-time']);
  } else if (req.responseTime) {
    responseTime = req.responseTime;
  } else if (req._startTime) {
    responseTime = Date.now() - req._startTime;
  }
  
  // Calculate real burst and 404 counts
  const burst = calculateBurstCount(ip);
  const total404 = get404Count(ip);
  
  const features = [pathLen, kwHits, statusIdx, responseTime, burst, total404];

  // Write back to cache with shorter TTL for dynamic features
  const ttl = 30; // 30 seconds TTL
  
  if (customCache?.set) {
    try {
      await customCache.set(cacheKey, features, ttl);
    } catch (err) {
      console.warn('Custom cache write failed:', err.message);
    }
  } else if (redis) {
    try {
      await redis.set(cacheKey, JSON.stringify(features), { EX: ttl });
    } catch (err) {
      console.warn('⚠️ Redis write failed. Ignoring.', err.message);
    }
  }

  return features;
}

module.exports = { 
  extractFeatures, 
  init, 
  recordRequest, 
  calculateBurstCount, 
  get404Count,
  markRequestStart,
  getResponseTime,
  cleanup,
  STATIC_KW,
  STATUS_IDX
};
