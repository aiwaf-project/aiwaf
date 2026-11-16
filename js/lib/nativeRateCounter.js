'use strict';

// Shared rate counts for the native pipeline. Redis counts; aiwaf-wasm still decides.
const crypto = require('crypto');

// The regular Python limiter's sliding window, so both runtimes keep the same bucket format.
const SCRIPT = `
local raw = redis.call('GET', KEYS[1])
local old = raw and cjson.decode(raw) or {}
local now, window = tonumber(ARGV[1]), tonumber(ARGV[2])
local current = {}
for _, timestamp in ipairs(old) do
    if now - timestamp < window then table.insert(current, timestamp) end
end
table.insert(current, now)
redis.call('SET', KEYS[1], cjson.encode(current), 'PX', math.ceil(window * 1000))
return #current
`;

// A fixed-length key, so long paths can't make long Redis keys.
function boundedKey(namespace, key) {
  return `${namespace}:${crypto.createHash('sha256').update(key).digest('hex')}`;
}

/**
 * Counts native rate limits in Redis, atomically across workers. Takes a node-redis v4
 * client (sendCommand) or an ioredis client (call).
 */
function redisRateCounter(client, { namespace = 'aiwaf:native:rate:v1' } = {}) {
  const call = typeof client?.call === 'function' ? args => client.call(...args)
    : typeof client?.sendCommand === 'function' ? args => client.sendCommand(args) : null;
  if (!call) throw new TypeError('redisRateCounter needs a node-redis or ioredis client');
  if (typeof namespace !== 'string' || !namespace || Buffer.byteLength(namespace) > 100) {
    throw new TypeError('namespace must be a nonempty string of at most 100 bytes');
  }
  return {
    async count(key, now, windowSeconds) {
      const count = Number(await call(['EVAL', SCRIPT, '1', boundedKey(namespace, key), String(now), String(windowSeconds)]));
      if (!Number.isInteger(count) || count < 1) throw new Error('Invalid Redis rate count');
      return count;
    }
  };
}

module.exports = { redisRateCounter, boundedKey, SCRIPT };
