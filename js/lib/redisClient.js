// lib/redisManager.js
const { createClient } = require('redis');

let client = null;
let isReady = false;

function getRedisUrl() {
  return process.env.REDIS_URL || process.env.AIWAF_REDIS_URL;
}

async function connect() {
  const redisUrl = getRedisUrl();
  if (!redisUrl) {
    console.warn('⚠️ No REDIS_URL set. Redis disabled.');
    return;
  }

  try {
    client = createClient({ url: redisUrl });
    client.on('error', err => console.warn('⚠️ Redis error:', err));
    await client.connect();
    isReady = true;
    console.log('Redis connected.');
  } catch (err) {
    console.warn('Redis connection failed. Falling back.');
    client = null;
    isReady = false;
  }
}

function getClient() {
  return isReady && client?.isOpen ? client : null;
}

module.exports = { connect, getClient, isReady: () => isReady };
