const NodeCache = require('node-cache');
const redis = require('redis');
module.exports = (useRedis=false) => useRedis ? redis.createClient() : new NodeCache();