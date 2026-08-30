const db = require('../utils/db');
const redisClient = require('../lib/redisClient');
const rateLimiter = require('../lib/rateLimiter');
const featureUtils = require('../lib/featureUtils');

module.exports = async () => {
  try {
    rateLimiter.cleanup();
  } catch (err) {
    // ignore
  }

  try {
    featureUtils.cleanup();
  } catch (err) {
    // ignore
  }

  try {
    const client = redisClient.getClient();
    if (client && client.quit) {
      await client.quit();
    }
  } catch (err) {
    // ignore
  }

  try {
    await db.destroy();
  } catch (err) {
    // ignore
  }
};
