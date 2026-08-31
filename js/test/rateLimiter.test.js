const blacklistManager = require('../lib/blacklistManager');
const rateLimiter = require('../lib/rateLimiter');

describe('rateLimiter', () => {
  afterEach(() => rateLimiter.cleanup());

  test('uses fallback cache operations and performs cleanup', async () => {
    jest.spyOn(blacklistManager, 'isBlocked').mockResolvedValue(false);
    await rateLimiter.init({ WINDOW_SEC: 1, MAX_REQ: 1, FLOOD_REQ: 20 });
    await rateLimiter.record('203.0.113.9');
    expect(await rateLimiter.isBlocked('203.0.113.9')).toBe(false);
    await rateLimiter.record('203.0.113.9');
    expect(await rateLimiter.isBlocked('203.0.113.9')).toBe(true);
    rateLimiter.cleanup();
    blacklistManager.isBlocked.mockRestore();
  });
});
