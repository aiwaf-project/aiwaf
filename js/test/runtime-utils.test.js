const runtimeUtils = require('../lib/runtimeUtils');

describe('runtime utils parity helpers', () => {
  test('detects static files and private IPs', () => {
    expect(runtimeUtils.isStaticFile('/assets/app.css')).toBe(true);
    expect(runtimeUtils.isStaticFile('/api/users')).toBe(false);
    expect(runtimeUtils.isPrivateIp('10.1.2.3')).toBe(true);
    expect(runtimeUtils.isPrivateIp('8.8.8.8')).toBe(false);
  });

  test('sanitizes headers and parses user agents', () => {
    expect(runtimeUtils.sanitizeHeaderValue(`abc${String.fromCharCode(1)}def`, 3)).toBe('abc...');
    expect(runtimeUtils.parseUserAgent('Mozilla/5.0 Windows Chrome/120').browser).toBe('chrome');
    expect(runtimeUtils.parseUserAgent('Mozilla/5.0 Windows Chrome/120').os).toBe('windows');
  });

  test('generates stable request fingerprints', () => {
    const req = {
      method: 'GET',
      headers: {
        'user-agent': 'ua',
        accept: '*/*',
        'accept-language': 'en',
        'accept-encoding': 'gzip',
        connection: 'keep-alive'
      }
    };
    expect(runtimeUtils.getRequestFingerprint(req)).toHaveLength(16);
    expect(runtimeUtils.getRequestFingerprint(req)).toBe(runtimeUtils.getRequestFingerprint(req));
  });

  test('rate limiter follows the python helper contract', () => {
    const limiter = new runtimeUtils.RateLimiter();
    expect(limiter.isRateLimited('1.2.3.4', '/a', 2, 60)).toBe(false);
    expect(limiter.isRateLimited('1.2.3.4', '/b', 2, 60)).toBe(false);
    expect(limiter.isRateLimited('1.2.3.4', '/c', 2, 60)).toBe(true);
  });
});
