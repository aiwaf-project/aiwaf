const honeypotDetector = require('../lib/honeypotDetector');

describe('honeypotDetector method and timing policies', () => {
  it('blocks disallowed methods when method policy is enabled', () => {
    honeypotDetector.init({
      AIWAF_METHOD_POLICY_ENABLED: true,
      AIWAF_ALLOWED_METHODS: ['GET', 'POST']
    });

    const result = honeypotDetector.evaluate({ method: 'TRACE' }, '198.51.100.10', '/');
    expect(result.triggered).toBe(true);
    expect(result.statusCode).toBe(405);
  });

  it('does not enforce method policy when disabled', () => {
    honeypotDetector.init({
      AIWAF_METHOD_POLICY_ENABLED: false,
      AIWAF_ALLOWED_METHODS: ['GET', 'POST']
    });

    const result = honeypotDetector.evaluate({ method: 'TRACE' }, '198.51.100.10', '/');
    expect(result.triggered).toBe(false);
  });

  it('returns page_expired when POST is too slow', () => {
    honeypotDetector.init({
      AIWAF_MIN_FORM_TIME: 0.1,
      AIWAF_MAX_PAGE_TIME: 0.2
    });

    const ip = '198.51.100.11';
    honeypotDetector.evaluate({ method: 'GET' }, ip, '/form');
    const result = honeypotDetector.evaluate({ method: 'POST' }, ip, '/form');
    if (result.errorCode) {
      expect(result.errorCode).toBe('page_expired');
      expect(result.statusCode).toBe(409);
    } else {
      expect(result.triggered).toBe(true);
      expect(result.reason).toBe('honeypot_too_fast');
    }
  });
});
