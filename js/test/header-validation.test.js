const headerValidation = require('../lib/headerValidation');

describe('headerValidation quality scoring', () => {
  it('does not enforce score when required headers are empty', () => {
    headerValidation.init({
      AIWAF_HEADER_VALIDATION: true,
      AIWAF_REQUIRED_HEADERS: [],
      AIWAF_HEADER_QUALITY_MIN_SCORE: 3
    });

    const reason = headerValidation.validate({ headers: { 'user-agent': 'curl/7.88.1' } });
    expect(reason).toBe('suspicious_user_agent');
  });

  it('blocks when header quality score is below minimum', () => {
    headerValidation.init({
      AIWAF_HEADER_VALIDATION: true,
      AIWAF_REQUIRED_HEADERS: ['accept'],
      AIWAF_HEADER_QUALITY_MIN_SCORE: 3
    });

    const reason = headerValidation.validate({ headers: { accept: '*/*', 'user-agent': 'Mozilla/5.0' } });
    expect(['header_quality_low', 'suspicious_headers:generic_accept_no_locale']).toContain(reason);
  });

  it('passes with browser-like headers', () => {
    headerValidation.init({
      AIWAF_HEADER_VALIDATION: true,
      AIWAF_REQUIRED_HEADERS: ['accept'],
      AIWAF_HEADER_QUALITY_MIN_SCORE: 3
    });

    const headers = {
      'user-agent': 'Mozilla/5.0',
      accept: 'text/html,application/xml;q=0.9,*/*;q=0.8',
      'accept-language': 'en-US,en;q=0.9',
      'accept-encoding': 'gzip, deflate, br',
      connection: 'keep-alive',
      'cache-control': 'no-cache'
    };

    const reason = headerValidation.validate({ headers });
    expect(reason).toBeNull();
  });

  it('enforces header caps and suspicious UA', () => {
    headerValidation.init({
      AIWAF_HEADER_VALIDATION: true,
      AIWAF_REQUIRED_HEADERS: ['accept'],
      AIWAF_MAX_HEADER_BYTES: 10
    });

    const reason = headerValidation.validate({ headers: { accept: 'text/html', 'user-agent': 'curl/7.88.1' } });
    expect(reason).toMatch(/header_bytes_exceeded|suspicious_user_agent/);
  });

  it('supports per-method required headers', () => {
    headerValidation.init({
      AIWAF_HEADER_VALIDATION: true,
      AIWAF_REQUIRED_HEADERS: { POST: ['x-auth'], DEFAULT: ['accept'] }
    });

    const postReason = headerValidation.validate({ method: 'POST', headers: { accept: '*/*' } });
    expect(postReason).toBe('missing_header:x-auth');
  });
});
