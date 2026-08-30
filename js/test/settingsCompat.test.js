const { normalizeSettings } = require('../lib/settingsCompat');

describe('settingsCompat.normalizeSettings', () => {
  it('maps legacy nested AIWAF_SETTINGS to flat keys', () => {
    const out = normalizeSettings({
      AIWAF_SETTINGS: {
        rate: { window: 7, max: 9, flood: 11 },
        honeypot: { field: 'hp', minFormTime: 3, maxPageTime: 44 },
        exemptions: {
          paths: ['/health'],
          ips: ['1.2.3.4'],
          allowedPathKeywords: ['safe'],
          keywords: ['.env'],
          dbEnabled: false
        },
        headerValidation: {
          enabled: true,
          requiredHeaders: ['x-auth'],
          blockedUserAgents: ['scanner'],
          minScore: 5
        },
        geo: {
          enabled: true,
          blockedCountries: ['cn'],
          allowCountries: ['us'],
          mmdbPath: '/tmp/geo.mmdb'
        },
        logging: {
          enabled: true,
          path: 'logs/a.jsonl',
          dbEnabled: true,
          csvEnabled: true,
          csvPath: 'logs/a.csv'
        },
        honeypot: {
          field: 'hp',
          minFormTime: 3,
          maxPageTime: 44,
          loginPathPrefixes: ['/login/'],
          postOnlySuffixes: ['/submit/'],
          allowedMethods: ['GET', 'POST'],
          methodPolicyEnabled: false
        },
        training: { minAiLogs: 123 },
        errors: { forceJson: false },
        keywords: { static: ['.php'], dynamicTopN: 12, enableLearning: false }
      }
    });

    expect(out.WINDOW_SEC).toBe(7);
    expect(out.MAX_REQ).toBe(9);
    expect(out.FLOOD_REQ).toBe(11);
    expect(out.HONEYPOT_FIELD).toBe('hp');
    expect(out.AIWAF_MIN_FORM_TIME).toBe(3);
    expect(out.AIWAF_MAX_PAGE_TIME).toBe(44);
    expect(out.AIWAF_LOGIN_PATH_PREFIXES).toEqual(['/login/']);
    expect(out.AIWAF_POST_ONLY_SUFFIXES).toEqual(['/submit/']);
    expect(out.AIWAF_ALLOWED_METHODS).toEqual(['GET', 'POST']);
    expect(out.AIWAF_METHOD_POLICY_ENABLED).toBe(false);
    expect(out.AIWAF_EXEMPT_PATHS).toEqual(['/health']);
    expect(out.AIWAF_EXEMPT_IPS).toEqual(['1.2.3.4']);
    expect(out.AIWAF_EXEMPTIONS_DB).toBe(false);
    expect(out.AIWAF_HEADER_VALIDATION).toBe(true);
    expect(out.AIWAF_REQUIRED_HEADERS).toEqual(['x-auth']);
    expect(out.AIWAF_HEADER_QUALITY_MIN_SCORE).toBe(5);
    expect(out.AIWAF_GEO_BLOCK_ENABLED).toBe(true);
    expect(out.AIWAF_GEO_BLOCK_COUNTRIES).toEqual(['cn']);
    expect(out.AIWAF_GEO_ALLOW_COUNTRIES).toEqual(['us']);
    expect(out.AIWAF_GEO_MMDB_PATH).toBe('/tmp/geo.mmdb');
    expect(out.AIWAF_MIDDLEWARE_LOG_DB).toBe(true);
    expect(out.AIWAF_MIDDLEWARE_LOG_CSV).toBe(true);
    expect(out.AIWAF_MIDDLEWARE_LOG_CSV_PATH).toBe('logs/a.csv');
    expect(out.AIWAF_MIN_AI_LOGS).toBe(123);
    expect(out.AIWAF_ENABLE_KEYWORD_LEARNING).toBe(false);
    expect(out.AIWAF_FORCE_JSON_ERRORS).toBe(false);
    expect(out.staticKeywords).toEqual(['.php']);
    expect(out.dynamicTopN).toBe(12);
  });

  it('parses env-driven booleans and arrays', () => {
    process.env.AIWAF_MIDDLEWARE_LOG_DB = 'true';
    process.env.AIWAF_MIDDLEWARE_LOG_CSV = '1';
    process.env.AIWAF_EXEMPT_PATHS = '/a,/b';

    const out = normalizeSettings({});

    expect(out.AIWAF_MIDDLEWARE_LOG_DB).toBe(true);
    expect(out.AIWAF_MIDDLEWARE_LOG_CSV).toBe(true);
    expect(out.AIWAF_EXEMPT_PATHS).toEqual(['/a', '/b']);

    delete process.env.AIWAF_MIDDLEWARE_LOG_DB;
    delete process.env.AIWAF_MIDDLEWARE_LOG_CSV;
    delete process.env.AIWAF_EXEMPT_PATHS;
  });
});
