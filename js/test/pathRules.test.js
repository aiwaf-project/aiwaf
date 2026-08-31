const request = require('supertest');
const express = require('express');
const aiwaf = require('../index');
const {
  normalizePaths,
  createRoutePlan,
  getPathRuleOverrides,
  isPathExempt,
  isMiddlewareDisabledForPath,
  exempt,
  exemptFrom,
  only,
  requireProtection
} = require('../lib/pathRules');
const { planEnabledMiddlewares } = require('../lib/middlewarePlan');

function testIp(segment) {
  return `203.0.113.${segment}`;
}

describe('path route capability', () => {
  it('uses longest-prefix path rule when disabling middleware', () => {
    const rules = [
      { PREFIX: '/api/', DISABLE: ['header_validation'] },
      { PREFIX: '/api/v1/', DISABLE: ['rate_limit'] }
    ];

    expect(isMiddlewareDisabledForPath('/api/v1/users', rules, 'rate_limit')).toBe(true);
    expect(isMiddlewareDisabledForPath('/api/v1/users', rules, 'header_validation')).toBe(false);
  });

  it('reads section overrides from the best path rule', () => {
    const rules = [{ PREFIX: '/api/', RATE_LIMIT: { WINDOW: 60, MAX: 10 } }];

    expect(getPathRuleOverrides('/api/data', rules, 'RATE_LIMIT')).toEqual({ WINDOW: 60, MAX: 10 });
  });

  it('honors required route middleware over path-rule disables', () => {
    const plan = createRoutePlan('/api/data', [{ PREFIX: '/api/', DISABLE: ['rate_limit'] }], {
      fullyExempt: true,
      exemptMiddlewares: ['rate_limit'],
      requiredMiddlewares: ['rate_limit']
    });

    expect(plan.shouldApply('rate_limit')).toBe(true);
    expect(plan.shouldApply('header_validation')).toBe(false);
  });

  it('lets path rules disable IP and keyword blocking for matching routes', async () => {
    const app = express();
    app.use(aiwaf({
      staticKeywords: ['.php'],
      AIWAF_WASM_VALIDATION: false,
      AIWAF_PATH_RULES: [{ PREFIX: '/allowed/', DISABLE: ['ip_keyword_block'] }]
    }));
    app.get('/allowed/test.php', (_req, res) => res.send('ok'));

    await request(app)
      .get('/allowed/test.php')
      .set('X-Forwarded-For', testIp(1))
      .expect(200, 'ok');
  });

  it('lets path rules disable header validation for matching routes', async () => {
    const app = express();
    app.use(aiwaf({
      AIWAF_HEADER_VALIDATION: true,
      AIWAF_WASM_VALIDATION: false,
      AIWAF_PATH_RULES: [{ PREFIX: '/health/', DISABLE: ['header_validation'] }]
    }));
    app.get('/health/check', (_req, res) => res.send('ok'));

    await request(app)
      .get('/health/check')
      .set('X-Forwarded-For', testIp(2))
      .set('user-agent', 'curl/8.0.1')
      .set('accept', '*/*')
      .expect(200, 'ok');
  });
});

describe('auto/all middleware planning', () => {
  it('auto disables logging when an external access log is configured', () => {
    const enabled = planEnabledMiddlewares({
      orderedAvailable: ['geo_block', 'ip_keyword_block', 'logging'],
      requested: ['all'],
      disabled: [],
      accessLog: '/var/log/nginx/access.log',
      geoEnabledFlag: false,
      staticBlockCountries: [],
      dynamicBlockCountries: []
    });

    expect(enabled.has('logging')).toBe(false);
  });

  it('auto disables uuid tamper when routes are known and none are UUID routes', () => {
    const enabled = planEnabledMiddlewares({
      orderedAvailable: ['uuid_tamper', 'ip_keyword_block', 'logging'],
      requested: ['auto'],
      disabled: [],
      accessLog: null,
      geoEnabledFlag: false,
      staticBlockCountries: [],
      dynamicBlockCountries: [],
      hasUuidRoutes: false
    });

    expect(enabled.has('uuid_tamper')).toBe(false);
  });
});

describe('path exemptions', () => {
  it('supports exact, prefix, and wildcard matching', () => {
    expect(isPathExempt('/health', ['/health'])).toBe(true);
    expect(isPathExempt('/api/health/live', ['/api/health'])).toBe(true);
    expect(isPathExempt('/assets/app.js', ['/assets/*'])).toBe(true);
  });

  it('normalizes lists and records route-level decorators', () => {
    expect(normalizePaths(['API//Users/', 'health'])).toEqual(['/api/users', '/health']);
    const req = {};
    const next = jest.fn();
    exempt(req, {}, next);
    exemptFrom('rate_limit')(req, {}, next);
    only('logging')(req, {}, next);
    requireProtection('header_validation')(req, {}, next);
    expect(req.aiwafRoute.fullyExempt).toBe(true);
    expect(req.aiwafRoute.exemptMiddlewares).toContain('rate_limit');
    expect(req.aiwafRoute.requiredMiddlewares).toContain('header_validation');
    expect(next).toHaveBeenCalledTimes(4);
  });
});
