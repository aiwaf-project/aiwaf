// Converting regular aiwaf(options) to native pipeline options.
if (process.env.AIWAF_NATIVE_TEST_MODULE) {
  jest.doMock(require.resolve('aiwaf-wasm'), () => require(process.env.AIWAF_NATIVE_TEST_MODULE));
}
const fs = require('fs');
const os = require('os');
const path = require('path');
const express = require('express');
const request = require('supertest');
const aiwaf = require('../index');
const { nativeOptionsFromSettings } = require('../lib/nativeSettings');
const createNativeForTest = require('../lib/nativePipeline');

const outcome = (conversion, setting) => conversion.notes.find(n => n.setting === setting)?.outcome;
const roots = [];
const temp = () => { const root = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-native-settings-')); roots.push(root); return root; };
afterAll(() => { for (const root of roots) fs.rmSync(root, { recursive: true, force: true }); });

describe('nativeOptionsFromSettings', () => {
  const saved = { ...process.env };
  beforeEach(() => { for (const key of Object.keys(process.env)) if (key.startsWith('AIWAF_')) delete process.env[key]; });
  afterAll(() => { process.env = saved; });

  test('writes out the regular defaults, which differ from the native ones', () => {
    const { profile, options } = nativeOptionsFromSettings({});
    expect([profile.rate_window, profile.rate_max, profile.rate_flood]).toEqual([60, 100, 200]);
    expect([profile.min_form_time, profile.max_page_time]).toEqual([0, 3600]);
    expect(profile.static_keywords).toEqual([]);
    expect(profile.exempt_paths).toEqual([]);
    expect(profile.enabled).not.toContain('header_validation');
    expect(profile.enabled).not.toContain('logging');
    expect(options).toEqual({ pathRules: [], manifestPath: '.aiwaf/paths.json', injectionRiskMode: 'block',
      stores: expect.any(Object), storesFallback: 'csv', rateFailureMode: 'closed', trustedProxies: 'default' });
  });

  test('carries the regular options over', () => {
    const { profile, options } = nativeOptionsFromSettings({
      windowSec: 30, MAX_REQ: 10, FLOOD_REQ: 20, staticKeywords: ['.php'], AIWAF_HEADER_VALIDATION: true,
      AIWAF_REQUIRED_HEADERS: ['user-agent'], AIWAF_GEO_BLOCK_COUNTRIES: ['CN'], AIWAF_MIDDLEWARE_LOGGING: true,
      AIWAF_METHOD_POLICY_ENABLED: true, AIWAF_ALLOWED_METHODS: ['get', 'post'], HONEYPOT_FIELD: 'website',
      AIWAF_SQL_INJECTION_MODE: 'monitor', pathRules: [{ PREFIX: '/api/', DISABLE: ['honeypot'] }]
    });
    expect([profile.rate_window, profile.rate_max, profile.rate_flood]).toEqual([30, 10, 20]);
    expect(profile.enabled).toEqual(expect.arrayContaining(['header_validation', 'logging', 'geo_block']));
    expect(profile.required_headers).toEqual(['HTTP_USER_AGENT']);
    expect(profile.allowed_methods).toEqual(['GET', 'POST']);
    expect([profile.honeypot_field, profile.sql_mode, profile.geo_block]).toEqual(['website', 'monitor', ['CN']]);
    expect(options.pathRules).toEqual([{ PREFIX: '/api/', DISABLE: ['honeypot'] }]);
  });

  test('no required headers also means no quality score, as in the regular check', () => {
    expect(nativeOptionsFromSettings({ AIWAF_HEADER_VALIDATION: true }).profile.header_min_score).toBe(0);
    const perMethod = nativeOptionsFromSettings({ AIWAF_REQUIRED_HEADERS: { GET: ['accept'], DEFAULT: ['user-agent'] } });
    expect(perMethod.profile.required_headers).toEqual(['HTTP_USER_AGENT']);
    expect(outcome(perMethod, 'AIWAF_REQUIRED_HEADERS')).toBe('changed');
  });

  test('auto selection turns logging on unless it was set, as in the regular middleware', () => {
    expect(nativeOptionsFromSettings({ AIWAF_MIDDLEWARES: ['auto'] }).profile.enabled).toContain('logging');
    expect(nativeOptionsFromSettings({ AIWAF_MIDDLEWARES: ['auto'], AIWAF_MIDDLEWARE_LOGGING: false }).profile.enabled)
      .not.toContain('logging');
  });

  test('values outside the native limits are adjusted and reported', () => {
    const conversion = nativeOptionsFromSettings({ MAX_REQ: 10, FLOOD_REQ: 200000, AIWAF_MAX_PAGE_TIME: 0 });
    expect(conversion.profile.rate_flood).toBe(100000);
    expect(conversion.profile.max_page_time).toBeUndefined();
    expect(outcome(conversion, 'FLOOD_REQ')).toBe('changed');
    expect(outcome(conversion, 'AIWAF_MAX_PAGE_TIME')).toBe('not carried');
  });

  test('reports everything that does not carry over exactly', () => {
    const conversion = nativeOptionsFromSettings({
      MAX_REQ: 50, FLOOD_REQ: 40, AIWAF_EXEMPT_IPS: ['10.0.0.0/8', '203.0.113.4'], AIWAF_EXEMPT_PATHS: ['/up', '*.css'],
      AIWAF_MODEL_STORAGE: 'db', AIWAF_TRUSTED_PROXIES: ['cloudflare']
    });
    expect(conversion.profile.rate_max).toBe(40);
    expect(conversion.profile.exempt_ips).toEqual(['203.0.113.4']);
    expect(conversion.profile.exempt_paths).toEqual(['/up']);
    for (const [setting, expected] of [['MAX_REQ', 'carried'], ['AIWAF_EXEMPT_IPS', 'changed'], ['AIWAF_EXEMPT_PATHS', 'changed'],
      ['AIWAF_MODEL_STORAGE', 'not carried'], ['AIWAF_TRUSTED_PROXIES', 'carried'], ['stores', 'carried']]) {
      expect(outcome(conversion, setting)).toBe(expected);
    }
    expect(conversion.report()).toContain('AIWAF_MODEL_STORAGE');
    expect(outcome(nativeOptionsFromSettings({}), 'AIWAF_MODEL_STORAGE')).toBeUndefined();
  });
});

describe('the report matches the Python converter', () => {
  test('notes carried, changed and unreadable options, and counts the carried ones', () => {
    const conversion = nativeOptionsFromSettings({ MAX_REQ: 30, FLOOD_REQ: 'lots', staticKeywords: ['.php'], AIWAF_SQL_INJECTION_MODE: 'off' });
    expect(outcome(conversion, 'MAX_REQ')).toBe('carried');
    expect(outcome(conversion, 'staticKeywords')).toBe('carried');
    expect(outcome(conversion, 'AIWAF_SQL_INJECTION_MODE')).toBe('carried');
    expect(outcome(conversion, 'FLOOD_REQ')).toBe('not carried');
    expect(conversion.profile.rate_flood).toBe(200);
    expect(conversion.report()).toMatch(/^Native pipeline settings: 4 carried over unchanged\./);
    expect(conversion.differences().map(n => n.setting)).toEqual(['FLOOD_REQ']);
    expect(conversion.config).toEqual({ ...conversion.options, profile: conversion.profile });
  });

  test('log() warns only when something differs', () => {
    const logger = { warn: jest.fn(), info: jest.fn() };
    nativeOptionsFromSettings({}).log(logger);
    expect(logger.info).toHaveBeenCalledWith('Native pipeline settings: 1 carried over unchanged.');
    const conversion = nativeOptionsFromSettings({ AIWAF_MODEL_STORAGE: 'db' });
    conversion.log(logger);
    expect(logger.warn).toHaveBeenCalledTimes(1);
    conversion.notes.splice(0, conversion.notes.length);
    conversion.log(logger);
    expect(logger.info).toHaveBeenCalledWith('Native pipeline settings: 0 carried over unchanged.');
  });
});

describe('aiwaf({ AIWAF_NATIVE_PIPELINE: "from_settings" })', () => {
  test('starts the native pipeline from the regular options', async () => {
    const logger = { warn: jest.fn(), info: jest.fn() };
    const middleware = aiwaf({
      MAX_REQ: 1, FLOOD_REQ: 5,
      AIWAF_NATIVE_PIPELINE: { fromSettings: true, dataDir: temp(), manifestPath: path.join(temp(), 'none.json'), logger }
    });
    const native = await middleware.ready;
    const app = express();
    app.use(middleware);
    app.get('/items', (_req, res) => res.json({ ok: true }));
    expect([(await request(app).get('/items')).status, (await request(app).get('/items')).status]).toEqual([200, 429]);
    expect(logger.info).toHaveBeenCalledWith(expect.stringContaining('Native pipeline settings:'));
    native.close();
  });

  test('profile keys given with it win over converted ones', async () => {
    const middleware = aiwaf({ MAX_REQ: 1, FLOOD_REQ: 5, AIWAF_NATIVE_PIPELINE: { fromSettings: true, dataDir: temp(),
      manifestPath: path.join(temp(), 'none.json'), logger: { warn() {} }, profile: { rate_max: 3 } } });
    const native = await middleware.ready;
    const app = express();
    app.use(middleware);
    app.get('/items', (_req, res) => res.json({ ok: true }));
    const statuses = [];
    for (let i = 0; i < 4; i++) statuses.push((await request(app).get('/items')).status);
    expect(statuses).toEqual([200, 200, 200, 429]);
    native.close();
  });

  test('regular options still cannot be mixed with a native profile', () => {
    expect(() => aiwaf({ MAX_REQ: 1, AIWAF_NATIVE_PIPELINE: { profile: {} } })).toThrow(/do not mix legacy options/);
  });
});

describe('settings without differences always start natively', () => {
  // The automatic switch relies on it. Edge values the regular middleware tolerates.
  const EDGE = {
    WINDOW_SEC: [0, 0.5, -1, '10', 'abc'], MAX_REQ: [0, -1, '5', 2.5], FLOOD_REQ: [0, -1, 3, 1e6],
    AIWAF_MIN_FORM_TIME: [-1, '2'], AIWAF_MAX_PAGE_TIME: [0, -1], AIWAF_HEADER_QUALITY_MIN_SCORE: [-1, 2.5, 1e12],
    AIWAF_SQL_INJECTION_MODE: ['BLOCK', 'Monitor', 'observe', ''], AIWAF_PAYLOAD_MAX_BYTES: [0, 1e7, '1000', 2.5],
    staticKeywords: [1, 'a,b', []], AIWAF_GEO_BLOCK_COUNTRIES: [10, 'US,CA'], AIWAF_REQUIRED_HEADERS: [[], ['x-custom']],
    AIWAF_PATH_RULES: [[{ PREFIX: '/', RATE_LIMIT: { WINDOW: 0 } }], [{ PREFIX: '/', RATE_LIMIT: { MAX: -1 } }],
      [{ PREFIX: '/', RATE_LIMIT: { MAX: '5' } }], [{ PREFIX: '/', RATE_LIMIT: { FLOOD: 0 } }],
      [{ PREFIX: '/', RATE_LIMIT: { MAX: 100 } }], ['nope'], 'nope'],
    AIWAF_INJECTION_RISK_MODE: ['LOG', 'warn']
  };
  const starts = async conversion => {
    const { stores: _stores, ...options } = conversion.options;
    const middleware = await createNativeForTest({ ...options, dataDir: temp(), manifestPath: path.join(temp(), 'none.json'), profile: conversion.profile });
    middleware.close();
  };
  test('one setting at a time and in combinations', async () => {
    const cases = Object.entries(EDGE).flatMap(([name, values]) => values.map(value => ({ [name]: value })));
    let seed = 7;
    const pick = list => { seed = (seed * 16807) % 2147483647; return list[seed % list.length]; };
    for (let i = 0; i < 300; i++) {
      const options = {};
      for (let k = 0; k < 4; k++) { const name = pick(Object.keys(EDGE)); options[name] = pick(EDGE[name].concat([1, 10, 100])); }
      cases.push(options);
    }
    for (const options of cases) {
      const conversion = nativeOptionsFromSettings(options);
      if (conversion.differences().length) continue;
      await expect(starts(conversion)).resolves.toBeUndefined();
    }
  });

  test('values native can\'t take are reported, and equivalent ones carry over', () => {
    const conversion = nativeOptionsFromSettings({ MAX_REQ: 2.5, AIWAF_MIN_FORM_TIME: -1, AIWAF_REQUIRED_HEADERS: ['accept'],
      AIWAF_HEADER_QUALITY_MIN_SCORE: 2.5, AIWAF_SQL_INJECTION_MODE: 'Monitor', AIWAF_INJECTION_RISK_MODE: 'LOG', staticKeywords: 'a, b' });
    expect(conversion.profile).toMatchObject({ rate_max: 2, min_form_time: 0, header_min_score: 3, sql_mode: 'monitor', static_keywords: ['a', 'b'] });
    expect(conversion.options.injectionRiskMode).toBe('log');
    const refused = nativeOptionsFromSettings({ WINDOW_SEC: 0, AIWAF_PAYLOAD_MAX_BYTES: 0, AIWAF_INJECTION_RISK_MODE: 'warn',
      staticKeywords: 1, AIWAF_PATH_RULES: [{ PREFIX: '/a', RATE_LIMIT: { WINDOW: 0 }, DISABLE: ['geo_block'] }, 'nope'] });
    expect(refused.differences().map(n => n.setting)).toEqual(expect.arrayContaining(
      ['WINDOW_SEC', 'AIWAF_PAYLOAD_MAX_BYTES', 'AIWAF_INJECTION_RISK_MODE', 'staticKeywords', 'AIWAF_PATH_RULES']));
    expect(refused.options.pathRules).toEqual([{ PREFIX: '/a', DISABLE: ['geo_block'] }]);
    expect(nativeOptionsFromSettings({ AIWAF_PATH_RULES: 'nope' }).options.pathRules).toEqual([]);
  });
});
