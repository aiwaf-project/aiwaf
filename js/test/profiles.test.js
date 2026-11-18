const profiles = require('../lib/profiles');
const { normalizeSettings } = require('../lib/settingsCompat');

process.env.NODE_ENV = 'test';

async function statuses(options) {
  const express = require('express');
  const request = require('supertest');
  const aiwaf = require('../lib/wafMiddleware');
  const app = express();
  app.use(aiwaf({ AIWAF_NATIVE_PIPELINE: false, MAX_REQ: 1, FLOOD_REQ: 50, AIWAF_PATH_MANIFEST: require('path').join(require('os').tmpdir(), 'aiwaf-profile-none.json'), ...options }));
  app.get('/items', (_req, res) => res.json({ ok: true }));
  return [(await request(app).get('/items')).status, (await request(app).get('/items')).status];
}

describe('AIWAF_PROFILE', () => {
  beforeAll(() => require('../lib/wasmChecks').load());
  afterEach(() => {
    profiles.activate('balanced');
    delete process.env.AIWAF_PROFILE;
    jest.restoreAllMocks();
  });

  test('strict halves the rate limits and turns the method policy on, unless set explicitly', () => {
    expect(normalizeSettings({}).MAX_REQ).toBe(100);
    const strict = normalizeSettings({ AIWAF_PROFILE: 'strict' });
    expect([strict.MAX_REQ, strict.FLOOD_REQ, strict.AIWAF_METHOD_POLICY_ENABLED]).toEqual([50, 100, true]);
    expect(normalizeSettings({ AIWAF_PROFILE: 'strict', MAX_REQ: 7, AIWAF_METHOD_POLICY_ENABLED: false }))
      .toMatchObject({ MAX_REQ: 7, AIWAF_METHOD_POLICY_ENABLED: false });
    process.env.AIWAF_PROFILE = 'monitor';
    expect(normalizeSettings({})).toMatchObject({ AIWAF_PROFILE: 'monitor', AIWAF_SQL_INJECTION_MODE: 'monitor', AIWAF_INJECTION_RISK_MODE: 'log' });
    expect(() => normalizeSettings({ AIWAF_PROFILE: 'loose' })).toThrow('AIWAF_PROFILE must be one of strict, balanced, monitor');
  });

  test('monitor lets a request it would block through, and says so', async () => {
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
    jest.spyOn(console, 'info').mockImplementation(() => {});
    expect(await statuses({})).toEqual([200, 429]);
    expect(await statuses({ AIWAF_PROFILE: 'monitor' })).toEqual([200, 200]);
    expect(warn.mock.calls.map(call => call[0]).join('\n')).toMatch(/AIWAF monitor: would block GET \/items from \S+ with 429 \(rate_limit\)/);
  });

  test('monitor adds no IP to the blocklist', async () => {
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
    profiles.activate('monitor');
    expect(profiles.skipBlocklist('203.0.113.7', 'flood')).toBe(true);
    expect(await require('../lib/blocklistManager').block('203.0.113.7', 'flood')).toBe(false);
    expect(warn).toHaveBeenCalledWith('AIWAF monitor: would add 203.0.113.7 to the blocklist (flood)');
    profiles.activate('balanced');
    expect(profiles.skipBlocklist('203.0.113.7', 'flood')).toBe(false);
  });

  test('monitor runs the regular middleware, so an explicit native pipeline is refused', () => {
    jest.spyOn(console, 'warn').mockImplementation(() => {});
    const aiwaf = require('../lib/wafMiddleware');
    expect(() => aiwaf({ AIWAF_PROFILE: 'monitor', AIWAF_NATIVE_PIPELINE: 'from_settings' })).toThrow('leave AIWAF_NATIVE_PIPELINE unset');
    profiles.activate('balanced');
    process.env.AIWAF_PROFILE = 'monitor';
    expect(profiles.resolve()).toBe('monitor');
    expect(profiles.monitoring()).toBe(false);  // The profile the app chose wins over the environment.
  });
});
