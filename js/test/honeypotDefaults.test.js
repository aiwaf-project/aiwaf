const { normalizeSettings } = require('../lib/settingsCompat');
const { nativeOptionsFromSettings } = require('../lib/nativeSettings');
const detector = require('../lib/honeypotDetector');

describe('login timing defaults shared by regular and native', () => {
  const previous = process.env.AIWAF_LOGIN_PATH_PREFIXES;
  beforeEach(() => { delete process.env.AIWAF_LOGIN_PATH_PREFIXES; });
  afterAll(() => {
    if (previous === undefined) delete process.env.AIWAF_LOGIN_PATH_PREFIXES;
    else process.env.AIWAF_LOGIN_PATH_PREFIXES = previous;
  });
  afterEach(() => jest.restoreAllMocks());

  test.each([undefined, [], ['/custom/']])('respects default and explicit prefixes: %p', prefixes => {
    const raw = { AIWAF_MIN_FORM_TIME: 5 };
    if (prefixes !== undefined) raw.AIWAF_LOGIN_PATH_PREFIXES = prefixes;
    const opts = normalizeSettings(raw);
    const profile = nativeOptionsFromSettings(raw).profile;
    expect(profile.login_prefixes).toEqual(opts.AIWAF_LOGIN_PATH_PREFIXES);
    if (prefixes === undefined) expect(profile.login_prefixes).toContain('/login/');
    detector.init(opts);
    let now = 1000;
    jest.spyOn(Date, 'now').mockImplementation(() => now);
    detector.evaluate({ method: 'GET' }, '198.51.100.10', '/form');
    now += 200;
    expect(detector.evaluate({ method: 'POST' }, '198.51.100.10', '/login/').triggered)
      .toBe(prefixes !== undefined);
    expect(detector.evaluate({ method: 'POST' }, '198.51.100.10', '/ordinary/').triggered).toBe(true);
  });

  test('direct detector initialization restores defaults after an explicit override', () => {
    detector.init({ AIWAF_LOGIN_PATH_PREFIXES: [] });
    detector.init({ AIWAF_MIN_FORM_TIME: 5 });
    let now = 2000;
    jest.spyOn(Date, 'now').mockImplementation(() => now);
    detector.evaluate({ method: 'GET' }, '198.51.100.11', '/form');
    now += 200;
    expect(detector.evaluate({ method: 'POST' }, '198.51.100.11', '/login/').triggered).toBe(false);
  });
});
