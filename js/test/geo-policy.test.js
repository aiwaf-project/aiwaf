const { normalizeCountryList, evaluateGeoPolicy } = require('../lib/geoPolicy');

describe('geo policy parity helpers', () => {
  test('normalizes country lists', () => {
    expect([...normalizeCountryList([' us ', 'ca', ''])].sort()).toEqual(['CA', 'US']);
    expect([...normalizeCountryList('gb')]).toEqual(['GB']);
  });

  test('allowlist takes precedence over block lists', () => {
    expect(evaluateGeoPolicy({
      country: 'ca',
      allowCountries: ['US'],
      blockCountries: ['CA'],
      dynamicBlocked: []
    })).toEqual({ shouldBlock: true, country: 'CA', reason: 'Geo blocked: CA' });
  });

  test('block and dynamic lists deny when no allowlist exists', () => {
    expect(evaluateGeoPolicy({
      country: 'de',
      allowCountries: [],
      blockCountries: [],
      dynamicBlocked: ['DE']
    }).shouldBlock).toBe(true);
  });
});
