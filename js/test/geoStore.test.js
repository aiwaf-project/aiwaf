const geoStore = require('../lib/geoStore');

describe('geoStore module contract', () => {
  test('exports the complete country storage API', () => {
    for (const name of ['initialize', 'addBlockedCountry', 'removeBlockedCountry', 'listBlockedCountries', 'isBlockedCountry']) {
      expect(geoStore[name]).toEqual(expect.any(Function));
    }
  });
});
