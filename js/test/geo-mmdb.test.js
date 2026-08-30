const fs = require('fs');
const path = require('path');

describe('geoBlocker MMDB lookup', () => {
  let mmdbPath;

  beforeEach(() => {
    jest.resetModules();
    const baseDir = path.join(process.cwd(), 'logs', 'test-mmdb');
    fs.mkdirSync(baseDir, { recursive: true });
    mmdbPath = path.join(baseDir, 'test.mmdb');
    fs.writeFileSync(mmdbPath, '');

    jest.doMock('maxmind', () => ({
      openSync: () => ({
        get: () => ({ country: { iso_code: 'US' } })
      })
    }));

    jest.doMock('../lib/geoStore', () => ({
      initialize: jest.fn(async () => {}),
      isBlockedCountry: jest.fn(async () => false)
    }));
  });

  afterEach(() => {
    try {
      fs.rmSync(path.dirname(mmdbPath), { recursive: true, force: true });
    } catch (err) {
      // noop
    }
  });

  it('blocks based on MMDB lookup', async () => {
    let result;
    jest.isolateModules(() => {
      const geoBlocker = require('../lib/geoBlocker');
      geoBlocker.init({
        AIWAF_GEO_BLOCK_ENABLED: true,
        AIWAF_GEO_BLOCK_COUNTRIES: ['US'],
        AIWAF_GEO_MMDB_PATH: mmdbPath
      });
      result = geoBlocker.check({
        headers: { 'x-forwarded-for': '203.0.113.10' }
      });
    });
    result = await result;

    expect(result.blocked).toBe(true);
    expect(result.country).toBe('US');
  });

  it('skips blocking when allow/block lists are empty', async () => {
    let result;
    jest.doMock('../lib/geoStore', () => ({
      initialize: jest.fn(async () => {}),
      isBlockedCountry: jest.fn(async () => true)
    }));

    jest.isolateModules(() => {
      const geoBlocker = require('../lib/geoBlocker');
      geoBlocker.init({
        AIWAF_GEO_BLOCK_ENABLED: true,
        AIWAF_GEO_BLOCK_COUNTRIES: [],
        AIWAF_GEO_ALLOW_COUNTRIES: [],
        AIWAF_GEO_MMDB_PATH: mmdbPath
      });
      result = geoBlocker.check({
        headers: { 'x-forwarded-for': '203.0.113.10' }
      });
    });
    result = await result;

    expect(result.blocked).toBe(false);
    expect(result.country).toBe('');
  });
});
