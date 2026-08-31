const {
  isAutoSelection,
  planEnabledMiddlewares,
  shouldEnableGeo,
  shouldEnableLogging,
  shouldEnableUuidTamper
} = require('../lib/middlewarePlan');

describe('middlewarePlan', () => {
  test('recognizes all auto-selection forms', () => {
    expect(isAutoSelection('all')).toBe(true);
    expect(isAutoSelection(['AIWAF.ALL'])).toBe(true);
    expect(isAutoSelection('logging,geo_block')).toBe(false);
    expect(isAutoSelection(null)).toBe(false);
  });

  test('derives automatic feature availability', () => {
    expect(shouldEnableLogging('')).toBe(true);
    expect(shouldEnableLogging('/var/log/access.log')).toBe(false);
    expect(shouldEnableGeo({ geoEnabledFlag: true })).toBe(true);
    expect(shouldEnableGeo({ staticBlockCountries: ['US'] })).toBe(true);
    expect(shouldEnableGeo({ dynamicBlockCountries: ['CA'] })).toBe(true);
    expect(shouldEnableGeo({})).toBe(false);
    expect(shouldEnableUuidTamper({})).toBe(true);
    expect(shouldEnableUuidTamper({ hasUuidRoutes: false })).toBe(false);
  });

  test('plans automatic, explicit, and disabled middleware sets', () => {
    const available = ['logging', 'geo_block', 'uuid_tamper', 'header_validation'];
    expect(planEnabledMiddlewares({
      orderedAvailable: available,
      requested: 'all',
      disabled: ['header_validation'],
      accessLog: '/var/log/access.log',
      geoEnabledFlag: false,
      staticBlockCountries: [],
      dynamicBlockCountries: [],
      hasUuidRoutes: false
    })).toEqual(new Set());

    expect(planEnabledMiddlewares({
      orderedAvailable: available,
      requested: ['logging', 'unknown'],
      disabled: []
    })).toEqual(new Set(['logging']));

    expect(planEnabledMiddlewares({
      orderedAvailable: available,
      requested: null,
      disabled: ['geo_block']
    })).toEqual(new Set(['logging', 'uuid_tamper', 'header_validation']));
  });
});
