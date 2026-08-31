const featureUtils = require('../lib/featureUtils');

describe('featureUtils', () => {
  afterEach(() => featureUtils.cleanup());

  test('tracks request timing and clears its in-memory state', () => {
    jest.spyOn(Date, 'now').mockReturnValueOnce(1000).mockReturnValueOnce(1025);
    const req = {};
    featureUtils.markRequestStart(req);
    expect(featureUtils.getResponseTime(req)).toBe(25);
    featureUtils.recordRequest('203.0.113.8', 404);
    expect(featureUtils.get404Count('203.0.113.8')).toBe(1);
    featureUtils.cleanup();
    expect(featureUtils.get404Count('203.0.113.8')).toBe(0);
    Date.now.mockRestore();
  });
});
