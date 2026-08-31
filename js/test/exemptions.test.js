const exemptions = require('../lib/exemptions');

describe('exemptions module contract', () => {
  test('exports initialization and decision helpers', () => {
    expect(exemptions).toEqual(expect.objectContaining({
      init: expect.any(Function),
      isExemptRequest: expect.any(Function),
      shouldSkipKeyword: expect.any(Function)
    }));
  });
});
