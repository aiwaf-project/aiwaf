const middlewareLogger = require('../lib/middlewareLogger');

describe('middlewareLogger module contract', () => {
  test('exports lifecycle, attachment, and block helpers', () => {
    expect(middlewareLogger.init).toEqual(expect.any(Function));
    expect(middlewareLogger.attach).toEqual(expect.any(Function));
    expect(middlewareLogger.markBlocked).toEqual(expect.any(Function));
  });
});
