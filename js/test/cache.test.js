describe('cache factory', () => {
  beforeEach(() => jest.resetModules());

  test('creates memory and Redis cache implementations', () => {
    const createClient = jest.fn(() => ({ kind: 'redis' }));
    jest.doMock('redis', () => ({ createClient }));
    const cache = require('../utils/cache');
    expect(cache(false).constructor.name).toBe('NodeCache');
    expect(cache(true)).toEqual({ kind: 'redis' });
    expect(createClient).toHaveBeenCalledTimes(1);
  });
});
