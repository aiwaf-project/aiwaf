describe('requestLogStore', () => {
  beforeEach(() => jest.resetModules());

  test('exposes an idempotent initialize operation', async () => {
    const db = jest.fn();
    db.schema = { hasTable: jest.fn(async () => true) };
    db.fn = { now: jest.fn() };
    jest.doMock('../utils/db', () => db);
    const store = require('../lib/requestLogStore');
    await expect(store.initialize()).resolves.toBeUndefined();
    await expect(store.initialize()).resolves.toBeUndefined();
    expect(db.schema.hasTable).toHaveBeenCalledTimes(1);
  });
});
