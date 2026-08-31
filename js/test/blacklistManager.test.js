describe('blacklistManager', () => {
  beforeEach(() => jest.resetModules());

  test('exposes an idempotent initialize operation', async () => {
    const db = jest.fn();
    db.schema = { hasTable: jest.fn(async () => true) };
    db.fn = { now: jest.fn() };
    jest.doMock('../utils/db', () => db);
    const manager = require('../lib/blacklistManager');
    await expect(manager.initialize()).resolves.toBe(true);
    await expect(manager.initialize()).resolves.toBe(true);
    expect(db.schema.hasTable).toHaveBeenCalledTimes(1);
  });
});
