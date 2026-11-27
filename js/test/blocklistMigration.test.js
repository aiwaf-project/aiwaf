process.env.NODE_ENV = 'test';

describe('blocklist schema migration', () => {
  test('upgrades a legacy SQLite table without losing existing blocks', async () => {
    jest.resetModules();
    jest.dontMock('../utils/db');
    const db = require('../utils/db');
    await db.schema.createTable('blocked_ips', table => {
      table.increments('id').primary();
      table.string('ip_address').unique().notNullable();
      table.string('reason');
      table.timestamp('blocked_at');
    });
    await db('blocked_ips').insert({
      ip_address: '203.0.113.70', reason: 'legacy scanner', blocked_at: new Date().toISOString()
    });

    const manager = require('../lib/blocklistManager');
    await expect(manager.initialize()).resolves.toBe(true);
    expect(await db.schema.hasColumn('blocked_ips', 'reputation_reason')).toBe(true);
    expect(await db.schema.hasColumn('blocked_ips', 'extended_request_info')).toBe(true);
    expect(await manager.isBlocked('203.0.113.70')).toBe(true);
    const result = await manager.migrateLegacy();
    expect(result).toEqual({ total: 1, changed: 1 });
    const info = await manager.getBlockInfo('203.0.113.70');
    expect(info.permanent).toBe(true);
    expect(info.reputation_reason).toBe('legacy_blocklist');
    await db.destroy();
  });

  test('rows an earlier version marked legacy_blacklist still convert', async () => {
    jest.resetModules();
    jest.dontMock('../utils/db');
    const db = require('../utils/db');
    const manager = require('../lib/blocklistManager');
    await manager.initialize();
    await db('blocked_ips').insert({
      ip_address: '203.0.113.71', reason: 'old', reputation_reason: 'legacy_blacklist',
      blocked_at: new Date().toISOString(), permanent: true
    });
    await db('blocked_ips').insert({
      ip_address: '203.0.113.72', reason: 'sql', reputation_reason: 'sql_injection',
      blocked_at: new Date().toISOString(), permanent: true
    });
    const result = await manager.migrateLegacy({ duration: 3600, now: 1000 });
    expect(result.changed).toBe(1);
    expect((await manager.getBlockInfo('203.0.113.71')).reputation_reason).toBe('legacy_blocklist_converted');
    expect((await manager.getBlockInfo('203.0.113.72')).reputation_reason).toBe('sql_injection');
    await db.destroy();
  });
});
