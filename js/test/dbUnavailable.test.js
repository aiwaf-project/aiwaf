const fs = require('fs');
const os = require('os');
const path = require('path');

// npm 12 skips sqlite3's install script unless the app approves it, so its native binary is
// missing. aiwaf must still load, say how to fix it, and keep its data in the CSV files.
function withoutSqlite(env = {}) {
  jest.resetModules();
  jest.doMock('sqlite3', () => { throw new Error("Could not locate the bindings file 'node_sqlite3.node'"); });
  const saved = {};
  for (const [key, value] of Object.entries(env)) {
    saved[key] = process.env[key];
    process.env[key] = value;
  }
  return () => {
    for (const [key, value] of Object.entries(saved)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    jest.dontMock('sqlite3');
    jest.resetModules();
  };
}

test('aiwaf loads without the sqlite3 binary', () => {
  const restore = withoutSqlite();
  try {
    expect(typeof require('../index')).toBe('function');
    expect(require('../utils/db').available()).toBe(false);
  } finally {
    restore();
  }
});

test('using the database explains the fix instead of blaming a missing package', () => {
  const restore = withoutSqlite();
  try {
    const db = require('../utils/db');
    let error;
    try { db('blocked_ips'); } catch (err) { error = err; }
    expect(error).toBeInstanceOf(db.SqliteUnavailableError);
    expect(error.code).toBe('AIWAF_SQLITE_UNAVAILABLE');
    expect(error.message).toContain('npm approve-scripts sqlite3 && npm rebuild sqlite3');
    expect(error.message).not.toContain('npm install sqlite3');
    expect(() => db.schema).toThrow(db.SqliteUnavailableError);
    expect(error.cause.message).toMatch(/node_sqlite3\.node/);
  } finally {
    restore();
  }
});

test('the warning is printed once, outside tests', () => {
  const restore = withoutSqlite({ NODE_ENV: 'production' });
  const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
  try {
    const db = require('../utils/db');
    expect(db.available()).toBe(false);
    expect(() => db('blocked_ips')).toThrow();
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0][0]).toMatch(/^\[aiwaf\] .*npm approve-scripts sqlite3/);
  } finally {
    warn.mockRestore();
    restore();
  }
});

test('inspecting the database object does not try to connect', () => {
  const restore = withoutSqlite();
  try {
    const db = require('../utils/db');
    expect(() => require('util').inspect(db)).not.toThrow();
    expect(db.then).toBeUndefined();
    expect('schema' in db).toBe(false);
  } finally {
    restore();
  }
});

test('the blocklist keeps working in its CSV file', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-nosqlite-'));
  const csv = path.join(dir, 'blocked_ips.csv');
  const restore = withoutSqlite({ AIWAF_BLOCKED_IPS_CSV_PATH: csv });
  try {
    const blocklist = require('../lib/blocklistManager');
    await blocklist.block('203.0.113.5', 'test');
    expect(await blocklist.isBlocked('203.0.113.5')).toBe(true);
    expect(await blocklist.isBlocked('203.0.113.6')).toBe(false);
    expect(fs.readFileSync(csv, 'utf8')).toContain('203.0.113.5');
  } finally {
    restore();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('available() reports other connection failures instead of hiding them', () => {
  jest.resetModules();
  jest.doMock('knex', () => () => { throw new Error('bad knex config'); });
  try {
    const db = require('../utils/db');
    expect(() => db.available()).toThrow('bad knex config');
  } finally {
    jest.dontMock('knex');
    jest.resetModules();
  }
});

test('with the binary, the database connects on first use', async () => {
  jest.resetModules();
  const db = require('../utils/db');
  expect(db.available()).toBe(true);
  const rows = await db.raw('select 1 as one');
  expect(rows[0].one).toBe(1);
});
