// Store fallbacks and per-request work in the regular middleware's helpers.
const fs = require('fs');
const os = require('os');
const path = require('path');
const csvStore = require('../lib/csvStore');
const pathRules = require('../lib/pathRules');

const temp = () => fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-store-eff-'));

describe('reading the end of an append-only CSV', () => {
  const headers = ['id', 'created_at', 'path'];
  test('gives the same rows as reading the whole file, newest last', () => {
    const file = path.join(temp(), 'log.csv');
    for (let i = 1; i <= 3000; i++) csvStore.appendRow(file, headers, { id: i, created_at: String(i), path: i % 5 ? `/p${i}` : `/q,"${i}"` });
    const all = csvStore.readRows(file, headers);
    for (const limit of [1, 7, 2999, 3000, 5000]) expect(csvStore.readTailRows(file, headers, limit)).toEqual(all.slice(-limit));
    expect(csvStore.nextId(csvStore.readTailRows(file, headers, 1))).toBe(3001);
  });

  test('handles missing, empty and headerless files', () => {
    const dir = temp();
    expect(csvStore.readTailRows(path.join(dir, 'none.csv'), headers, 5)).toEqual([]);
    fs.writeFileSync(path.join(dir, 'empty.csv'), '');
    expect(csvStore.readTailRows(path.join(dir, 'empty.csv'), headers, 5)).toEqual([]);
    fs.writeFileSync(path.join(dir, 'bare.csv'), '1,a,/x\n2,b,/y\n');
    expect(csvStore.readTailRows(path.join(dir, 'bare.csv'), headers, 1)).toEqual([{ id: '2', created_at: 'b', path: '/y' }]);
    expect(csvStore.readTailRows(path.join(dir, 'bare.csv'), headers, 0)).toEqual([]);
  });
});

describe('the request log store without a database', () => {
  test('appends with increasing ids and returns the newest first', async () => {
    const dir = temp();
    jest.resetModules();
    process.env.AIWAF_REQUEST_LOGS_CSV_PATH = path.join(dir, 'request_logs.csv');
    jest.doMock('../utils/db', () => {
      const failing = () => { throw new Error('no database'); };
      failing.schema = { hasTable: async () => { throw new Error('no database'); } };
      return failing;
    });
    const store = require('../lib/requestLogStore');
    for (let i = 0; i < 5; i++) await store.insert({ timestamp: new Date(1e12 + i * 1000).toISOString(), ip: '1.1.1.1', path: `/p${i}` });
    const recent = await store.recent(2);
    expect(recent.map(row => row.path)).toEqual(['/p4', '/p3']);
    expect(recent.map(row => row.id)).toEqual(['5', '4']);
    jest.dontMock('../utils/db');
    delete process.env.AIWAF_REQUEST_LOGS_CSV_PATH;
  });
});

test('a failing database query falls back to the CSV store instead of throwing', async () => {
  jest.resetModules();
  const dir = temp();
  process.env.AIWAF_GEO_BLOCKED_COUNTRIES_CSV_PATH = path.join(dir, 'geo.csv');
  // The table exists, but queries on it fail, as when a database goes away.
  jest.doMock('../utils/db', () => {
    const query = () => ({ select: () => ({ orderBy: async () => { throw new Error('database is locked'); } }) });
    query.schema = { hasTable: async () => true };
    return query;
  });
  const geoStore = require('../lib/geoStore');
  await expect(geoStore.listBlockedCountries()).resolves.toEqual([]);
  jest.dontMock('../utils/db');
  delete process.env.AIWAF_GEO_BLOCKED_COUNTRIES_CSV_PATH;
});

test('route plans are built once per path and markers, and patterns are matched literally', () => {
  const rules = [{ PREFIX: '/api/', DISABLE: ['rate_limit'] }, { PREFIX: '/files/*.php', DISABLE: ['geo_block'] }];
  const plan = pathRules.createRoutePlan('/api/x', rules, {});
  expect(pathRules.createRoutePlan('/api/x', rules, {})).toBe(plan);
  expect(pathRules.createRoutePlan('/api/x', rules, { exemptMiddlewares: ['geo_block'] })).not.toBe(plan);
  expect(plan.shouldApply('rate_limit')).toBe(false);
  expect(pathRules.createRoutePlan('/x', 'not a list', {}).shouldApply('rate_limit')).toBe(true);
  expect(pathRules.isPathExempt('/static/app.css', ['/static/*.css'])).toBe(true);
  expect(pathRules.isPathExempt('/static/appXcss', ['/static/*.css'])).toBe(false);

});

test('the UUID route prefix is matched as literal text', async () => {
  const uuidDetector = require('../lib/uuidDetector');
  uuidDetector.init({ uuidRoutePrefix: '/u.x' });
  // A malformed UUID under the prefix is suspicious; "." in the prefix doesn't match any character.
  expect(await uuidDetector.isSuspicious({ path: '/u.x/not-a-uuid' })).toBe(true);
  expect(await uuidDetector.isSuspicious({ path: '/uXx/not-a-uuid' })).toBe(false);
  uuidDetector.init({ uuidRoutePrefix: '/user' });
  expect(await uuidDetector.isSuspicious({ path: '/user/not-a-uuid' })).toBe(true);
});
