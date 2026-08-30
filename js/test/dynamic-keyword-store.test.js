const fs = require('fs');
const path = require('path');

function makeFailingDbMock() {
  const db = function dbCall() {
    return {
      where() {
        return {
          first: async () => { throw new Error('db down'); },
          update: async () => { throw new Error('db down'); },
          del: async () => { throw new Error('db down'); }
        };
      },
      insert: async () => { throw new Error('db down'); },
      select() {
        return {
          orderBy: async () => { throw new Error('db down'); },
          limit: async () => { throw new Error('db down'); }
        };
      }
    };
  };

  db.schema = {
    hasTable: async () => { throw new Error('db down'); },
    createTable: async () => { throw new Error('db down'); }
  };
  db.fn = { now: () => new Date() };
  return db;
}

function setupCsvPath(prefix) {
  const baseDir = path.join(process.cwd(), 'logs', 'test-dk-fallback', prefix);
  fs.mkdirSync(baseDir, { recursive: true });
  process.env.AIWAF_DYNAMIC_KEYWORDS_CSV_PATH = path.join(baseDir, 'dynamic_keywords.csv');
  return baseDir;
}

describe('dynamicKeywordStore CSV fallback', () => {
  let baseDir;

  beforeEach(() => {
    jest.resetModules();
    baseDir = setupCsvPath(`run-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    jest.doMock('../utils/db', () => makeFailingDbMock());
  });

  afterEach(() => {
    try {
      fs.rmSync(baseDir, { recursive: true, force: true });
    } catch (err) {
      // noop
    }
  });

  it('increments and lists keywords', async () => {
    const store = require('../lib/dynamicKeywordStore');

    await store.increment('Admin');
    await store.increment('Admin');
    await store.increment('login');

    const rows = await store.list(10);
    const admin = rows.find(row => row.keyword === 'admin');

    expect(admin).toBeDefined();
    expect(Number(admin.count)).toBe(2);
  });

  it('clears keywords', async () => {
    const store = require('../lib/dynamicKeywordStore');

    await store.increment('test');
    const cleared = await store.clear();
    expect(cleared).toBe(1);
    expect((await store.list(10)).length).toBe(0);
  });
});
