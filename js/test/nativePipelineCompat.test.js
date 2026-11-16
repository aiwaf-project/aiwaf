// The native adapter must keep working with older published aiwaf-wasm builds.
jest.mock('../lib/wasmAdapter', () => ({
  createRequestPipeline: async profile => {
    // Older engines reject monitor mode when they're constructed.
    if (profile.sql_mode === 'monitor') throw new Error('Unsupported native policy');
    return {
    // An older engine: no supports() method.
    native: { begin: () => '{"context":"1","status":0,"reason":""}', finish: () => '{}', free: jest.fn(), diagnostics: () => '{}' },
    setModel: () => {}
    };
  }
}));
const fs = require('fs');
const os = require('os');
const path = require('path');
const createNative = require('../lib/nativePipeline');

const temp = () => fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-native-compat-'));

test('path rules without RATE_LIMIT work with an older engine', async () => {
  const middleware = await createNative({ dataDir: temp(), persistBlocks: false, profile: { enabled: [] },
    manifestPath: path.join(temp(), 'missing.json'), pathRules: [{ PREFIX: '/health/', DISABLE: ['rate_limit'] }] });
  middleware.close();
});

test('RATE_LIMIT rules need a newer engine and fail at startup', async () => {
  await expect(createNative({ dataDir: temp(), persistBlocks: false, profile: { enabled: [] },
    manifestPath: path.join(temp(), 'missing.json'), pathRules: [{ PREFIX: '/api/', RATE_LIMIT: { MAX: 1 } }] }))
    .rejects.toThrow(/needs a newer aiwaf-wasm/);
});

test('monitor mode on an older engine fails at startup with an upgrade message', async () => {
  await expect(createNative({ dataDir: temp(), persistBlocks: false, profile: { sql_mode: 'monitor' },
    manifestPath: path.join(temp(), 'missing.json') })).rejects.toThrow(/sql_mode "monitor" needs a newer aiwaf-wasm/);
});

test('uuidResolver on an older engine fails at startup with an upgrade message', async () => {
  await expect(createNative({ dataDir: temp(), persistBlocks: false, profile: { enabled: [] },
    manifestPath: path.join(temp(), 'missing.json'), uuidResolver: async () => true })).rejects.toThrow(/uuidResolver needs a newer aiwaf-wasm/);
});

test('stores on an older engine fail at startup, or fall back to dataDir when asked', async () => {
  const stores = { isBlocked() {}, isExempted() {}, topKeywords() {}, blockedCountries() {}, block() {}, learn() {} };
  const base = { dataDir: temp(), profile: { enabled: [] }, manifestPath: path.join(temp(), 'missing.json'), stores };
  await expect(createNative(base)).rejects.toThrow(/stores needs a newer aiwaf-wasm/);
  const logger = { warn: jest.fn() };
  const middleware = await createNative({ ...base, storesFallback: 'csv', logger });
  expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining('kept in dataDir'));
  expect(fs.existsSync(path.join(base.dataDir, 'native_blocks.csv'))).toBe(true);
  middleware.close();
});

test('a rate counter on an older engine fails at startup, or counts per worker when asked', async () => {
  const base = { dataDir: temp(), persistBlocks: false, profile: { enabled: [] }, manifestPath: path.join(temp(), 'missing.json'),
    rateCounter: { count: async () => 1 } };
  await expect(createNative(base)).rejects.toThrow(/rateCounter needs a newer aiwaf-wasm/);
  const logger = { warn: jest.fn() };
  const middleware = await createNative({ ...base, rateCounterFallback: 'memory', logger });
  expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining('counted in each worker'));
  middleware.close();
});
