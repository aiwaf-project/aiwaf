const fs = require('fs');
const path = require('path');
const modelStore = require('../lib/modelStore');
const db = require('../utils/db');

describe('modelStore', () => {
  it('saves and loads from file storage', async () => {
    const baseDir = path.join(process.cwd(), 'logs', 'test-model-store');
    fs.mkdirSync(baseDir, { recursive: true });
    const modelPath = path.join(baseDir, 'model.json');

    const payload = { nTrees: 1, sampleSize: 2, trees: [], metadata: { createdAt: 'now' } };
    await modelStore.save({ AIWAF_MODEL_STORAGE: 'file', AIWAF_MODEL_PATH: modelPath }, payload, payload.metadata);

    const loaded = await modelStore.load({ AIWAF_MODEL_STORAGE: 'file', AIWAF_MODEL_PATH: modelPath });
    expect(loaded.nTrees).toBe(1);
    expect(loaded.metadata.createdAt).toBe('now');
  });

  it('supports cache storage', async () => {
    const payload = { nTrees: 2, sampleSize: 3, trees: [], metadata: { createdAt: 'cache' } };
    await modelStore.save({ AIWAF_MODEL_STORAGE: 'cache', AIWAF_MODEL_CACHE_KEY: 'test-cache', AIWAF_MODEL_CACHE_TTL: 5 }, payload, payload.metadata);

    const loaded = await modelStore.load({ AIWAF_MODEL_STORAGE: 'cache', AIWAF_MODEL_CACHE_KEY: 'test-cache' });
    expect(loaded.nTrees).toBe(2);
  });
});

afterAll(async () => {
  await db.destroy();
});
