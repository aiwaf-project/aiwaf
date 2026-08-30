const modelStore = require('../lib/modelStore');
const db = require('../utils/db');

describe('modelStore db backend', () => {
  afterAll(async () => {
    try {
      await db('model_artifacts').del();
    } catch (err) {
      // noop
    }
    await db.destroy();
  });

  it('saves and loads model payload from db', async () => {
    const payload = { nTrees: 5, sampleSize: 8, trees: [], metadata: { createdAt: 'db' } };
    await modelStore.save({ AIWAF_MODEL_STORAGE: 'db' }, payload, payload.metadata);

    const loaded = await modelStore.load({ AIWAF_MODEL_STORAGE: 'db' });
    expect(loaded.nTrees).toBe(5);
    expect(loaded.metadata.createdAt).toBe('db');
  });
});
