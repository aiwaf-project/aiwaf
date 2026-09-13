describe('anomalyDetector behavior', () => {
  beforeEach(() => {
    jest.resetModules();
  });
  afterEach(() => {
    jest.dontMock('../lib/wasmAdapter');
  });

  it('flags scanning paths and computes behavior stats', () => {
    jest.isolateModules(() => {
      const anomalyDetector = require('../lib/anomalyDetector');
      const stats = anomalyDetector.analyzeRecentBehavior([
        { timestamp: Date.now() - 1000, path: '/wp-admin', status: 404 },
        { timestamp: Date.now() - 2000, path: '/.env', status: 404 },
        { timestamp: Date.now() - 3000, path: '/safe', status: 200 }
      ]);

      expect(anomalyDetector.isScanningPath('/wp-admin')).toBe(true);
      expect(stats.scanning_404s).toBeGreaterThan(0);
      expect(typeof stats.should_block).toBe('boolean');
    });
  });

  it('uses the WASM recent-behavior result and sends explicit millisecond timestamps', async () => {
    const rustStats = {
      avg_kw_hits: 1,
      max_404s: 1,
      avg_burst: 2,
      total_requests: 2,
      scanning_404s: 1,
      legitimate_404s: 0,
      should_block: false
    };
    const analyze = jest.fn(async () => rustStats);
    jest.doMock('../lib/wasmAdapter', () => ({ analyzeRecentBehavior: analyze }));
    const anomalyDetector = require('../lib/anomalyDetector');
    const rows = [
      { timestamp: 10000, path: '/wp-admin', status: 404 },
      { timestamp: 12000, path: '/safe', status: 200 }
    ];

    expect(await anomalyDetector.analyzeRecentBehaviorAccelerated(rows)).toBe(rustStats);
    expect(analyze).toHaveBeenCalledWith([
      expect.objectContaining({ timestamp_ms: 10000, kw_check: true }),
      expect.objectContaining({ timestamp_ms: 12000, kw_check: true })
    ], expect.any(Array));
  });

  it('falls back to JS and counts inclusive burst boundaries when WASM is unavailable', async () => {
    jest.doMock('../lib/wasmAdapter', () => ({ analyzeRecentBehavior: jest.fn(async () => null) }));
    const anomalyDetector = require('../lib/anomalyDetector');
    const rows = [
      { timestamp: 1, path: '/safe', status: 200 },
      { timestamp: 10001, path: '/safe', status: 200 },
      { timestamp: 20002, path: '/safe', status: 200 }
    ];
    const stats = await anomalyDetector.analyzeRecentBehaviorAccelerated(rows);
    expect(stats.avg_burst).toBe(5 / 3);
    expect(stats).toEqual(anomalyDetector.analyzeRecentBehavior(rows));
  });

  it('keeps JS scan classification for discovery paths with older WASM releases', async () => {
    const analyze = jest.fn(async () => ({ scanning_404s: 1, should_block: true }));
    jest.doMock('../lib/wasmAdapter', () => ({ analyzeRecentBehavior: analyze }));
    const anomalyDetector = require('../lib/anomalyDetector');
    const rows = [{ timestamp: 10000, path: '/robots.txt', status: 404 }];

    const stats = await anomalyDetector.analyzeRecentBehaviorAccelerated(rows);
    expect(stats.scanning_404s).toBe(0);
    expect(analyze).not.toHaveBeenCalled();
  });

  it('disables model when logs are insufficient', async () => {
    jest.doMock('../lib/requestLogStore', () => ({
      recent: jest.fn(async () => [])
    }));
    jest.doMock('../lib/modelStore', () => ({
      load: jest.fn(async () => null)
    }));

    const anomalyDetector = require('../lib/anomalyDetector');
    await anomalyDetector.init({ AIWAF_MIN_AI_LOGS: 10 });

    const info = anomalyDetector.getModelInfo();
    expect(info.aiLogsSufficient).toBe(false);
    expect(anomalyDetector.hasModel()).toBe(false);
  });

  it('restores a persisted WASM model instead of parsing it as a JS forest', async () => {
    const model = { __aiwafWasm: true, isAnomaly: jest.fn(() => false) };
    const restore = jest.fn(async () => model);
    jest.doMock('../lib/wasmAdapter', () => ({ createIsolationForestFromJSON: restore }));
    jest.doMock('../lib/modelStore', () => ({ load: jest.fn(async () => ({
      model_type: 'aiwaf_wasm.IsolationForest',
      model_state: { n_estimators: 2, trees: [] },
      metadata: { samplesCount: 20 }
    })) }));
    const anomalyDetector = require('../lib/anomalyDetector');
    await anomalyDetector.init({ AIWAF_MIN_AI_LOGS: 0 });

    expect(restore).toHaveBeenCalledWith({ n_estimators: 2, trees: [] });
    expect(anomalyDetector.hasModel()).toBe(true);
  });

  it('trains the active model through the public API', async () => {
    jest.doMock('../lib/requestLogStore', () => ({
      recent: jest.fn(async () => [{ id: 1, created_at: new Date().toISOString() }])
    }));
    jest.doMock('../lib/modelStore', () => ({ load: jest.fn(async () => null) }));
    const anomalyDetector = require('../lib/anomalyDetector');
    await anomalyDetector.init({ AIWAF_MIN_AI_LOGS: 1 });
    anomalyDetector.train([[1, 0, 0, 1, 1, 0], [2, 0, 0, 1, 1, 0]]);
    expect(anomalyDetector.hasModel()).toBe(true);
  });
});
