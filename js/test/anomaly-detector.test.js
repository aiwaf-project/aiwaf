describe('anomalyDetector behavior', () => {
  beforeEach(() => {
    jest.resetModules();
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
});
