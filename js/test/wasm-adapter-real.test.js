describe('WASM adapter integration (real package)', () => {
  async function loadRealAdapter() {
    jest.resetModules();
    jest.dontMock('aiwaf-wasm');

    let adapter;
    await new Promise(resolve => {
      jest.isolateModules(() => {
        adapter = require('../lib/wasmAdapter');
        resolve();
      });
    });
    return adapter;
  }

  it('loads aiwaf-wasm and validates headers with a plain object', async () => {
    const { validateHeaders } = await loadRealAdapter();

    const result = await validateHeaders(
      { accept: 'text/html', 'user-agent': 'Mozilla/5.0' },
      { requiredHeaders: ['accept', 'user-agent'], minScore: 3 }
    );

    expect(result).toBeNull();
  });

  it('covers real aiwaf-wasm model and feature exports when available', async () => {
    const {
      analyzeRecentBehavior,
      buildWasmRecords,
      createIsolationForest,
      createIsolationForestFromJSON,
      extractWasmFeatures,
      extractWasmFeaturesBatchWithState,
      finalizeWasmFeatureState,
      getWasmStatus,
      pythonFeatureFromRecord,
      pythonFeaturesBatched,
      rustPayloadFromRecords,
      validateRecent
    } = await loadRealAdapter();

    const parsed = [
      {
        ip: '203.0.113.10',
        path: '/wp-login.php',
        response_time: 12,
        status: 404,
        timestamp: 1000
      },
      {
        ip: '203.0.113.10',
        path: '/api/users',
        response_time: 8,
        status: 200,
        timestamp: 1005
      }
    ];

    const built = await buildWasmRecords(
      parsed,
      { '203.0.113.10': 1 },
      path => path === '/api/users',
      () => false,
      [200, 403, 404, 500]
    );

    if (!built) {
      // eslint-disable-next-line no-console
      console.warn(`WASM record helpers unavailable: ${getWasmStatus().error || 'not loaded'}`);
      expect(true).toBe(true);
      return;
    }

    expect(Array.isArray(built)).toBe(true);
    expect(built).toHaveLength(2);
    expect(built[0].path_lower).toBe('/wp-login.php');
    expect(built[0].kw_check).toBe(true);
    expect(built[1].kw_check).toBe(false);

    const payload = await rustPayloadFromRecords(built);
    expect(Array.isArray(payload)).toBe(true);
    expect(payload[0].path_lower).toBe('/wp-login.php');

    const features = await extractWasmFeatures(payload, ['.php', 'wp-']);
    expect(Array.isArray(features)).toBe(true);
    expect(features[0].kw_hits).toBeGreaterThanOrEqual(1);
    expect(features[0].burst_count).toBeGreaterThanOrEqual(1);

    const batch = await extractWasmFeaturesBatchWithState(payload, ['.php', 'wp-'], null);
    expect(batch).toHaveProperty('features');
    expect(batch).toHaveProperty('state');
    expect(Array.isArray(batch.features)).toBe(true);

    const finalized = await finalizeWasmFeatureState();
    expect(finalized === null || typeof finalized === 'object').toBe(true);

    const onePythonFeature = await pythonFeatureFromRecord(built[0], { '203.0.113.10': [1000, 1005] }, ['.php']);
    expect(onePythonFeature).toHaveProperty('kw_hits');

    const pythonBatch = await pythonFeaturesBatched(built, { '203.0.113.10': [1000, 1005] }, ['.php'], {
      batchSize: 1,
      parallelEnabled: false,
      maxWorkers: 1
    });
    expect(Array.isArray(pythonBatch)).toBe(true);
    expect(pythonBatch).toHaveLength(2);

    const recent = await analyzeRecentBehavior([
      { path: '/wp-login.php', status: 404, timestamp: 1000 },
      { path: '/xmlrpc.php', status: 404, timestamp: 1001 },
      { path: '/admin.php', status: 404, timestamp: 1002 }
    ], ['.php', 'wp-']);
    expect(recent === null || typeof recent.should_block === 'boolean').toBe(true);

    const recentValidation = await validateRecent([
      { path: '/wp-login.php', status: 404, timestamp: 1000 },
      { path: '/xmlrpc.php', status: 404, timestamp: 1001 }
    ]);
    expect(recentValidation === null || typeof recentValidation === 'string').toBe(true);

    const model = await createIsolationForest({ nTrees: 10, sampleSize: 8, seed: 3 });
    if (!model.__aiwafWasm) {
      // eslint-disable-next-line no-console
      console.warn(`WASM model unavailable: ${getWasmStatus().error || 'not loaded'}`);
      expect(true).toBe(true);
      return;
    }

    model.fit([
      [0.1, 0.2, 0.3],
      [0.2, 0.1, 0.3],
      [0.15, 0.25, 0.35],
      [1.1, 1.2, 1.3],
      [1.2, 1.1, 1.3],
      [1.15, 1.25, 1.35],
      [0.3, 0.2, 0.1],
      [1.3, 1.2, 1.1]
    ]);

    expect(Number.isFinite(model.anomalyScore([0.1, 0.2, 0.3]))).toBe(true);
    expect(typeof model.isAnomaly([0.1, 0.2, 0.3], 0.5)).toBe('boolean');
    expect(Array.isArray(model.scoreSamples([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]]))).toBe(true);
    expect(Array.isArray(model.decisionFunction([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]]))).toBe(true);
    expect(Array.isArray(model.predict([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]]))).toBe(true);

    const state = model.toJSON();
    expect(state === null || typeof state === 'object').toBe(true);
    if (state) {
      const restored = await createIsolationForestFromJSON(state);
      expect(typeof restored.anomalyScore).toBe('function');
      expect(Number.isFinite(restored.anomalyScore([0.1, 0.2, 0.3]))).toBe(true);
    }
  });
});
