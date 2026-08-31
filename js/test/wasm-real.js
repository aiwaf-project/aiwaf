async function run() {
  const {
    createIsolationForest,
    createIsolationForestFromJSON,
    getWasmStatus,
    validateHeaders
  } = require('../lib/wasmAdapter');
  const headers = {
    'user-agent': 'Mozilla/5.0',
    accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'accept-language': 'en-US,en;q=0.9',
    'accept-encoding': 'gzip, deflate, br',
    connection: 'keep-alive'
  };

  const result = await validateHeaders(
    headers,
    { requiredHeaders: ['accept', 'user-agent'], minScore: 3 }
  );

  const status = getWasmStatus();
  console.log(`aiwaf-wasm version: ${require('../node_modules/aiwaf-wasm/package.json').version}`);
  console.log(`header keys: ${Object.keys(headers).join(', ')}`);
  console.log(`WASM loaded: ${status.loaded} error=${status.error || 'none'}`);
  console.log(`WASM validate_headers result: ${result === null ? 'null' : JSON.stringify(result)}`);

  if (!status.loaded) {
    process.exit(1);
  }
  if (result !== null) {
    process.exit(2);
  }

  const training = [[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]];
  const model = await createIsolationForest({ nTrees: 8, sampleSize: 6, seed: 7 });
  if (!model.__aiwafWasm) {
    throw new Error('createIsolationForest silently used the JavaScript fallback');
  }
  model.fit(training);
  const score = model.anomalyScore([0.15]);
  if (!Number.isFinite(score)) {
    throw new Error(`WASM anomaly score is not finite: ${score}`);
  }
  if (typeof model.isAnomaly([0.15], 0.5) !== 'boolean') {
    throw new Error('WASM isAnomaly did not return a boolean');
  }
  if (model.scoreSamples([[0.15], [1.15]]).length !== 2) {
    throw new Error('WASM scoreSamples returned the wrong number of scores');
  }
  if (model.predict([[0.15], [1.15]]).length !== 2) {
    throw new Error('WASM predict returned the wrong number of predictions');
  }

  model.retrain([[0.05], [0.25], [0.95], [1.25]]);
  if (!Number.isFinite(model.anomalyScore([0.15]))) {
    throw new Error('WASM retrain left the model in an invalid state');
  }

  const state = model.toJSON();
  if (!state) {
    throw new Error('WASM model did not produce serialized state');
  }
  const restored = await createIsolationForestFromJSON(state);
  if (!restored.__aiwafWasm || !Number.isFinite(restored.anomalyScore([0.15]))) {
    throw new Error('WASM model serialization round trip failed');
  }
}

run().catch(err => {
  console.error(err);
  process.exit(1);
});
