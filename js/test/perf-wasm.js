const { createIsolationForest, getWasmStatus } = require('../lib/wasmAdapter');
const { IsolationForest } = require('../lib/isolationForest');

function nowMs() {
  const [sec, nano] = process.hrtime();
  return (sec * 1000) + (nano / 1e6);
}

function randomVec(dim) {
  const out = [];
  for (let i = 0; i < dim; i += 1) {
    out.push(Math.random());
  }
  return out;
}

async function run() {
  const dims = 6;
  const training = Array.from({ length: 200 }, () => randomVec(dims));
  const samples = Array.from({ length: 5000 }, () => randomVec(dims));

  const wasmModel = await createIsolationForest({ nTrees: 100, sampleSize: 256, threshold: 0.5, seed: 42 });
  wasmModel.fit(training);

  const wasmStart = nowMs();
  for (const s of samples) {
    wasmModel.anomalyScore(s);
  }
  const wasmEnd = nowMs();
  const wasmTotal = wasmEnd - wasmStart;
  const wasmPerSample = wasmTotal / samples.length;
  const wasmRate = samples.length / (wasmTotal / 1000);

  const jsModel = new IsolationForest({ nTrees: 100, sampleSize: 256 });
  jsModel.fit(training);
  const jsStart = nowMs();
  for (const s of samples) {
    jsModel.anomalyScore(s);
  }
  const jsEnd = nowMs();
  const jsTotal = jsEnd - jsStart;
  const jsPerSample = jsTotal / samples.length;
  const jsRate = samples.length / (jsTotal / 1000);

  const wasmStatus = getWasmStatus();
  // eslint-disable-next-line no-console
  console.log('IsolationForest benchmark (wasm)');
  // eslint-disable-next-line no-console
  console.log(`Samples: ${samples.length}, Total: ${wasmTotal.toFixed(2)}ms, Per-sample: ${wasmPerSample.toFixed(4)}ms, Rate: ${wasmRate.toFixed(0)}/sec`);
  if (!wasmModel.__aiwafWasm) {
    // eslint-disable-next-line no-console
    console.log(`WASM load status: loaded=${wasmStatus.loaded} error=${wasmStatus.error || 'none'}`);
  }
  // eslint-disable-next-line no-console
  console.log('IsolationForest benchmark (js)');
  // eslint-disable-next-line no-console
  console.log(`Samples: ${samples.length}, Total: ${jsTotal.toFixed(2)}ms, Per-sample: ${jsPerSample.toFixed(4)}ms, Rate: ${jsRate.toFixed(0)}/sec`);
}

run().catch(err => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exit(1);
});
