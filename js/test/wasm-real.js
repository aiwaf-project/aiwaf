async function run() {
  const { validateHeaders, getWasmStatus } = require('../lib/wasmAdapter');
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
}

run().catch(err => {
  console.error(err);
  process.exit(1);
});
