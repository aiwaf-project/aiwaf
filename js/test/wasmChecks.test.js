describe('aiwaf-wasm loading', () => {
  afterEach(() => {
    jest.dontMock('../lib/wasmAdapter');
    jest.resetModules();
  });

  test('loads once and returns the same module', async () => {
    const checks = require('../lib/wasmChecks');
    const mod = await checks.load();
    expect(await checks.load()).toBe(mod);
    expect(checks.get()).toBe(mod);
    expect(typeof mod.sql_inspect).toBe('function');
  });

  test('a missing or old aiwaf-wasm is reported, and loading is retried', async () => {
    let calls = 0;
    jest.doMock('../lib/wasmAdapter', () => ({ loadWasm: async () => { calls += 1; return calls === 1 ? null : { sql_inspect() {} }; } }));
    const checks = require('../lib/wasmChecks');
    expect(() => checks.get()).toThrow("isn't loaded yet");
    await expect(checks.load()).rejects.toThrow('aiwaf-wasm 0.2.6');
    await expect(checks.load()).rejects.toThrow(checks.MESSAGE);
    expect(calls).toBe(2);
  });
});
