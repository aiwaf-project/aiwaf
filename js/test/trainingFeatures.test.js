jest.mock('../lib/wasmAdapter', () => ({
  extractWasmTrainingFeatures: jest.fn(),
  extractWasmRawTrainingFeatures: jest.fn()
}));

const { extractWasmTrainingFeatures, extractWasmRawTrainingFeatures } = require('../lib/wasmAdapter');
const { calculateTrainingFeatures } = require('../lib/trainingFeatures');

describe('training feature acceleration', () => {
  beforeEach(() => {
    delete process.env.AIWAF_EXPERIMENTAL_WASM_RAW_TRAINING;
    extractWasmTrainingFeatures.mockReset();
    extractWasmRawTrainingFeatures.mockReset();
    extractWasmRawTrainingFeatures.mockResolvedValue(null);
  });

  it('passes raw rows to the new WASM batch API without JS feature preparation', async () => {
    process.env.AIWAF_EXPERIMENTAL_WASM_RAW_TRAINING = '1';
    const rows = [{
      ip: '203.0.113.1', path: '/a.php', status: '404', responseTime: 7, timestamp: new Date(20000)
    }];
    extractWasmRawTrainingFeatures.mockResolvedValue([{
      path_len: 6, kw_hits: 1, resp_time: 7, status_idx: 1, burst_count: 1, total_404: 1
    }]);

    expect(await calculateTrainingFeatures(rows, ['.php'], ['200', '404']))
      .toEqual([[6, 1, 1, 7, 1, 1]]);
    expect(extractWasmRawTrainingFeatures).toHaveBeenCalledWith(rows, ['.php'], ['200', '404']);
    expect(extractWasmTrainingFeatures).not.toHaveBeenCalled();
    delete process.env.AIWAF_EXPERIMENTAL_WASM_RAW_TRAINING;
  });

  it('uses Rust batch output while preserving the persisted JS feature order', async () => {
    extractWasmTrainingFeatures.mockResolvedValue([{
      path_len: 5, kw_hits: 2, resp_time: 7, status_idx: 1, burst_count: 3, total_404: 4
    }]);
    const features = await calculateTrainingFeatures([
      { ip: '203.0.113.1', path: '/a.php', status: '404', responseTime: 7, timestamp: new Date(20000) }
    ], ['.php'], ['200', '404']);

    expect(features).toEqual([[5, 2, 1, 7, 3, 4]]);
    expect(extractWasmTrainingFeatures).toHaveBeenCalledWith([
      expect.objectContaining({ timestamp: 20, status_idx: 1, total_404: 1 })
    ], ['.php']);
    expect(extractWasmRawTrainingFeatures).not.toHaveBeenCalled();
  });

  it('falls back to a symmetric ten-second window when WASM is unavailable', async () => {
    extractWasmTrainingFeatures.mockResolvedValue(null);
    const rows = [
      { ip: '203.0.113.1', path: '/a.php', status: '404', responseTime: 7, timestamp: new Date(0) },
      { ip: '203.0.113.1', path: '/safe', status: '200', responseTime: 8, timestamp: new Date(10000) },
      { ip: '203.0.113.1', path: '/safe', status: '200', responseTime: 9, timestamp: new Date(21000) }
    ];
    expect(await calculateTrainingFeatures(rows, ['.php'], ['200', '404'])).toEqual([
      [6, 1, 1, 7, 2, 1],
      [5, 0, 0, 8, 2, 1],
      [5, 0, 0, 9, 1, 1]
    ]);
  });
});
