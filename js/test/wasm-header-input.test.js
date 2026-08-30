process.env.NODE_ENV = 'test';

describe('WASM header validation input shape', () => {
  afterEach(() => {
    jest.resetModules();
    jest.clearAllMocks();
  });

  it('passes a plain object into WASM validator in Node', async () => {
    const validateWithConfig = jest.fn(() => null);

    jest.doMock('aiwaf-wasm', () => ({
      default: jest.fn(async () => {}),
      validate_headers_with_config: validateWithConfig,
      validate_headers: jest.fn(() => null),
      AiwafIsolationForest: class {
        constructor() {}
        fit() {}
        anomaly_score() { return 0.1; }
      }
    }));

    let validateHeaders;
    await new Promise(resolve => {
      jest.isolateModules(() => {
        ({ validateHeaders } = require('../lib/wasmAdapter'));
        resolve();
      });
    });

    const result = await validateHeaders(
      { accept: 'text/html', 'user-agent': 'Mozilla/5.0' },
      { requiredHeaders: ['accept', 'user-agent'], minScore: 3 }
    );

    expect(validateWithConfig).toHaveBeenCalledTimes(1);
    const [headerInput] = validateWithConfig.mock.calls[0];
    expect(typeof headerInput).toBe('object');
    expect(headerInput['user-agent']).toBe('Mozilla/5.0');
    expect(result).toBeNull();
  });
});
