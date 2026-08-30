process.env.NODE_ENV = 'test';

describe('WASM header validation config passthrough', () => {
  afterEach(() => {
    jest.resetModules();
    jest.clearAllMocks();
  });

  it('passes requiredHeaders + minScore into validate_headers_with_config', async () => {
    const validateWithConfig = jest.fn(() => null);
    const validateHeaders = jest.fn(() => null);

    jest.doMock('aiwaf-wasm', () => ({
      default: jest.fn(async () => {}),
      validate_headers: validateHeaders,
      validate_headers_with_config: validateWithConfig,
      AiwafIsolationForest: class {
        constructor() {}
        fit() {}
        anomaly_score() { return 0.1; }
      }
    }));

    let middleware;
    await new Promise(resolve => {
      jest.isolateModules(() => {
        const aiwafLocal = require('../index');
        middleware = aiwafLocal({
          AIWAF_HEADER_VALIDATION: true,
          AIWAF_REQUIRED_HEADERS: ['accept', 'user-agent'],
          AIWAF_HEADER_QUALITY_MIN_SCORE: 3,
          AIWAF_WASM_VALIDATION: true
        });
        resolve();
      });
    });

    const req = {
      headers: {
        host: 'localhost:3002',
        connection: 'keep-alive',
        'sec-ch-ua': '"Chromium";v="146", "Not?A_Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36 Edg/146.0.0.0',
        accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'sec-fetch-site': 'none',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-user': '?1',
        'sec-fetch-dest': 'document',
        'accept-encoding': 'gzip, deflate, br, zstd',
        'accept-language': 'en-US,en;q=0.9,hi;q=0.8',
        cookie: 'sid=test',
        'server-protocol': 'HTTP/1.1'
      },
      method: 'GET',
      path: '/',
      url: '/',
      ip: '198.51.100.100'
    };
    const res = {
      statusCode: 200,
      headersSent: false,
      writableEnded: false,
      locals: {},
      status() { return res; },
      json() { res.headersSent = true; res.writableEnded = true; return res; },
      send() { res.headersSent = true; res.writableEnded = true; return res; },
      on() {}
    };

    const next = jest.fn();
    await middleware(req, res, next);

    expect(validateWithConfig).toHaveBeenCalledTimes(1);
    const [, requiredHeaders, minScore] = validateWithConfig.mock.calls[0];
    expect(requiredHeaders).toEqual(['accept', 'user-agent']);
    expect(minScore).toBe(3);
  });
});
