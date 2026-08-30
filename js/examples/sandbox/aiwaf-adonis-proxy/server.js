const { createProxyMiddleware } = require('http-proxy-middleware');
const aiwaf = require('aiwaf-js');

const PORT = process.env.PORT || 3007;
const TARGET_BASE_URL = process.env.TARGET_BASE_URL || 'http://localhost:3001';

const proxy = createProxyMiddleware({
  target: TARGET_BASE_URL,
  changeOrigin: true,
  ws: true,
  logLevel: 'warn'
});

function createCtx(req, res) {
  const response = {
    response: res,
    statusCode: res.statusCode || 200,
    status(code) {
      response.statusCode = code;
      res.statusCode = code;
      return response;
    },
    send(payload) {
      res.end(payload);
      return response;
    },
    json(payload) {
      res.setHeader('content-type', 'application/json');
      res.end(JSON.stringify(payload));
      return response;
    }
  };

  const request = {
    request: req,
    url: () => req.url,
    headers: () => req.headers || {},
    ip: () => (req.headers?.['x-forwarded-for'] || req.socket?.remoteAddress),
    method: () => req.method
  };

  return { request, response };
}

const middleware = aiwaf.adonis({
  staticKeywords: ['.php', '.env', '.git', '../'],
  dynamicTopN: 5,
  WINDOW_SEC: 10,
  MAX_REQ: 25,
  FLOOD_REQ: 50,
  HONEYPOT_FIELD: 'hp_field',
  AIWAF_METHOD_POLICY_ENABLED: true,
  AIWAF_ALLOWED_METHODS: ['GET', 'POST', 'HEAD', 'OPTIONS'],
  AIWAF_HEADER_VALIDATION: true,
  AIWAF_REQUIRED_HEADERS: [],
  AIWAF_MIDDLEWARE_LOGGING: true,
  AIWAF_MIDDLEWARE_LOG_PATH: process.env.AIWAF_MIDDLEWARE_LOG_PATH || 'logs/aiwaf-requests.jsonl'
});

require('http')
  .createServer(async (req, res) => {
    console.log(`[sandbox-adonis] ${req.method} ${req.url}`);
    const ctx = createCtx(req, res);
    await middleware(ctx, () => new Promise(resolve => {
      proxy(req, res, resolve);
    }));
  })
  .listen(PORT, '0.0.0.0', () => {
    console.log(`AIWAF Adonis sandbox proxy running on port ${PORT}`);
    console.log(`Forwarding traffic to ${TARGET_BASE_URL}`);
  });
