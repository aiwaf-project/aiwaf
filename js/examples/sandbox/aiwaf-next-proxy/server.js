const next = require('next');
const { createProxyMiddleware } = require('http-proxy-middleware');
const aiwaf = require('aiwaf-js');

const PORT = process.env.PORT || 3006;
const TARGET_BASE_URL = process.env.TARGET_BASE_URL || 'http://localhost:3001';
const dev = process.env.NODE_ENV !== 'production';

const app = next({ dev });
const nextHandler = app.getRequestHandler();

async function start() {
  await app.prepare();

  const proxy = createProxyMiddleware({
    target: TARGET_BASE_URL,
    changeOrigin: true,
    ws: true,
    logLevel: 'warn'
  });

  const handler = (req, res) => {
    if (req.url.startsWith('/_next') || req.url === '/favicon.ico') {
      return nextHandler(req, res);
    }
    return proxy(req, res, (err) => {
      if (err) {
        res.statusCode = 502;
        res.end('bad_gateway');
        return;
      }
      nextHandler(req, res);
    });
  };

  const wrapped = aiwaf.next(handler, {
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
    .createServer((req, res) => {
      console.log(`[sandbox-next] ${req.method} ${req.url}`);
      return wrapped(req, res);
    })
    .listen(PORT, '0.0.0.0', () => {
      console.log(`AIWAF Next.js sandbox proxy running on port ${PORT}`);
      console.log(`Forwarding traffic to ${TARGET_BASE_URL}`);
    });
}

start().catch(err => {
  console.error(err);
  process.exit(1);
});
