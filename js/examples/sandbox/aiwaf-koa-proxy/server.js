const Koa = require('koa');
const bodyParser = require('koa-bodyparser');
const proxy = require('koa-proxies');
const aiwaf = require('aiwaf-js');

const PORT = process.env.PORT || 3004;
const TARGET_BASE_URL = process.env.TARGET_BASE_URL || 'http://localhost:3001';

const app = new Koa();

app.use(bodyParser());
app.use((ctx, next) => {
  console.log(`[sandbox-koa] ${ctx.method} ${ctx.url}`);
  return next();
});

app.use(aiwaf.koa({
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
}));

app.use(proxy('/', {
  target: TARGET_BASE_URL,
  changeOrigin: true,
  logs: true
}));

app.listen(PORT, () => {
  console.log(`AIWAF Koa sandbox proxy running on port ${PORT}`);
  console.log(`Forwarding traffic to ${TARGET_BASE_URL}`);
});
