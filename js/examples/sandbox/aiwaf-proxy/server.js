const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const aiwaf = require('aiwaf-js');

const app = express();
const PORT = process.env.PORT || 3000;
const TARGET_BASE_URL = process.env.TARGET_BASE_URL || 'http://localhost:3001';

app.use(express.json());

app.use((req, res, next) => {
  console.log(`[sandbox] ${req.method} ${req.originalUrl}`);
  next();
});

app.use(aiwaf({
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

app.use(
  '/',
  createProxyMiddleware({
    target: TARGET_BASE_URL,
    changeOrigin: true,
    ws: true,
    logLevel: 'warn'
  })
);

app.listen(PORT, () => {
  console.log(`AIWAF sandbox proxy running on port ${PORT}`);
  console.log(`Forwarding traffic to ${TARGET_BASE_URL}`);
});
