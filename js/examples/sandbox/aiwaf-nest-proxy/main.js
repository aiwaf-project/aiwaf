require('reflect-metadata');
const { NestFactory } = require('@nestjs/core');
const { Module } = require('@nestjs/common');
const { createProxyMiddleware } = require('http-proxy-middleware');
const aiwaf = require('aiwaf-js');

const PORT = process.env.PORT || 3005;
const TARGET_BASE_URL = process.env.TARGET_BASE_URL || 'http://localhost:3001';

class AppModule {}

Module({})(AppModule);

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

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

  app.use((req, res, next) => {
    console.log(`[sandbox-nest] ${req.method} ${req.originalUrl}`);
    next();
  });

  app.use(
    '/',
    createProxyMiddleware({
      target: TARGET_BASE_URL,
      changeOrigin: true,
      ws: true,
      logLevel: 'warn'
    })
  );

  await app.listen(PORT, '0.0.0.0');
  console.log(`AIWAF NestJS sandbox proxy running on port ${PORT}`);
  console.log(`Forwarding traffic to ${TARGET_BASE_URL}`);
}

bootstrap().catch(err => {
  console.error(err);
  process.exit(1);
});
