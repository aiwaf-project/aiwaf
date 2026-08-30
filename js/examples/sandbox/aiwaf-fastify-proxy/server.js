const fastify = require('fastify')({ logger: true });
const aiwaf = require('aiwaf-js');
const { createProxyMiddleware } = require('http-proxy-middleware');

const PORT = process.env.PORT || 3002;
const TARGET_BASE_URL = process.env.TARGET_BASE_URL || 'http://localhost:3001';

fastify.register(aiwaf.fastify, {
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

fastify.addHook('onRequest', async (request, reply) => {
  fastify.log.info(`[sandbox-fastify] ${request.method} ${request.url}`);
});

fastify.register(require('@fastify/middie')).then(() => {
  fastify.use(
    '/',
    createProxyMiddleware({
      target: TARGET_BASE_URL,
      changeOrigin: true,
      ws: true,
      logLevel: 'warn'
    })
  );
});

fastify.listen({ port: PORT, host: '0.0.0.0' }).then(() => {
  fastify.log.info(`AIWAF Fastify sandbox proxy running on port ${PORT}`);
  fastify.log.info(`Forwarding traffic to ${TARGET_BASE_URL}`);
});
