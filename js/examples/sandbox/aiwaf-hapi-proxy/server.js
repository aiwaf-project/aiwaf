const Hapi = require('@hapi/hapi');
const H2o2 = require('@hapi/h2o2');
const aiwaf = require('aiwaf-js');

const PORT = process.env.PORT || 3003;
const TARGET_BASE_URL = process.env.TARGET_BASE_URL || 'http://localhost:3001';

async function start() {
  const server = Hapi.server({ port: PORT, host: '0.0.0.0' });

  await server.register(H2o2);
  await server.register({
    plugin: aiwaf.hapi,
    options: {
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
    }
  });

  server.ext('onRequest', (request, h) => {
    console.log(`[sandbox-hapi] ${request.method.toUpperCase()} ${request.url.pathname}`);
    return h.continue;
  });

  server.route({
    method: '*',
    path: '/{path*}',
    handler: {
      proxy: {
        uri: TARGET_BASE_URL,
        passThrough: true,
        xforward: true
      }
    }
  });

  await server.start();
  console.log(`AIWAF Hapi sandbox proxy running on port ${PORT}`);
  console.log(`Forwarding traffic to ${TARGET_BASE_URL}`);
}

start().catch(err => {
  console.error(err);
  process.exit(1);
});
