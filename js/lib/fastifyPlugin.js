const createExpressMiddleware = require('./wafMiddleware');

function createExpressLikeResponse(reply) {
  const raw = reply.raw;
  const res = {
    locals: {},
    on: (...args) => raw.on(...args),
    get statusCode() {
      return raw.statusCode;
    },
    set statusCode(code) {
      raw.statusCode = code;
    },
    status(code) {
      res.statusCode = code;
      reply.code(code);
      return res;
    },
    json(payload) {
      reply.type('application/json').send(payload);
      return res;
    },
    send(payload) {
      reply.send(payload);
      return res;
    }
  };
  return res;
}

function fastifyPlugin(fastify, opts = {}, done) {
  const middleware = createExpressMiddleware(opts);

  fastify.addHook('onRequest', async (request, reply) => {
    const res = createExpressLikeResponse(reply);
    await middleware(request.raw, res, () => {});

    if (reply.sent) {
      return reply;
    }
  });

  fastify.addHook('preValidation', async (request, reply) => {
    const finding = await middleware.inspectPayload({
      headers: request.headers, ip: request.ip, url: request.raw.url,
      body: request.body, query: request.query, aiwafRoute: request.raw.aiwafRoute
    });
    if (finding) return reply.code(finding.status).send({ error: finding.rule });
  });

  done();
}

fastifyPlugin[Symbol.for('skip-override')] = true;

module.exports = fastifyPlugin;
