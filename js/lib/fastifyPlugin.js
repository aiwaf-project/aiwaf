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
    await new Promise(resolve => {
      const res = createExpressLikeResponse(reply);
      middleware(request.raw, res, resolve);
    });

    if (reply.sent) {
      return reply;
    }
  });

  done();
}

fastifyPlugin[Symbol.for('skip-override')] = true;

module.exports = fastifyPlugin;
