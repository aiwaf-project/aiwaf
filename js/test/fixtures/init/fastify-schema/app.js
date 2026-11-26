// A Fastify route whose JSON Schema types quantity as an integer, with the AIWAF plugin
// registered; the handler reads quantity untyped.
const fastify = require('fastify')();
const aiwaf = require('../../../../index');

fastify.register(aiwaf.fastify, { AIWAF_MIDDLEWARES: [] });
fastify.register(async app => {
  app.post('/orders', {
    schema: {
      body: {
        type: 'object',
        properties: { sku: { type: 'string' }, quantity: { type: 'integer' } },
        required: ['sku', 'quantity']
      }
    }
  }, async request => {
    const { sku, quantity } = request.body;
    return { sku, quantity };
  });
});
module.exports = fastify;
