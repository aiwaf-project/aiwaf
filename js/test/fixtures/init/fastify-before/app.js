// Routes registered before the AIWAF plugin, which the plugin's onRoute hook never sees.
const fastify = require('fastify')();
const aiwaf = require('../../../../index');

const orderSchema = { body: { type: 'object', properties: { sku: { type: 'string' }, quantity: { type: 'integer' } } } };

fastify.get('/early', async request => ({ page: Number(request.query.page) }));
fastify.post('/early-order', { schema: { body: { type: 'object', properties: { quantity: { type: 'integer' } } } } }, async request => {
  const { quantity } = request.body;
  return { quantity };
});
fastify.route({ method: 'PUT', url: '/early-route', schema: orderSchema, handler: async request => ({ sku: request.body.sku, q: request.body.quantity }) });
fastify.register(aiwaf.fastify, { AIWAF_MIDDLEWARES: [] });
fastify.register(async app => {
  app.post('/late', { schema: { body: { type: 'object', properties: { n: { type: 'integer' } } } } }, async request => ({ n: request.body.n }));
});
module.exports = fastify;
