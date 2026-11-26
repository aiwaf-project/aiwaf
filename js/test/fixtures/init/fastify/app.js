const fastify = require('fastify')();

fastify.get('/health', async () => ({ ok: true }));
fastify.post('/orders', async (request, reply) => reply.send({ sku: request.body.sku }));
fastify.register(async scope => {
  scope.get('/profile', async () => ({ user: 1 }));
}, { prefix: '/account' });
module.exports = fastify;
