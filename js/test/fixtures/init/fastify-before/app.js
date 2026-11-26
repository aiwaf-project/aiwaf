// A route registered before the AIWAF plugin, which the plugin's onRoute hook never sees.
const fastify = require('fastify')();
const aiwaf = require('../../../../index');

fastify.get('/early', async request => ({ page: Number(request.query.page) }));
fastify.register(aiwaf.fastify, { AIWAF_MIDDLEWARES: [] });
fastify.register(async app => {
  app.post('/late', { schema: { body: { type: 'object', properties: { n: { type: 'integer' } } } } }, async request => ({ n: request.body.n }));
});
module.exports = fastify;
