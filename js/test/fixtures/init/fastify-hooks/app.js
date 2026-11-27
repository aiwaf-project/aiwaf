// An app-level hook reads tenant for every route; a route-level preHandler reads region.
const fastify = require('fastify')();
fastify.addHook('preHandler', async request => { request.tenant = request.query.tenant; });
fastify.get('/invoices', { schema: { querystring: { type: 'object', properties: { page: { type: 'integer' } } } } }, async request => ({ page: request.query.page }));
fastify.get('/scoped', { preHandler: async request => { request.region = request.query.region; } }, async request => ({ ok: true }));
module.exports = fastify;
