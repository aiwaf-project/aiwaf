// Named hooks defined inside the plugin callback; a sibling route uses none; one schema stays open.
const fastify = require('fastify')();
fastify.register(async app => {
  async function readTenant(request) { request.tenant = request.query.tenant; }
  const readRegion = async request => { request.region = request.query.region; };
  app.get('/invoices', { preHandler: readTenant, schema: { querystring: { type: 'object', properties: { page: { type: 'integer' } } } } }, async request => ({ page: request.query.page }));
  app.get('/regions', { preHandler: [readRegion] }, async () => ({ ok: true }));
  app.get('/sibling', { schema: { querystring: { type: 'object', properties: { page: { type: 'integer' } } } } }, async request => ({ page: request.query.page }));
  app.get('/open', { preHandler: readTenant, schema: { querystring: { type: 'object', properties: { page: { type: 'integer' } }, additionalProperties: true } } }, async request => ({ page: request.query.page }));
});
module.exports = fastify;
