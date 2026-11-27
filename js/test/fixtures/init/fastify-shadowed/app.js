// A plugin-local readTenant shadows the module-level one of the same name; JavaScript runs the local.
const fastify = require('fastify')();

async function readTenant(request) { request.region = request.query.region; }

fastify.get('/module', { preHandler: readTenant }, async () => ({ ok: true }));
fastify.register(async app => {
  async function readTenant(request) { request.tenant = request.query.tenant; }
  app.get('/local', { preHandler: readTenant }, async () => ({ ok: true }));
  app.register(async inner => {
    inner.get('/nested', { preHandler: readTenant }, async () => ({ ok: true }));
  });
});
fastify.register(async (app, { readTenant }) => {
  app.get('/option', { preHandler: readTenant }, async () => ({ ok: true }));
}, { readTenant: async () => {} });
fastify.register(async app => {
  const readTenant = require('./tenant-hook');
  app.get('/required', { preHandler: readTenant }, async () => ({ ok: true }));
});
module.exports = fastify;
