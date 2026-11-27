// The outer hook is at module scope; a block inside the plugin declares its own const of that name.
const fastify = require('fastify')();

async function readTenant(request) { request.region = request.query.region; }

fastify.register(async app => {
  {
    const readTenant = async request => { request.tenant = request.query.tenant; };
    app.get('/inside', { preHandler: readTenant }, async () => ({ ok: true }));
  }
  app.get('/outside', { preHandler: readTenant }, async () => ({ ok: true }));
});
{
  const blockOnly = async request => { request.other = request.query.other; };
  fastify.get('/block-route', { preHandler: blockOnly }, async () => ({ ok: true }));
}
module.exports = fastify;
