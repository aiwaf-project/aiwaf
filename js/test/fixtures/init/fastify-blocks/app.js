// A hook declared inside a {} block shadows a same-named outer one only inside that block.
const fastify = require('fastify')();

fastify.register(async app => {
  const readTenant = async request => { request.region = request.query.region; };
  {
    const readTenant = async request => { request.tenant = request.query.tenant; };
    app.get('/inside', { preHandler: readTenant }, async () => ({ ok: true }));
  }
  app.get('/outside', { preHandler: readTenant }, async () => ({ ok: true }));
});
fastify.register(async app => {
  for (const readTenant of [async request => { request.x = request.query.x; }]) {
    app.get('/loop', { preHandler: readTenant }, async () => ({ ok: true }));
  }
});
module.exports = fastify;
