// Route-local hooks: a named preHandler reads tenant for /invoices, an inline one region for /inline.
const fastify = require('fastify')();
async function requireTenant(request) { if (!request.query.tenant) throw new Error('tenant'); }
fastify.get('/invoices', { preHandler: requireTenant }, async request => ({ page: Number(request.query.page) }));
fastify.get('/inline', { preHandler: [async request => { request.region = request.query.region; }] }, async () => ({ ok: true }));
fastify.get('/plain', async request => ({ page: Number(request.query.page) }));
module.exports = fastify;
