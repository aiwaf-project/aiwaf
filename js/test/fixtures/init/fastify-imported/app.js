// A webhook registered before the AIWAF plugin, with a schema imported from another module.
const fastify = require('fastify')();
const aiwaf = require('../../../../index');
const { webhookSchema } = require('./schemas');

fastify.post('/webhooks/payments', { schema: webhookSchema }, async request => ({ ok: request.body.successful }));
fastify.register(aiwaf.fastify, { AIWAF_MIDDLEWARES: [] });
fastify.post('/webhooks/after', { schema: webhookSchema }, async request => ({ ok: request.body.successful }));
module.exports = fastify;
