// Two sibling plugins reuse one schema $id for different schemas: each route sees its own scope's.
// AIWAF_SCOPED_ORDER=reversed registers them the other way round.
const fastify = require('fastify')();
const aiwaf = require('../../../../index');

const counts = async scope => {
  scope.addSchema({ $id: 'payload', type: 'object', properties: { value: { type: 'integer' } } });
  scope.post('/counts', { schema: { body: { $ref: 'payload#' } } }, async request => ({ v: request.body.value }));
};
const flags = async scope => {
  scope.addSchema({ $id: 'payload', type: 'object', properties: { value: { type: 'boolean' } } });
  scope.post('/flags', { schema: { body: { $ref: 'payload#' } } }, async request => ({ v: request.body.value }));
};
fastify.register(aiwaf.fastify, { AIWAF_MIDDLEWARES: [] });
for (const plugin of process.env.AIWAF_SCOPED_ORDER === 'reversed' ? [flags, counts] : [counts, flags]) fastify.register(plugin);
module.exports = fastify;
