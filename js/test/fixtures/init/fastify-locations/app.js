// One POST takes limit twice: an integer in the query, text in the JSON body.
const fastify = require('fastify')();
fastify.post('/reports', { schema: {
  querystring: { type: 'object', properties: { limit: { type: 'integer' } } },
  body: { type: 'object', properties: { limit: { type: 'string' }, since: { type: 'integer' } } }
} }, async request => ({ q: request.query.limit, b: request.body.limit }));
module.exports = fastify;
