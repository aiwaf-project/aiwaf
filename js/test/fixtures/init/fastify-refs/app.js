// GET and POST on one path with different types for ref; shared schemas by $ref; a closed nested object.
const fastify = require('fastify')();

const customer = { $id: 'customer', type: 'object', properties: { email: { type: 'string' }, tier: { type: 'integer' } }, additionalProperties: false };
fastify.addSchema(customer);
fastify.addSchema({ $id: 'order', type: 'object', properties: { ref: { type: 'integer' }, customer: { $ref: 'customer#' } } });
fastify.get('/orders', { schema: { querystring: { type: 'object', properties: { ref: { type: 'string' } } } } }, async request => ({ ref: request.query.ref }));
fastify.post('/orders', { schema: { body: { $ref: 'order#' } } }, async request => ({ ref: request.body.ref }));
fastify.post('/notes', async request => ({ size: request.body.length }));
module.exports = fastify;
