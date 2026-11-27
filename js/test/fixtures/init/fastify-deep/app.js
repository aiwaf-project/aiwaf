// Closed objects nested two levels, an array of closed objects, and an open object.
const fastify = require('fastify')();
const address = { type: 'object', properties: { city: { type: 'string' }, zip: { type: 'string' } }, additionalProperties: false };
fastify.post('/orders', { schema: { body: { type: 'object', properties: {
  customer: { type: 'object', properties: { email: { type: 'string' }, address }, additionalProperties: false },
  lines: { type: 'array', items: { type: 'object', properties: { sku: { type: 'string' }, qty: { type: 'integer' } }, additionalProperties: false } },
  meta: { type: 'object', additionalProperties: true, properties: { any: { type: 'string' } } }
} } } }, async request => ({ n: request.body.lines.length }));
module.exports = fastify;
