// Hapi gives a route its body as request.payload: named reads are inputs, and a payload used
// whole is only complete when a closed validate.payload schema names its keys.
const Hapi = require('@hapi/hapi');
const Joi = require('joi');

const saved = [];
function save(order) { saved.push(order); }

const server = Hapi.server();
server.route({
  method: 'POST', path: '/orders',
  handler: request => ({ sku: request.payload.sku, quantity: Number(request.payload.quantity) })
});
server.route({
  method: 'POST', path: '/raw',
  handler: request => { save(request.payload); return { ok: true }; }
});
server.route({
  method: 'POST', path: '/closed',
  options: { validate: { payload: Joi.object({ sku: Joi.string(), quantity: Joi.number().integer() }) } },
  handler: request => { save(request.payload); return { ok: true }; }
});
server.route({
  method: 'POST', path: '/open',
  options: { validate: { payload: Joi.object({ sku: Joi.string() }).unknown() } },
  handler: request => { save(request.payload); return { ok: true }; }
});
server.route({
  method: 'GET', path: '/search',
  handler: request => ({ q: request.query.q, page: parseInt(request.query.page, 10) })
});
module.exports = server;
