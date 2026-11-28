// Three methods on one path: each keeps its own inputs and types, and a handler that reads a
// name from the query and the body keeps both locations' types.
const Hapi = require('@hapi/hapi');
const server = Hapi.server();
server.route({ method: 'GET', path: '/items', handler: request => ({ q: request.query.q, page: parseInt(request.query.page, 10) }) });
server.route({ method: 'POST', path: '/items', handler: request => ({ title: request.payload.title, page: request.payload.page }) });
server.route({ method: 'PATCH', path: '/items', handler: request => ({ note: request.payload.note }) });
server.route({
  method: 'POST', path: '/search',
  handler: request => ({ limit: parseInt(request.query.limit, 10), label: request.payload.limit })
});
module.exports = server;
