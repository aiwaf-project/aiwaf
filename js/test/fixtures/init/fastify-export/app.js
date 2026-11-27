// A factory makes one handler per export format.
const fastify = require('fastify')();
function makeExport(format) {
  return async function exportReport(request) { return { format, since: request.query.since }; };
}
for (const format of ['csv', 'pdf']) {
  fastify.get(`/exports/${format}`, { schema: { querystring: { type: 'object', properties: { since: { type: 'string' } } } } }, makeExport(format));
}
module.exports = fastify;
