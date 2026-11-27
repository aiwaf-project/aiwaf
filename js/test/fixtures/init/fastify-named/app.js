// Named handlers on one path, each method with its own schema: the audit links them by name.
const fastify = require('fastify')();
async function listPlans(request) { return { mode: request.query.mode }; }
async function createPlan(request) { const { dryRun, weeks, mode } = request.body; return { dryRun, weeks, mode }; }
async function removePlan(request) { return { force: request.query.force }; }
fastify.get('/plans', { schema: { querystring: { type: 'object', properties: { mode: { type: 'string' } } } } }, listPlans);
fastify.post('/plans', { schema: { body: { type: 'object', properties: { dryRun: { type: 'boolean' }, weeks: { type: 'integer' }, mode: { type: 'integer' } } } } }, createPlan);
fastify.delete('/plans', { schema: { querystring: { type: 'object', properties: { force: { type: 'boolean' } } } } }, removePlan);
module.exports = fastify;
