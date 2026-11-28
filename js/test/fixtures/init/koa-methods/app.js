// Koa, in the shape @koa/router's routes() returns: each method's handler keeps its own inputs
// and types, and a name read from the query and the body keeps each location's type.
const Koa = require('koa');

const app = new Koa();
async function listItems(ctx) { ctx.body = { q: ctx.query.q, page: parseInt(ctx.query.page, 10) }; }
async function createItem(ctx) { ctx.body = { title: ctx.request.body.title, page: ctx.request.body.page }; }
async function search(ctx) { ctx.body = { limit: parseInt(ctx.query.limit, 10), label: ctx.request.body.limit }; }
const routes = async (_ctx, next) => next();
routes.router = { stack: [
  { path: '/items', methods: ['HEAD', 'GET'], stack: [listItems] },
  { path: '/items', methods: ['POST'], stack: [createItem] },
  { path: '/search', methods: ['POST'], stack: [search] }
] };
app.use(routes);
module.exports = app;
