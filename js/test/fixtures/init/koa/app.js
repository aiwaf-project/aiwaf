// A Koa app with the middleware shape @koa/router's routes() returns: a function carrying its router.
const Koa = require('koa');

const app = new Koa();
const show = async ctx => { ctx.body = { id: ctx.params.id }; };
const routes = async (_ctx, next) => next();
routes.router = { stack: [
  { path: '/items/:id', methods: ['HEAD', 'GET'], stack: [show], name: 'item' },
  { path: '/items', methods: ['POST'], stack: [async ctx => { ctx.body = { name: ctx.request.body.name }; }] },
  { path: '/ignored', methods: [], stack: [] }
] };
app.use(routes);
module.exports = app;
