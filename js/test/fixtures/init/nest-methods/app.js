// A Nest controller with GET, POST and PATCH on one path, each reading the request through
// @Req(): each method keeps its own inputs, and the GET handler's survive the others.
require('reflect-metadata');
const { NestFactory } = require('@nestjs/core');
const { Controller, Get, Module, Patch, Post, Req } = require('@nestjs/common');

class ItemsController {
  list(req) { return { q: req.query.q, page: parseInt(req.query.page, 10) }; }
  create(req) { return { title: req.body.title, page: req.body.page }; }
  update(req) { return { note: req.body.note }; }
}
for (const [name, verb] of [['list', Get()], ['create', Post()], ['update', Patch()]]) {
  verb(ItemsController.prototype, name, Object.getOwnPropertyDescriptor(ItemsController.prototype, name));
  Req()(ItemsController.prototype, name, 0);
}
Controller('items')(ItemsController);

class AppModule {}
Module({ controllers: [ItemsController] })(AppModule);

module.exports = async () => {
  const app = await NestFactory.create(AppModule, { logger: false });
  await app.init();
  return app;
};
