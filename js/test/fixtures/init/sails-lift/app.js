// A stand-in for a Sails app: lift() starts an HTTP server and calls back once it listens, and the
// exported factory waits for that. init records listen() without binding a port, so the callback
// has to run for the factory to return. The routes are Sails' config.routes.
const express = require('express');
const http = require('http');

function createSails() {
  const app = express();
  return {
    config: {
      appPath: __dirname,
      routes: {
        'GET /users': 'UserController.find',
        'POST /users': { controller: 'UserController', action: 'create' },
        'GET /users/search': 'user/search',
        'POST /signup': 'entrance/signup',
        'POST /legacy': 'entrance/legacy',
        '/health': { view: 'health' }
      }
    },
    hooks: { http: { app } },
    lift(_options, done) {
      http.createServer(app).listen(1337, () => done());
    },
    lower(done) { if (done) done(); },
    load(_options, done) { done(); }
  };
}

module.exports = async () => {
  const sails = createSails();
  await new Promise(resolve => sails.lift({}, resolve));
  return sails;
};
