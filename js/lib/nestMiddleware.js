const createExpressMiddleware = require('./wafMiddleware');

module.exports = function createNestMiddleware(opts = {}) {
  const middleware = createExpressMiddleware(opts);

  return class AIWAFNestMiddleware {
    use(req, res, next) {
      // Express middleware - req/res already have headers set by Express itself
      return middleware(req, res, next);
    }
  };
};
