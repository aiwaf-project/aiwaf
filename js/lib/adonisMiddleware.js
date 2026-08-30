const createExpressMiddleware = require('./wafMiddleware');

function createExpressLikeResponse(ctx) {
  const rawRes = ctx.response?.response;
  const res = {
    locals: {},
    on: (...args) => rawRes?.on?.(...args),
    get statusCode() {
      return rawRes?.statusCode ?? ctx.response?.statusCode ?? 200;
    },
    set statusCode(code) {
      if (rawRes) rawRes.statusCode = code;
      if (ctx.response) ctx.response.statusCode = code;
    },
    status(code) {
      if (ctx.response?.status) {
        ctx.response.status(code);
      } else {
        res.statusCode = code;
      }
      res._handled = true;
      return res;
    },
    json(payload) {
      if (ctx.response?.json) {
        ctx.response.json(payload);
      } else if (ctx.response?.send) {
        ctx.response.send(payload);
      }
      res._handled = true;
      return res;
    },
    send(payload) {
      if (ctx.response?.send) {
        ctx.response.send(payload);
      } else if (rawRes?.end) {
        rawRes.end(payload);
      }
      res._handled = true;
      return res;
    }
  };

  return res;
}

module.exports = function createAdonisMiddleware(opts = {}) {
  const middleware = createExpressMiddleware(opts);

  return async (ctx, next) => {
    const req = ctx.request?.request || ctx.request || {};
    const res = createExpressLikeResponse(ctx);

    // Only set path/url/ip if not already available
    if (!req.path) req.path = ctx.request?.url?.() || ctx.request?.url || req.url;
    if (!req.url) req.url = req.path;
    if (!req.ip) req.ip = ctx.request?.ip?.() || ctx.request?.ip;
    
    // Don't override headers - the raw request already has them from the HTTP server

    await new Promise(resolve => {
      let resolved = false;
      const finish = () => {
        if (resolved) return;
        resolved = true;
        resolve();
      };
      const maybePromise = middleware(req, res, finish);
      if (res._handled) {
        finish();
        return;
      }
      Promise.resolve(maybePromise).then(finish).catch(finish);
    });

    if (res._handled || ctx.response?.response?.writableEnded || ctx.response?.response?.headersSent) {
      return;
    }

    await next();
  };
};
