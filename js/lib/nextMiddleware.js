const createExpressMiddleware = require('./wafMiddleware');

function createExpressLikeResponse(res) {
  const wrapped = {
    locals: {},
    on: (...args) => res.on(...args),
    get statusCode() {
      return res.statusCode;
    },
    set statusCode(code) {
      res.statusCode = code;
    },
    status(code) {
      res.statusCode = code;
      wrapped._handled = true;
      return wrapped;
    },
    json(payload) {
      if (typeof res.setHeader === 'function') {
        res.setHeader('content-type', 'application/json');
      }
      res.end(JSON.stringify(payload));
      wrapped._handled = true;
      return wrapped;
    },
    send(payload) {
      if (payload !== undefined && typeof payload !== 'string' && !Buffer.isBuffer(payload)) {
        if (typeof res.setHeader === 'function') {
          res.setHeader('content-type', 'application/json');
        }
        res.end(JSON.stringify(payload));
      } else {
        res.end(payload);
      }
      wrapped._handled = true;
      return wrapped;
    }
  };
  return wrapped;
}

function wrapHandler(handler, opts) {
  const middleware = createExpressMiddleware(opts);

  return async (req, res) => {
    const handled = await new Promise(resolve => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        resolve(wrappedRes._handled === true);
      };
      res.on('finish', finish);
      res.on('close', finish);
      const wrappedRes = createExpressLikeResponse(res);
      const maybePromise = middleware(req, wrappedRes, () => finish());
      if (wrappedRes._handled) {
        finish();
        return;
      }
      Promise.resolve(maybePromise).then(finish).catch(finish);
    });

    if (handled) return;
    if (res.writableEnded || res.headersSent) return;
    if (res.finished) return;
    return handler(req, res);
  };
}

module.exports = function createNextHandler(handlerOrOpts, maybeOpts) {
  if (typeof handlerOrOpts === 'function') {
    return wrapHandler(handlerOrOpts, maybeOpts || {});
  }

  const opts = handlerOrOpts || {};
  return (handler) => wrapHandler(handler, opts);
};
