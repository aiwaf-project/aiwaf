const { EventEmitter } = require('events');
const aiwaf = require('../index');

process.env.NODE_ENV = 'test';

function createMockRes() {
  const res = new EventEmitter();
  res.statusCode = 200;
  res.headers = {};
  res.locals = {};
  res.status = (code) => {
    res.statusCode = code;
    return res;
  };
  res.json = (payload) => {
    res.body = payload;
    res.emit('finish');
    return res;
  };
  res.send = (payload) => {
    res.body = payload;
    res.emit('finish');
    return res;
  };
  return res;
}

function createMockReq({ path = '/', method = 'GET', headers = {}, ip = '198.51.100.40' } = {}) {
  return {
    path,
    url: path,
    method,
    headers,
    ip,
    app: { _router: { stack: [] } }
  };
}

describe('AIWAF NestJS middleware', () => {
  it('allows safe requests', async () => {
    const NestMiddleware = aiwaf.nest({ WINDOW_SEC: 60, MAX_REQ: 1000, FLOOD_REQ: 2000 });
    const middleware = new NestMiddleware();
    const req = createMockReq({ path: '/safe' });
    const res = createMockRes();

    const next = jest.fn();
    await new Promise(resolve => middleware.use(req, res, () => {
      next();
      resolve();
    }));

    expect(next).toHaveBeenCalled();
    expect(res.statusCode).toBe(200);
  });

  it('blocks static keyword paths', async () => {
    const NestMiddleware = aiwaf.nest({ staticKeywords: ['.php'] });
    const middleware = new NestMiddleware();
    const req = createMockReq({ path: '/wp-config.php' });
    const res = createMockRes();

    await new Promise(resolve => {
      res.on('finish', resolve);
      middleware.use(req, res, () => resolve());
    });

    expect(res.statusCode).toBe(403);
    expect(res.body).toEqual({ error: 'blocked' });
  });

  it('enforces method policy when enabled', async () => {
    const NestMiddleware = aiwaf.nest({
      AIWAF_METHOD_POLICY_ENABLED: true,
      AIWAF_ALLOWED_METHODS: ['GET']
    });
    const middleware = new NestMiddleware();
    const req = createMockReq({ path: '/safe', method: 'POST' });
    const res = createMockRes();

    await new Promise(resolve => {
      res.on('finish', resolve);
      middleware.use(req, res, () => resolve());
    });

    expect(res.statusCode).toBe(405);
    expect(res.body).toEqual({ error: 'blocked' });
  });
});
