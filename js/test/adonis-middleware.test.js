const { EventEmitter } = require('events');
const aiwaf = require('../index');

process.env.NODE_ENV = 'test';

function createMockCtx({ path = '/', method = 'GET', headers = {}, ip = '198.51.100.60' } = {}) {
  const rawReq = {
    path,
    url: path,
    method,
    headers
  };
  const rawRes = new EventEmitter();
  rawRes.statusCode = 200;
  rawRes.headersSent = false;
  rawRes.writableEnded = false;
  rawRes.end = (payload) => {
    rawRes.body = payload;
    rawRes.headersSent = true;
    rawRes.writableEnded = true;
    rawRes.emit('finish');
  };

  const response = {
    response: rawRes,
    statusCode: 200,
    status(code) {
      response.statusCode = code;
      rawRes.statusCode = code;
      return response;
    },
    send(payload) {
      response.body = payload;
      rawRes.end(payload);
      return response;
    },
    json(payload) {
      response.body = payload;
      rawRes.end(JSON.stringify(payload));
      return response;
    }
  };

  const request = {
    request: rawReq,
    url: () => path,
    headers: () => headers,
    ip: () => ip,
    method: () => method
  };

  return { request, response };
}

describe('AIWAF Adonis middleware', () => {
  it('allows safe requests', async () => {
    const middleware = aiwaf.adonis({ WINDOW_SEC: 60, MAX_REQ: 1000, FLOOD_REQ: 2000 });
    const ctx = createMockCtx({ path: '/safe' });
    const next = jest.fn();

    await middleware(ctx, next);

    expect(next).toHaveBeenCalled();
    expect(ctx.response.statusCode).toBe(200);
  });

  it('blocks static keyword paths', async () => {
    const middleware = aiwaf.adonis({ staticKeywords: ['.php'] });
    const ctx = createMockCtx({ path: '/wp-config.php', headers: { accept: 'application/json' } });
    const next = jest.fn();

    await middleware(ctx, next);

    expect(next).not.toHaveBeenCalled();
    expect(ctx.response.statusCode).toBe(403);
    expect(ctx.response.body).toEqual({ error: 'blocked' });
  });

  it('enforces method policy when enabled', async () => {
    const middleware = aiwaf.adonis({
      AIWAF_METHOD_POLICY_ENABLED: true,
      AIWAF_ALLOWED_METHODS: ['GET']
    });
    const ctx = createMockCtx({ path: '/safe', method: 'POST', headers: { accept: 'application/json' } });
    const next = jest.fn();

    await middleware(ctx, next);

    expect(next).not.toHaveBeenCalled();
    expect(ctx.response.statusCode).toBe(405);
    expect(ctx.response.body).toEqual({ error: 'blocked' });
  });

  it('enforces header validation requirements', async () => {
    const middleware = aiwaf.adonis({
      AIWAF_HEADER_VALIDATION: true,
      AIWAF_REQUIRED_HEADERS: ['x-required-security-header']
    });
    const ctx = createMockCtx({ path: '/safe', headers: { accept: 'application/json' } });
    const next = jest.fn();

    await middleware(ctx, next);

    expect(next).not.toHaveBeenCalled();
    expect(ctx.response.statusCode).toBe(403);
    expect(ctx.response.body).toEqual({ error: 'blocked' });
  });

  it('rate limits repeated requests', async () => {
    const middleware = aiwaf.adonis({
      WINDOW_SEC: 60,
      MAX_REQ: 1,
      FLOOD_REQ: 2
    });
    const headers = { accept: 'application/json', 'x-forwarded-for': '198.51.100.61' };

    const ctx1 = createMockCtx({ path: '/safe', headers });
    const next1 = jest.fn();
    await middleware(ctx1, next1);
    expect(next1).toHaveBeenCalled();

    const ctx2 = createMockCtx({ path: '/safe', headers });
    const next2 = jest.fn();
    await middleware(ctx2, next2);
    expect(next2).not.toHaveBeenCalled();
    expect(ctx2.response.statusCode).toBe(429);
    expect(ctx2.response.body).toEqual({ error: 'too_many_requests' });
  });
});
