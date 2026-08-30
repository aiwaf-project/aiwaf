const { EventEmitter } = require('events');
const aiwaf = require('../index');

process.env.NODE_ENV = 'test';

function createMockRes() {
  const res = new EventEmitter();
  res.statusCode = 200;
  res.headersSent = false;
  res.writableEnded = false;
  res.finished = false;
  res.headers = {};
  res.setHeader = (key, value) => {
    res.headers[key.toLowerCase()] = value;
  };
  res.status = (code) => {
    res.statusCode = code;
    return res;
  };
  res.json = (payload) => {
    res.body = payload;
    res.headersSent = true;
    res.writableEnded = true;
    res.finished = true;
    res.emit('finish');
    return res;
  };
  res.send = (payload) => {
    res.body = payload;
    res.headersSent = true;
    res.writableEnded = true;
    res.finished = true;
    res.emit('finish');
    return res;
  };
  res.end = (payload) => {
    if (payload !== undefined) {
      if (res.headers['content-type'] === 'application/json' && typeof payload === 'string') {
        try {
          res.body = JSON.parse(payload);
        } catch (err) {
          res.body = payload;
        }
      } else {
        res.body = payload;
      }
    }
    res.headersSent = true;
    res.writableEnded = true;
    res.finished = true;
    res.emit('finish');
    return res;
  };
  return res;
}

function createMockReq({ path = '/', method = 'GET', headers = {}, ip = '198.51.100.50' } = {}) {
  return {
    url: path,
    method,
    headers,
    ip,
    app: { _router: { stack: [] } }
  };
}

describe('AIWAF Next.js handler wrapper', () => {
  it('allows safe requests', async () => {
    const handler = jest.fn((req, res) => res.status(200).json({ ok: true }));
    const wrapped = aiwaf.next(handler, { WINDOW_SEC: 60, MAX_REQ: 1000, FLOOD_REQ: 2000 });
    const req = createMockReq({ path: '/safe' });
    const res = createMockRes();

    await wrapped(req, res);

    expect(handler).toHaveBeenCalled();
    expect(res.statusCode).toBe(200);
  });

  it('supports factory wrapper signature', async () => {
    const handler = jest.fn((req, res) => res.status(200).json({ ok: true }));
    const wrap = aiwaf.next({ WINDOW_SEC: 60, MAX_REQ: 1000, FLOOD_REQ: 2000 });
    const wrapped = wrap(handler);
    const req = createMockReq({ path: '/safe' });
    const res = createMockRes();

    await wrapped(req, res);

    expect(handler).toHaveBeenCalled();
    expect(res.statusCode).toBe(200);
  });

  it('blocks static keyword paths', async () => {
    const handler = jest.fn();
    const wrapped = aiwaf.next(handler, { staticKeywords: ['.php'] });
    const req = createMockReq({ path: '/wp-config.php', headers: { accept: 'application/json' } });
    const res = createMockRes();

    await wrapped(req, res);

    expect(handler).not.toHaveBeenCalled();
    expect(res.statusCode).toBe(403);
    expect(res.body).toEqual({ error: 'blocked' });
  });

  it('returns text payload when JSON is not requested', async () => {
    const handler = jest.fn();
    const wrapped = aiwaf.next(handler, { staticKeywords: ['.php'] });
    const req = createMockReq({ path: '/wp-config.php' });
    const res = createMockRes();

    await wrapped(req, res);

    expect(handler).not.toHaveBeenCalled();
    expect(res.statusCode).toBe(403);
    expect(res.body).toEqual({ error: 'blocked' });
  });

  it('enforces method policy when enabled', async () => {
    const handler = jest.fn();
    const wrapped = aiwaf.next(handler, {
      AIWAF_METHOD_POLICY_ENABLED: true,
      AIWAF_ALLOWED_METHODS: ['GET']
    });
    const req = createMockReq({ path: '/safe', method: 'POST', headers: { accept: 'application/json' } });
    const res = createMockRes();

    await wrapped(req, res);

    expect(handler).not.toHaveBeenCalled();
    expect(res.statusCode).toBe(405);
    expect(res.body).toEqual({ error: 'blocked' });
  });

  it('enforces header validation requirements', async () => {
    const handler = jest.fn();
    const wrapped = aiwaf.next(handler, {
      AIWAF_HEADER_VALIDATION: true,
      AIWAF_REQUIRED_HEADERS: ['x-required-security-header']
    });
    const req = createMockReq({
      path: '/safe',
      headers: { accept: 'application/json' }
    });
    const res = createMockRes();

    await wrapped(req, res);

    expect(handler).not.toHaveBeenCalled();
    expect(res.statusCode).toBe(403);
    expect(res.body).toEqual({ error: 'blocked' });
  });

  it('rate limits repeated requests', async () => {
    const handler = jest.fn((req, res) => res.status(200).json({ ok: true }));
    const wrapped = aiwaf.next(handler, {
      WINDOW_SEC: 60,
      MAX_REQ: 1,
      FLOOD_REQ: 2
    });
    const headers = { accept: 'application/json', 'x-forwarded-for': '198.51.100.51' };

    const firstReq = createMockReq({ path: '/safe', headers });
    const firstRes = createMockRes();
    await wrapped(firstReq, firstRes);
    expect(firstRes.statusCode).toBe(200);

    const secondReq = createMockReq({ path: '/safe', headers });
    const secondRes = createMockRes();
    await wrapped(secondReq, secondRes);
    expect(secondRes.statusCode).toBe(429);
    expect(secondRes.body).toEqual({ error: 'too_many_requests' });
  });
});
