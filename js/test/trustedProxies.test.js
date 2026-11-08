const { getRequestIp } = require('../lib/runtimeUtils');

const PUBLIC_PEER = '198.51.100.7';
const PROXY = '10.0.0.1';
const CLIENT = '203.0.113.5';
const SPOOFED = '127.0.0.1';

function request(peer, xff, extra = {}) {
  return {
    socket: { remoteAddress: peer },
    ip: peer,
    headers: xff === undefined ? {} : { 'x-forwarded-for': xff },
    ...extra
  };
}

describe('getRequestIp trusted proxies', () => {
  const originalEnv = process.env.AIWAF_TRUSTED_PROXIES;

  beforeEach(() => {
    delete process.env.AIWAF_TRUSTED_PROXIES;
  });

  afterAll(() => {
    if (originalEnv === undefined) delete process.env.AIWAF_TRUSTED_PROXIES;
    else process.env.AIWAF_TRUSTED_PROXIES = originalEnv;
  });

  test('a direct public client cannot spoof its address', () => {
    expect(getRequestIp(request(PUBLIC_PEER, SPOOFED))).toBe(PUBLIC_PEER);
    expect(getRequestIp(request(`::ffff:${PUBLIC_PEER}`, SPOOFED))).toBe(PUBLIC_PEER);
    expect(getRequestIp(request(PUBLIC_PEER, undefined, { headers: { 'x-real-ip': SPOOFED } }))).toBe(PUBLIC_PEER);
  });

  test.each([
    ['ignores a spoofed prefix', `${SPOOFED}, ${CLIENT}`, CLIENT],
    ['skips trusted hops', `${CLIENT}, 10.0.0.2, 10.0.0.3`, CLIENT],
    ['keeps internal clients from choosing an address', '10.0.0.9, 10.0.0.5', '10.0.0.5'],
    ['stops at a malformed hop', `${SPOOFED}, garbage`, PROXY],
    ['stops at an unknown hop', `${CLIENT}, unknown`, PROXY],
    ['accepts an IPv4 port', `${CLIENT}:4711`, CLIENT],
    ['accepts a bracketed IPv6 port', '[2001:db8::1]:4711', '2001:db8::1'],
    ['normalizes IPv4-mapped addresses', `::ffff:${CLIENT}`, CLIENT],
    ['ignores an oversized header', '1.1.1.1,'.repeat(1000), PROXY]
  ])('from a trusted proxy it %s', (_name, xff, expected) => {
    expect(getRequestIp(request(PROXY, xff))).toBe(expected);
  });

  test('uses X-Real-IP only from a trusted peer without X-Forwarded-For', () => {
    expect(getRequestIp(request(PROXY, undefined, { headers: { 'x-real-ip': CLIENT } }))).toBe(CLIENT);
  });

  test('treats a missing peer as a local Unix socket', () => {
    expect(getRequestIp({ headers: { 'x-forwarded-for': CLIENT } })).toBe(CLIENT);
  });

  test('respects an address the framework already resolved', () => {
    const req = request(PUBLIC_PEER, SPOOFED, { ip: CLIENT });
    expect(getRequestIp(req)).toBe(CLIENT);
  });

  test('explicit configuration overrides the defaults', () => {
    expect(getRequestIp(request(PROXY, CLIENT), [PUBLIC_PEER])).toBe(PROXY);
    expect(getRequestIp(request(PUBLIC_PEER, CLIENT), [`${PUBLIC_PEER}/32`])).toBe(CLIENT);
    expect(getRequestIp(request(PROXY, CLIENT), [])).toBe(PROXY);
  });

  test('reads AIWAF_TRUSTED_PROXIES from the environment', () => {
    process.env.AIWAF_TRUSTED_PROXIES = `${PUBLIC_PEER}/32, 2001:db8::/32`;
    expect(getRequestIp(request(PUBLIC_PEER, CLIENT))).toBe(CLIENT);
    process.env.AIWAF_TRUSTED_PROXIES = '';
    expect(getRequestIp(request(PROXY, CLIENT))).toBe(PROXY);
  });

  test('invalid entries do not widen trust', () => {
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
    expect(getRequestIp(request(PUBLIC_PEER, SPOOFED), ['not-a-network', '10.0.0.0/99'])).toBe(PUBLIC_PEER);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  test('reuses the address the middleware resolved', () => {
    expect(getRequestIp(request(PROXY, CLIENT, { aiwafClientIp: '192.0.2.9' }))).toBe('192.0.2.9');
  });
});
