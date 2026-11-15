// aiwaf-wasm's TrustedProxySet must give the same answers as the JavaScript walk it replaces.
const wasmChecks = require('../lib/wasmChecks');
const {
  CLOUDFLARE_PROXIES, getRequestIp, resetUntrustedPeerWarning, _resolveRequestIpInJavaScript
} = require('../lib/runtimeUtils');

const DEFAULT_TRUST = [
  '127.0.0.0/8', '10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', '169.254.0.0/16',
  '::1/128', 'fc00::/7', 'fe80::/10'
];
const TRUST_CONFIGS = {
  default: DEFAULT_TRUST,
  cloudflare: [...DEFAULT_TRUST, ...CLOUDFLARE_PROXIES],
  hosts: ['2001:db8::/32', '203.0.113.9'],
  none: ''
};
const PEERS = [
  '', 'unknown', '10.0.0.1', '10.0.0.1:8080', '203.0.113.5', '::1', '[::1]:443', '[2001:DB8::1]:8443',
  '2001:DB8:0:0:0:0:0:1', '::ffff:10.0.0.1', '::ffff:a00:1', '::ffff:203.0.113.5', '104.16.0.1',
  '2606:4700::1', '2c0f:f248::1', '2001:0db8::0001', '1:2:3:4:5:6:7:8', '1::', '010.0.0.1', '256.1.1.1'
];
const FORWARDED = [
  undefined, '', '203.0.113.5', '127.0.0.1, 203.0.113.5, 10.0.0.2', '203.0.113.5, junk, 10.0.0.2',
  '10.0.0.3, 10.0.0.2', ' 203.0.113.5 ,10.0.0.2 ', '2001:DB8::5, 104.16.0.1', '203.0.113.5,,10.0.0.2',
  Array(40).fill('10.0.0.2').join(', '), ['203.0.113.7', ...Array(31).fill('104.16.0.1')].join(', '),
  '1'.repeat(4096), `${'1'.repeat(4100)},203.0.113.5`, '[2001:DB8::9]:1234, 10.0.0.2', '::ffff:a00:2, 10.0.0.2'
];
const REAL_IPS = [undefined, '', '203.0.113.9', 'unknown', ' 2001:DB8::7 '];

function request(peer, forwardedFor, realIp) {
  const headers = {};
  if (forwardedFor !== undefined) headers['x-forwarded-for'] = forwardedFor;
  if (realIp !== undefined) headers['x-real-ip'] = realIp;
  return { socket: { remoteAddress: peer }, headers };
}

describe('aiwaf-wasm trusted proxy resolution', () => {
  let warn;

  beforeAll(async () => {
    await wasmChecks.load();
  });

  beforeEach(() => {
    delete process.env.AIWAF_TRUSTED_PROXIES;
    warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
    resetUntrustedPeerWarning();
  });

  afterEach(() => {
    warn.mockRestore();
    resetUntrustedPeerWarning();
  });

  test('is used once aiwaf-wasm has loaded', () => {
    const { TrustedProxySet } = wasmChecks.loadedModule();
    expect(typeof TrustedProxySet).toBe('function');
    const resolve = jest.spyOn(TrustedProxySet.prototype, 'resolve');
    expect(getRequestIp(request('10.0.0.1', '2001:DB8::5'))).toBe('2001:db8::5');
    expect(resolve).toHaveBeenCalledWith('10.0.0.1', '2001:DB8::5', '');
    resolve.mockRestore();
  });

  test.each(Object.entries(TRUST_CONFIGS))('agrees with the JavaScript walk: %s', (_name, trusted) => {
    const mismatches = [];
    for (const peer of PEERS) {
      for (const forwardedFor of FORWARDED) {
        for (const realIp of REAL_IPS) {
          const req = request(peer, forwardedFor, realIp);
          const viaWasm = getRequestIp(req, trusted);
          const viaJs = _resolveRequestIpInJavaScript(req, trusted);
          if (viaWasm !== viaJs) mismatches.push({ peer, forwardedFor, realIp, viaWasm, viaJs });
        }
      }
    }
    expect(mismatches).toEqual([]);
  });

  test('reports an untrusted peer once, naming the header', () => {
    expect(getRequestIp(request('198.51.100.7', '203.0.113.5'))).toBe('198.51.100.7');
    expect(getRequestIp(request('198.51.100.8', '203.0.113.5'))).toBe('198.51.100.8');
    const messages = warn.mock.calls.map(([m]) => String(m)).filter(m => m.includes('untrusted peer'));
    expect(messages).toHaveLength(1);
    expect(messages[0]).toContain('X-Forwarded-For');
    expect(messages[0]).toContain('198.51.100.7');
  });

  test('names X-Real-IP when only that header is sent', () => {
    getRequestIp(request('198.51.100.7', undefined, '203.0.113.5'));
    const messages = warn.mock.calls.map(([m]) => String(m)).filter(m => m.includes('untrusted peer'));
    expect(messages[0]).toContain('X-Real-IP');
  });

  test('the cloudflare shortcut works through aiwaf-wasm', () => {
    expect(getRequestIp(request('104.16.0.1', '203.0.113.5'), 'default,cloudflare')).toBe('203.0.113.5');
    expect(getRequestIp(request('198.51.100.7', '203.0.113.5'), 'cloudflare')).toBe('198.51.100.7');
  });
});
