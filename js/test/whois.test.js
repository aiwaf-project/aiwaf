jest.mock('dns', () => ({
  promises: {
    reverse: jest.fn(async () => ['example.com'])
  }
}));

jest.mock('child_process', () => ({
  execFile: jest.fn((_binary, _args, _options, cb) => cb(null, 'whois output', ''))
}));

const dns = require('dns').promises;
const { execFile } = require('child_process');
const whois = require('../lib/whois');

describe('whois parity helper', () => {
  test('returns domains unchanged', async () => {
    await expect(whois.resolveDomain('example.com')).resolves.toBe('example.com');
  });

  test('reverse-resolves IPs before lookup', async () => {
    await expect(whois.resolveDomain('93.184.216.34')).resolves.toBe('example.com');
    expect(dns.reverse).toHaveBeenCalledWith('93.184.216.34');
  });

  test('runs system whois binary against resolved domain', async () => {
    await expect(whois.runWhoisLookup('example.com')).resolves.toBe('whois output');
    expect(execFile).toHaveBeenCalledWith('whois', ['example.com'], { timeout: 10000 }, expect.any(Function));
  });
});
