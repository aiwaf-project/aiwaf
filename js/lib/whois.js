const dns = require('dns').promises;
const { execFile } = require('child_process');
const net = require('net');

async function resolveDomain(target) {
  const candidate = String(target || '').trim();
  if (!candidate) throw new Error('Target is required');
  if (!net.isIP(candidate)) return candidate;

  try {
    const [host] = await dns.reverse(candidate);
    if (host) return host;
  } catch (err) {
    // Convert DNS-specific failures to the Python-compatible contract below.
  }

  throw new Error(`Cannot resolve reverse DNS for IP ${candidate}`);
}

async function runWhoisLookup(target, options = {}) {
  const domain = await resolveDomain(target);
  const binary = options.binary || 'whois';
  return new Promise((resolve, reject) => {
    execFile(binary, [domain], { timeout: Number(options.timeoutMs || 10000) }, (err, stdout, stderr) => {
      if (err) {
        err.stderr = stderr;
        reject(err);
        return;
      }
      resolve(stdout);
    });
  });
}

module.exports = {
  resolveDomain,
  runWhoisLookup
};
