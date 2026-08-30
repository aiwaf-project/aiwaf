const exemptionStore = require('./exemptionStore');

let config = {
  exemptPaths: [],
  exemptIps: [],
  allowedPathKeywords: [],
  exemptKeywords: [],
  useDb: true
};

function normalize(items) {
  return (items || []).map(value => String(value).toLowerCase());
}

module.exports = {
  init(opts = {}) {
    config = {
      exemptPaths: normalize(opts.AIWAF_EXEMPT_PATHS),
      exemptIps: normalize(opts.AIWAF_EXEMPT_IPS),
      allowedPathKeywords: normalize(opts.AIWAF_ALLOWED_PATH_KEYWORDS),
      exemptKeywords: normalize(opts.AIWAF_EXEMPT_KEYWORDS),
      useDb: opts.AIWAF_EXEMPTIONS_DB !== false
    };
    if (config.useDb) {
      exemptionStore.initialize().catch(() => {});
    }
  },

  async isExemptRequest(ip, path) {
    const ipLower = String(ip || '').toLowerCase();
    const pathLower = String(path || '').toLowerCase();

    if (config.exemptIps.includes(ipLower)) return true;
    if (config.exemptPaths.some(prefix => pathLower.startsWith(prefix))) return true;

    if (config.useDb) {
      try {
        if (await exemptionStore.isIpExempt(ipLower)) return true;
        if (await exemptionStore.isPathExempt(pathLower)) return true;
      } catch (err) {
        // Exemption storage failure should fail open.
      }
    }

    return false;
  },

  shouldSkipKeyword(keyword, path) {
    const normalizedKeyword = String(keyword || '').toLowerCase();
    const pathLower = String(path || '').toLowerCase();

    if (config.exemptKeywords.includes(normalizedKeyword)) return true;
    if (config.allowedPathKeywords.some(fragment => pathLower.includes(fragment))) return true;
    return false;
  }
};
