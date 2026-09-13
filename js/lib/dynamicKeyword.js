// lib/dynamicKeyword.js
const dynamicKeywordStore = require('./dynamicKeywordStore');
const { createKeywordMatcher } = require('./wasmAdapter');

let opts, counts;
let activeKeywords = [];
let matcherPromise = null;

function refreshActiveKeywords() {
  activeKeywords = Object.entries(counts)
    .filter(([, count]) => count > opts.dynamicTopN)
    .map(([keyword]) => keyword);
  matcherPromise = null;
}

module.exports = {
  init(o = {}) {
    // normalize option name
    const topN = o.dynamicTopN ?? o.DYNAMIC_TOP_N ?? 10;
    opts = { dynamicTopN: topN };
    counts = {};
    refreshActiveKeywords();
    dynamicKeywordStore.initialize().catch(() => {});
    dynamicKeywordStore.list(2000).then(rows => {
      rows.forEach(row => {
        counts[row.keyword] = Number(row.count || 0);
      });
      refreshActiveKeywords();
    }).catch(() => {});
  },

  learnSegments(segments = []) {
    segments.forEach(s => {
      const key = String(s).toLowerCase();
      if (!key) return;
      const prior = counts[key] || 0;
      counts[key] = (counts[key] || 0) + 1;
      if (prior <= opts.dynamicTopN && counts[key] > opts.dynamicTopN) refreshActiveKeywords();
      dynamicKeywordStore.increment(key).catch(() => {});
    });
  },

  learn(path) {
    const segments = String(path || '').split('/').filter(s => s.length > 3);
    module.exports.learnSegments(segments);
  },

  check(path) {
    // only block segments whose count exceeds the threshold
    const pathLower = String(path || '').toLowerCase();
    return activeKeywords.find(keyword => pathLower.includes(keyword)) || null;
  },

  async checkAccelerated(path) {
    if (!activeKeywords.length) return null;
    const pathLower = String(path || '').toLowerCase();
    if (!matcherPromise) matcherPromise = createKeywordMatcher(activeKeywords);
    const pending = matcherPromise;
    const matcher = await pending;
    if (pending !== matcherPromise) return module.exports.check(path);
    return matcher ? matcher.firstMatch(pathLower) : module.exports.check(path);
  }
};
