// lib/dynamicKeyword.js
const dynamicKeywordStore = require('./dynamicKeywordStore');

let opts, counts;

module.exports = {
  init(o = {}) {
    // normalize option name
    const topN = o.dynamicTopN ?? o.DYNAMIC_TOP_N ?? 10;
    opts = { dynamicTopN: topN };
    counts = {};
    dynamicKeywordStore.initialize().catch(() => {});
    dynamicKeywordStore.list(2000).then(rows => {
      rows.forEach(row => {
        counts[row.keyword] = Number(row.count || 0);
      });
    }).catch(() => {});
  },

  learnSegments(segments = []) {
    segments.forEach(s => {
      const key = String(s).toLowerCase();
      if (!key) return;
      counts[key] = (counts[key] || 0) + 1;
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
    return Object.entries(counts)
      .find(([seg, cnt]) => cnt > opts.dynamicTopN && pathLower.includes(seg))
      ?. [0]  // return the segment string
    || null;
  }
};
