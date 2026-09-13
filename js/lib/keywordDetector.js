const { createKeywordMatcher } = require('./wasmAdapter');
let staticKeywords = [];
let matcherPromise = null;
module.exports = {
  init(o) {
    staticKeywords = o.staticKeywords || [];
    matcherPromise = null;
  },
  check(path) { return staticKeywords.find(kw => path.includes(kw)); },
  async checkAccelerated(path) {
    if (!staticKeywords.length) return undefined;
    if (!matcherPromise) matcherPromise = createKeywordMatcher(staticKeywords);
    const pending = matcherPromise;
    const matcher = await pending;
    if (pending !== matcherPromise) return module.exports.check(path);
    return matcher ? matcher.firstMatch(path) : module.exports.check(path);
  }
};
