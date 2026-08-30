let staticKeywords = [];
module.exports = {
  init(o) { staticKeywords = o.staticKeywords || []; },
  check(path) { return staticKeywords.find(kw => path.includes(kw)); }
};