const fs = require('fs');
const os = require('os');
const path = require('path');
const KeywordFallbackStore = require('../lib/keywordFallbackStore');

describe('keywordFallbackStore', () => {
  test('persists and ranks keyword counters', () => {
    const file = path.join(os.tmpdir(), `aiwaf-keywords-${Date.now()}.json`);
    const store = new KeywordFallbackStore(file);
    store.add('probe', 2);
    store.add('scan', 1);
    store.add('probe', 3);
    expect(store.top(1)).toEqual([['probe', 5]]);
    fs.unlinkSync(file);
  });
});
