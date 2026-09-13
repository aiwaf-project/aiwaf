jest.mock('../lib/wasmAdapter', () => ({ createKeywordMatcher: jest.fn() }));
jest.mock('../lib/dynamicKeywordStore', () => ({
  initialize: jest.fn(async () => {}),
  list: jest.fn(async () => [{ keyword: 'admin', count: 5 }]),
  increment: jest.fn(async () => {})
}));

const { createKeywordMatcher } = require('../lib/wasmAdapter');
const keywordDetector = require('../lib/keywordDetector');
const dynamicKeyword = require('../lib/dynamicKeyword');

describe('keyword matcher acceleration', () => {
  beforeEach(() => createKeywordMatcher.mockReset());

  it('uses the Rust matcher for static keywords and preserves configured order', async () => {
    const firstMatch = jest.fn(() => '.env');
    createKeywordMatcher.mockResolvedValue({ firstMatch });
    keywordDetector.init({ staticKeywords: ['.env', 'admin'] });

    expect(await keywordDetector.checkAccelerated('/admin/.env')).toBe('.env');
    expect(createKeywordMatcher).toHaveBeenCalledWith(['.env', 'admin']);
    expect(firstMatch).toHaveBeenCalledWith('/admin/.env');
  });

  it('falls back to JS when the new WASM export is unavailable', async () => {
    createKeywordMatcher.mockResolvedValue(null);
    keywordDetector.init({ staticKeywords: ['.env', 'admin'] });
    expect(await keywordDetector.checkAccelerated('/admin/.env')).toBe('.env');
  });

  it('uses active dynamic keywords and refreshes when a count crosses the threshold', async () => {
    createKeywordMatcher.mockResolvedValue({ firstMatch: path => path.includes('admin') ? 'admin' : null });
    dynamicKeyword.init({ dynamicTopN: 3 });
    await new Promise(resolve => setTimeout(resolve, 0));

    expect(await dynamicKeyword.checkAccelerated('/admin/settings')).toBe('admin');
    expect(createKeywordMatcher).toHaveBeenCalledWith(['admin']);
    dynamicKeyword.learnSegments(['probe', 'probe', 'probe', 'probe']);
    expect(dynamicKeyword.check('/probe')).toBe('probe');
    expect(createKeywordMatcher).toHaveBeenCalledTimes(1);
    await dynamicKeyword.checkAccelerated('/probe');
    expect(createKeywordMatcher).toHaveBeenCalledWith(['admin', 'probe']);
  });
});
