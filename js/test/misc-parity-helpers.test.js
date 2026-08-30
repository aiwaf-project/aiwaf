const fs = require('fs');
const os = require('os');
const path = require('path');
const KeywordFallbackStore = require('../lib/keywordFallbackStore');
const { blockedPayload, blockedResponse, throttleResponse } = require('../lib/blockResponses');
const { isTrustedModelPath } = require('../lib/modelSecurity');
const pathRules = require('../lib/pathRules');

describe('misc python parity helpers', () => {
  test('keyword fallback store persists counters', () => {
    const file = path.join(os.tmpdir(), `aiwaf-keywords-${Date.now()}.json`);
    const store = new KeywordFallbackStore(file);
    store.add('probe', 2);
    store.add('scan', 1);
    store.add('probe', 3);
    expect(store.top(1)).toEqual([['probe', 5]]);
    fs.unlinkSync(file);
  });

  test('block response helpers match shared contract', () => {
    expect(blockedPayload('nope')).toEqual({ error: 'blocked', message: 'nope' });
    expect(blockedResponse('nope')).toEqual({ payload: { error: 'blocked', message: 'nope' }, statusCode: 403 });
    expect(throttleResponse()).toEqual({ payload: { error: 'too_many_requests' }, statusCode: 429 });
  });

  test('model path trust defaults to built-in path only', () => {
    expect(isTrustedModelPath('/tmp/model.json', { defaultPath: '/tmp/model.json' })).toBe(true);
    expect(isTrustedModelPath('/tmp/other.json', { defaultPath: '/tmp/model.json' })).toBe(false);
    expect(isTrustedModelPath('/tmp/other.json', { defaultPath: '/tmp/model.json', allowCustom: true })).toBe(true);
  });

  test('path exemption helper supports exact, prefix, and wildcard matching', () => {
    expect(pathRules.isPathExempt('/health', ['/health'])).toBe(true);
    expect(pathRules.isPathExempt('/api/health/live', ['/api/health'])).toBe(true);
    expect(pathRules.isPathExempt('/assets/app.js', ['/assets/*'])).toBe(true);
  });
});
