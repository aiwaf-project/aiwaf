const { isTrustedModelPath } = require('../lib/modelSecurity');

describe('modelSecurity', () => {
  test('trusts the built-in path and explicitly allowed custom paths', () => {
    expect(isTrustedModelPath('/tmp/model.json', { defaultPath: '/tmp/model.json' })).toBe(true);
    expect(isTrustedModelPath('/tmp/other.json', { defaultPath: '/tmp/model.json' })).toBe(false);
    expect(isTrustedModelPath('/tmp/other.json', {
      defaultPath: '/tmp/model.json',
      allowCustom: true
    })).toBe(true);
  });
});
