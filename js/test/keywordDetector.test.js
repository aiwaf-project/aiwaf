const keywordDetector = require('../lib/keywordDetector');

describe('keywordDetector', () => {
  test('initializes and checks every configured static keyword', () => {
    keywordDetector.init({ staticKeywords: ['.env', 'wp-admin'] });
    expect(keywordDetector.check('/backup/.env')).toBe('.env');
    expect(keywordDetector.check('/safe')).toBeUndefined();
  });

  test('defaults to an empty keyword list', () => {
    keywordDetector.init({});
    expect(keywordDetector.check('/backup/.env')).toBeUndefined();
  });
});
