const trainingLogic = require('../lib/trainingLogic');

describe('training logic parity helpers', () => {
  test('detects scanning paths', () => {
    expect(trainingLogic.isScanningPath('/wp-admin/install.php')).toBe(true);
    expect(trainingLogic.isScanningPath('/products/123')).toBe(false);
    expect(trainingLogic.isScanningPath('/safe/%2e%2e/passwd')).toBe(true);
  });

  test('provides default legitimate keywords', () => {
    const keywords = trainingLogic.getDefaultLegitimateKeywords();
    expect(keywords.has('profile')).toBe(true);
    expect(keywords.has('dashboard')).toBe(true);
    expect(keywords.has('oauth')).toBe(true);
  });

  test('detects malicious keyword contexts', () => {
    expect(trainingLogic.isMaliciousContext('/a/../.env', 'env', '404', ['env'], () => false)).toBe(true);
    expect(trainingLogic.isMaliciousContext('/real/profile', 'profile', '200', ['profile'], () => true)).toBe(false);
  });
});
