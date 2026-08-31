const reputation = require('../lib/reputation');

describe('reputation policy parity', () => {
  test('matches Python reason weights and progressive durations', () => {
    expect(reputation.reasonWeight('SQL injection probe')).toBe(40);
    expect(reputation.reasonWeight('unknown event')).toBe(10);
    expect(reputation.progressiveDuration(59, 1)).toBeNull();
    expect(reputation.progressiveDuration(60, 1)).toBe(900);
    expect(reputation.progressiveDuration(70, 2)).toBe(3600);
    expect(reputation.progressiveDuration(80, 2)).toBe(86400);
  });

  test('accumulates unique reasons, score, and offenses', () => {
    const first = reputation.evaluateReputation({ reason: 'scanner', now: 100 });
    const second = reputation.evaluateReputation({ existing: first, reason: 'scanner', now: 200 });
    expect(second).toEqual(expect.objectContaining({ score: 40, offenses: 2, reasons: ['scanner'] }));
    expect(reputation.formatBlockReason(second)).toContain('score=40; offenses=2');
  });
});
