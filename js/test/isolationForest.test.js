const { IsolationForest } = require('../lib/isolationForest');

describe('IsolationForest', () => {
  const originalRandom = Math.random;

  afterEach(() => {
    Math.random = originalRandom;
  });

  it('serializes and deserializes consistently', () => {
    let seed = 0.42;
    Math.random = () => {
      seed = (seed * 9301 + 49297) % 233280;
      return seed / 233280;
    };

    const data = [
      [1, 0, 0],
      [2, 1, 0],
      [3, 0, 1],
      [10, 2, 2],
      [11, 2, 3]
    ];

    const forest = new IsolationForest({ nTrees: 10, sampleSize: 5 });
    forest.fit(data);

    const point = [10, 2, 2];
    const score = forest.anomalyScore(point);

    const restored = IsolationForest.fromJSON(forest.toJSON());
    const restoredScore = restored.anomalyScore(point);

    expect(typeof score).toBe('number');
    expect(typeof restoredScore).toBe('number');
    expect(restoredScore).toBeCloseTo(score, 8);
  });
});
