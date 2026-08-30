class IsolationTree {
  constructor(depth = 0, maxDepth = 0) {
    this.depth = depth;
    this.maxDepth = maxDepth;
    this.left = null;
    this.right = null;
    this.splitAttr = null;
    this.splitValue = null;
    this.size = 0;
  }

  fit(data) {
    this.size = data.length;
    if (this.depth >= this.maxDepth || data.length <= 1) return;
    const numFeatures = data[0].length;
    this.splitAttr = Math.floor(Math.random() * numFeatures);
    let vals = data.map(row => row[this.splitAttr]);
    const min = Math.min(...vals), max = Math.max(...vals);
    if (min === max) return;
    this.splitValue = min + Math.random() * (max - min);
    const leftData = [], rightData = [];
    for (let row of data) {
      (row[this.splitAttr] < this.splitValue ? leftData : rightData).push(row);
    }
    this.left  = new IsolationTree(this.depth + 1, this.maxDepth);
    this.right = new IsolationTree(this.depth + 1, this.maxDepth);
    this.left.fit(leftData);
    this.right.fit(rightData);
  }

  pathLength(point) {
    if (!this.left || !this.right) {
      return this.depth + c(this.size);
    }
    return point[this.splitAttr] < this.splitValue
      ? this.left.pathLength(point)
      : this.right.pathLength(point);
  }

  serialize() {
    return {
      depth: this.depth,
      maxDepth: this.maxDepth,
      splitAttr: this.splitAttr,
      splitValue: this.splitValue,
      size: this.size,
      left: this.left ? this.left.serialize() : null,
      right: this.right ? this.right.serialize() : null
    };
  }

  static deserialize(obj) {
    const tree = new IsolationTree(obj.depth, obj.maxDepth);
    tree.splitAttr = obj.splitAttr;
    tree.splitValue = obj.splitValue;
    tree.size = obj.size;
    tree.left = obj.left ? IsolationTree.deserialize(obj.left) : null;
    tree.right = obj.right ? IsolationTree.deserialize(obj.right) : null;
    return tree;
  }
}

// Average path length
function c(n) {
  if (n <= 1) return 1;
  return 2 * (Math.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n);
}

class IsolationForest {
  constructor({ nTrees = 100, sampleSize = 256 } = {}) {
    this.nTrees = nTrees;
    this.sampleSize = sampleSize;
    this.trees = [];
  }

  fit(data) {
    const heightLimit = Math.ceil(Math.log2(this.sampleSize));
    for (let i = 0; i < this.nTrees; i++) {
      const sample = [];
      for (let j = 0; j < this.sampleSize; j++) {
        sample.push(data[Math.floor(Math.random() * data.length)]);
      }
      const tree = new IsolationTree(0, heightLimit);
      tree.fit(sample);
      this.trees.push(tree);
    }
  }

  anomalyScore(point) {
    const pathLens = this.trees.map(t => t.pathLength(point));
    const avgPath = pathLens.reduce((a, b) => a + b, 0) / this.nTrees;
    return Math.pow(2, -avgPath / c(this.sampleSize));
  }

  isAnomaly(point, thresh = 0.5) {
    return this.anomalyScore(point) > thresh;
  }

  toJSON() {
    return {
      nTrees: this.nTrees,
      sampleSize: this.sampleSize,
      trees: this.trees.map(t => t.serialize())
    };
  }

  static fromJSON(obj) {
    const forest = new IsolationForest({
      nTrees: obj.nTrees,
      sampleSize: obj.sampleSize
    });
    forest.trees = obj.trees.map(IsolationTree.deserialize);
    return forest;
  }
}

module.exports = { IsolationForest };
