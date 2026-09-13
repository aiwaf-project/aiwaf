const { IsolationForest } = require('./isolationForest');
const {
  createIsolationForest,
  createIsolationForestFromJSON,
  getWasmStatus,
  analyzeRecentBehavior: analyzeRecentBehaviorWasm
} = require('./wasmAdapter');
const modelStore = require('./modelStore');
const requestLogStore = require('./requestLogStore');
const { STATIC_KW } = require('./featureUtils');
const dynamicKeyword = require('./dynamicKeyword');
const exemptions = require('./exemptions');
const trainingLogic = require('./trainingLogic');

let model;
let trained = false;
let modelMetadata = null;
let loadStarted = false;
let minAiLogs = 0;
let aiLogsSufficient = true;
let aiLogCount = null;
let wasmStatus = { loaded: false, error: null };

async function loadModel(opts = {}) {
  if (loadStarted) return;
  loadStarted = true;
  try {
    const modelData = await modelStore.load(opts);
    if (!modelData) return;

    if (modelData.model_type === 'aiwaf_wasm.IsolationForest' && modelData.model_state) {
      const restored = await createIsolationForestFromJSON(modelData.model_state);
      if (!restored.__aiwafWasm) throw new Error('WASM model cannot be restored without aiwaf-wasm');
      model = restored;
      modelMetadata = modelData.metadata || null;
      trained = true;
      return;
    }

    if (modelData.metadata) {
      modelMetadata = modelData.metadata;
      model = IsolationForest.fromJSON(modelData);
      trained = true;
      console.log(`Pretrained anomaly model loaded (${modelMetadata.samplesCount} samples, created: ${modelMetadata.createdAt})`);
      return;
    }

    model = IsolationForest.fromJSON(modelData);
    trained = true;
    console.log('Pretrained anomaly model loaded (legacy format)');
  } catch (err) {
    console.warn('Failed to load pretrained model:', err.message);
  }
}

async function countRecentDbLogs(days = 30) {
  try {
    const rows = await requestLogStore.recent(20000);
    const cutoff = Date.now() - days * 24 * 60 * 60 * 1000;
    return rows.filter(row => {
      const ts = row.created_at ? new Date(row.created_at).getTime() : 0;
      return ts >= cutoff;
    }).length;
  } catch (err) {
    return 0;
  }
}

async function checkAiLogSufficiency() {
  if (minAiLogs <= 0) {
    aiLogCount = null;
    aiLogsSufficient = true;
    return;
  }

  const dbCount = await countRecentDbLogs(30);
  aiLogCount = dbCount;
  aiLogsSufficient = dbCount >= minAiLogs;
}

const { isScanningPath } = trainingLogic;

function analyzeRecentBehavior(recentData = []) {
  const recentKwHits = [];
  let recent404s = 0;
  const recentBurstCounts = [];
  const timestamps = recentData.map(entry => Number(entry.timestamp || 0)).sort((a, b) => a - b);

  const lowerBound = target => {
    let left = 0;
    let right = timestamps.length;
    while (left < right) {
      const mid = (left + right) >>> 1;
      if (timestamps[mid] < target) left = mid + 1;
      else right = mid;
    }
    return left;
  };
  const upperBound = target => {
    let left = 0;
    let right = timestamps.length;
    while (left < right) {
      const mid = (left + right) >>> 1;
      if (timestamps[mid] <= target) left = mid + 1;
      else right = mid;
    }
    return left;
  };

  for (const entry of recentData) {
    const entryTime = entry.timestamp || 0;
    const entryPath = String(entry.path || '');
    const entryStatus = entry.status;

    let entryKwHits = 0;
    if (!exemptions.shouldSkipKeyword('', entryPath)) {
      entryKwHits = STATIC_KW.reduce(
        (count, kw) => count + (entryPath.toLowerCase().includes(kw) ? 1 : 0),
        0
      );
    }
    recentKwHits.push(entryKwHits);

    if (entryStatus === 404) {
      recent404s += 1;
    }

    const entryBurst = upperBound(entryTime + 10000) - lowerBound(entryTime - 10000);
    recentBurstCounts.push(entryBurst);
  }

  const avgKwHits = recentKwHits.length
    ? recentKwHits.reduce((sum, v) => sum + v, 0) / recentKwHits.length
    : 0;
  const max404s = recent404s;
  const avgBurst = recentBurstCounts.length
    ? recentBurstCounts.reduce((sum, v) => sum + v, 0) / recentBurstCounts.length
    : 0;
  const totalRequests = recentData.length;
  const scanning404s = recentData.filter(entry => entry.status === 404 && isScanningPath(entry.path)).length;
  const legitimate404s = Math.max(max404s - scanning404s, 0);

  let shouldBlock = true;
  if (max404s === 0 && avgKwHits === 0 && scanning404s === 0) {
    shouldBlock = false;
  } else if (
    avgKwHits < 3
    && scanning404s < 5
    && legitimate404s < 20
    && avgBurst < 25
    && totalRequests < 150
  ) {
    shouldBlock = false;
  }

  return {
    avg_kw_hits: avgKwHits,
    max_404s: max404s,
    avg_burst: avgBurst,
    total_requests: totalRequests,
    scanning_404s: scanning404s,
    legitimate_404s: legitimate404s,
    should_block: shouldBlock
  };
}

async function analyzeRecentBehaviorAccelerated(recentData = []) {
  // Older aiwaf-wasm releases classify these ordinary discovery paths as scans.
  // Also keep the JS semantics for malformed timestamps or non-numeric statuses.
  const safeForWasm = recentData.every(entry =>
    typeof entry.status === 'number'
    && Number.isFinite(entry.timestamp)
    && !/robots\.txt|sitemap\.xml/.test(String(entry.path || '').toLowerCase())
  );
  if (recentData.length > 0 && safeForWasm) {
    const entries = recentData.map(entry => ({
      ...entry,
      timestamp_ms: Number(entry.timestamp || 0),
      kw_check: !exemptions.shouldSkipKeyword('', String(entry.path || ''))
    }));
    const stats = await analyzeRecentBehaviorWasm(entries, STATIC_KW);
    if (stats && typeof stats.should_block === 'boolean' && Number.isFinite(Number(stats.avg_burst))) {
      return stats;
    }
  }
  return analyzeRecentBehavior(recentData);
}

module.exports = {
  async init(opts = {}) {
    minAiLogs = Number.isFinite(Number(opts.AIWAF_MIN_AI_LOGS))
      ? Number(opts.AIWAF_MIN_AI_LOGS)
      : 10000; // Default: require 10k training samples
    await checkAiLogSufficiency();
    await loadModel(opts);
    if (!model) {
      model = await createIsolationForest({
        nTrees: opts.nTrees || 100,
        sampleSize: opts.sampleSize || 256,
        threshold: opts.threshold || 0.5
      });
      wasmStatus = getWasmStatus();
    }

    if (model && !aiLogsSufficient) {
      model = null;
      trained = false;
      if (aiLogCount !== null) {
        console.log(`AIWAF AI model disabled due to insufficient logs (${aiLogCount}/${minAiLogs}). Require at least ${minAiLogs} logs before using AI detection.`);
      } else {
        console.log(`AIWAF AI model disabled due to insufficient logs (unknown/${minAiLogs}). Require at least ${minAiLogs} logs before using AI detection.`);
      }
    }
  },

  train(data) {
    model.fit(data);
    trained = true;
  },

  hasModel() {
    return !!model && trained;
  },

  isModelSufficientlyTrained() {
    // Model must exist AND have enough training data
    return !!model && trained && aiLogsSufficient;
  },

  // Expects a feature vector: [pathLen, kwHits, statusIdx, responseTime, burst, total404]
  isAnomalous(features, threshold = 0.5) {
    if (!trained || !model) {
      return false;
    }

    try {
      return model.isAnomaly(features, threshold);
    } catch (err) {
      console.warn('Error in anomaly detection:', err.message);
      return false;
    }
  },

  analyzeRecentBehavior,
  analyzeRecentBehaviorAccelerated,
  isScanningPath,

  maybeLearnKeyword(path, statusCode, opts = {}) {
    if (!opts.AIWAF_ENABLE_KEYWORD_LEARNING) return;
    if (statusCode !== 404) return;
    const pathLower = String(path || '').toLowerCase();
    if (!pathLower || exemptions.shouldSkipKeyword('', pathLower)) return;

    const segments = pathLower.split(/\W+/).filter(seg => seg.length > 3);
    if (segments.length === 0) return;

    const suspicious = segments.some(seg => {
      if (STATIC_KW.includes(seg)) return false;
      if (pathLower.includes('../') || pathLower.includes('..\\')) return true;
      if (pathLower.includes('%2e%2e') || pathLower.includes('%252e') || pathLower.includes('%c0%ae')) return true;
      if (pathLower.includes(seg) && isScanningPath(pathLower)) return true;
      return false;
    });

    if (!suspicious) return;
    dynamicKeyword.learnSegments(segments);
  },

  getModelInfo() {
    return {
      trained,
      metadata: modelMetadata,
      threshold: 0.5,
      minAiLogs,
      aiLogsSufficient,
      aiLogCount,
      wasm: wasmStatus
    };
  }
};
