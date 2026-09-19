const { extractWasmTrainingFeatures, extractWasmRawTrainingFeatures } = require('./wasmAdapter');

function modelFeaturesFromRust(features, expectedLength) {
  if (!Array.isArray(features) || features.length !== expectedLength
      || !features.every(rec => rec && Number.isFinite(Number(rec.burst_count)))) return null;
  return features.map(rec => [
    rec.path_len, rec.kw_hits, rec.status_idx, rec.resp_time, rec.burst_count, rec.total_404
  ]);
}

function lowerBound(values, target) {
  let left = 0;
  let right = values.length;
  while (left < right) {
    const mid = (left + right) >>> 1;
    if (values[mid] < target) left = mid + 1;
    else right = mid;
  }
  return left;
}

function upperBound(values, target) {
  let left = 0;
  let right = values.length;
  while (left < right) {
    const mid = (left + right) >>> 1;
    if (values[mid] <= target) left = mid + 1;
    else right = mid;
  }
  return left;
}

async function calculateTrainingFeatures(parsedRequests, staticKeywords, statusIndices) {
  const keywords = staticKeywords || [];
  const statuses = statusIndices || [];
  // Direct JS-object parsing currently costs more than the prepared-record path
  // on typical Node workloads; keep it opt-in until workload benchmarks improve.
  if (process.env.AIWAF_EXPERIMENTAL_WASM_RAW_TRAINING === '1'
      && parsedRequests.length && keywords.every(kw => kw === String(kw).toLowerCase())
      && typeof extractWasmRawTrainingFeatures === 'function') {
    const rawFeatures = await extractWasmRawTrainingFeatures(parsedRequests, keywords, statuses);
    const mapped = modelFeaturesFromRust(rawFeatures, parsedRequests.length);
    if (mapped) return mapped;
  }
  const ip404Counts = new Map();
  const timestampsByIp = new Map();

  for (const req of parsedRequests) {
    if (!timestampsByIp.has(req.ip)) timestampsByIp.set(req.ip, []);
    const timestamp = req.timestamp instanceof Date
      ? req.timestamp.getTime()
      : new Date(req.timestamp).getTime();
    if (Number.isFinite(timestamp)) timestampsByIp.get(req.ip).push(timestamp);
    if (req.status === '404') {
      ip404Counts.set(req.ip, (ip404Counts.get(req.ip) || 0) + 1);
    }
  }
  for (const timestamps of timestampsByIp.values()) timestamps.sort((a, b) => a - b);

  // JS's persisted model format uses statusIdx before responseTime. Preserve that order.
  const records = parsedRequests.map(req => {
    const path = String(req.path || '');
    const timestampMs = req.timestamp instanceof Date
      ? req.timestamp.getTime()
      : new Date(req.timestamp).getTime();
    return {
      ip: String(req.ip),
      path_lower: path.toLowerCase(),
      path_len: path.length,
      timestamp: timestampMs / 1000,
      response_time: Number(req.responseTime),
      status_idx: statuses.indexOf(req.status),
      kw_check: true,
      total_404: ip404Counts.get(req.ip) || 0
    };
  });

  const canUseRust = records.every(rec => Number.isFinite(rec.timestamp) && Number.isFinite(rec.response_time))
    && keywords.every(kw => kw === String(kw).toLowerCase());
  if (canUseRust && records.length > 0) {
    const rustFeatures = await extractWasmTrainingFeatures(records, keywords);
    const mapped = modelFeaturesFromRust(rustFeatures, records.length);
    if (mapped) return mapped;
  }

  return parsedRequests.map((req, index) => {
    const path = String(req.path || '');
    const timestampMs = records[index].timestamp * 1000;
    const times = timestampsByIp.get(req.ip) || [];
    const burst = Number.isFinite(timestampMs)
      ? upperBound(times, timestampMs + 10000) - lowerBound(times, timestampMs - 10000)
      : 0;
    const kwHits = keywords.reduce((sum, kw) => sum + (path.toLowerCase().includes(kw) ? 1 : 0), 0);
    return [path.length, kwHits, statuses.indexOf(req.status), req.responseTime, burst, ip404Counts.get(req.ip) || 0];
  });
}

module.exports = { calculateTrainingFeatures };
