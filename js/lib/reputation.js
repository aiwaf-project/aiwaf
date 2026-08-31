const REASON_WEIGHTS = Object.freeze({
  scanner: 20, scan: 20, sqli: 40, 'sql injection': 40, xss: 30,
  bruteforce: 25, 'brute force': 25, flood: 25, 'rate limit': 20,
  honeypot: 30, uuid: 25, header: 15, geo: 20, keyword: 20
});
const DEFAULT_REASON_WEIGHT = 10;
const BLOCK_THRESHOLD = 60;
const LONG_BLOCK_THRESHOLD = 80;
const FIRST_BLOCK_SECONDS = 15 * 60;
const SECOND_BLOCK_SECONDS = 60 * 60;
const REPEATED_BLOCK_SECONDS = 24 * 60 * 60;

function normalizeReason(reason) {
  return String(reason || 'unknown').trim() || 'unknown';
}

function reasonWeight(reason) {
  const normalized = normalizeReason(reason).toLowerCase();
  for (const [token, weight] of Object.entries(REASON_WEIGHTS)) {
    if (normalized.includes(token)) return weight;
  }
  return DEFAULT_REASON_WEIGHT;
}

function uniqueReasons(values = []) {
  const seen = new Set();
  return values.reduce((result, value) => {
    const reason = normalizeReason(value);
    const key = reason.toLowerCase();
    if (!seen.has(key)) { seen.add(key); result.push(reason); }
    return result;
  }, []);
}

function progressiveDuration(score, offenses) {
  if (score < BLOCK_THRESHOLD) return null;
  if (score >= LONG_BLOCK_THRESHOLD || offenses >= 3) return REPEATED_BLOCK_SECONDS;
  if (offenses === 2) return SECOND_BLOCK_SECONDS;
  return FIRST_BLOCK_SECONDS;
}

function evaluateReputation({ existing = {}, reason, now = Date.now() / 1000 } = {}) {
  const previousReasons = Array.isArray(existing.reasons)
    ? existing.reasons
    : (existing.reasons || existing.reason ? [existing.reasons || existing.reason] : []);
  const offenses = Number(existing.offenses || 0) + 1;
  const score = Math.min(100, Number(existing.score || 0) + reasonWeight(reason));
  const duration = progressiveDuration(score, offenses);
  return {
    score,
    offenses,
    reasons: uniqueReasons([...previousReasons, normalizeReason(reason)]),
    shouldBlock: score >= BLOCK_THRESHOLD,
    duration,
    expiresAt: duration ? Number(now) + duration : null
  };
}

function formatBlockReason(decision) {
  return `${decision.reasons.join(', ')}; score=${decision.score}; offenses=${decision.offenses}`;
}

module.exports = {
  REASON_WEIGHTS, DEFAULT_REASON_WEIGHT, BLOCK_THRESHOLD, LONG_BLOCK_THRESHOLD,
  FIRST_BLOCK_SECONDS, SECOND_BLOCK_SECONDS, REPEATED_BLOCK_SECONDS,
  normalizeReason, reasonWeight, progressiveDuration, evaluateReputation, formatBlockReason
};
