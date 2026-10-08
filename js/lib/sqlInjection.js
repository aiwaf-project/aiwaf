// Deterministic, bounded signatures. Findings never include request values.
const RULES = [
  ['sql_union_select', /\bunion\s+(?:all\s+)?select\b/i],
  ['sql_boolean_tautology', /['"`]\s*(?:or|and)\s+(?:true\b|\d+\s*=\s*\d+|['"][^'"\r\n]{0,80}['"]\s*=\s*['"])/i],
  ['sql_quote_comment', /['"`]\s*(?:--|#)/],
  ['sql_stacked_statement', /;\s*(?:drop\s+table|delete\s+from|insert\s+into|update\s+\w+\s+set)\b/i]
];

function createInspector(options = {}) {
  const mode = String(options.AIWAF_SQL_INJECTION_MODE ?? process.env.AIWAF_SQL_INJECTION_MODE ?? 'block').toLowerCase();
  if (!['block', 'monitor', 'off'].includes(mode)) throw new Error('Invalid AIWAF_SQL_INJECTION_MODE');
  const maxBytes = Number(options.AIWAF_PAYLOAD_MAX_BYTES ?? process.env.AIWAF_PAYLOAD_MAX_BYTES ?? 65536);
  if (!Number.isInteger(maxBytes) || maxBytes < 1 || maxBytes > 1048576) throw new Error('AIWAF_PAYLOAD_MAX_BYTES must be 1..1048576');
  return function inspect(req) {
    if (mode === 'off') return null;
    const stack = [[req.body, 0], [req.query, 0]];
    // Also handles raw Node requests whose framework has not populated query.
    const query = String(req.originalUrl || req.url || '').split('?').slice(1).join('?');
    if (query) stack.push([query, 0]);
    let bytes = 0, nodes = 0;
    const seen = new Set();
    const finding = (rule, status = 403) => ({ rule, status, mode });
    while (stack.length) {
      const [value, depth] = stack.pop();
      if (++nodes > 4096 || depth > 16) return finding('payload_inspection_limit', 413);
      if (value === undefined || value === null) continue;
      // Frameworks may expose an unparsed upload stream; never traverse or consume it.
      if (typeof value.pipe === 'function') continue;
      if (Buffer.isBuffer(value)) {
        if (value.length + bytes > maxBytes) return finding('payload_inspection_limit', 413);
        stack.push([value.toString('utf8'), depth + 1]);
      } else if (typeof value === 'object') {
        if (seen.has(value)) return finding('payload_inspection_limit', 413);
        seen.add(value);
        for (const [key, item] of Object.entries(value)) {
          stack.push([key, depth + 1], [item, depth + 1]);
          if (stack.length > 4096) return finding('payload_inspection_limit', 413);
        }
      } else if (typeof value === 'string') {
        bytes += Buffer.byteLength(value);
        if (bytes > maxBytes) return finding('payload_inspection_limit', 413);
        let normalized = value;
        for (let round = 0; round < 2; round++) {
          try { normalized = decodeURIComponent(normalized.replace(/\+/g, ' ')); } catch (_) { break; }
        }
        // Decode serialized JSON escapes before signature matching.
        if (/^\s*[\[{]/.test(normalized)) {
          try {
            const parsed = JSON.parse(normalized);
            stack.push([parsed, depth + 1]);
          } catch (_) { /* Non-JSON strings still receive signature checks. */ }
        }
        normalized = normalized.replace(/\/\*[\s\S]*?\*\//g, ' ');
        for (const [rule, pattern] of RULES) if (pattern.test(normalized)) return finding(rule);
      }
    }
    return null;
  };
}

module.exports = { createInspector };
