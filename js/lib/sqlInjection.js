'use strict';

// SQL injection checks. The check runs in aiwaf-wasm (sql_inspect), which walks the query and
// body, decodes values, parses JSON, applies the size limits and runs the SQL tokenizer.
// Findings never include request values.
const wasmChecks = require('./wasmChecks');

const EMPTY = new Uint8Array(0);

// Returns body bytes, or null when the body can't be serialized, such as one that contains itself.
function bodyBytes(body) {
  if (body === undefined || body === null) return EMPTY;
  if (Buffer.isBuffer(body) || body instanceof Uint8Array) return body;
  if (typeof body === 'string') return Buffer.from(body);
  // Frameworks may expose an unparsed upload stream; never read or consume it.
  if (typeof body.pipe === 'function') return EMPTY;
  try {
    return Buffer.from(JSON.stringify(body) ?? '');
  } catch (_) {
    return null;
  }
}

function createInspector(options = {}) {
  const mode = String(options.AIWAF_SQL_INJECTION_MODE ?? process.env.AIWAF_SQL_INJECTION_MODE ?? 'block').toLowerCase();
  if (!['block', 'monitor', 'off'].includes(mode)) throw new Error('Invalid AIWAF_SQL_INJECTION_MODE');
  const maxBytes = Number(options.AIWAF_PAYLOAD_MAX_BYTES ?? process.env.AIWAF_PAYLOAD_MAX_BYTES ?? 65536);
  if (!Number.isInteger(maxBytes) || maxBytes < 1 || maxBytes > 1048576) throw new Error('AIWAF_PAYLOAD_MAX_BYTES must be 1..1048576');
  /** Checks a request. Await wasmChecks.load() before the first call. */
  return function inspect(req) {
    if (mode === 'off') return null;
    const wasm = wasmChecks.get();
    const body = bodyBytes(req.body);
    if (body === null) return { rule: 'payload_inspection_limit', status: 413, mode };
    // Raw Node requests have only a URL. Frameworks may also have parsed the query.
    const raw = String(req.originalUrl || req.url || '').split('?').slice(1).join('?');
    let found = JSON.parse(wasm.sql_inspect(Buffer.from(raw), body, maxBytes));
    if (!found && !raw && req.query && typeof req.query === 'object') {
      const query = bodyBytes(req.query);
      found = query === null
        ? { rule: 'payload_inspection_limit', status: 413 }
        : JSON.parse(wasm.sql_inspect(EMPTY, query, maxBytes));
    }
    return found ? { rule: found.rule, status: found.status, mode } : null;
  };
}

module.exports = { createInspector, bodyBytes };
