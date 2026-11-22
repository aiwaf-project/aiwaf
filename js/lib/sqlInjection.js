'use strict';

// SQL injection checks. The check runs in aiwaf-wasm (sql_inspect), which walks the query and
// body, decodes values, parses JSON, applies the size limits and runs the SQL tokenizer.
// Findings never include request values.
const zlib = require('zlib');
const wasmChecks = require('./wasmChecks');
const { peekBody, isMultipart } = require('./bodyPeek');

// The start of a multipart body, read by peekUpload before a body parser has run.
const UPLOAD = Symbol('aiwaf.upload-prefix');

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

// Every header is passed on. aiwaf-wasm decides which to check: cookies, the Referer and any
// other header except those holding credentials, framing or browser-set values.
function headerPairs(headers) {
  if (!headers || typeof headers !== 'object') return [];
  const found = [];
  for (const [name, value] of Object.entries(headers)) {
    if (value === undefined || value === null) continue;
    for (const item of Array.isArray(value) ? value : [value]) {
      found.push([String(name), String(item)]);
      // One past what aiwaf-wasm checks, so it refuses a request with too many, not skips them.
      if (found.length > 256) return found;
    }
  }
  return found;
}

// A raw body in the encoding it was sent in. aiwaf-wasm reads gzip and deflate itself; br is
// decoded here, to at most maxBytes + 1 bytes, so the check refuses a larger one. Null for an
// encoding that can't be read.
function decoded(body, encoding, maxBytes) {
  const name = String(encoding || 'identity').trim().toLowerCase();
  if (['', 'identity', 'gzip', 'x-gzip', 'deflate'].includes(name)) return body;
  if (name !== 'br') return null;
  try {
    return zlib.brotliDecompressSync(body, { maxOutputLength: maxBytes + 1 });
  } catch (_) {
    return null;
  }
}

/**
 * Reads the start of an unparsed multipart body, then puts it back on the stream, so the
 * inspector checks its text fields, file names and file headers before an upload parser runs.
 * A body that isn't multipart, or was already read, is left alone.
 */
async function peekUpload(req, maxBytes) {
  if (req.body !== undefined || req[UPLOAD] || !isMultipart(req) || typeof req.read !== 'function'
      || typeof req.unshift !== 'function' || req.readableEnded) return;
  const declared = Number(req.headers?.['content-length'] || 0) > 0 || !!req.headers?.['transfer-encoding'];
  if (declared) req[UPLOAD] = await peekBody(req, maxBytes);
}

function createInspector(options = {}) {
  const mode = String(options.AIWAF_SQL_INJECTION_MODE ?? process.env.AIWAF_SQL_INJECTION_MODE ?? 'block').toLowerCase();
  if (!['block', 'monitor', 'off'].includes(mode)) throw new Error('Invalid AIWAF_SQL_INJECTION_MODE');
  const maxBytes = Number(options.AIWAF_PAYLOAD_MAX_BYTES ?? process.env.AIWAF_PAYLOAD_MAX_BYTES ?? 65536);
  if (!Number.isInteger(maxBytes) || maxBytes < 1 || maxBytes > 1048576) throw new Error('AIWAF_PAYLOAD_MAX_BYTES must be 1..1048576');
  /** Checks a request. Await wasmChecks.load(), and inspect.prepare(req), before the call. */
  inspect.prepare = req => (mode === 'off' ? Promise.resolve() : peekUpload(req, maxBytes));
  inspect.maxBytes = maxBytes;
  return inspect;
  function inspect(req) {
    if (mode === 'off') return null;
    const wasm = wasmChecks.get();
    // Raw Node requests have only a URL. Frameworks may also have parsed the query.
    const raw = String(req.originalUrl || req.url || '').split('?').slice(1).join('?');
    let body = req.body === undefined && req[UPLOAD]
      // A large upload is checked by its start: its text fields and file headers come first.
      ? req[UPLOAD].subarray(0, Math.max(0, maxBytes - Buffer.byteLength(raw)))
      : bodyBytes(req.body);
    // Only a raw body is still encoded: a parsed one was decoded by its parser.
    const unparsed = Buffer.isBuffer(req.body) || req.body instanceof Uint8Array || typeof req.body === 'string';
    if (body !== null && body.length && unparsed) body = decoded(body, req.headers?.['content-encoding'], maxBytes);
    if (body === null) return { rule: 'payload_inspection_limit', status: 413, mode };
    // Where the finding came from, so the route's saved inputs can review it.
    let source = 'payload';
    let found = JSON.parse(wasm.sql_inspect(Buffer.from(raw), body, maxBytes));
    if (!found && !raw && req.query && typeof req.query === 'object') {
      const query = bodyBytes(req.query);
      found = query === null
        ? { rule: 'payload_inspection_limit', status: 413 }
        : JSON.parse(wasm.sql_inspect(EMPTY, query, maxBytes));
    }
    if (!found) {
      source = 'headers';
      found = JSON.parse(wasm.sql_inspect_headers(JSON.stringify(headerPairs(req.headers)), maxBytes));
    }
    // The URL path, one segment at a time, as in /post/1 OR 1=1.
    const path = String(req.originalUrl || req.url || '').split('?')[0];
    if (!found && path) {
      source = 'path';
      found = JSON.parse(wasm.sql_inspect_path(path, maxBytes));
    }
    return found ? { rule: found.rule, status: found.status, mode, source } : null;
  }
}

module.exports = { createInspector, bodyBytes, peekUpload };
