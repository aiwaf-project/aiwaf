'use strict';

// WebSocket messages, checked like request bodies. HTTP checks never see what a WebSocket
// carries after its handshake, so this checks each message a client sends with the payload
// check in aiwaf-wasm: SQL injection, remote code execution and their disguises, and JSON
// messages by their values. A message with a finding closes the socket with 1008 (policy
// violation), or 1009 when it's too large to check, and never reaches the application's
// handlers. It follows AIWAF_SQL_INJECTION_MODE: monitor logs the rule and passes the message,
// off checks nothing. Log events carry the rule and path only, never message content.
const wasmChecks = require('./wasmChecks');
const { bodyBytes } = require('./sqlInjection');

const EMPTY = new Uint8Array(0);

function settings(options = {}) {
  const mode = String(options.AIWAF_SQL_INJECTION_MODE ?? process.env.AIWAF_SQL_INJECTION_MODE ?? 'block').toLowerCase();
  if (!['block', 'monitor', 'off'].includes(mode)) throw new Error('Invalid AIWAF_SQL_INJECTION_MODE');
  const maxBytes = Number(options.AIWAF_PAYLOAD_MAX_BYTES ?? process.env.AIWAF_PAYLOAD_MAX_BYTES ?? 65536);
  if (!Number.isInteger(maxBytes) || maxBytes < 1 || maxBytes > 1048576) throw new Error('AIWAF_PAYLOAD_MAX_BYTES must be 1..1048576');
  return { mode, maxBytes, logger: options.logger || console };
}

// A message as bytes: ws gives a Buffer, an array of Buffers or an ArrayBuffer; socket.io gives
// event arguments, which are checked as JSON.
function messageBytes(data) {
  if (Buffer.isBuffer(data) || data instanceof Uint8Array) return data;
  if (Array.isArray(data) && data.every(item => Buffer.isBuffer(item))) return Buffer.concat(data);
  if (data instanceof ArrayBuffer) return Buffer.from(data);
  return bodyBytes(data);
}

/** The finding for one message, or null. Await wasmChecks.load() first. Ignores the mode. */
function inspectMessage(data, options = {}) {
  const { maxBytes } = settings(options);
  const bytes = messageBytes(data);
  if (bytes === null) return { rule: 'payload_inspection_limit', status: 413 };
  return JSON.parse(wasmChecks.get().sql_inspect(EMPTY, bytes, maxBytes));
}

function report(logger, mode, path, finding) {
  try {
    logger.warn?.({ event: 'aiwaf.websocket', mode, path: String(path || ''), rule: finding.rule });
  } catch (_) { /* Logging must not decide the outcome. */ }
}

/**
 * Checks every message clients send to a `ws` WebSocketServer. Resolves once the checks are
 * loaded; messages are checked from then on.
 */
async function guardWebSocketServer(wss, options = {}) {
  const config = settings(options);
  if (config.mode === 'off') return wss;
  await wasmChecks.load();
  wss.on('connection', (socket, req) => {
    const path = String(req?.url || '').split('?')[0];
    let closed = false;
    const emit = socket.emit;
    socket.emit = function emitChecked(event, ...args) {
      if (event === 'message') {
        if (closed) return false;
        const finding = inspectMessage(args[0], options);
        if (finding) {
          report(config.logger, config.mode, path, finding);
          if (config.mode === 'block') {
            closed = true;
            socket.close(finding.status === 413 ? 1009 : 1008, 'aiwaf');
            return false;
          }
        }
      }
      return emit.call(this, event, ...args);
    };
  });
  return wss;
}

/** Checks every event clients send to a socket.io server, with per-socket packet middleware. */
async function guardSocketIo(io, options = {}) {
  const config = settings(options);
  if (config.mode === 'off') return io;
  await wasmChecks.load();
  io.on('connection', socket => {
    const path = String(socket.handshake?.url || '').split('?')[0];
    socket.use(([event, ...args], next) => {
      // The event name and its arguments, except an acknowledgement callback.
      const finding = inspectMessage([event, ...args.filter(item => typeof item !== 'function')], options);
      if (!finding) return next();
      report(config.logger, config.mode, path, finding);
      if (config.mode !== 'block') return next();
      next(new Error(finding.rule));
      socket.disconnect(true);
    });
  });
  return io;
}

module.exports = { guardSocketIo, guardWebSocketServer, inspectMessage };
