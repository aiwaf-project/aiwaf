'use strict';

// Records which query and body names each Express (or Nest on Express) handler reads, while tests
// or staging traffic run. Set AIWAF_RECORD_INPUTS to a file path (or 1 for
// .aiwaf/observed_inputs.jsonl) and add app.use(aiwaf.recordInputs()) before your routes.
//
// Each request appends a line per name read: {"route", "method", "key"}, with key * when req.query
// or req.body was read whole (spread, Object.keys, JSON.stringify). Names are what the code read,
// not what the client sent. init reads the file (see inputSources.js). Without AIWAF_RECORD_INPUTS
// the middleware only calls next().

const fs = require('fs');
const path = require('path');

const ENVIRONMENT = 'AIWAF_RECORD_INPUTS';
const DEFAULT_PATH = path.join('.aiwaf', 'observed_inputs.jsonl');
const WHOLE = '*';
// Read by CSRF middleware on every form post, not by the handler.
const IGNORED = new Set(['_csrf', 'csrf_token', 'csrfmiddlewaretoken']);
const SKIPPED = new Set(['toJSON', 'then', 'constructor', '__proto__', 'inspect']);

/** Where to record, from AIWAF_RECORD_INPUTS (or value), or null when recording is off. */
function destination(value) {
  const raw = String(value === undefined ? (process.env[ENVIRONMENT] || '') : value).trim();
  if (!raw || ['0', 'false', 'no', 'off'].includes(raw.toLowerCase())) return null;
  if (['1', 'true', 'yes', 'on'].includes(raw.toLowerCase())) return DEFAULT_PATH;
  return raw;
}

/** A container that notes the names read from it into keys; a whole read stops individual ones. */
function recorded(value, keys, state) {
  if (!value || typeof value !== 'object') return value;
  return new Proxy(value, {
    get(target, key, receiver) {
      if (typeof key === 'string' && !SKIPPED.has(key) && !(key in Object.prototype) && !state.whole && !IGNORED.has(key)) keys.add(key);
      return Reflect.get(target, key, receiver);
    },
    ownKeys(target) {
      state.whole = true;
      keys.add(WHOLE);
      return Reflect.ownKeys(target);
    }
  });
}

/** Express middleware that records what each handler reads, when recording is on. */
function recordInputs(options = {}) {
  const target = destination(options.path);
  if (!target) return (req, res, next) => next();
  return (req, res, next) => {
    const keys = new Set();
    for (const name of ['query', 'body']) {
      const own = Object.getOwnPropertyDescriptor(req, name);
      let stored = own ? own.value : undefined;
      let assigned = Boolean(own);
      const state = { whole: false };
      Object.defineProperty(req, name, {
        configurable: true,
        enumerable: true,
        get() {
          // Express 5's req.query is a getter on the prototype; Express 4 and body parsers assign it.
          const value = assigned ? stored : Reflect.get(Object.getPrototypeOf(req), name, req);
          return recorded(value, keys, state);
        },
        set(value) {
          stored = value;
          assigned = true;
        }
      });
    }
    res.on('finish', () => {
      const route = req.route && req.route.path !== undefined ? `${req.baseUrl || ''}${req.route.path}` : null;
      if (!route || !keys.size) return;
      const lines = [...keys].sort().map(key => `${JSON.stringify({ route, method: req.method, key })}\n`).join('');
      try {
        fs.mkdirSync(path.dirname(target), { recursive: true });
        fs.appendFileSync(target, lines);
      } catch (error) {
        console.warn(`AIWAF couldn't record inputs to ${target}: ${error.message || error}`);
      }
    });
    next();
  };
}

module.exports = { DEFAULT_PATH, ENVIRONMENT, WHOLE, destination, recordInputs };
