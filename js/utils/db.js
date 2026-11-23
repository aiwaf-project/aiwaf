const knex = require('knex');

const isTest = process.env.NODE_ENV === 'test';

// npm 12 no longer runs dependencies' install scripts unless the app allows them, and sqlite3's
// install script is what fetches or builds its native binary. Without it, sqlite3 can't load.
// So the database connects on first use rather than when aiwaf is required: when sqlite3 can't
// load, each use throws SqliteUnavailableError, and the stores fall back to their CSV files.
const FIX = 'To store data in SQLite instead, run: npm approve-scripts sqlite3 && npm rebuild sqlite3';

class SqliteUnavailableError extends Error {
  constructor(cause) {
    super(`AIWAF can't load sqlite3's native binary, so it stores data in CSV files under logs/storage. `
      + `npm 12 skips install scripts unless they're approved, and sqlite3 needs its install script. ${FIX}`);
    this.name = 'SqliteUnavailableError';
    this.code = 'AIWAF_SQLITE_UNAVAILABLE';
    this.cause = cause;
  }
}

let instance = null;
let failure = null;
let warned = false;

function loadDriver() {
  // Required for its side effect: it throws when the native binary is missing.
  // eslint-disable-next-line global-require
  require('sqlite3');
}

function connect() {
  if (instance) return instance;
  if (failure) throw failure;
  try {
    loadDriver();
  } catch (err) {
    failure = new SqliteUnavailableError(err);
    if (!warned && !isTest) {
      warned = true;
      console.warn(`[aiwaf] ${failure.message}`);
    }
    throw failure;
  }
  instance = knex({
    client: 'sqlite3',
    connection: {
      filename: isTest ? ':memory:' : './aiwaf.sqlite'
    },
    useNullAsDefault: true,
  });
  return instance;
}

// Properties that inspection and module tooling read; answering them must not connect.
const PASSIVE = new Set(['then', '__esModule', 'default', 'inspect', 'toJSON', 'constructor', 'prototype']);

// Calling db('table') runs this function; reading a property goes through the get trap below.
function aiwafDb(...args) {
  return connect()(...args);
}

const db = new Proxy(aiwafDb, {
  get(target, prop) {
    if (prop === 'available') return available;
    if (prop === 'SqliteUnavailableError') return SqliteUnavailableError;
    if (typeof prop === 'symbol' || PASSIVE.has(prop)) return Reflect.get(target, prop);
    const conn = connect();
    const value = conn[prop];
    return typeof value === 'function' ? value.bind(conn) : value;
  },
  has(_target, prop) {
    try { return prop in connect(); } catch (_) { return false; }
  }
});

/** Whether sqlite3 loads. False means the stores use their CSV files. */
function available() {
  try {
    connect();
    return true;
  } catch (err) {
    if (err instanceof SqliteUnavailableError) return false;
    throw err;
  }
}

module.exports = db;
