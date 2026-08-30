const fs = require('fs');
const path = require('path');
const requestLogStore = require('./requestLogStore');

let enabled = false;
let outputPath = 'logs/aiwaf-requests.jsonl';
let logToDb = false;
let logToCsv = false;
let csvPath = 'logs/aiwaf-requests.csv';
let csvHeaderWritten = false;

function escapeCsv(value) {
  const str = String(value ?? '');
  if (str.includes('"') || str.includes(',') || str.includes('\n')) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

function csvLineFromEvent(event) {
  const cells = [
    event.timestamp || '',
    event.ip || '',
    event.method || '',
    event.path || '',
    event.status || '',
    event.responseTime || '',
    event.blocked ? '1' : '0',
    event.reason || '',
    event.country || '',
    event.userAgent || ''
  ];
  return `${cells.map(escapeCsv).join(',')}\n`;
}

function writeCsvEvent(event) {
  if (!logToCsv) return;
  try {
    ensureDirectoryExists(csvPath);
    if (!csvHeaderWritten && !fs.existsSync(csvPath)) {
      fs.appendFileSync(csvPath, 'timestamp,ip,method,path,status,response_time,blocked,reason,country,user_agent\n');
      csvHeaderWritten = true;
    }
    fs.appendFileSync(csvPath, csvLineFromEvent(event), 'utf8');
  } catch (err) {
    // CSV logging should never take down request handling.
  }
}

function ensureDirectoryExists(filePath) {
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function writeEvent(event) {
  if (!enabled) return;

  try {
    ensureDirectoryExists(outputPath);
    fs.appendFileSync(outputPath, `${JSON.stringify(event)}\n`, 'utf8');
  } catch (err) {
    // Logging should never take down request handling.
  }

  writeCsvEvent(event);

  if (logToDb) {
    requestLogStore.insert(event).catch(() => {
      // Keep a durable non-DB copy when DB storage fails.
      if (!logToCsv) {
        logToCsv = true;
      }
      writeCsvEvent(event);
    });
  }
}

module.exports = {
  init(opts = {}) {
    enabled = !!opts.AIWAF_MIDDLEWARE_LOGGING;
    outputPath = opts.AIWAF_MIDDLEWARE_LOG_PATH || outputPath;
    logToDb = !!opts.AIWAF_MIDDLEWARE_LOG_DB;
    logToCsv = !!opts.AIWAF_MIDDLEWARE_LOG_CSV || logToDb;
    csvPath = opts.AIWAF_MIDDLEWARE_LOG_CSV_PATH || csvPath;
    csvHeaderWritten = false;
    if (logToDb) {
      requestLogStore.initialize().catch(() => {});
    }
  },

  attach(req, res, context) {
    if (!enabled) return;

    const start = Date.now();
    res.on('finish', () => {
      const decision = res.locals?.aiwafDecision || {};

      writeEvent({
        timestamp: new Date().toISOString(),
        ip: context.ip,
        method: req.method,
        path: req.path || req.url,
        status: res.statusCode,
        responseTime: Date.now() - start,
        blocked: !!decision.blocked,
        reason: decision.reason || '',
        country: decision.country || '',
        userAgent: req.headers?.['user-agent'] || ''
      });
    });
  },

  markBlocked(res, reason, country = '') {
    if (!res.locals) res.locals = {};
    res.locals.aiwafDecision = {
      blocked: true,
      reason,
      country
    };
  }
};
