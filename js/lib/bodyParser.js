const express = require('express');

// Use before AIWAF in raw Node/custom Next servers and early Nest middleware.
// Keep original bytes available to proxies after parsing consumes the stream.
module.exports = function bodyParser(options = {}) {
  const limit = Number(options.AIWAF_PAYLOAD_MAX_BYTES ?? process.env.AIWAF_PAYLOAD_MAX_BYTES ?? 65536);
  if (!Number.isInteger(limit) || limit < 1 || limit > 1048576) {
    throw new Error('AIWAF_PAYLOAD_MAX_BYTES must be 1..1048576');
  }
  const verify = (req, _res, bytes) => { req.aiwafRawBody = Buffer.from(bytes); };
  const parsers = [express.json({ limit, verify, inflate: false }),
    express.urlencoded({ limit, verify, extended: false, inflate: false })];
  return (req, res, next) => {
    const run = index => {
      if (index === parsers.length) return next();
      parsers[index](req, res, error => {
        if (!error) return run(index + 1);
        res.statusCode = [400, 413, 415].includes(error.status) ? error.status : 400;
        res.setHeader('content-type', 'application/json');
        res.end(JSON.stringify({ error: res.statusCode === 413 ? 'payload_inspection_limit' : 'invalid_request_body' }));
      });
    };
    run(0);
  };
};
