'use strict';

// Reading the start of a request body without consuming it, for checks that run before the
// application's body parser.

function isMultipart(req) {
  return String(req.headers?.['content-type'] || '').split(';', 1)[0].trim().toLowerCase() === 'multipart/form-data';
}

// Reads at most limit + 1 bytes of the body, then puts them back on the stream, so body
// parsers after this adapter still read the whole body. Python's adapters do the same.
function peekBody(req, limit) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    const expected = Number(req.headers?.['content-length']) || null;
    const finish = () => {
      req.removeListener('readable', onReadable);
      req.removeListener('error', onError);
      const prefix = Buffer.concat(chunks);
      if (prefix.length) req.unshift(prefix);
      resolve(prefix);
    };
    const onError = error => { req.removeListener('readable', onReadable); reject(error); };
    function onReadable() {
      let chunk;
      while ((chunk = req.read()) !== null) {
        chunks.push(chunk);
        size += chunk.length;
        if (size > limit || size === expected) return finish();
      }
      // The whole body is buffered: put it back before the stream emits 'end'.
      if (req._readableState?.ended) finish();
    }
    req.on('error', onError);
    req.on('readable', onReadable);
  });
}

module.exports = { peekBody, isMultipart };
