function blockedPayload(message = null) {
  const payload = { error: 'blocked' };
  if (message) payload.message = String(message);
  return payload;
}

function blockedResponse(message = null, statusCode = 403) {
  return {
    payload: blockedPayload(message),
    statusCode: Number(statusCode || 403)
  };
}

function throttleResponse() {
  return {
    payload: { error: 'too_many_requests' },
    statusCode: 429
  };
}

module.exports = {
  blockedPayload,
  blockedResponse,
  throttleResponse
};
