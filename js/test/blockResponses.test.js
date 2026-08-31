const { blockedPayload, blockedResponse, throttleResponse } = require('../lib/blockResponses');

describe('blockResponses', () => {
  test('builds blocked and throttled response contracts', () => {
    expect(blockedPayload('nope')).toEqual({ error: 'blocked', message: 'nope' });
    expect(blockedResponse('nope')).toEqual({
      payload: { error: 'blocked', message: 'nope' },
      statusCode: 403
    });
    expect(throttleResponse()).toEqual({
      payload: { error: 'too_many_requests' },
      statusCode: 429
    });
  });
});
