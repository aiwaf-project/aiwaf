const { createInspector } = require('../lib/sqlInjection');

describe('bounded SQL injection inspection', () => {
  const inspect = createInspector();
  test.each([
    "admin@juice-sh.op' OR 1=1--", "admin@juice-sh.op'--",
    "x'/**/OR/**/1=1--", "x%2527%2520OR%25201%253D1--",
    "x' OR 'a'='a'--", "UNION ALL SELECT password FROM users"
  ])('finds SQL signatures inside nested payloads: %s', email => {
    expect(inspect({ body: { login: [{ email }] } }).status).toBe(403);
  });
  test.each(["O'Reilly", 'select a union membership', 'ordinary-password#123', 'email@example.com'])('allows ordinary values: %s', email => {
    expect(inspect({ body: { email } })).toBeNull();
  });
  test('inspects query parameters and serialized JSON escapes', () => {
    expect(inspect({ url: '/search?q=union%20select%20password' }).rule).toBe('sql_union_select');
    expect(inspect({ body: '{"email":"admin\\u0027--"}' }).rule).toBe('sql_quote_comment');
  });
  test('bounds bytes, nesting and cyclic objects', () => {
    expect(createInspector({ AIWAF_PAYLOAD_MAX_BYTES: 8 })({ body: '123456789' }).status).toBe(413);
    const cyclic = {}; cyclic.self = cyclic;
    expect(inspect({ body: cyclic }).status).toBe(413);
    let nested = 'ok'; for (let i = 0; i < 20; i++) nested = [nested];
    expect(inspect({ body: nested }).status).toBe(413);
  });
  test('supports off and monitor without including secrets in findings', () => {
    expect(createInspector({ AIWAF_SQL_INJECTION_MODE: 'off' })({ body: "admin'--" })).toBeNull();
    expect(createInspector({ AIWAF_SQL_INJECTION_MODE: 'monitor' })({ body: "admin'--" }))
      .toEqual({ rule: 'sql_quote_comment', status: 403, mode: 'monitor' });
    expect(() => createInspector({ AIWAF_SQL_INJECTION_MODE: 'typo' })).toThrow();
  });
  test('does not consume unparsed upload streams', () => {
    const { Readable } = require('stream');
    const body = Readable.from(["admin'--"]);
    expect(inspect({ body })).toBeNull();
    expect(body.readableFlowing).toBeNull();
    body.destroy();
  });
});
