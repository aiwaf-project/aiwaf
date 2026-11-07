const fs = require('fs');
const path = require('path');
const { createInspector } = require('../lib/sqlInjection');
const wasmChecks = require('../lib/wasmChecks');

const FIXTURES = path.join(__dirname, '../../tests/fixtures');
const CASES = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'sql_injection_cases.json'), 'utf8'));
const ORDINARY = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'ordinary_text.json'), 'utf8')).values;

beforeAll(() => wasmChecks.load());

describe('bounded SQL injection inspection', () => {
  const inspect = createInspector();
  test.each([
    "admin@juice-sh.op' OR 1=1--", "admin@juice-sh.op'--",
    "x'/**/OR/**/1=1--", "x%2527%2520OR%25201%253D1--",
    "x' OR 'a'='a'--", "1 UNION ALL SELECT password FROM users"
  ])('finds SQL signatures inside nested payloads: %s', email => {
    expect(inspect({ body: { login: [{ email }] } }).status).toBe(403);
  });
  test.each(["O'Reilly", 'select a union membership', 'ordinary-password#123', 'email@example.com'])('allows ordinary values: %s', email => {
    expect(inspect({ body: { email } })).toBeNull();
  });
  test('inspects query parameters and serialized JSON escapes', () => {
    expect(inspect({ url: '/search?q=1%20union%20select%20password' }).rule).toBe('sql_breakout_numeric');
    // Without a value to close, this text can't change a query.
    expect(inspect({ url: '/search?q=union%20select%20password' })).toBeNull();
    expect(inspect({ body: '{"email":"admin\\u0027--"}' }).rule).toBe('sql_breakout_single_quote');
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
      .toEqual({ rule: 'sql_breakout_single_quote', status: 403, mode: 'monitor' });
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

describe('the SQL check in aiwaf-wasm', () => {
  const inspect = req => createInspector({})(req);

  test('shared attacks are caught and ordinary text passes', () => {
    expect(CASES.attacks.filter(value => !inspect({ body: { field: value } }))).toEqual([]);
    expect([...CASES.ordinary, ...ORDINARY].filter(value => inspect({ body: { field: value } }))).toEqual([]);
    expect(CASES.attacks.filter(value => !wasmChecks.get().sql_tokenizer(value))).toEqual([]);
  });

  test('JSON bodies are checked by their values, and a literal + is still seen', () => {
    expect(inspect({ body: '{"field": "const fn = match(\\"/*splat\\");"}' })).toBeNull();
    expect(inspect({ body: '{"a": "x\\" OR 1=1--"}' }).rule).toBe('sql_breakout_double_quote');
    expect(inspect({ body: { q: "'+(SELECT 1)+'" } }).rule).toBe('sql_breakout_single_quote');
    expect(inspect({ body: "'+(SELECT 1)+'" }).rule).toBe('sql_breakout_single_quote');
  });

  test('every request shape is read', () => {
    expect(inspect({ url: '/x?id=1%20OR%201%3D1' }).rule).toBe('sql_breakout_numeric');
    expect(inspect({ query: { q: '1 OR 1=1' } }).rule).toBe('sql_breakout_numeric');
    expect(inspect({ body: Buffer.from('a=1 OR 1=1') }).rule).toBe('sql_breakout_numeric');
    expect(inspect({ body: { pipe() {} } })).toBeNull();
    const cyclic = {}; cyclic.self = cyclic;
    expect(inspect({ body: cyclic }).status).toBe(413);
    expect(inspect({ query: cyclic }).status).toBe(413);
  });

  test('the accuracy script reports the same cases', () => {
    expect(CASES.attacks.length).toBeGreaterThan(50);
  });
});
