const { analyzeHandlerAst } = require('../lib/sourceAst');
const {
  detectApiEndpoint,
  detectAuthEndpoint
} = require('../lib/pathManifest');

describe('source AST detection', () => {
  it('detects methods and JSON body usage from handler AST', () => {
    function handler(req, res) {
      if (req.method === 'POST') {
        return res.json({ ok: Boolean(req.body?.name) });
      }
      return res.json([]);
    }

    const result = analyzeHandlerAst(handler);

    expect(result.available).toBe(true);
    expect(result.methods).toContain('POST');
    expect(result.apiScore).toBeGreaterThanOrEqual(80);
    expect(result.requestBody).toBe(true);
    expect(result.payloadType).toBe('json');
  });

  it('prefers form classification when render handles request body', () => {
    function contact(req, res) {
      const name = req.body.name;
      return res.render('contact', { name });
    }

    const result = detectApiEndpoint(contact, '/contact', ['POST']);

    expect(result.responseType).toBe('mixed');
    expect(result.payloadType).toBe('form');
    expect(result.formConfidence).toBeGreaterThanOrEqual(0.5);
  });

  it('detects auth signals from common auth libraries', () => {
    function login(req, res) {
      const token = jwt.sign({ id: req.body.id }, 'secret');
      return res.json({ token });
    }

    const result = detectAuthEndpoint(login, '/token');

    expect(result.isAuth).toBe(true);
    expect(result.action).toBe('token_login');
    expect(result.signals.length).toBeGreaterThan(0);
  });
});
