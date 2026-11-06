const fs = require('fs');
const os = require('os');
const path = require('path');
const { analyzePath, analyzeSource, formatReport, runAuditCommand } = require('../lib/injectionAnalysis');

const sinks = source => analyzeSource(source).map(finding => [finding.sink, finding.call]);

const FRAMEWORKS = {
  express: "app.get('/x', (req, res) => { db.query('SELECT ' + req.query.id); });",
  fastify: "fastify.get('/x', async (request, reply) => { db.query(`SELECT ${request.query.id}`); });",
  hapi: "server.route({ method: 'GET', path: '/x', handler: (request, h) => db.query('SELECT ' + request.payload.id) });",
  koa: "app.use(async ctx => { db.query('SELECT ' + ctx.query.id); });",
  koaRequest: "app.use(async ctx => { db.query('SELECT ' + ctx.request.body.id); });",
  nest: `class Items {
    @Get(':id')
    find(@Param('id') id: string) { return this.db.query('SELECT ' + id); }
  }`,
  next: "export default function handler(req, res) { db.query('SELECT ' + req.query.id); }",
  adonis: "export default class Items { async show({ request }) { db.rawQuery; db.query('SELECT ' + request.input('id')); } }",
  sails: "module.exports = { find: async function (req, res) { db.query('SELECT ' + req.param('id')); } };"
};

describe('injection analysis', () => {
  test.each(Object.entries(FRAMEWORKS))('%s reports input reaching SQL', (_name, source) => {
    expect(analyzeSource(source).map(finding => finding.sink)).toEqual(['sql']);
  });

  test.each([
    ["db.query('SELECT ' + value)", 'sql'],
    ['db.query(`SELECT ${value}`)', 'sql'],
    ['knex.raw(value)', 'sql'],
    ['qb.whereRaw(value)', 'sql'],
    ['prisma.$queryRawUnsafe(value)', 'sql'],
    ['sequelize.query(value)', 'sql'],
    ['db.prepare(value)', 'sql'],
    ['users.find(value)', 'nosql'],
    ['User.findOne(value)', 'nosql'],
    ['exec(value)', 'command'],
    ['child_process.execSync(value)', 'command'],
    ['spawn(value, { shell: true })', 'command'],
    ['fs.readFile(value)', 'path'],
    ['res.sendFile(value)', 'path'],
    ['fs.readFile(path.join("/data", value))', 'path'],
    ['fs.readFile(path.resolve("/data", value))', 'path'],
    ['fetch(value)', 'ssrf'],
    ['axios.get(value)', 'ssrf'],
    ['axios({ url: value })', 'ssrf'],
    ['axios.request({ method: "GET", url: value })', 'ssrf'],
    ['eval(value)', 'code'],
    ['new Function(value)', 'code'],
    ['vm.runInNewContext(value)', 'code'],
    ['ejs.render(value)', 'template'],
    ['serialize.unserialize(value)', 'deserialization']
  ])('reports %s', (statement, sink) => {
    const source = `function view(req, res) { const value = req.query.x; ${statement}; }`;
    expect(analyzeSource(source).map(finding => finding.sink)).toEqual([sink]);
  });

  test.each([
    "db.query('SELECT * FROM items WHERE id = ?', [value])",
    "db.query('SELECT * FROM items WHERE id = $1', [value])",
    "spawn('ping', [value])",
    "spawn('ping', [value], { shell: false })",
    'items.find(item => item.id === value)',
    'console.log(value)',
    'axios.post("http://fixed.example", { name: value })',
    'redisClient.get(value)',
    'res.json(value)'
  ])('does not report %s', statement => {
    const source = `function view(req, res) { const value = req.query.x; ${statement}; }`;
    expect(sinks(source)).toEqual([]);
  });

  test.each(['parseInt', 'Number', 'encodeURIComponent', 'path.basename'])('%s stops tracking', cleaner => {
    expect(sinks(`function view(req) { const v = ${cleaner}(req.query.x); exec('ls ' + v); }`)).toEqual([]);
  });

  test('tracks input through assignments, destructuring and string methods', () => {
    const source = `function view(req) {
      const { term } = req.body;
      const clean = term.trim().toLowerCase();
      const sql = 'SELECT ' + clean;
      db.query(sql);
    }`;
    const [finding] = analyzeSource(source);
    expect([finding.sink, finding.source, finding.line]).toEqual(['sql', 'req.body.term', 5]);
  });

  test('overwriting a value stops tracking it, but not inside a branch', () => {
    expect(sinks("function v(req) { let x = req.query.a; x = 'safe'; exec(x); }")).toEqual([]);
    expect(sinks("function v(req) { let x = req.query.a; if (!x) { x = 'safe'; } exec(x); }")).toEqual([['command', 'exec']]);
  });

  test('tracks input into callbacks', () => {
    const source = "function v(req) { const x = req.query.a; items.forEach(item => { exec(x); }); }";
    expect(sinks(source)).toEqual([['command', 'exec']]);
  });

  test('follows input into helpers', () => {
    const source = `
      function run(text) { db.query(text); }
      function outer(value) { run('SELECT ' + value); }
      function view(req) { outer(req.query.x); }`;
    const finding = analyzeSource(source).find(item => item.function.includes('->'));
    expect(finding.function).toBe('view -> outer -> run');
  });

  test('recursive helpers finish', () => {
    const source = 'function loop(v) { loop(v); exec(v); } function view(req) { loop(req.query.x); }';
    expect(sinks(source)).toEqual([['command', 'exec']]);
  });

  test('ignores functions without request input', () => {
    expect(sinks('function task(value) { exec(value); }')).toEqual([]);
  });

  test('reading from the request object itself is not a query sink', () => {
    expect(sinks('function v(req, res) { res.json(req.query); ctx; }')).toEqual([]);
  });

  describe('path scans', () => {
    let directory;
    beforeEach(() => { directory = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-audit-')); });
    afterEach(() => fs.rmSync(directory, { recursive: true, force: true }));
    const write = (name, content) => {
      const file = path.join(directory, name);
      fs.mkdirSync(path.dirname(file), { recursive: true });
      fs.writeFileSync(file, content);
      return file;
    };

    test('reports syntax errors and keeps going', () => {
      write('bad.js', 'function (');
      write('good.js', FRAMEWORKS.express);
      const report = analyzePath(directory);
      expect(report.files_scanned).toBe(1);
      expect(report.errors).toHaveLength(1);
      expect(report.findings[0].sink).toBe('sql');
    });

    test('skips tests and dependencies unless asked', () => {
      for (const name of ['test/a.js', 'node_modules/b.js', 'c.test.js', 'dist/d.js']) write(name, FRAMEWORKS.express);
      expect(analyzePath(directory).files_scanned).toBe(0);
      expect(analyzePath(directory, { includeTests: true }).files_scanned).toBe(2);
    });

    test('scans one file, TypeScript and other extensions', () => {
      const file = write('a.ts', FRAMEWORKS.nest);
      expect(analyzePath(file).findings).toHaveLength(1);
      write('b.mjs', FRAMEWORKS.next);
      write('types.d.ts', 'declare const x: string;');
      expect(analyzePath(directory).files_scanned).toBe(2);
    });

    test('missing paths raise', () => {
      expect(() => analyzePath(path.join(directory, 'nope'))).toThrow('Path not found');
    });

    test('reports use the shared format and are readable', () => {
      write('a.js', FRAMEWORKS.express);
      const report = analyzePath(directory);
      expect(Object.keys(report).sort()).toEqual(['errors', 'files_scanned', 'findings', 'language', 'schema_version']);
      expect(Object.keys(report.findings[0]).sort()).toEqual(['call', 'confidence', 'file', 'function', 'line', 'sink', 'source']);
      expect(formatReport(report)).toContain('req.query.id reaches db.query in app.get /x');
    });

    test('a clean report says it is not proof', () => {
      write('a.js', 'const x = 1;');
      expect(formatReport(analyzePath(directory))).toContain("doesn't prove the code is safe");
    });

    test('source is never executed', () => {
      const marker = path.join(directory, 'ran');
      write('a.js', `require('fs').writeFileSync(${JSON.stringify(marker)}, 'x');`);
      analyzePath(directory);
      expect(fs.existsSync(marker)).toBe(false);
    });

    test('the command returns exit codes and writes output', () => {
      write('a.js', FRAMEWORKS.express);
      const output = [];
      expect(runAuditCommand([directory], text => output.push(text))).toBe(0);
      expect(output[0]).toContain('found 1 possible');
      expect(runAuditCommand([directory, '--fail-on-findings'], () => {})).toBe(1);
      const target = path.join(directory, 'report.json');
      expect(runAuditCommand([directory, '--format', 'json', '--output', target], () => {})).toBe(0);
      expect(JSON.parse(fs.readFileSync(target, 'utf8')).language).toBe('javascript');
      expect(runAuditCommand([directory, '--format', 'xml'], () => {})).toBe(2);
      expect(runAuditCommand([path.join(directory, 'nope')], () => {})).toBe(2);
    });
  });
});
