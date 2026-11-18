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

  test.each(['parseInt', 'Number', 'encodeURIComponent'])('%s stops tracking', cleaner => {
    expect(sinks(`function view(req) { const v = ${cleaner}(req.query.x); exec('ls ' + v); }`)).toEqual([]);
  });

  test('encoders are safe only for what they encode for', () => {
    // basename keeps a file in its folder, but its result can still hold shell syntax.
    expect(sinks(`function view(req) { const v = path.basename(req.query.x); fs.readFileSync(v); exec('ls ' + v); }`))
      .toEqual([['command', 'exec']]);
    expect(sinks(`function view(req, res) { const v = escapeHtml(req.query.x); res.send('<p>' + v); db.query('SELECT ' + v); }`))
      .toEqual([['sql', 'db.query']]);
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

  test.each([
    ['a function', "function build(n) { return 'SELECT ' + n; }"],
    ['an arrow function', "const build = n => 'SELECT ' + n;"]
  ])("a helper's return value carries its input, from %s", (_name, helper) => {
    const source = `${helper} app.get('/u', (req, res) => { db.query(build(req.query.n)); });`;
    expect(analyzeSource(source).map(item => [item.sink, item.source])).toEqual([['sql', 'req.query.n']]);
  });

  test('a helper that cleans its input returns a clean value', () => {
    const source = "function build(n) { return parseInt(n); } app.get('/u', (req, res) => { db.query('x' + build(req.query.n)); });";
    expect(sinks(source)).toEqual([]);
  });

  test('a collection built in a loop carries its input', () => {
    const source = `app.get('/u', (req, res) => {
      const parts = [];
      for (const v of req.query.ids) { parts.push(v); }
      db.query('SELECT ' + parts.join(','));
    });`;
    expect(sinks(source)).toEqual([['sql', 'db.query']]);
  });

  test('NestJS values parsed as numbers or UUIDs are not input', () => {
    const source = `class C {
      a(@Param('id', ParseIntPipe) id: number) { return this.db.query('SELECT ' + id); }
      b(@Param('id', new ParseUUIDPipe()) id: string) { return this.db.query('SELECT ' + id); }
      c(@Param('n') n: string) { return this.db.query('SELECT ' + n); }
    }`;
    expect(analyzeSource(source).map(item => item.function)).toEqual(['c']);
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

  // Each case: what it shows, the operations reported, and the code. Safe look-alikes report nothing.
  const FLOW_CASES = {
  // [expected sinks], source
  'xss send (vuln)': [['xss'], "app.get('/a', (req, res) => { res.send('<h1>' + req.query.name + '</h1>'); });"],
  'xss json object (safe)': [[], "app.get('/a', (req, res) => { res.send(req.query); res.json({ q: req.query.q }); });"],
  'xss escaped (safe)': [[], "app.get('/a', (req, res) => { res.send(`<p>${escapeHtml(req.query.q)}</p>`); });"],
  'koa body (vuln)': [['xss'], "router.get('/a', (ctx) => { ctx.body = `<p>${ctx.query.q}</p>`; });"],
  'redirect (vuln)': [['redirect'], "app.get('/a', (req, res) => { res.redirect(req.query.next); });"],
  'redirect allowlist (safe)': [[], "const OK = ['/home', '/me']; app.get('/a', (req, res) => { const n = req.query.next; if (!OK.includes(n)) return res.status(400).end(); res.redirect(n); });"],
  'prototype (vuln)': [['prototype'], "app.post('/a', (req, res) => { const o = {}; o[req.body.key] = req.body.value; });"],
  'prototype nested (vuln)': [['prototype'], "app.post('/a', (req, res) => { config[req.body.a][req.body.b] = 1; });"],
  'prototype bare (safe)': [[], "app.post('/a', (req, res) => { const o = Object.create(null); o[req.body.key] = 1; });"],
  'merge (vuln)': [['prototype'], "app.post('/a', (req, res) => { _.merge(settings, req.body); });"],
  'redos (vuln)': [['redos'], "app.get('/a', (req, res) => { new RegExp(req.query.q).test(s); });"],
  'redos escaped (safe)': [[], "app.get('/a', (req, res) => { new RegExp(escapeRegExp(req.query.q)); });"],
  'xpath (vuln)': [['xpath'], "app.get('/a', (req, res) => { xpath.select(`//user[@name='${req.query.n}']`, doc); });"],
  'ldap (vuln)': [['ldap'], "app.get('/a', (req, res) => { client.search('ou=x', { filter: `(uid=${req.query.u})` }, cb); });"],
  'xxe (vuln)': [['xxe'], "app.post('/a', (req, res) => { libxmljs.parseXml(req.body.xml, { noent: true }); });"],
  'xxe default (safe)': [[], "app.post('/a', (req, res) => { libxmljs.parseXml(req.body.xml); });"],
  'regex guard (safe)': [[], "app.get('/a', (req, res) => { const id = req.query.id; if (!/^\\w+$/.test(id)) return res.sendStatus(400); db.query('SELECT ' + id); });"],
  'dotdot guard (path safe, sql not)': [['sql'], "app.get('/a', (req, res) => { const f = req.query.f; if (f.includes('..')) return; fs.readFileSync(f); db.query(f); });"],
  'positive allowlist (safe)': [[], "app.get('/a', (req, res) => { const s = req.query.sort; if (['name', 'date'].includes(s)) { db.query('ORDER BY ' + s); } });"],
  'denylist continue (vuln)': [['sql'], "app.get('/a', (req, res) => { for (const k of Object.keys(req.query)) { if (SKIP.includes(k)) continue; db.query(k); } });"],
  'zod number (safe)': [[], "app.get('/a', (req, res) => { const id = z.coerce.number().int().parse(req.query.id); db.query('SELECT ' + id); });"],
  'zod object (vuln)': [['sql'], "app.get('/a', (req, res) => { const b = z.object({ q: z.string() }).parse(req.body); db.query('SELECT ' + b.q); });"],
  'const condition (safe)': [[], "app.get('/a', (req, res) => { let bar; const num = 106; if ((7 * 18) + num > 200) bar = 'safe'; else bar = req.query.q; db.query(bar); });"],
  'object key (safe)': [[], "app.get('/a', (req, res) => { const m = { a: 'safe', b: req.query.q }; db.query(m.a); });"],
  'object key (vuln)': [['sql'], "app.get('/a', (req, res) => { const m = { a: 'safe', b: req.query.q }; db.query(m.b); });"],
  'array position (safe)': [[], "app.get('/a', (req, res) => { const l = ['safe']; l.push(req.query.q); l.shift(); l.push('x'); db.query(l[1]); });"],
  'switch const (safe)': [[], "app.get('/a', (req, res) => { let bar; switch ('ABC'.charAt(1)) { case 'A': bar = req.query.q; break; case 'B': bar = 'bob'; break; default: bar = req.query.q; } db.query(bar); });"],
  'then callback (vuln)': [['sql'], "app.get('/a', (req, res) => { Promise.resolve(req.query.q).then(v => db.query('SELECT ' + v)); });"],
  'forEach callback (vuln)': [['command'], "app.post('/a', (req, res) => { req.body.files.forEach(f => exec('rm ' + f)); });"],
  'middleware req (vuln)': [['sql'], "app.use((req, res, next) => { req.search = req.query.q; next(); }); app.get('/s', (req, res) => { db.query('SELECT ' + req.search); });"],
  'class state (vuln)': [['sql'], "class Repo { constructor(term) { this.term = term; } run() { db.query('SELECT ' + this.term); } } app.get('/a', (req, res) => { const r = new Repo(req.query.q); r.run(); });"],
  'graphql (vuln)': [['sql'], "const resolvers = { Query: { user: (parent, args) => db.query('SELECT ' + args.id) } };"],
  'graphql destructured (vuln)': [['sql'], "const resolvers = { Mutation: { async update(_, { name }) { return db.query(`UPDATE x SET n='${name}'`); } } };"],
  'trpc (vuln)': [['sql'], "const r = t.router({ get: t.procedure.input(z.string()).query(({ input }) => db.query('SELECT ' + input)) });"],
  'server action (vuln)': [['sql'], "async function save(formData) { 'use server'; db.query('UPDATE ' + formData.get('name')); }"],
  'remix loader (vuln)': [['sql'], "export async function loader({ request }) { const q = new URL(request.url).searchParams.get('q'); return db.query('SELECT ' + q); }"],
  'next route json (vuln)': [['command'], "export async function POST(request) { const body = await request.json(); exec('ls ' + body.dir); }"],
  'websocket (vuln)': [['sql'], "wss.on('connection', ws => { ws.on('message', data => { db.query('SELECT ' + data); }); });"],
  'socketio (vuln)': [['command'], "io.on('connection', socket => { socket.on('run', cmd => exec(cmd)); });"],
  'nest graphql args (vuln)': [['sql'], "class R { @Query() user(@Args('id') id: string) { return this.db.query('SELECT ' + id); } }"],
  };

  test.each([
    ['10 - 4 === 6', true], ['9 / 3 !== 3', false], ['7 % 4 == 3', true], ['2 < 1', false], ['2 <= 2', true],
    ['3 >= 4', false], ['1 != 1', false], ['(5 / 0) > 1', null]
  ])('a condition decided before running: %s', (condition, known) => {
    const source = `app.get('/a', (req, res) => { let bar = req.query.q; if (${condition}) bar = 'safe'; db.query(bar); });`;
    // A true condition always runs the safe assignment; a false or unknown one may not.
    expect(sinks(source)).toEqual(known === true ? [] : [['sql', 'db.query']]);
  });

  test('checks joined with && and || count when all of them hold', () => {
    const both = "app.get('/a', (req, res) => { const a = req.query.a; const b = req.query.b;"
      + ' if (!/^\\w+$/.test(a) || !/^\\w+$/.test(b)) return; db.query(a + b); });';
    expect(sinks(both)).toEqual([]);
    const either = "app.get('/a', (req, res) => { const a = req.query.a; if (/^\\w+$/.test(a) || a.length) { db.query(a); } });";
    expect(sinks(either)).toEqual([['sql', 'db.query']]);
  });

  test('destructured callback parameters get the input', () => {
    expect(sinks("app.post('/a', (req, res) => { req.body.items.forEach(({ name, tags: [first] }) => exec(name + first)); });"))
      .toEqual([['command', 'exec']]);
  });

  test.each(Object.entries(FLOW_CASES))('%s', (_name, [expected, source]) => {
    expect([...new Set(analyzeSource(source).map(finding => finding.sink))].sort()).toEqual([...expected].sort());
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

    test('calls into other project files are followed and linked to the handler', () => {
      write('lib/db.js', 'module.exports = { query() {} };');
      write('lib/sql.js', `const db = require('./db');
exports.search = function (term) { return db.query("SELECT * FROM t WHERE n = '" + term + "'"); };
function harmless(n) { return n; }
module.exports.harmless = harmless;`);
      write('app.js', `const sql = require('./lib/sql');
import { search } from './lib/sql';
app.get('/search', (req, res) => { sql.search(req.query.q); });
app.get('/named', (req, res) => { search(req.query.q); });
app.get('/harmless', (req, res) => { sql.harmless(req.query.q); });`);
      const findings = analyzePath(directory).findings;
      expect(findings.map(item => item.function)).toEqual(['app.get /named -> search', 'app.get /search -> search']);
      for (const finding of findings) {
        expect(finding.file).toMatch(/lib\/sql\.js$/);
        expect(finding.entry_file).toMatch(/app\.js$/);
      }
    });

    test('tsconfig paths, baseUrl and workspace packages are followed', () => {
      write('tsconfig.base.json', `{
        // comments, trailing commas and URLs are allowed, as TypeScript allows them
        "$schema": "https://json.schemastore.org/tsconfig",
        "compilerOptions": { "baseUrl": "apps/api/src", "paths": { "@/*": ["*"], "~lib": ["lib/index.ts"], }, },
      }`);
      write('apps/api/tsconfig.json', '{ "extends": "../../tsconfig.base.json" }');
      write('packages/db/package.json', '{ "name": "@acme/db", "main": "dist/index.js", "exports": { ".": { "import": "./src/index.ts" } } }');
      write('packages/db/src/index.ts', 'export function raw(sql: string) { return pool.query(sql); }');
      write('apps/api/src/lib/index.ts', 'export function run(q: string) { exec(q); }');
      write('apps/api/src/lib/sql.ts', `export function search(t: string) { return db.query("SELECT '" + t + "'"); }`);
      write('apps/api/src/app.ts', `
        import { search } from '@/lib/sql';
        import { run } from '~lib';
        import { search as bare } from 'lib/sql';
        import { raw } from '@acme/db';
        app.get('/e', (req, res) => search(req.query.q));
        app.get('/f', (req, res) => run(req.query.q));
        app.get('/g', (req, res) => bare(req.query.q));
        app.get('/h', (req, res) => raw('SELECT ' + req.query.q));`);
      const findings = analyzePath(directory).findings;
      expect(findings.map(item => item.function).sort()).toEqual(
        ['app.get /e -> search', 'app.get /f -> run', 'app.get /g -> bare', 'app.get /h -> raw']);
      for (const finding of findings) expect(finding.entry_file).toMatch(/apps\/api\/src\/app\.ts$/);
    });

    test('methods on instances of project classes are followed', () => {
      write('users.service.ts', `
        export class UsersService {
          find(id: string) { return this.db.query('SELECT * FROM u WHERE id = ' + id); }
          safe(id: string) { return id.length; }
        }
        export default UsersService;`);
      write('users.controller.ts', `
        import { UsersService } from './users.service';
        import Default from './users.service';
        import * as svc from './users.service';
        export class UsersController {
          private readonly extra = new UsersService();
          private other: UsersService;
          constructor(private readonly users: UsersService, made: UsersService) { this.other = made; }
          @Get('a') a(@Query('id') id: string) { return this.users.find(id); }
          @Get('b') b(@Query('id') id: string) { return this.extra.find(id); }
          @Get('c') c(@Query('id') id: string) { const s = new Default(); return s.find(id); }
          @Get('d') d(@Query('id') id: string) { return new svc.UsersService().find(id); }
          @Get('e') e(@Query('id') id: string) { const r: UsersService = make(); return r.find(id); }
          @Get('f') f(@Query('id') id: string) { return this.other.find(id); }
          @Get('g') g(@Query('id') id: string) { return this.users.safe(id); }
        }`);
      const findings = analyzePath(directory).findings;
      expect(findings.map(item => item.function).sort()).toEqual(['a -> find', 'b -> find', 'c -> find', 'd -> find', 'e -> find', 'f -> find']);
      expect(findings.every(item => item.file.endsWith('users.service.ts') && item.sink === 'sql')).toBe(true);
    });

    test('ignore comments, baselines and SARIF', () => {
      const { applyBaseline, fingerprint, toSarif } = require('../lib/injectionAnalysis');
      write('a.js', "app.get('/x', (req, res) => {\n  exec(req.query.a);\n  exec(req.query.b); // aiwaf: ignore[command]\n"
        + "  // aiwaf-ignore\n  db.query(req.query.c);\n  db.query(req.query.d); // aiwaf: ignore[xss]\n});\n");
      const report = analyzePath(directory);
      expect(report.findings.map(item => item.line)).toEqual([2, 6]);
      expect(report.suppressed).toBe(2);
      expect(formatReport(report)).toContain('Left out 2 marked with an aiwaf: ignore comment.');
      const baseline = write('baseline.json', JSON.stringify(report));
      fs.appendFileSync(path.join(directory, 'a.js'), "app.get('/y', (req, res) => { exec(req.query.z); });\n");
      const later = applyBaseline(analyzePath(directory), JSON.parse(fs.readFileSync(baseline, 'utf8')));
      expect(later.findings.map(item => item.function)).toEqual(['app.get /y']);
      expect(later.baselined).toBe(2);
      // The line isn't part of a finding's identity, so moved code still matches the baseline.
      expect(fingerprint({ ...later.findings[0], line: 99 })).toBe(fingerprint(later.findings[0]));
      const sarif = JSON.parse(toSarif(report));
      expect(sarif.version).toBe('2.1.0');
      expect(sarif.runs[0].tool.driver.rules.map(rule => rule.id)).toEqual(['aiwaf.command', 'aiwaf.sql']);
      expect(sarif.runs[0].results[0].locations[0].physicalLocation.region).toEqual({ startLine: 2 });
      const out = [];
      expect(runAuditCommand([directory, '--baseline', baseline, '--format', 'sarif'], text => out.push(text))).toBe(0);
      expect(JSON.parse(out[0]).runs[0].results).toHaveLength(1);
      expect(runAuditCommand([directory, '--baseline', path.join(directory, 'missing.json')], () => {})).toBe(2);
    });

    test('a shared helper walked once still links its findings to every route', () => {
      write('helpers.js', "function inner(v) { db.query('SELECT ' + v); return v; }\n"
        + 'function outer(v) { inner(v); return inner(v); }\nmodule.exports = { outer };\n');
      for (const index of [0, 1, 2]) {
        write(`route${index}.js`, `const { outer } = require('./helpers');\napp.get('/r${index}', (req, res) => { exec(outer(req.query.q)); });\n`);
      }
      const functions = analyzePath(directory).findings.map(item => item.function).sort();
      expect(functions).toEqual([0, 1, 2].flatMap(index => [`app.get /r${index}`, `app.get /r${index} -> outer -> inner`]));
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

    test('the command prints to the console by default', () => {
      write('a.js', FRAMEWORKS.express);
      const log = jest.spyOn(console, 'log').mockImplementation(() => {});
      try {
        expect(runAuditCommand([directory])).toBe(0);
        expect(log.mock.calls[0][0]).toContain('found 1 possible');
      } finally {
        log.mockRestore();
      }
    });
  });
});
