# AIWAF

AIWAF is a multi-runtime web application firewall toolkit for Python, Node.js, WebAssembly, and Java applications. It combines deterministic request controls with route-aware policies, reputation tracking, request logging, training pipelines, and optional Isolation Forest anomaly detection.

This is the single authoritative README for the monorepo and every published AIWAF package. Release workflows stage this file, the root `LICENSE`, `NOTICE`, and `licenses/` into package artifacts when a registry requires package-local copies.

AIWAF is a defense-in-depth component. It does not replace secure application code, authentication and authorization, dependency patching, TLS, a reverse proxy, or a managed edge WAF.

## Packages

The next local drafts add bounded Java SQL inspection, a JavaScript `bodyParser`
helper, and atomic Python Redis rate buckets. They have not been published.

For custom Node HTTP servers or middleware registered before a framework's
parser, call `aiwaf.bodyParser()` before AIWAF and before forwarding requests.
It parses JSON and URL-encoded bodies, retains the original forwarding bytes,
and rejects oversized bodies (default 64 KiB). Native Fastify/Hapi integrations
must run AIWAF after body parsing and before their proxy handler.

Java SQL inspection defaults to `security.sql_injection_mode: "block"`;
`monitor` records findings without denying requests and `off` disables it.
`AIWAF_SQL_INJECTION_MODE` provides the environment override. Inspection uses
the configured body byte limit and rejects over-budget inspection with 413.
Previously learned benign keywords no longer block ordinary Java routes.

Redis rate-cache failures now produce generic HTTP 503 responses in JavaScript
and Python rate-limit middleware. JavaScript defaults to
`AIWAF_RATE_CACHE_FAILURE_MODE: 'closed'`; explicitly setting `'open'` preserves
availability at the cost of enforcement when the cache fails. Flask rejects
invalid Redis configuration at initialization, and FastAPI rejects requests
instead of silently selecting a process-local cache. Python `rediss://` URLs
are delegated to redis-py for TLS support when that driver is installed.

Python Flask/FastAPI Redis buckets now update atomically across workers.
Django's native `django.core.cache.backends.redis.RedisCache` uses Redis
transactions while preserving its existing serialized bucket values. Memory
and other custom cache backends serialize updates within one process; use
the supported Redis backends when sharing limits across workers.

Each package is versioned and released independently.

Python 1.3.3, Rust/Python 0.2.17, Node.js 1.3.3, WASM 0.2.13, and Java 1.4.0 are local source targets. See the package table below and [CHANGELOG](CHANGELOG.md) for release changes.

| Runtime | Package | Repository version | Primary use |
| --- | --- | ---: | --- |
| Python | [`aiwaf`](https://pypi.org/project/aiwaf/) | `1.3.3` | Django, Flask, and FastAPI protection |
| Node.js | [`aiwaf`](https://www.npmjs.com/package/aiwaf) | `1.3.3` | Express and Node framework middleware |
| Rust/Python | [`aiwaf-rust`](https://pypi.org/project/aiwaf-rust/) | `0.2.17` | Native Python acceleration and JSON model inference |
| WebAssembly | [`aiwaf-wasm`](https://www.npmjs.com/package/aiwaf-wasm) | `0.2.13` | Rust detection and model primitives for JavaScript |
| Java | `io.github.aiwaf-project:aiwaf-java` | `1.4.0` | Spring MVC and Jakarta Servlet protection |

Supported framework integrations include Django, Flask, FastAPI, Express, Fastify, Hapi, Koa, NestJS, Next.js API routes, AdonisJS, Sails.js, Spring MVC, and Jakarta Servlet.

## What AIWAF provides

- IP blocklist, exemption, reputation, expiry, and progressive-block handling
- Sliding-window rate limiting and flood detection
- Header, method, request-size, content-type, and parameter validation
- Static and learned malicious keyword detection
- Honeypot field and form-timing checks
- UUID probing and tamper detection
- GeoIP allow/block policies using the bundled MMDB data
- Route manifests, path-specific overrides, and decorator/annotation exemptions
- Memory, file, CSV, SQLite, database, and cache-backed runtime state, depending on the runtime
- Structured request logging and offline training
- Six-feature anomaly models with Rust, JavaScript, and Java implementations
- Required Rust runtime for Python 1.1.4 and optional WebAssembly acceleration for Node.js

The runtimes share policy intent and training feature names, but their framework adapters and model implementations remain language-native. Model artifacts are not generally interchangeable across languages.

## Profiles

`AIWAF_PROFILE` sets sensible defaults with one setting, in Django and Flask settings,
FastAPI's `AIWAFConfig` (or `AIWAF(app, profile=...)`), the Node.js options, the Java config
(`profile`) or the environment. A setting you give explicitly always wins over the profile.

| Profile | What it does |
| --- | --- |
| `balanced` (default) | Every default as it is. |
| `strict` | Halves the rate and flood limits (Django, Flask, FastAPI and Java: 10 and 20 a window; Node.js: 50 and 100), turns on UUID scoring, header validation and the method policy where the runtime has them as settings, and keeps the SQL and route checks blocking. |
| `monitor` | A dry run. Every check runs, and what it would block is logged instead, as `AIWAF monitor: would block GET /.env from 203.0.113.9 with 403 (...)`. Nothing is blocked and no IP is added to the blocklist. |

Use `monitor` to try AIWAF on real traffic before turning it on. Its log lines go to the
`aiwaf.monitor` logger in Python, `console.warn` in Node.js and `com.aiwaf.monitor` in Java.
Monitor runs the regular middleware, where every block decision can be logged instead: the
automatic native pipeline stays off, and a native pipeline chosen explicitly refuses to start.
Java's request body size limit still applies.

## Client IP addresses and proxies

Blocks, exemptions, rate limits, and geo policies all depend on the client address. Python, Node.js, and Java read `X-Forwarded-For` and `X-Real-IP` only when the connecting peer is a trusted proxy. They walk `X-Forwarded-For` from the right and stop at the first untrusted address, because anything further left is client-supplied. `CF-Connecting-IP` and other vendor headers are not read for this purpose. In Python and Node.js the walk runs in `aiwaf-rust` and `aiwaf-wasm`, which also write IPv6 addresses in canonical form.

By default, Python and Node.js trust loopback, RFC 1918, link-local, and IPv6 unique-local peers, so a proxy on a private network needs no configuration. Java trusts loopback only. Set `AIWAF_TRUSTED_PROXIES` to a list or comma-separated string of addresses and CIDR ranges to replace those defaults. An empty value trusts no proxy. Two names can be used alongside addresses: `default` for the runtime's built-in ranges and `cloudflare` for [Cloudflare's published ranges](https://www.cloudflare.com/ips/). Behind Cloudflare, use `AIWAF_TRUSTED_PROXIES=default,cloudflare` to keep trusting a local proxy as well.

When a peer that isn't trusted sends `X-Forwarded-For` or `X-Real-IP`, all three runtimes log one warning per process naming that peer. If the peer is your proxy or CDN, add it to `AIWAF_TRUSTED_PROXIES`. A client sending the header directly causes the same warning, and then no action is needed.

| Runtime | Where to set it |
| --- | --- |
| Django | `AIWAF_TRUSTED_PROXIES` in settings, or the environment variable |
| Flask | `app.config["AIWAF_TRUSTED_PROXIES"]`, or the environment variable |
| FastAPI | The environment variable |
| Node.js | The `AIWAF_TRUSTED_PROXIES` or `trustedProxies` option, or the environment variable |
| Java | `aiwaf.trusted-proxies` in Spring properties, the environment variable, `security.trusted_proxies` in the JSON config file, or the `trustedProxyCidrs` field |

> [!IMPORTANT]
> If your proxy or CDN connects from a public address (Cloudflare or Google Cloud load balancers, for example), add its ranges to `AIWAF_TRUSTED_PROXIES`. Otherwise every visitor appears to come from the proxy, and rate limiting or a block can then affect all of them. The `cloudflare` name trusts every Cloudflare address, and anyone can send requests through Cloudflare. If clients could also send requests through Cloudflare to your origin from outside your zone, restrict the origin to your zone, for example with authenticated origin pulls.

If untrusted hosts can reach the application directly from your private network, narrow the list to your proxy's addresses. Addresses already resolved by Flask's `ProxyFix`, Uvicorn's `--proxy-headers`, or Express `trust proxy` are respected, except that AIWAF still walks the headers when the resolved address is itself in a trusted range. The Rust-owned Node pipeline resolves the address the same way when it runs from the regular options (the default, or `from_settings`); configured on its own, it uses `req.ip` unless its `trustedProxies` option is set.

## Injection audit

`audit` reads your source code and reports where request input can reach a dangerous
operation. It parses files without running them, and it doesn't change how requests are
handled.

```bash
aiwaf audit path/to/app                                    # Python
npx aiwaf audit path/to/app                                # Node.js
java -cp <classpath> com.aiwaf.cli.AiwafConsole audit src  # Java, needs a JDK
```

| Language | Frameworks | Parser |
| --- | --- | --- |
| Python | Django, Flask, FastAPI, Graphene, Strawberry, Ariadne, Channels, Flask-SocketIO | `ast` |
| Node.js | Express, Fastify, Hapi, Koa, NestJS, Next.js (routes and server actions), Remix, tRPC, GraphQL resolvers, AdonisJS, Sails, ws and Socket.IO | `@babel/parser` |
| Java | Spring MVC and WebFlux, Spring GraphQL, Netflix DGS, Jakarta Servlet, JAX-RS, Spring messaging and Jakarta WebSocket | JDK Compiler Tree API |

It reports these operations:

| Operation | Examples |
| --- | --- |
| `sql`, `nosql` | `cursor.execute`, `db.query`, `jdbcTemplate.update`, `collection.find` |
| `command` | `os.system`, `exec`, `Runtime.exec`, `ProcessBuilder` |
| `path` | `open`, `fs.readFile`, `new File`, `Path.read_text` |
| `ssrf` | `requests.get`, `fetch`, `RestTemplate` |
| `code`, `template`, `deserialization` | `eval`, `render_template_string`, `pickle.loads` |
| `xss` | text returned from a Flask view, `res.send`, `ctx.body`, `response.getWriter()` |
| `redirect` | `redirect`, `res.redirect`, `sendRedirect` |
| `ldap`, `xpath` | `conn.search`, `client.search`, `xpath`, `XPath.evaluate` |
| `xxe` | XML parsers with external entities turned on |
| `redos` | regular expressions built from input |
| `prototype` (Node.js) | `obj[key] = value` with a key from input, `_.merge` |

It follows input through helpers, return values, other files and classes, objects that
keep input in fields, middleware that keeps input on the request, and callbacks such as
`.then()` and `.map()`. It understands checks: `if not name.isalnum(): return`, allowlists,
regex tests and `'..'` checks make a value safe, and encoders such as `html.escape` make
it safe only for what they encode for (HTML, not SQL). Values decided before the program
runs, dict keys and list positions are followed, so `m['safe']` isn't confused with
`m['input']`.

On the [OWASP Benchmark](https://owasp.org/www-project-benchmark/), whose labelled test
cases are real vulnerabilities or safe look-alikes, the Java audit finds every one of
1,572 injection cases with no false alarms, and the Python audit every one of 677
(scores of +100 on OWASP's scale). The benchmarks are small synthetic cases, so this isn't
proof of the same on real applications; `scripts/audit_benchmark.py` repeats the
measurement on a checkout of either.

All three accept `--format text|json|sarif`, `--output <file>`, `--include-tests`,
`--fail-on-findings`, `--manifest <file>` and `--baseline <file>`. The exit status is 0
when the scan completes, 1 with `--fail-on-findings` when something is found, and 2 for a
bad path or argument.

- `--format sarif` writes SARIF 2.1.0, which GitHub code scanning and other tools read.
- `--baseline report.json` leaves out findings already in an earlier JSON report, so CI
  fails only on new ones. Findings match by file, operation, call, function and input,
  not line, so moved code still matches.
- A comment `aiwaf: ignore` on a finding's line or the line above leaves it out, and
  `aiwaf: ignore[sql, xss]` leaves out only those operations. The report counts them.

A finding is a path worth reading, not a confirmed vulnerability. A reverse proxy, for
example, will report its upstream request because forwarding input is its job. A clean
report doesn't prove the code is safe: reflection, dynamic calls and library code can be
missed, and helpers are followed three calls deep.

Java can also build a route manifest from source, without starting the application.
This is how plain Servlet applications get a manifest:

```bash
java -cp <classpath> com.aiwaf.cli.AiwafConsole init --source src --output .aiwaf/paths.json
```

`init --app <SpringApplicationClass>` starts the Spring application instead and reads its live
routes, then closes it. Its web server listens on a free port meanwhile (`--server.port=0`), so
`init` works while the app, or anything else, holds the configured port. As in Python and
Node.js, `init` writes nothing when it finds no routes, and every failure (no routes, a class
or folder it can't find, a usage error) exits non-zero, so a build step running it fails too.

### SQL detection

AIWAF checks request values for SQL injection with a tokenizer. It tries each value as a
bare number, inside single quotes, and inside double quotes, and flags it only when the
value closes that SQL value and what follows reads as SQL. It catches a number followed by an
always-true condition or a time delay, which needs no quote, and MySQL's version comments,
whose contents MySQL executes. It passes text like `"Item" #3`, `o'reilly`, `union select committee`
and `Drop table tennis; update your set list`.

SQL keywords on their own, with no value to close, aren't flagged, because they can't change a
query.

Besides the query string and body, the check reads the URL path and the headers, where SQL
injection also reaches applications. The path is checked one segment at a time, so a SQL
condition after a post ID in the path is found. Cookies are checked by their names and values, and the `Referer`,
`X-Original-URL` and `X-Rewrite-URL` by their path and query string. Every other header is
checked as one value, such as `User-Agent`, `X-Forwarded-For`, `Accept-Language` or an app's own
`X-` headers. Headers that hold credentials, framing or browser-set values aren't checked:
`Authorization`, `Accept`, `Content-*`, `Sec-*`, `If-*`, `Host`, CSRF tokens and tracing IDs.
Values are decoded the same way, including IIS `%uXXXX` escapes and null bytes, and JSON and XML
documents are checked by their values. Real browser, app, crawler, client and proxy headers,
analytics and session cookies, and search referers are in `tests/fixtures/sql_header_cases.json`,
and ordinary and attack paths in `tests/fixtures/sql_path_cases.json`. Every implementation runs
both. None of the 11,743 URL paths found in the documentation of this project's dependencies is
flagged. `AIWAF_SQL_INJECTION_MODE` still chooses whether a finding
blocks (the default), is only logged, or is ignored.

In Python, the whole request check runs in `aiwaf-rust`, which is required: one call walks
the query and body, decodes values, reads JSON bodies by their values, applies the size limits
and runs the tokenizer. It runs whatever `AIWAF_USE_RUST` says, because there's no Python copy.
Node.js runs the same Rust code through `aiwaf-wasm`, which it requires in the same way, so
`AIWAF_DISABLE_WASM` doesn't turn these checks off. Java has its own copy.

The cases in `tests/fixtures/sql_injection_cases.json` run against every copy in CI, and
`python scripts/sql_accuracy.py` reports the result. They were written alongside the
tokenizer, so passing them doesn't prove real-world accuracy. Add any miss or false block you
see there.

Outside test data is measured too. `python scripts/sql_benchmark.py` runs the whole SQL check
against the OWASP Core Rule Set's SQL injection tests and libinjection's test data, and CI
fails if the results get worse:

| Test set | Attacks caught | Ordinary values flagged |
| --- | --- | --- |
| libinjection `sqli-*.txt` | 84,830 of 85,791 (98.9%) | 1 of 421 in `false_positives.txt` |
| CRS 942, paranoia level 1 | 230 of 324 (71%) | 0 of 49 |
| CRS 942, paranoia level 2 | 209 of 498 (42%) | 2 of 90 |

CRS also matches SQL keywords and function names on their own, such as `concat(` or
`cast(`, especially at higher paranoia levels. AIWAF flags values that break out of a
SQL value, plus names that only appear in SQL, such as delay functions and the schema catalog, so
it misses many of those CRS fragments by design: matching them would block code snippets and
ordinary text. Each CRS test's whole request is checked, including its path and headers. Three
CRS tests marked benign carry SQL injection, such as a post path ending in a UNION query,
because CRS only means that one rule shouldn't match them. AIWAF flags them, and the benchmark
lists them apart from false positives.

The SQL tokenizer, request walker and route checks are fuzzed: for two minutes per target on
each pull request, and for 30 minutes each week (`rust/fuzz`). Short random-input tests also
run with every `cargo test`.

### Remote code execution

The same payload check looks in every request value for text that runs code on the server:

| Rule | What it matches |
| --- | --- |
| `rce_jndi_lookup` | Log4Shell JNDI lookups, including nested lookups that spell out the keyword |
| `rce_shellshock` | Shellshock function definitions in headers and values |
| `rce_serialized_object` | Serialized Java, Python pickle, PHP, .NET and node-serialize objects, and unsafe YAML tags |
| `rce_gadget` | Deserialization gadgets: Fastjson and Jackson type keys that name known gadget classes, XStream dynamic proxies and `XMLDecoder` documents that start a process |
| `rce_expression` | OGNL, SpEL, EL, Thymeleaf preprocessing and Java reflection that reach a process, and Spring4Shell's class-loader property path |
| `rce_template_escape` | Jinja, Twig, Smarty, Velocity, Mako, Ruby ERB and Node.js sandbox escapes that reach Python internals, shell filters or Node's process module |
| `rce_webshell` | JSP, ASP and ASP.NET webshell code, and PHP that runs code or function names taken from the request, including base64-encoded code |
| `rce_script_upload` | Uploads named as server scripts: PHP, JSP, ASPX, `.htaccess` and similar |
| `rce_command_injection` | A probe or timing command chained onto the end of a value, as scanners send it: an identity, system-information or delay command after a semicolon, pipe or `&&`, or inside backticks or command substitution. A short probe must touch its separator, so ordinary text with a semicolon passes |
| `rce_shell_payload` | Reverse shells, download-and-run commands, reads of system account files, shell field-separator tricks, and Windows download commands and PowerShell download cradles |
| `rce_php` | PHP opening tags, PHP calls that run request input, and the PHP stream wrappers that run or unpack code |
| `rce_prototype_pollution` | Prototype-pollution keys in JSON bodies, forms and query strings |

Each value is checked as sent, then again with the disguises attackers use undone: `\u0024`,
`\x24` and `%u0024` escapes, HTML entities such as `&#36;` and `&dollar;`, fullwidth
characters such as `％`, invisible characters inside a word, Log4j lookups nested inside other
lookups, and a whole value sent as base64.

Each field of a query string or form body is also checked on its own, so a serialized Java
object in one field is found whether it's base64 or percent-encoded, and base64 is unwrapped up to three layers deep,
including base64 of gzip or deflate. A raw binary serialized body (Java, pickle, .NET
`BinaryFormatter`, Ruby `Marshal`) is checked before it's read as text. A prototype key in a
JSON body, form or query string is blocked as `rce_prototype_pollution`, since in Node.js it turns into code execution through
options that libraries pass to `child_process` and template compilers.

A multipart form is checked by its first `AIWAF_PAYLOAD_MAX_BYTES`, part by part: each
part's headers (such as a file name ending in `.php`) and field name, each text field as any
other value, and each file for a binary serialized object or PHP, JSP or ASP code, such as PHP
hidden in an image.
A file isn't checked for SQL or the other rules, since an ordinary document can quote them,
and a large upload isn't refused: the rest of it reaches your app unread. Node.js reads the
start of an unparsed multipart body and puts it back for your upload parser; Java reads the
parts the container parsed, when the servlet has a multipart config.

A gzip or deflate body (`Content-Encoding`) is read inflated, within
`AIWAF_PAYLOAD_MAX_BYTES`, so one that inflates past it is refused with `413`. Node.js also
reads `br`. A body in an encoding that can't be read is refused with `413`, since it can't be
checked (Java refuses compressed bodies unless `allowCompressedRequestBodies` is on). Every
header is checked; a request with more than 256 headers or 256 cookies is refused with `413`
rather than checked in part.

It reads the query, the body (JSON, XML and forms), path segments and headers, including
`User-Agent`, `Accept` and `Content-Disposition`, which the SQL check skips. Every rule is checked against the shared
list of ordinary text and must match none of it, and shell syntax on its own (`;`, `|`, a
backtick) isn't matched: only the probes `rce_command_injection` lists are, and routes whose input
reaches a command get the full command checks from `init`.
The rules are in `aiwaf-rust` and `aiwaf-wasm`. Java's copy is generated from them by
`scripts/sync_rce_rules.py`, and a test fails when it's out of date. The cases in
`tests/fixtures/rce_cases.json` (73 attacks and 57 ordinary look-alikes, plus whole form and JSON bodies, such as JSON-LD
`"@type": "Person"` and `filename="report.pdf"`) run against both. It follows `AIWAF_SQL_INJECTION_MODE`:
`block` (the default) answers `403`, `monitor` logs the rule, and `off` turns both checks off.

### WebSocket messages

HTTP checks don't see what a WebSocket carries after its handshake, so each message a client
sends can be checked like a request body, with the same rules and `AIWAF_SQL_INJECTION_MODE`.
A message with a finding closes the socket with `1008` (`1009` when it's too large to check)
and never reaches your handlers; log events carry the rule and path, never the message.

- Python (FastAPI, Starlette, Django Channels): `app.add_middleware(WebSocketGuard)`, from
  `aiwaf.core.websocket` (also `aiwaf.fast.WebSocketGuard`), or wrap your ASGI application.
  Flask-SocketIO: `guard_socketio(socketio)` for every event, or `@aiwaf_message()` under
  `@socketio.on(...)` for one handler, from `aiwaf.flask.websocket`. `inspect_message(data)`
  checks one message anywhere else.
- Node.js: `await aiwaf.guardWebSocketServer(wss)` for `ws`, `await aiwaf.guardSocketIo(io)` for
  socket.io, or `aiwaf.inspectMessage(data)`.
- Java: `new AiwafWebSocketHandler(handler, config)` for Spring WebSocket, and
  `new AiwafEndpoint(endpoint, config)` for a Jakarta WebSocket endpoint (return it from your
  `ServerEndpointConfig.Configurator`). In an annotated `@ServerEndpoint`, call
  `AiwafMessageCore.check(message, config, path)` and close with `AiwafMessageCore.closeCode(rule)`.

These are tested against real servers: Starlette, Flask-SocketIO's test server, `ws` and
socket.io with their clients, and Tomcat's WebSocket container with the JDK client.

### Route-specific checks from init

`init` audits your source as it builds the route manifest, and saves which operation each
route's input reaches. Requests to those routes are then checked for patterns that matter to
that operation, in the parameter that reaches it. A route whose input reaches a file path is
checked for traversal, and a route whose input reaches SQL gets the SQL tokenizer on its route
values too. A match blocks the request with `403` and logs an event like this:

```json
{"event": "aiwaf.injection_risk", "mode": "block", "method": "GET", "route": "/files", "sink": "path", "input": "name", "rule": "path_traversal", "confidence": "medium"}
```

| Operation | Rules: what a value must contain to match |
| --- | --- |
| `sql` | The SQL tokenizer: a value that closes its SQL literal and continues as SQL |
| `command` | `command_chain`, `command_substitution`: `;`, `&` or `\|` and a command, `$(...)`, backticks |
| `path` | `path_traversal`, `path_sensitive_location`, `path_null_byte` |
| `ssrf` | `ssrf_scheme`, `ssrf_internal_host`: `file://` and similar, private and metadata hosts |
| `nosql`, `code`, `template`, `deserialization` | query operators, code evaluation calls, template expressions, serialized object markers |
| `xss` | `xss_markup`, `xss_event_handler`, `xss_script_url`: script tags, event-handler attributes, script URLs |
| `redirect` | `redirect_external`: protocol-relative, backslash, script-URL, user-info and scheme-only targets |
| `ldap` | `ldap_filter`: filter breakouts, OR filters and leading wildcards |
| `xpath` | `xpath_quote_break`, `xpath_tautology`: quote breakouts and always-true conditions |
| `xxe` | `xxe_entity`: document type and entity declarations |
| `redos` | `redos_nested_quantifier`: nested quantifiers |
| `prototype` | `prototype_key`: prototype-pollution keys |

Every rule is checked against a shared list of ordinary values (names, addresses, prose with
quotes and parentheses) that must never match. A plain absolute URL passes the redirect rule,
because the site's own host isn't known there.

```bash
aiwaf init --framework flask --app app:app                 # audits the current folder
aiwaf init --framework flask --app app:app --source src    # audits another folder
aiwaf init --framework flask --app app:app --no-audit      # saves no route checks
```

Node.js `aiwaf init` and `aiwaf manifest` accept `--source` and `--no-audit` too. Java `init --source` uses
the same parse for routes and findings, and `init --app` audits `src/main/java` (or
`--audit-source`). `aiwaf audit --manifest <file>` refreshes the saved findings without
rebuilding the manifest.

`AIWAF_INJECTION_RISK_MODE` is `block` by default. Set it to `log` to only log matches, or
`off` to skip route checks. Flask and Django settings and the Node.js options accept the same
name, Spring uses `aiwaf.injection-risk-mode`, and the Java config file uses
`security.injection_risk_mode`. The Node.js native pipeline reads `AIWAF_INJECTION_RISK_MODE`
and `AIWAF_PATH_MANIFEST` from the environment.

- **Only routes with findings:** a route with no saved findings keeps its normal checks, and
  saved findings never turn a protection off.
- **Logs hold no request values:** events contain route, rule and parameter names from the
  manifest only.
- **What it reads:** query strings, route values, and buffered JSON or form bodies. Headers,
  cookies and multipart uploads aren't checked.
- **Unlinked findings:** `init` reports findings it can't tie to a manifest route, and those
  get no route-specific checks. Handlers are matched by path or function name, so routes
  missing from the manifest, or reused function names in Node.js, can stay unlinked.
- **Stale results:** run `init` again when code changes, for example in your build.
- **Where they run:** route checks run in Rust: through `aiwaf-rust` in Python and
  `aiwaf-wasm` in Node.js. Java has its own copy.
- **Accuracy isn't measured on real traffic yet:** if a rule blocks real users on a route,
  set `AIWAF_INJECTION_RISK_MODE=log` while you report it.

### Route schemas from init

`init` also saves what each route reads from the query and body, and a type for each input,
from your code: `int`, `float`, `bool`, `uuid` or `text`.

```json
"/comments": {"inputs": {"body": "text", "post_id": "int"}, "inputs_complete": true}
```

Types come from what the code says:
- Python: `int(request.GET["page"])`, `type=int`, FastAPI type hints and Pydantic models, Django
  forms and DRF serializers.
- Node.js: `parseInt(req.query.page)`, `Number(...)`, destructuring, and zod object schemas.
- Java: `@RequestParam int page`, `@RequestBody` and `@ModelAttribute` classes and records,
  `Pageable`, and `Integer.parseInt(request.getParameter("n"))`.

Forms, serializers, models, zod schemas, helper functions and string constants are followed
into the project's other files through imports: `from .forms import ContactForm`, `import {
Body } from './schemas'`, `require('./helpers')`, or a static `Paging.page(request)` in Java. A key
held in a constant, as in `request.GET[PAGE_PARAM]` or `getParameter(PAGE)`, is known.

`inputs_complete` is true only when every read was understood. When it isn't, the route's
`inputs_incomplete` says why, with the file and line, and `init`'s summary counts the reasons:

| Code | Meaning |
| --- | --- |
| `helper` | The request is passed to code that isn't followed. |
| `container_whole` | `request.POST`, `req.body` or the parameter map is used whole. |
| `dynamic_key` | An input is read with a key that isn't a constant. |
| `raw_body` | The raw body, query string or a multipart part is read. |
| `class_view` | A class view inherits handlers that aren't read, or `@InitBinder` runs first. |
| `dependency` | A FastAPI dependency takes the request itself. |
| `external_class` | A form, schema or body class isn't in the scanned sources. |
| `model_fields` | A ModelForm, ModelSerializer or `@JsonAnySetter` class takes fields from elsewhere. |
| `middleware` | Node.js route middleware that isn't a body parser may read the body. |
| `observed` | A recording saw the code read a name the analysis didn't find. |
| `unread_view` | Several handlers serve the path, and one's code wasn't found. |

Then `init` fills in what reading the code can't see:

- The framework. `init` runs with your app loaded, so it asks the framework: Django generic
  views (forms, `ModelForm` fields, pagination, `next`), DRF views (serializer fields,
  pagination, search, ordering and filterset parameters, `format`), FastAPI's resolved
  parameters and dependencies, and in Java each Spring handler method and its body classes by
  reflection, including classes from libraries. Python also reads the fields of any form,
  serializer, Pydantic model, marshmallow schema, WTForms form or dataclass the code reading
  couldn't, from the modules the app already imported. Flask and Django views read the fields of their
  webargs `@use_args`/`@use_kwargs` and flask-smorest `@blp.arguments` schemas (marshmallow),
  both from the live app and from source; a view only receives those fields, plus anything it
  reads from `request` itself. When both the query and the body arguments reject unknown names
  (`unknown=RAISE`), requests with any other name fail before the view runs, so the route is
  complete. In Node.js, when you generate the
  manifest from the live app (`generateFrameworkManifest(framework, app)`), it reads Fastify
  route schemas (`querystring` and `body`, including TypeBox; register the aiwaf plugin before
  your routes so it sees them), Hapi `validate.query` and `validate.payload` (Joi), and each
  NestJS handler's `@Query()` and `@Body()` parameters, with their pipes and the fields of
  their DTO classes from class-validator or `@nestjs/swagger` metadata, at the paths Nest
  registers (global prefix and its exclusions, `RouterModule` paths, URI versioning). On Express
  routes it reads validation middleware: celebrate schemas and express-validator chains
  (`query()`, `body()`, `check()`, `checkSchema()`). Validators only check the names they list,
  so the handler's code still says what it reads; when every middleware before the handler is a
  validator, its fields replace the `middleware` reason. A validation middleware of the app's
  own that keeps its zod or Joi schema in its closure, as `validate(schema, 'query')` does, is
  read too, through Node's inspector while the manifest is built (never per request). Route
  middleware the project defines itself, as `app.post('/x', checkBody, handler)`, is read from
  source like the handler, so a schema it parses with (even one defined at the top of the
  module) and anything it reads count for the route. zod schemas (zod 3 and 4) are read
  wherever these accept one: Fastify route schemas (as fastify-type-provider-zod uses them) and
  nestjs-zod DTOs.

  Reading the source does the same for validator calls written on the route
  (`app.post('/x', body('name'), ...)`, `checkSchema({...})`, `celebrate({...})`, also through a
  namespace import such as `ev.body('name')`, an object on `this`, or a function's result). A route behind
  `checkExact(...)`, or a celebrate schema closing both query and body, rejects requests with
  any other name before the handler runs, so it's complete with the names its validators list
  plus what the handler reads.
- Declarations, for code that can't be read. In Python, `@aiwaf.inputs(q=str, page=int)`; in
  Node.js, `aiwaf.inputs({ q: 'text', page: 'int' })(handler)`; in Java,
  `@AiwafInputs({"q", "page:int"})`, on a handler or its class. Or in `.aiwaf/inputs.json`, for
  any language: `{"/search": {"inputs": {"q": "text", "page": "int"}}}`. A declaration makes the
  route's list complete: what it lists, plus anything the code is seen reading. Pass
  `complete=False` (`{ complete: false }`, `complete = false`) to only add names. In Node.js,
  `aiwaf.inputs()` also takes the schema the handler validates with, a zod or Joi schema or a
  DTO class: `aiwaf.inputs(SearchQuery)(handler)`. A schema that allows names it doesn't list
  only adds them.
- An OpenAPI spec, with `AIWAF_OPENAPI_SPEC=openapi.json`. An operation whose body schema
  names every property, and whose route's methods it all lists, is a declaration too.
- Recordings, from tests or staging. Set `AIWAF_RECORD_INPUTS=1` (or a file path). Flask's and
  FastAPI's `AIWAF(app)`, the Node.js middleware, Spring Boot (`aiwaf.record-inputs`) and
  Django (the `AIWAF_RECORD_INPUTS` setting or variable, with aiwaf in `INSTALLED_APPS`, which
  adds `DjangoInputRecorder` last to `MIDDLEWARE`) then record on their own, after AIWAF's own
  checks so its reads aren't counted. Elsewhere use
  `record_flask_inputs(app)`, `app.add_middleware(AsgiInputRecorder)`,
  `app.use(aiwaf.recordInputs())` before Express routes, or `AiwafInputRecorderFilter` in a
  servlet app. This is how handlers no reading or declaration covers get their lists. It records the names each handler reads, not what clients send, to
  `.aiwaf/observed_inputs.jsonl`, which the next `init` reads. Recorded names are added to the
  routes' inputs, and a route the analysis called complete that read something else is made
  incomplete. Without `AIWAF_RECORD_INPUTS` the recorders do nothing. The Java recorder records
  `getParameter` reads, and the application's own whole reads (`getParameterMap`, the body
  stream) as reading everything; reads Spring itself makes for binding aren't counted. Django's
  can't see `json.loads(request.body)`.

Requests to those routes are then reviewed after the general check:

- A typed input sent with a value of another type, as in a SQL condition sent as a numeric
  ID, is `schema_type_mismatch`.
- A finding from `sql_function`, `sql_comment` or `sql_breakout_numeric` is checked again
  without the inputs the schema clears: typed inputs of their type, text that reaches no
  operation, and, when the list is complete, inputs the route never reads. When nothing else is
  found, the finding is cleared. This is where prose and pasted code, such as a question about
  a SQL function, stops being blocked.
- A finding from the URL path, such as a blog slug that names a SQL function, is reviewed by the
  route's parameters: static segments are code, typed parameters (`<int:id>`, `{id:uuid}`, or a
  saved type) must fit, and text parameters that reach no operation are cleared the same way.
- Quote breakouts and every remote code execution rule still block on every
  input. An incomplete list, a sink reached by an input the audit couldn't name, a body that
  isn't a form or JSON object, or a finding from headers or the path keeps the finding.

`AIWAF_ROUTE_SCHEMA_MODE` is `monitor` by default: it logs what it would do
(`{"event": "aiwaf.route_schema", "action": "relax", "rule": "sql_function", ...}`, never
request values) and changes nothing. Set it to `enforce` to act on it, or `off`. Java reads
`security.route_schema_mode` and `aiwaf.route-schema-mode`. The native pipelines review in the
engine too: Python's reads `route_schema_mode` (or `AIWAF_ROUTE_SCHEMA_MODE`), and Node.js's
takes `routeSchemaMode`, including its `inspectPayload()`. The cases in `tests/fixtures/route_schema_cases.json` run against aiwaf-rust, aiwaf-wasm
and Java.

### Strict inputs

With `AIWAF_STRICT_INPUTS=1`, a route whose saved input list is complete accepts only the names
on it: a query or body name the route never reads is `schema_unexpected_input`. It follows
`AIWAF_ROUTE_SCHEMA_MODE`, so it's logged in `monitor` (`"action": "unexpected"` with the name,
cut to 64 characters) and blocked in `enforce`. The query and a form, multipart or JSON object
body are checked, a multipart file part by its field name. Routes whose list is incomplete,
routes with no saved inputs, path parameters, headers and other bodies are never checked.

Keys inside an object-valued input are checked too when `init` knows them all
(`inputs_nested`): from code that reads only constant keys inside it, as
`data["user"]["name"]` or `req.body.user.name`, in every handler that reads it, from a FastAPI
body model whose field is itself a Pydantic model (`customer: Customer` gives `customer`'s
declared fields, one level down), or from a Java body class's nested classes. A JSON
`{"user": {"role": "admin"}}`, or a form's `user[role]`, is then `user.role`, unexpected. An
input a handler passes on whole, a model that allows undeclared fields, or one that another
framework schema or declaration names, has no such list and isn't checked inside.

`AIWAF_STRICT_INPUTS_EXEMPT=/webhooks/,/legacy/*` lists path prefixes strict inputs skips, for
routes that take whatever their senders add.

When a path's methods are served by different handlers, as a GET view that reads `q` and a POST
view that reads `title`, `init` saves each method's own list (`inputs_by_method`), and strict
inputs check a request against its method's: `title` on the GET is unexpected. A method takes
the names its own handler's code (or the framework's per-route schema, or its OpenAPI
operation) reads, plus every name nothing ties to one method: declarations, recordings and the
like. HEAD uses GET's list, and a method without its own list uses the whole path's.

CSRF tokens (`csrfmiddlewaretoken`, `csrf_token`, `_csrf`, `_token`, `authenticity_token`) and
tracking and cache-busting parameters (`utm_*`, `fbclid`, `gclid`, `msclkid`, `_`) are always
allowed. Add more with `AIWAF_STRICT_INPUTS_ALLOW=trace_id,ref_*`; a trailing `*` matches any rest.

Turn it on in `monitor` first and read the log: a name that turns up there for a real client is
either one the analysis missed (record it, or declare it) or one to allow. The setting is
`AIWAF_STRICT_INPUTS`/`AIWAF_STRICT_INPUTS_ALLOW` in Django settings, Flask config and FastAPI
config; `strictInputs`/`strictInputsAllow` for the Node.js native pipeline; and
`security.strict_inputs`, `aiwaf.strict-inputs` or `AiwafConfig.strictInputs` in Java. It needs
`aiwaf-rust` 0.2.16 or `aiwaf-wasm` 0.2.12; with an older engine it logs a warning and stays
off. The cases under `strict_cases` in `tests/fixtures/route_schema_cases.json` run against all
three.

## Repository layout

```text
.
├── py/aiwaf/                       Python package and adapters
├── js/                             Node.js package and adapters
├── rust/                           Rust core and Python bindings
│   └── crates/aiwaf_wasm/          WebAssembly bindings
├── java/                           Java 17 package
├── tests/                          Python integration and parity tests
├── examples/                       Focused Python examples
├── .github/workflows/              Tests, security checks, and releases
├── README.md                       Canonical project documentation
└── LICENSE                         Canonical Elastic License 2.0 (see NOTICE)
```

## Quick installation

Choose the package for the application runtime:

```bash
# Python core or a framework extra
pip install aiwaf
pip install "aiwaf[django]"
pip install "aiwaf[flask]"
pip install "aiwaf[fastapi]"

# Backward-compatible alias; Python 1.3.0 already installs aiwaf-rust
pip install "aiwaf[rust]"

# Node.js middleware
npm install aiwaf

# Standalone WebAssembly bindings
npm install aiwaf-wasm
```

npm 12 runs a dependency's install script only after you approve it, and `sqlite3`, where
Node.js AIWAF keeps its blocklist, exemptions and logs, gets its native binary from its install
script. Without approval AIWAF still runs, keeps that data in CSV files under `logs/storage`, and
prints one warning. To use SQLite, approve and build it once in your app:

```bash
npm approve-scripts sqlite3
npm rebuild sqlite3
```

Java 1.4.0, after it is published to Maven Central:

```xml
<dependency>
  <groupId>io.github.aiwaf-project</groupId>
  <artifactId>aiwaf-java</artifactId>
  <version>1.4.0</version>
</dependency>
```

## Python package

The Python package requires Python 3.10 or newer. Framework dependencies are optional, so install only the extra needed by the application.

Python 1.3.3 requires `aiwaf-rust>=0.2.17`; the `rust` extra remains an installation
alias. Publish the matching Rust wheel with `NativeCSVPipeline` support before
the Python release. The Rust release workflow builds Linux x86_64/aarch64,
musl x86_64, macOS x86_64/arm64 and Windows x86_64 wheels and checks the native
pipeline API. Windows and WSL builds are checked locally; the other wheel
targets must pass CI before publication.

### Rust-owned CSV pipeline

The complete CSV request pipeline is available through explicit adapters below.
Install it **instead of the existing eight-middleware chain**. The existing
`AIWAF` adapters remain available for compatibility with custom stores and hooks;
they do not switch automatically to this pipeline.

Rust owns the immutable configuration, enforcement order, SQL and keyword
inspection, geo and blocklist decisions, rate buckets, form timing state, UUID
scores, anomaly inference/history, keyword learning and CSV logging. Python
supplies framework routing/authentication facts, obtains GeoIP country data,
buffers a bounded body prefix and constructs responses. Each request crosses
into the engine once before the handler and once when its response is available.
The engine never calls Python policy callbacks or falls back to Python execution.
Missing native support, unsupported settings and invalid CSV schemas fail at
startup.

```python
# Flask
from flask import Flask
from aiwaf.flask import NativeAIWAF

app = Flask(__name__)
NativeAIWAF(app, config={"data_dir": "aiwaf_data"})
```

```python
# FastAPI / pure ASGI
from fastapi import FastAPI
from aiwaf.fast import NativeAIWAFMiddleware

app = FastAPI()
app.add_middleware(NativeAIWAFMiddleware, config={"data_dir": "aiwaf_data"})
```

```python
# Django settings: replace the existing AIWAF middleware entries
AIWAF_CSV_PIPELINE = {"data_dir": "aiwaf_data"}
MIDDLEWARE = [
    # Keep the application's other middleware in the required order.
    "aiwaf.django.csv_pipeline.NativeAIWAFMiddleware",
]
```

The configuration uses a dedicated lowercase profile; existing uppercase
framework settings, `AIWAFConfig` dictionaries, custom stores and policy callbacks
are not imported into it. Unknown keys are rejected. Path rules and the route
manifest work as in the regular chain, through three adapter keys:

- `path_rules`: the same rules as `AIWAF_PATH_RULES`. `DISABLE` turns stages off for
  a prefix, and `RATE_LIMIT` (`WINDOW`, `MAX`, `FLOOD`) replaces the profile's rate
  limits there. Invalid limits fail at startup.
- `manifest_path`: default `.aiwaf/paths.json`. Its route rules are added after
  `path_rules`, and the route-specific checks saved by `init` run on matching requests.
- `injection_risk_mode`: `block` (default), `log` or `off`, defaulting to
  `AIWAF_INJECTION_RISK_MODE`. A blocked request is logged with the matched rule.
Route decorators for full/partial exemptions and required middleware are read
once. Required decorators override decorator exemptions; profile-level path/IP
exemptions remain global. The profile is immutable: construct a new engine and
reload workers to change it.

**The regular setup runs as the native pipeline by default** when the installed aiwaf-rust
has every native feature (0.2.9 or newer) and every setting carries over exactly,
so the report below lists no differences. Nothing in the app changes:
- **Django:** the regular AIWAF classes in `MIDDLEWARE` run as one native middleware, at
  the place of the last of them, so sessions and authentication have run before it. The
  others step aside with `MiddlewareNotUsed`.
- **Flask and FastAPI:** `AIWAF(app)` installs the native pipeline instead of the regular
  middleware, with the same stages its plan would register.

Request logs go to `aiwaf_native_access.csv` in the regular log directory, which the Flask
and FastAPI trainers read. A model the regular middleware has loaded carries over if
aiwaf-rust made it; an sklearn model, or a Flask model still held back for lack of logs,
is a difference. AIWAF logs one line on the `aiwaf.native` logger when it switches. To keep
the regular middleware, set `AIWAF_NATIVE_PIPELINE = False` in Django settings or Flask
config, pass `native_pipeline=False` to Flask's or FastAPI's `AIWAF(...)`, or set the
`AIWAF_NATIVE_PIPELINE=false` environment variable. `aiwaf init` says whether a project
switches and, if not, why.

To start from the settings the regular middleware already uses explicitly, pass
`"from_settings"`: `NativeAIWAF(app, "from_settings")` in Flask,
`AIWAF_CSV_PIPELINE = "from_settings"` in Django, or
`app.add_middleware(NativeAIWAFMiddleware, config="from_settings")` in FastAPI, which
converts the `AIWAFConfig` the regular middleware uses (or one passed as `aiwaf_config`). `{"from_settings": True, ...}` converts
and then applies the other keys on top. The same `AIWAF_*` names are read with the same
defaults, including Django's built-in login and admin exemptions and keyword list, and
Django's stages follow the regular classes in `MIDDLEWARE`. At startup AIWAF logs, on the
`aiwaf.native` logger, how many settings carried over unchanged and every one that didn't,
with the reason: for example, Flask's `*.css`-style exemptions and IP ranges in
`AIWAF_EXEMPT_IPS`, which the native pipeline can't match. It also uses the regular
stores (below), so blocks, exemptions and keywords already stored carry over, and Flask's
pattern exemptions and FastAPI's private-range exemptions keep working. FastAPI's header-only `exempt_paths` become path rules
that turn off header validation alone, and its private-range auto-exemptions are reported
because only exact addresses carry over. `aiwaf.core.native_settings.native_config()` and
`native_config_fastapi()` return the same profile and report without starting anything.

By default the native pipeline keeps its blocklist, exemptions, learned keywords and
blocked countries in CSV files in `data_dir`. To use the stores the rest of the app
already uses, set `"store"` in the native config: `"django"`, `"flask"` or `"fastapi"` use
that framework's regular stores, following its storage settings (`AIWAF_STORAGE_MODE`,
`AIWAF_USE_CSV` or `AIWAF_DATA_DIR`, or `AIWAFConfig`'s `storage`), so the native and regular
middleware and the management commands share one blocklist. Flask's store also matches its
`*.css`-style exemptions, and FastAPI's its private-range IP exemptions, which native
profiles can't. `"from_settings"` turns this on when the installed aiwaf-rust supports it.
Elsewhere, pass `stores=` to `CSVPipeline`. Before
each decision AIWAF reads what that request's stages use: whether it's exempt, then, if
not, whether the IP is blocked (IP/keyword and anomaly stages), the learned keywords
(IP/keyword stage) and blocked countries (geo stage). A route whose path rules turn those
stages off reads nothing. Answers are cached for `store_cache_seconds` (default 5):
keywords and countries, and the blocklist and exemption answers per IP and path, up to
10,000 of each. So a block or exemption made in another worker, the admin or the CLI can
take that long to apply; blocks this worker makes apply at once. `store_cache_seconds: 0`
reads the stores on every request. It then
writes the blocks and keywords the decision produced back through the stores. The
decision itself still runs in aiwaf-rust. A custom store is any object with
`is_blocked(ip)`, `is_exempted(ip, path)`, `top_keywords(n)`, `blocked_countries()`,
`block(ip, reason)` and `learn(keywords)`; `aiwaf.core.csv_pipeline.PipelineStores`
documents them. Rate limits, honeypot timing, UUID scores and anomaly history stay in
the engine.

Common profile keys and defaults:

| Setting | Default / meaning |
| --- | --- |
| `enabled` | All eight names: `geo_block`, `ip_keyword_block`, `rate_limit`, `ai_anomaly`, `honeypot`, `uuid_tamper`, `header_validation`, `logging` |
| `data_dir` | `aiwaf_data`; policy CSVs use the shared allowlist/blocklist/keyword/geo/path schemas |
| `exempt_paths`, `exempt_ips` | `/health`, `/robots.txt`, `/sitemap.xml`; empty IP list. Paths are literal, case-insensitive prefixes, not globs. Stored path exemptions use the same literal-prefix rule |
| `sql_mode`, `max_bytes` | `block`, `65536`; also `monitor`, which logs findings on the `aiwaf.payload` logger without blocking, and `off`. Structured JSON/form bodies and nested escaped strings are inspected with bounded traversal |
| `learning`, `top_n` | `True`, `10`; presence keywords use a sorted, stable subset instead of Python set iteration order |
| `static_keywords`, `legitimate_keywords`, `exempt_keywords`, `safe_prefixes` | Native policy defaults or explicit arrays; set route-specific safe vocabulary in the profile |
| `geo_allow`, `geo_block` | Empty lists; includes stored blocked countries. Unknown country remains allowed |
| `rate_backend`, `rate_key_mode` | `memory`, `ip_path`; `csv` shares buckets across workers; `redis` and `django_cache` (Django) share counts through a cache, below; `ip` aggregates paths |
| `rate_redis_url`, `rate_cache_namespace`, `rate_cache_alias` | Required for `rate_backend="redis"`; key prefix, default `aiwaf:native:rate:v1`; Django cache alias for `django_cache`, default `default` |
| `rate_window`, `rate_max`, `rate_flood` | `10`, `20`, `40`; throttle at 429, flood at 403 with blocklist persistence |
| `soft_block_blocklist`, `ip_blocking`, `block_duration` | `False`, `True`, reputation-based duration; explicit seconds or nonpositive permanent duration |
| `honeypot_backend` | Django defaults to `django_cache`; Flask/FastAPI and direct construction default to `csv`. Optional `memory` is local to one engine; `redis` and `cache` delegate timestamp I/O while Rust enforces timing |
| `honeypot_cache_alias` | Django-only; `default`. Uses the named `CACHES` entry for honeypot timestamps, not other policy or rate state |
| `honeypot_redis_url` | Required for `honeypot_backend="redis"`; stripped before native profile serialization. `rediss://` requires redis-py |
| `honeypot_cache_namespace` | Redis/custom cache key prefix, default `aiwaf:native:honeypot:v1`; use the same value across workers and a distinct value for unrelated applications |
| `min_form_time`, `max_page_time`, `skip_authenticated` | `1.0`, `240`, `True`; native Rust applies the timing policy |
| `uuid_score_enabled`, `uuid_window`, `uuid_threshold` | `True`, `60`, `5`; weights: `uuid_malformed_weight=5`, `uuid_not_found_weight=1`, `uuid_success_decay=2` |
| `required_headers`, `header_min_score` | `HTTP_USER_AGENT` and `HTTP_ACCEPT`, `3`; empty required list disables required-header/quality checks, retains safety caps |
| `anomaly_window`, `anomaly_keywords` | `60`, standard anomaly vocabulary; history/scoring is held in Rust per worker |
| `log_path`, `log_fsync` | `data_dir/requests.csv`, `False`; immediate CSV append, optional per-row fsync |

Pass a trained `aiwaf_rust.IsolationForest` as `model=` to Flask/FastAPI, or
set `AIWAF_CSV_PIPELINE_MODEL` in Django. Without a model, the anomaly stage
maintains history and learns suspicious 404 keywords but performs no model-based
blocking. `pipeline.native.set_model(model)` replaces the model with an owned
snapshot; `set_model(None)` disables inference. Training and management APIs
remain available through the existing Python interface.

For deployments that accept per-worker form timing, set
`"honeypot_backend": "memory"` in the native profile. Rust then holds GET
timestamps in an expiring map and evaluates/clears POST timing atomically,
avoiding a durable CSV rewrite on each GET. Timing state is isolated per engine,
is not shared across workers and resets on restart. Backend switches do not
transfer timestamps automatically. Keep `csv` for shared or persistent timing;
CSV writes remain immediately synced. Blocklist, keyword, geo and allowlist
policy files and immediate request logging keep their configured CSV behavior.

### Shared native rate limits

With `rate_backend: "memory"` each worker counts on its own, so N workers allow N times
the limit. `rate_backend: "redis"` with `rate_redis_url` counts in Redis instead, with the
regular limiter's atomic sliding-window script, and in Django `"django_cache"` counts in a
Django cache (`rate_cache_alias`), as the regular rate limiter does. Only the count is
shared: aiwaf-rust still applies `rate_max`, `rate_flood` and path-rule limits to it.
`CSVPipeline(config, rate_counter=...)` with `rate_backend: "counter"` takes any object
with a synchronous, atomic `count(key, now, window_seconds)` that records the request and
returns the window's count. A failing counter answers 503, as the regular limiter does.
`from_settings` carries this over: Django counts in its default cache, and Flask
(`AIWAF_RATE_CACHE_BACKEND="redis"` with `AIWAF_REDIS_URL`) and FastAPI
(`rate_limiting.cache_backend`, `rate_limiting.redis_url`) count in the same Redis. It
needs `aiwaf-rust` 0.2.9 or newer; with 0.2.8 a counter fails at startup and `from_settings`
reports that rate limits stay per worker.

### Flask and FastAPI native honeypot cache

Both native adapters accept Redis or a supplied synchronous cache. CSV remains
their default. This changes honeypot timestamp storage only; other policies and
rate-limit backends retain their configured behavior.

```python
import os
from aiwaf.flask.csv_pipeline import NativeAIWAF

profile = {
    "data_dir": "aiwaf_data",
    "honeypot_backend": "redis",
    "honeypot_redis_url": os.environ["AIWAF_REDIS_URL"],
    "honeypot_cache_namespace": "myapp:honeypot:v1",
}
NativeAIWAF(app, config=profile)  # Flask

# FastAPI alternative:
# from aiwaf.fast.csv_pipeline import NativeAIWAFMiddleware
# app.add_middleware(NativeAIWAFMiddleware, config=profile)
```

For an existing Flask-Caching instance or another compatible object, select
`honeypot_backend="cache"` and pass `honeypot_cache=cache` to either native adapter.
The contract is `get(key)` returning a stored value or `None`, and
`set(key, value, timeout=seconds)` returning success or `None`. Explicit `False`
means failure. Methods must be synchronous, preserve a timestamp/expiry pair,
honor TTL, and propagate storage errors. Disable any ignored-error option;
no-op cache backends are unsuitable. Custom adapters must translate their own
cache's TTL argument if it differs. A local cache still isolates workers.

Rust retains timing decisions, exemptions and logical expiry. Entries have a
300-second TTL; misses/evictions permit the POST timing check, while exceptions,
invalid records and failed writes return generic HTTP 503. Expired-page POSTs
return 409 until a new GET or TTL expiry, without deleting a concurrent refresh.
FastAPI offloads the cache-backed native request check to its worker thread pool
so synchronous network I/O does not block the event loop. Pool saturation and
cache latency still affect request latency. Existing CSV/memory timing behavior
and the legacy middleware chain are unchanged.

To exercise independent workers against a dedicated test Redis instance, set
`AIWAF_TEST_REDIS_URL` and run
`python -m pytest tests/integration/test_native_honeypot_cache.py`.

### Django native honeypot cache

Starting with Python 1.1.7 / aiwaf-rust 0.2.6, Django's native middleware uses
the existing Django cache for honeypot GET timestamps by default. Rust calls a
small Python storage adapter; form timing, logical expiry, exemptions and block
decisions remain native. Other policies retain their configured CSV storage.

```python
AIWAF_CSV_PIPELINE = {
    "data_dir": "aiwaf_data",
    "honeypot_cache_alias": "default",  # optional; uses CACHES["default"]
}
```

Use a shared Django cache backend for cross-worker timing. For example, Django's
built-in Redis cache can be configured as follows (install its Redis Python
driver and configure the URL for your deployment):

```python
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": "redis://127.0.0.1:6379/1",
        "KEY_PREFIX": "my-service",
    },
}
```

LocMemCache remains supported with a startup warning: it is worker-local and
resets on restart. DummyCache is rejected when the honeypot stage is enabled.
Configured ignored cache errors are rejected; runtime read/write failures,
failed writes and invalid timestamp records produce generic HTTP 503, without
silently selecting memory or CSV. Cache methods must be synchronous.

Timestamp entries use a 300-second TTL and bounded, hashed client-IP keys under
Django's key prefix/version. Expired-page POSTs return 409 until a fresh GET or
TTL expiry. The pipeline does not delete an expired snapshot, so it cannot erase
a concurrent GET refresh. As before, no timestamp means the timing check allows
the POST; cache outages are distinguished from misses. Backends can evict entries
before their TTL. Changing backends does not migrate existing timestamps.

Set `"honeypot_backend": "csv"` explicitly to retain shared filesystem timing,
or `"memory"` for the per-engine Rust map. Install the matching native extension
before upgrading Django workers. Flask and FastAPI defaults are unchanged.

Flask/FastAPI accept synchronous `country_lookup(ip)` and
`authenticated(request)` callbacks for adapter facts. Default GeoIP uses the
bundled lookup. Configure authentication explicitly in these adapters when
authenticated requests should bypass honeypot timing; Django reads user/session
state, so place it after authentication/session middleware. Django function views
have no standard method declaration and are accepted conservatively; class-based
views supply their declared method facts. Anomaly history stores route eligibility
as observed at arrival; historical requests are not re-resolved after routes change.

The ASGI adapter preserves streaming bodies and finalizes enforcement/logging at
response start. Logged response time measures time to response headers rather
than stream completion. WSGI adapters finalize when the framework response is
available. Session/route handlers, response construction, GeoIP lookup and legacy
custom integrations still execute in Python; the whole Python package has not
been rewritten in Rust.

### Flask

```python
from flask import Flask
from aiwaf.flask import AIWAF

app = Flask(__name__)
AIWAF(
    app,
    middlewares=["auto"],
)

@app.get("/")
def home():
    return {"protected": True}
```

`middlewares=["all"]` enables the canonical middleware set. `auto` starts from the same set and omits checks that are not useful for the detected application signals.

### FastAPI

```python
from fastapi import FastAPI
from aiwaf.fast import AIWAF

app = FastAPI()

AIWAF(
    app,
    middlewares=["auto"],
    storage={"backend": "memory"},
    header_validation={"enabled": True, "quality_threshold": 3},
    rate_limiting={"enabled": True, "window_seconds": 10, "max_requests": 20},
    logging_middleware={
        "enabled": True,
        "log_dir": "aiwaf_logs",
        "log_format": "json",
    },
)

@app.get("/")
async def home():
    return {"protected": True}
```

### Django

Add the app and the unified middleware alias in `settings.py`:

```python
INSTALLED_APPS = [
    # ...
    "aiwaf.django",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "aiwaf.django.middleware.all",
    # ...
]
```

For explicit ordering, replace the alias with the individual middleware classes:

```python
MIDDLEWARE = [
    "aiwaf.django.middleware.JsonExceptionMiddleware",
    "aiwaf.django.middleware.GeoBlockMiddleware",
    "aiwaf.django.middleware.IPAndKeywordBlockMiddleware",
    "aiwaf.django.middleware.RateLimitMiddleware",
    "aiwaf.django.middleware.AIAnomalyMiddleware",
    "aiwaf.django.middleware.HoneypotTimingMiddleware",
    "aiwaf.django.middleware.UUIDTamperMiddleware",
    "aiwaf.django.middleware.HeaderValidationMiddleware",
    "aiwaf.django.middleware_logger.AIWAFLoggerMiddleware",
]
```

Protection middleware should run early. Logging should run late enough to observe the application response.

### Route manifests

`aiwaf init` imports the live application and records all registered routes in `.aiwaf/paths.json`. The `--app` value must be `module:object`, not only a module name.

```bash
# Flask application object
aiwaf init --framework flask --app app:app

# Flask application factory
aiwaf init --framework flask --app app:create_app

# FastAPI
aiwaf init --framework fastapi --app app:app

# Django
aiwaf init --framework django --settings project.settings

# Custom output
aiwaf init --framework flask --app app:app --output .aiwaf/paths.json
```

`aiwaf init` then prints how to run the native pipeline for that framework, the
recommended setup for new installs, with the conversion report for the project's own
settings: what carries over and what wouldn't. In a Django project already using the
regular middleware, it says how to switch; nothing changes until you do. `--no-native`
skips this part.

Run the command in the same environment and checkout used to start the application. If it discovers only `/` and Flask's static route, verify the imported file and live route table:

```bash
python -c "import app; print(app.__file__); print(*sorted(r.rule for r in app.app.url_map.iter_rules()), sep='\n')"
```

### Python route exemptions

Django, Flask, and FastAPI expose equivalent route helpers:

```python
from aiwaf.flask import (
    aiwaf_exempt,
    aiwaf_exempt_from,
    aiwaf_only,
    aiwaf_require_protection,
)

@app.get("/health")
@aiwaf_exempt
def health():
    return {"ok": True}

@app.post("/webhook")
@aiwaf_only("header_validation")
def webhook():
    return {"accepted": True}
```

Canonical middleware names include `header_validation`, `rate_limit`, `ip_keyword_block`, `honeypot`, `geo_block`, `uuid_tamper`, `ai_anomaly`, and `logging`.

### Python configuration

Python adapters use flat `AIWAF_*` settings or their equivalent constructor options.

| Area | Important settings |
| --- | --- |
| Training | `AIWAF_ACCESS_LOG`, `AIWAF_MIN_TRAIN_LOGS`, `AIWAF_MIN_AI_LOGS`, `AIWAF_FORCE_AI_TRAINING` |
| Rate limiting | `AIWAF_RATE_WINDOW`, `AIWAF_RATE_MAX`, `AIWAF_RATE_FLOOD` |
| Headers | `AIWAF_REQUIRED_HEADERS`, `AIWAF_HEADER_QUALITY_MIN_SCORE` |
| Honeypot | `AIWAF_MIN_FORM_TIME`, `AIWAF_MAX_PAGE_TIME` |
| UUID checks | `AIWAF_UUID_SCORE_ENABLED`, `AIWAF_UUID_SCORE_BLOCK_THRESHOLD` |
| GeoIP | `AIWAF_GEO_BLOCK_ENABLED`, `AIWAF_GEO_BLOCK_COUNTRIES`, `AIWAF_GEO_ALLOW_COUNTRIES` |
| Exemptions | `AIWAF_EXEMPT_PATHS`, `AIWAF_EXEMPT_KEYWORDS`, `AIWAF_ALLOWED_PATH_KEYWORDS` |
| Models | `AIWAF_MODEL_PATH`, `AIWAF_MODEL_STORAGE`, `AIWAF_MODEL_STORAGE_FALLBACK` |

When `aiwaf_rust` is importable, supported acceleration paths are selected automatically. Python remains the fallback when a native capability is unavailable. Persisted runtime models use JSON state; AIWAF does not load pickle, joblib, or other executable Python-object formats.

### Rust-backed CSV runtime and training

The Python package and `aiwaf-rust` provide native CSV storage,
shared policy evaluation, and training preparation for Django, Flask, and FastAPI.
Install matching Python and Rust builds and reload application workers to activate
the new capabilities. Older extensions retain the compatible Python fallback.

- **CSV storage:** bounded snapshot caches and indexed IP/key lookups avoid repeated
  parsing and copying entire tables into Python. Native readers validate file
  metadata and use shared OS locks; mutations use exclusive locks. Blocklist
  reputation updates, keyword increments/removals, exemptions, audit entries, and
  geo-list changes use atomic transactions. Unchanged policy rows skip rewriting
  and disk synchronization. Changed policy writes still synchronize to disk.
- **Request policies:** native helpers handle keyword/probe decisions, ASCII SQL
  signatures, geo decisions, rate-window filtering, form timing, UUID format
  validation, reputation scoring, wildcard/CIDR exemptions, and longest-prefix
  route plans. Flask CSV keyword retrieval, evaluation, and learning share one
  native call. SQL checks and blocklist updates retain their existing order;
  custom evaluators and unsupported Unicode behavior use the Python adapter.
- **Exemption reuse:** Flask reuses CSV exemption results within a request and
  checks file identity, size, timestamps, and configuration before each reuse.
  Changes invalidate the result; decorator checks still execute. Set
  `AIWAF_CSV_REQUEST_CACHE=False` to disable this optimization.
- **Training ingestion:** plain and gzip CSV stream directly in bounded batches,
  including quoted fields and embedded newlines. Seconds and millisecond response
  fields are normalized. Apache logs use batched extraction and native timestamp
  parsing for supported English/C-locale forms. Unsupported dates/locales and
  Unicode keyword learning retain Python behavior. Route checks are reused for
  distinct paths, and learned keywords are persisted in batches.
- **Native model preparation:** `NativeTrainingBatch` keeps features and numeric
  matrices in Rust through Isolation Forest fitting, prediction, and per-IP anomaly
  aggregation. `NativeTrainingBuilder` preserves Django's chunk/state behavior.
  Supported native model paths train without a Pandas DataFrame; older APIs and
  custom parsers retain compatibility paths.
- **Synchronous CSV logging:** a bounded cache of writer handles avoids repeated
  opens. Every row is locked and flushed before returning. File identity checks
  handle rotation/replacement, and truncation recreates the header. Rows are not
  queued in an application-side batch.

Framework hooks, request/body extraction, route discovery, session/ORM access,
response construction, datetime object/epoch conversion, training orchestration,
and model artifact serialization still run in Python. The route plan selects
middleware; framework hooks continue to execute separately in their established
order. Database and Redis backends remain separate from the CSV implementation.

Flask and Django honor `AIWAF_USE_RUST=False` or `AIWAF_RUST_CSV=False` to disable
native CSV dispatch. Outside those application contexts, use the environment
variable `AIWAF_RUST_CSV=false`. FastAPI training also honors these application
options. ISO timestamps use Python's faster C parser by default;
`AIWAF_RUST_ISO_DATES=1` opts into native ISO parsing for measurement.

Flask can opt into shared, persistent CSV rate buckets:

```python
app.config["AIWAF_RATE_CACHE_BACKEND"] = "csv"
# Optional: otherwise stored under AIWAF_DATA_DIR.
app.config["AIWAF_RATE_CSV_PATH"] = "/absolute/path/to/rate_buckets.csv"
```

CSV rate decisions trim the window, consume the request, and persist the bucket
under one native lock. Missing or disabled native support returns HTTP 503.
Memory remains the default; CSV rewrites and synchronization can be slower than
memory or Redis. Workers sharing CSV state must use compatible schemas and a
filesystem with suitable locking. Legacy Python fallback writers on Windows only
have process-local thread locks, so mixed Python/native writers need particular
care. Explicit whole-file replacements are not concurrent incremental updates.

`AIWAF_CSV_LOG_FSYNC=True` adds a disk sync per log row; the default is `False`,
matching previous Python logging durability. Without per-row sync, a system crash
can lose recently flushed OS-buffered entries. Cached handles close at process
shutdown. After requests stop, call the Flask logger's `close()` or
`aiwaf.core.native_runtime.close_csv_logs()` before deleting temporary log
directories, especially on Windows.

Check the installed extension without touching application data:

```bash
python scripts/check_native_csv_runtime.py
```

`policy_sql_inspect` runs the whole SQL request check in one call, and Django, Flask and
FastAPI use it for every request. `RouteRiskChecker` and `policy_route_rule` run the
route-specific checks saved by `init`. `policy_sql_tokenizer` checks one value.
`policy_sql_signatures` and `csv_geo_policy` remain for older releases.

Flask with CSV geo state and an explicit allow/block list can read dynamic countries
and evaluate the geo policy in one native call. Django CSV and FastAPI CSV use
`csv_runtime_geo_policy` for their key/value geo state. Django additionally retains
its existing ORM country blocks; the combined call includes those countries as
extra policy input. ORM-only Django operation remains separate. FastAPI geo
middleware honors its `use_rust` option, passed from the application's Rust flags.
Dynamic-only FastAPI policy and custom getters/evaluators retain their fallback
path. Native reads respect TTLs and observe updates/removals; disabled native geo
reads refresh the Python CSV snapshot. Exemptions and blocklist writes still run
at their original stages. Rebuild the extension from this checkout to test these
unpublished capabilities.

### Request-time anomaly acceleration (development)

With a matching rebuilt extension and a real Rust Isolation Forest, Flask and
FastAPI retain request histories as immutable `NativeAnomalyHistory` snapshots.
Native code trims the time window, builds the six current features, and predicts
without exporting the numeric matrix. Records are shared between snapshots;
evaluating a response does not mutate another request's history. Existing
per-worker cache ownership and response/keyword/blocking order remain unchanged.

If the forest predicts an anomaly, Python still resolves the current route and
exemptions for each historical path. Rust then analyzes the retained records.
Dynamic route/exemption changes remain visible. Unicode paths/keywords, non-finite
current inputs, disabled native settings, older extensions, and other model types
retain the established Python-history path. Set
`AIWAF_RUST_ANOMALY_HISTORY=False` in Flask config or the FastAPI `AIWAFConfig`
object to disable native history retention; existing snapshots convert back to
tuple lists. Disabling or replacing a model also retains compatibility.

Django continues to cache serializable tuple lists, including with shared cache
backends. The shared recent-behavior bridge uses `analyze_recent_tuples` to avoid
building a dictionary per historical record. No native process-local object is
stored in Django's cache. These changes do not introduce model auto-reload or
shared histories for Flask/FastAPI.

Further request-path refinements use the existing native CSV log writer for
Flask/FastAPI anomaly training logs when supported. Each row is flushed before
returning; there is no background queue. `AIWAF_CSV_LOG_FSYNC=True` additionally
requests an OS sync for these logs, including in the Python fallback. Rust-disable
settings and older extensions retain the Python writer.

Flask reuses its routing adapter within a request while continuing to match each
path and evaluate exemptions. Route-map replacement and additions remain visible;
the `MethodNotAllowed` import was corrected so existing routes with another HTTP
method are recognized. Flask/FastAPI sweep inactive histories at most every 30
seconds during response processing, removing only histories with no entries in
the configured scoring window. Active histories remain uncapped, and a quiet
worker performs no background cleanup. Django continues to use cache timeouts.

Rust recent-behavior analysis now counts symmetric ten-second bursts with a
linear sweep after sorting finite timestamps, replacing two binary searches per
record. Non-finite timestamps retain the established search path. This shared-core
change also applies to the WASM recent-behavior API.

### Python operations

#### Experimental compiled request stage

`AIWAF_RUST_REQUEST_ENGINE=True` opts into `NativeRequestEngine` with a matching
rebuilt Rust extension. Set it in Flask/Django settings or the FastAPI `AIWAFConfig`.
It is **off by default**. This first stage combines bounded ASCII query SQL
inspection, default keyword-context evaluation, dynamic CSV keyword reads, and
keyword learning writes behind a compiled policy profile. Static policy changes
rebuild the profile; CSV state remains live. Flask/FastAPI can use one native call
for supported query requests. Django keeps SQL inspection before route resolution,
then uses the compiled keyword/CSV stage with the resolved route facts.

Framework extraction, body buffering/replay, exemptions, blocklist enforcement,
rate limits, honeypot/UUID hooks, header enforcement, and response processing keep
their existing positions. Structured SQL bodies, Unicode paths/decoding, monitor
mode, custom policy/storage hooks, custom Django converters, older extensions,
and dynamic-keyword subsets whose ranking requires Python retain the established
path. The engine does not replace the whole eight-middleware chain.

```bash
# Auto-detected CLI (one supported framework installed)
aiwaf --help
aiwaf blocklist migrate
aiwaf status

# Django management commands
python manage.py detect_and_train
python manage.py regenerate_model
python manage.py aiwaf_reset --keywords --confirm
python manage.py aiwaf_logging --status
python manage.py geo_block_country list

# Explicit adapter dispatch
aiwaf django aiwaf_list --all
aiwaf flask --help
aiwaf fast --help
aiwaf fastapi --help

# FastAPI route-aware training using the application's runtime storage
aiwaf fast train --app app:app --log-dir aiwaf_logs --disable-ai
aiwaf fast train --app app:app --log-dir aiwaf_logs --model-path model.json
```

Bare CLI commands select the single installed supported framework. If multiple
frameworks are installed, specify `aiwaf flask ...`, `aiwaf fastapi ...`, or
`aiwaf django <management-command> ...`. The `fast` alias remains supported.
Django uses its management command names, such as `aiwaf_list`.

The Flask CSV management CLI reuses existing policy files in the current project's
`aiwaf_data` (preferred) or `.aiwaf/data` before searching user and package data
directories. `AIWAF_DATA_DIR` or `aiwaf flask --data-dir /absolute/path status`
overrides discovery, including for a new directory. Set the same absolute
`AIWAF_DATA_DIR` for the Flask app and CLI when running them from different
working directories.

Flask, FastAPI, Django CSV mode, and the native Rust pipeline now use the same
policy files: `blacklist.csv`, `whitelist.csv`, `keywords.csv`,
`geo_blocked_countries.csv`, and `path_exemptions.csv`. For FastAPI, configure
`storage.backend = "csv"` and `storage.data_dir = "/absolute/path/aiwaf_data"`;
`AIWAF_DATA_DIR` also supplies the directory. Memory, JSON-file, database, and
Django ORM modes retain their existing storage behavior.

Startup imports a legacy `runtime_store.csv` in the chosen directory, or the
old default `aiwaf_data.csv` beside the project's `aiwaf_data` directory.
Existing custom `storage.file_path` settings remain accepted as migration
sources: `/path/custom.csv` maps to `/path/custom/`, while `runtime_store.csv`
maps to its parent. Prefer an explicit `data_dir` for the application and CLI.
Original legacy files remain intact, pre-import canonical files are backed up,
and a completion marker prevents replay after blocks or keywords are removed.
Existing canonical policy wins conflicts. Invalid legacy input stops startup
with an error; repair it and restart to retry an incomplete import. Stop old
workers before upgrading: subsequent writes to a legacy source are not reimported.

Keyword frequencies and non-policy runtime history are retained in
`runtime_metadata.csv`; policy enforcement reads the shared policy files.
The low-level `CSVStorage` key/value class remains available for explicit use,
but framework CSV initialization selects `PolicyCSVStorage`.

FastAPI also exposes `aiwaf.fast.trainer.train_from_logs(app, ...)`. Training
refreshes learned keywords and can write a Rust JSON model artifact. Both
regular and native middleware load that artifact at startup when
`config.set("ai_anomaly.model_path", "/path/to/model.json")` is configured
(`AIWAF_MODEL_PATH` is also accepted). The trainer uses the same configured path
unless `model_path` is supplied explicitly. Restart the application after
retraining to load the new model. Missing or invalid files produce a warning
and leave model scoring disabled.

## JavaScript package

The npm package uses CommonJS and exports the Express middleware as its default function plus adapters for the other supported Node frameworks.

### Express

```javascript
const express = require('express');
const aiwaf = require('aiwaf');

const app = express();
app.use(express.json());

app.use(aiwaf({
  AIWAF_MIDDLEWARES: ['auto'],
  staticKeywords: ['.php', '.env', '.git'],
  WINDOW_SEC: 10,
  MAX_REQ: 20,
  FLOOD_REQ: 40,
  AIWAF_HEADER_VALIDATION: true,
  AIWAF_METHOD_POLICY_ENABLED: true,
  AIWAF_ALLOWED_METHODS: ['GET', 'POST', 'HEAD', 'OPTIONS']
}));

app.get('/', (req, res) => res.json({ protected: true }));
app.listen(3000);
```

Attach AIWAF after the body parser when honeypot rules inspect parsed form or JSON fields.

### Node framework adapters

| Framework | Entry point |
| --- | --- |
| Express | `aiwaf(options)` |
| Fastify | `fastify.register(aiwaf.fastify, options)` |
| Hapi | `server.register({ plugin: aiwaf.hapi, options })` |
| Koa | `app.use(aiwaf.koa(options))` |
| NestJS/Express | `aiwaf.nest(options)` or the Express middleware |
| NestJS/Fastify | `app.register(aiwaf.fastify, options)` |
| Next.js API routes | `aiwaf.next(handler, options)` |
| AdonisJS | `aiwaf.adonis(options)` |
| Sails.js | `aiwaf.sails(options)` |

### JavaScript path policies

```javascript
app.use(aiwaf({
  AIWAF_PATH_RULES: [
    { PREFIX: '/health/', DISABLE: ['header_validation', 'rate_limit'] },
    { PREFIX: '/api/public/', RATE_LIMIT: { WINDOW: 60, MAX: 300 } }
  ]
}));
```

Applications can also use `aiwaf.exempt`, `aiwaf.exemptFrom`, `aiwaf.only`, and `aiwaf.requireProtection` to build route policies.

Generate the route manifest with `aiwaf init`, which loads your app, as Python's init does, and
records every route it registers, routers mounted with `app.use('/prefix', router)` included:

```bash
npx aiwaf init                                  # package.json main, else app.js, server.js or index.js
npx aiwaf init --app src/server.js              # a specific file
npx aiwaf init --app src/app.js --export createApp   # a named export, or a function returning the app
```

It reads Express (and Sails and Nest, which run on Express), Fastify, Hapi, and Koa apps that
use `@koa/router` or `koa-router`. Loading runs the app's module code, but any server it starts is
stopped from binding a port, so an app that only calls `app.listen()` is found too. It accepts
ES modules, and TypeScript that your Node version can run (otherwise pass the compiled file).

When it finds no routes it writes nothing and exits with status 1, so an empty manifest never
looks configured while protecting no route, or replaces one that works. The same holds for
`aiwaf manifest` and for Python's `init`.

From inside the app, after its routes are registered:

```javascript
aiwaf.generateExpressManifest(app, '.aiwaf/paths.json');
```

For an app that can't be loaded this way, list the routes in JSON. The handlers come from the
source (`--source`, default the current folder), so a route such as `POST /orders` that answers
with `res.json()` is classified as an API route as it is from the live app:

```bash
npx aiwaf manifest --framework express --routes routes.json --output .aiwaf/paths.json
```

### JavaScript configuration

| Area | Important options |
| --- | --- |
| Middleware | `AIWAF_MIDDLEWARES`, `AIWAF_DISABLE_MIDDLEWARES`, `AIWAF_PATH_RULES` |
| Rate limiting | `WINDOW_SEC`, `MAX_REQ`, `FLOOD_REQ`, `cache` |
| Headers | `AIWAF_HEADER_VALIDATION`, `AIWAF_REQUIRED_HEADERS`, `AIWAF_HEADER_QUALITY_MIN_SCORE` |
| Methods | `AIWAF_METHOD_POLICY_ENABLED`, `AIWAF_ALLOWED_METHODS` |
| Keywords | `staticKeywords`, `AIWAF_ENABLE_KEYWORD_LEARNING`, `AIWAF_DYNAMIC_TOP_N` |
| Honeypot | `HONEYPOT_FIELD`, timing settings |
| GeoIP | `AIWAF_GEO_BLOCK_ENABLED`, `AIWAF_GEO_BLOCK_COUNTRIES`, `AIWAF_GEO_ALLOW_COUNTRIES` |
| Models | `AIWAF_MODEL_STORAGE`, `AIWAF_MODEL_PATH`, `AIWAF_MODEL_CACHE_KEY` |
| Logging | `AIWAF_MIDDLEWARE_LOGGING`, `AIWAF_MIDDLEWARE_LOG_PATH`, DB/CSV options |
| WASM | `AIWAF_WASM_VALIDATION`, `AIWAF_WASM_VALIDATE_RECENT` |

AIWAF loads `aiwaf-wasm` when available and falls back to the JavaScript implementation when WASM initialization or conversion is not worthwhile. The SQL check and route checks are the exception: they run only in `aiwaf-wasm`, and a request fails with an install message if it can't load. Small, object-heavy operations can be faster in JavaScript because crossing the JS/WASM boundary and serializing objects has a fixed cost. WASM is most useful for sufficiently large feature batches and model scoring workloads.

The Python filesystem CSV transactions and retained training bindings remain
PyO3-specific. Node now also has an **explicit stateful WASM request pipeline**;
its transport and filesystem I/O remain in JavaScript.

### Rust-owned Node request pipeline

Node 1.1.2 now requires `aiwaf-wasm>=0.2.2,<0.3`. From the current source checkout, build the companion package with
`NativeRequestPipeline` support before using this adapter. Registry packages may
not yet contain this new API. Publish the WASM 0.2.2 companion first; refresh the
npm lockfile's registry integrity after publication, then run Node release gates
and publish the Node package. The existing `aiwaf(options)` integration remains
the legacy path; it does not switch automatically.

```javascript
const express = require('express');
const aiwaf = require('aiwaf');

async function main() {
  const app = express();
  // Configure trust proxy for your actual deployment; the native adapter uses req.ip.
  app.use(express.json({ limit: '64kb' }));
  app.use(express.urlencoded({ extended: false, limit: '64kb' }));
  const firewall = await aiwaf.native({
    dataDir: 'aiwaf_native_data',
    profile: { rate_key_mode: 'ip_path', min_form_time: 1.0 },
    // Optional synchronous adapter facts; use your application's authentication.
    authenticated: req => Boolean(req.user),
    routeExists: req => req.path === '/api/ok'
  });
  app.use(firewall);
  app.get('/api/ok', (_req, res) => res.json({ ok: true }));
  app.listen(3000);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
```

Configuration and native API checks happen once at startup. Rust runs geo,
IP/keyword/structured SQL inspection, rate/flood, anomaly block state, honeypot,
UUID and header checks in one `begin` call. A single `finish` call owns response
observations, UUID scores, anomaly features/history/inference, bounded keyword
learning, and escaped CSV row formatting. Node normalizes framework facts,
serializes the request envelope, emits responses, and appends the returned rows
using cached file descriptors. Logging is immediate, not queued or batched.

The dedicated lowercase `profile` rejects unknown keys. It does not import legacy
settings or Redis state on its own; see `from_settings` and `stores` below. Missing native support fails startup with no JavaScript policy fallback.
Install this adapter instead of the existing chain, not alongside it. Body parsers
can run before or after it. A parsed `req.body` is inspected as it is. An unparsed body
is handled as in Python: uploads and other bodies that aren't JSON, form or XML
(multipart, binary) pass through unread, so upload middleware such as multer still
gets the whole stream. An unparsed JSON, form or XML body is read up to `max_bytes`,
inspected, and put back on the stream for the parsers after it. A larger one is
refused with 413 (`payload_inspection_limit`), or logged in monitor mode. A compressed
or already-consumed inspectable body can't be read, and is refused with 413 as
before. `uuidResolver(uid, req)`, as in the regular middleware, is asked
whether the record named by a well-formed UUID under `uuid_prefix` exists. It can be
async; only those requests wait for it, `false` blocks with `uuid_not_found`, and a
failing resolver lets the request through. `countryLookup(req)`, `authenticated(req)` and `routeExists(req)`
are optional synchronous fact callbacks; country defaults to unknown and route
existence to `Boolean(req.route)`. Supply accurate facts when the firewall runs
before Express resolves routes. Country headers are not implicitly trusted.

**`aiwaf(options)` runs the native pipeline by default** when two things hold: the
installed aiwaf-wasm is 0.2.9 or newer (it has every native feature), and every option
carries over exactly, so the conversion report below lists no differences. Otherwise it
runs the regular middleware, as before. It logs one line when it picks the native
pipeline. To keep the regular middleware, set `AIWAF_NATIVE_PIPELINE: false` in the
options or `AIWAF_NATIVE_PIPELINE=false` in the environment. A model saved by
`aiwaf train` carries over; the sample model shipped in `resources/model.json` doesn't,
since it's in the old format and only a sample. Another model in the old format, or one
kept in the database or cache, is a difference, so those apps stay on the regular
middleware. The client address is resolved as the regular middleware resolves it,
through `AIWAF_TRUSTED_PROXIES` (the native adapter's `trustedProxies` option).

To start from the options the regular middleware already takes explicitly, add
`AIWAF_NATIVE_PIPELINE: 'from_settings'` to them, as in Python: `aiwaf({ MAX_REQ: 50, ...,
AIWAF_NATIVE_PIPELINE: 'from_settings' })`. Options are read through the same
normalization, defaults and environment variables, so the regular defaults (60-second
windows, no static keywords, header validation off) carry over rather than the native
ones. `{ fromSettings: true, dataDir, logger, profile: {...} }` also passes native options,
and its `profile` keys win. The startup report has the same form as Python's: how many
options carried over unchanged, then every one that didn't, with the reason, written to
`logger.warn` (or `logger.info` when nothing differs). `aiwaf.nativeOptionsFromSettings(options)`
returns the converted `config`, ready for `aiwaf.native()`, with its `report()`, `notes` and
`differences()`, without starting anything.

By default blocks and learned keywords stay in the adapter's own files (below). To share
them with the regular middleware and its CLI, pass `stores: 'regular'`. The adapter
then uses Node's regular blocklist, exemption, dynamic keyword and geo stores, from the
database when one is configured or otherwise their CSV files. Their blocks apply,
their IP and path exemptions keep logging only, a learned keyword applies once its count
passes `dynamicTopN`, and blocked countries add to `geo_block`. `from_settings` turns
this on, carrying `dynamicTopN` and `AIWAF_EXEMPTIONS_DB`. `stores` also takes any object
with six methods, each sync or async:
`isBlocked(ip)`, `isExempted(ip, path)`, `topKeywords(limit)`, `blockedCountries()`,
`block(ip, reason, expiresAt)` and `learn(keyword)`. The adapter reads them before the
engine decides, only for the stages the request runs, and keywords, countries and the
blocklist and exemption answers are cached for `storeCacheSeconds` (default 5), so a block
or exemption made elsewhere can take that long to apply; this worker's own blocks apply at
once. A failed read goes to `onError` and counts as not blocked, not exempt and nothing
learned. Blocks and keywords are written after the response, so a block applies from
the client's next request once the write lands. The decision itself stays in aiwaf-wasm.
Stores need `aiwaf-wasm` 0.2.7 or newer. With an older build, `stores` fails at startup, and
`from_settings` (which adds `storesFallback: 'csv'`) keeps the adapter's own files and
warns.

Rate limits are counted in each worker unless `rateCounter` is set. `aiwaf.redisRateCounter(client)`
takes a node-redis v4 or ioredis client and counts in Redis atomically, with the same
script and key format as Python's native Redis backend, so every worker shares one
limit. Any object with `count(key, now, windowSeconds)`, sync or async, also works. The
count is read before the engine decides, and the engine still applies the profile's and
path rules' limits to it; exempt requests aren't counted. A failing counter answers 503,
or lets the request through with `rateFailureMode: 'open'`. `from_settings` uses the
regular middleware's `cache` option when it's a Redis client, and carries
`AIWAF_RATE_CACHE_FAILURE_MODE`; with an older aiwaf-wasm it counts per worker and warns.

Path rules and the route manifest work as in the regular middleware, through three
adapter options. `pathRules` takes the same rules as `AIWAF_PATH_RULES`: `DISABLE`
turns stages off for a prefix, and `RATE_LIMIT` replaces the profile's rate limits
there, with invalid limits failing at startup. `manifestPath` (default
`AIWAF_PATH_MANIFEST`, then `.aiwaf/paths.json`) adds the manifest's route rules and
the route-specific checks saved by `init`. `injectionRiskMode` is `block` (default),
`log` or `off`, defaulting to `AIWAF_INJECTION_RISK_MODE`.

All eight stages are enabled by default. Native defaults include:

| Profile setting | Default / scope |
| --- | --- |
| `exempt_paths`, `exempt_ips` | `/health`, `/robots.txt`, `/sitemap.xml`; no IPs; literal case-insensitive path prefixes |
| `sql_mode`, `max_bytes` | `block`, `65536`; also `monitor`, which logs `{event: 'aiwaf.payload', rule, mode}` through the `logger` option (default `console`) without blocking, and `off`; bounded nested JSON/form values and double URL decoding |
| `static_keywords`, `dynamic_keywords`, `legitimate_keywords`, `learning` | Native vocabulary, empty dynamic set, legitimate exact route names, `true`; learned probe segments capped at 4096 |
| `geo_allow`, `geo_block`, `blocklist` | Empty; blocklist maps IP to Unix expiry seconds (`0` permanent) |
| `rate_window`, `rate_max`, `rate_flood`, `rate_key_mode` | `10` seconds, `20`, `40`, `ip_path`; `ip` also supported |
| `min_form_time`, `max_page_time`, `skip_authenticated`, `honeypot_field` | `1.0`, `240` seconds, `true`, empty field; local Rust timing state |
| `allowed_methods`, `uuid_prefix`, `uuid_window`, `uuid_threshold` | Empty method restriction, `/user/`, `60` seconds, `5` repeated UUID 404s; malformed UUID denied immediately |
| `required_headers`, `header_min_score` | `HTTP_USER_AGENT`, `HTTP_ACCEPT`; score `3`, plus header safety caps |
| `anomaly_window`, `max_history`, `max_clients`, `max_pending` | `60` seconds, `1000`, `100000`, `10000`; periodic cleanup and bounded state |
| `block_duration` | `86400` seconds for newly generated blocks |

Rate counters (without `rateCounter`), honeypot timestamps, UUID scores, anomaly history and loaded policy
snapshots are local to each engine/worker and reset on restart. Without `stores`, blocks and learned
keywords append to dedicated `native_blocks.csv`/`native_keywords.csv`, synced
immediately, and replay at startup (`persistBlocks: false` disables this disk
history). Policy files are **not live synchronized across workers**; do not share
the native state/log files across worker processes. These schemas and block rules
are separate from legacy reputation/CLI stores. Restart/reconstruct to apply
changed policies; no backend migration or automatic legacy CSV import occurs.

`native_requests.csv` is written immediately; `logFsync` defaults to false and
optionally syncs every log row. `logPath` overrides its location. CSV formatting
and policy decisions run in Rust; Node necessarily owns actual filesystem calls.
Finalization runs on response `finish` (or `close`, recorded as 499), uses a Node
monotonic elapsed clock, and executes once per engine-owned context. Stop requests,
wait for responses, then call `firewall.close()` to release writers/WASM state.
Logging/persistence errors invoke `onError(error)` or report to stderr; an already
completed response cannot be retracted.
After rotating native log files, reload workers to reopen the cached writers;
ordinary appends do not check file metadata or reopen descriptors per request.

Pass a fitted six-feature Rust/WASM forest as `model` (from
`aiwaf.wasm.createIsolationForest` with `__aiwafWasm === true`). The engine copies
its snapshot; `firewall.setModel(model)` replaces it and `setModel(null)` disables
inference. Node's feature order remains `[pathLen, kwHits, statusIdx, responseTime,
burst, total404]`, with response time in **milliseconds**, distinct from Python's
contract. Model scoring/behavior gates at response finalization can update block
state for later requests. No model means history/learning without model blocking.
`firewall.diagnostics()` reports native activation, actual inference count and
state sizes. There are no historical request-log database reads per request.

Fastify/Hapi/Koa/Nest/Next/Adonis/Sails can select the same engine with the sole
outer option `AIWAF_NATIVE_PIPELINE: { profile: {...}, dataDir: '...' }`. Do not mix
legacy options. Fastify initializes during registration and executes at
`preValidation`; Hapi initializes during registration, begins enforcement on
`onRequest` (including unknown routes), then makes an additional native call at
`onPreHandler` for parsed-body inspection and authenticated form timing.
Koa/Adonis wrappers and Next API handlers expose `.ready`;
Nest's middleware class exposes static `.ready`; await those promises before
starting the application. Fastify/Hapi close the engine on shutdown; use the
resolved native handle's `close()` for the other wrappers. Route metadata must be
installed before enforcement; required middleware overrides decorator exemptions,
while profile-level exemptions remain global. Next integration is Node API routes,
not Edge/Fetch middleware.

Run `npm run test:native` in `js/` against the built WASM package to validate the native integrations.

### JavaScript training and operations

```bash
cd js

AIWAF_ACCESS_LOG=/path/to/access.log npm run train
npm run aiwaf -- status
npm run aiwaf -- list blocklist
npm run aiwaf -- add blocklist 203.0.113.9 "manual block"
npm run aiwaf -- add ip-exemption 203.0.113.10 "trusted monitor"
npm run aiwaf -- geo summary
npm run aiwaf -- model info
npm run aiwaf -- export aiwaf-export.json
```

Use Redis or another shared cache for consistent rate limiting across multiple Node processes. Without shared state, the in-memory fallback is process-local.

## Rust accelerator for Python

`aiwaf-rust` exposes the shared Rust core through PyO3 and ABI3 wheels for Python 3.8 and newer.

```bash
pip install aiwaf-rust
```

The module provides:

- `validate_headers` and `validate_headers_with_config`
- `KeywordMatcher` and `RouteMatcher`
- request-record construction and six-feature extraction
- incremental feature extraction with explicit state
- recent-behavior analysis
- `IsolationForest` training, scoring, prediction, retraining, and JSON state
- locked CSV storage, indexed snapshot lookups, atomic policy mutations, and
  synchronous cached log writers
- shared request policies, combined route plans, and exemption matching
- batched log/timestamp preparation, `TrainingCSVReader`,
  `NativeTrainingBatch`, and `NativeTrainingBuilder`

```python
import aiwaf_rust

reason = aiwaf_rust.validate_headers({
    "HTTP_USER_AGENT": "Mozilla/5.0",
    "HTTP_ACCEPT": "text/html",
})

forest = aiwaf_rust.IsolationForest(
    n_estimators=100,
    max_samples="auto",
    contamination="auto",
    max_features=1.0,
    bootstrap=False,
    random_state=42,
    warm_start=False,
)
forest.fit([[0.1, 1.0], [0.2, 1.1], [9.0, 9.0]])
score = forest.anomaly_score([9.0, 9.0])
state = forest.to_json()
restored = aiwaf_rust.IsolationForest.from_json(state)
```

For a local native build, stage the two canonical repository assets and run maturin:

```bash
cp README.md rust/README.md
cp LICENSE rust/LICENSE
cp NOTICE rust/NOTICE
cp -R licenses rust/
python -m pip install maturin
cd rust
maturin develop
```

The staged files are ignored by Git. Release automation performs the same staging before building wheels and the source distribution.

## WebAssembly package

`aiwaf-wasm` exposes the shared Rust core to browsers and Node-compatible bundlers.

```javascript
import init, {
  IsolationForest,
  KeywordMatcher,
  RouteMatcher,
  analyze_recent_behavior,
  extract_features,
  validate_headers,
} from 'aiwaf-wasm';

await init();

const reason = validate_headers({
  'user-agent': 'Mozilla/5.0',
  accept: 'text/html'
});

const forest = new IsolationForest({
  n_estimators: 100,
  max_samples: 'auto',
  contamination: 'auto',
  random_state: 42
});
forest.fit([[0.1, 1.0], [0.2, 1.1], [9.0, 9.0]]);
const score = forest.anomaly_score([9.0, 9.0]);
```

The WASM API includes header, URL, and content validation; keyword and route matchers; record conversion; batched feature extraction; recent-behavior analysis; and Isolation Forest lifecycle methods. Browser use requires a bundler that understands WebAssembly assets.

Build and test it locally:

```bash
cargo test --manifest-path rust/Cargo.toml --locked -p aiwaf_core
cd rust/crates/aiwaf_wasm
wasm-pack test --node
wasm-pack build --release --target bundler
cd ../..
python scripts/patch_wasm_pkg.py
cp ../LICENSE crates/aiwaf_wasm/pkg/LICENSE
cp ../NOTICE crates/aiwaf_wasm/pkg/NOTICE
cp -R ../licenses crates/aiwaf_wasm/pkg/
```

## Java package

AIWAF Java 1.4.0 targets Java 17, Spring Framework 7.0, and Jakarta Servlet 6.1. It is a native Java implementation and does not load the Rust library.

### Core engine

Use the two-stage API when the integration can observe the handler response. Deterministic controls run before the application; anomaly scoring then receives the actual status and elapsed time.

```java
import com.aiwaf.core.AiwafConfig;
import com.aiwaf.core.AiwafDecision;
import com.aiwaf.core.AiwafEngine;
import com.aiwaf.core.AiwafRequest;

import java.util.Map;
import java.util.Set;

AiwafConfig config = new AiwafConfig();
config.rateLimitEnabled = true;
config.rateLimitMax = 20;
config.rateLimitWindowSeconds = 10;

AiwafEngine engine = new AiwafEngine(config);
AiwafRequest request = new AiwafRequest(
    "GET",
    "/api/profile",
    "203.0.113.10",
    "US",
    Map.of("User-Agent", "Mozilla/5.0", "Accept", "application/json"),
    Map.of(),
    System.currentTimeMillis(),
    Set.of()
);

long started = System.currentTimeMillis();
AiwafDecision decision = engine.evaluateBeforeResponse(request);
if (decision.allowed()) {
    decision = engine.evaluateAfterResponse(
        request,
        200,
        System.currentTimeMillis() - started
    );
}
```

`evaluate(request)` remains available for request-only integrations and uses provisional response values for AI features.

### Spring MVC

Spring Boot applications are protected automatically. Adding `aiwaf-java` creates `AiwafConfig`, `AiwafEngine`, and the ordered `AiwafFilter`; discovers live MVC routes, route annotations, and JPA UUID fields; and applies request controls before the controller plus post-response scoring with the real status and timing. Set `aiwaf.enabled=false` to disable auto-configuration.

```properties
aiwaf.rate-limit.max=60
aiwaf.rate-limit.window-seconds=60
aiwaf.geo.enabled=true
aiwaf.geo.blocked-countries=CN,RU
aiwaf.path-manifest.path=.aiwaf/paths.json
# Use one namespace and shared enforcement state across application instances.
aiwaf.storage.backend=redis
aiwaf.storage.redis-url=redis://localhost:6379/0
aiwaf.storage.key-prefix=my-service
# Redis is fail-closed by default. Every wait is bounded.
aiwaf.storage.redis-failure-mode=fail_closed
aiwaf.storage.redis-connection-timeout-millis=1000
aiwaf.storage.redis-socket-timeout-millis=2000
aiwaf.storage.redis-pool-max-total=32
aiwaf.storage.redis-pool-max-idle=16
aiwaf.storage.redis-pool-min-idle=0
aiwaf.storage.redis-pool-max-wait-millis=250
```

Set `aiwaf.config-file=/etc/aiwaf/config.json` (or `AIWAF_CONFIG_FILE`) to load the same nested JSON configuration used by the Python runtime. Environment variables and Spring properties override file values. Java-only keys include `security.trusted_proxies`, a list such as `["default", "cloudflare"]` or a comma-separated string; an empty list trusts no proxy, and loading fails on an entry that isn't an address, a CIDR range, `default`, or `cloudflare`. `AiwafConfigFileCore` also exposes load, save, deep-merge, and validation APIs for non-Spring applications.

Define an `AiwafConfig`, `AiwafEngine`, or `AiwafFilter` bean only when the application needs a custom replacement. For plain Spring MVC without Boot, register the filter directly; passing the application context enables the same live route discovery:

```java
@Bean
AiwafFilter aiwafFilter(AiwafEngine engine, ApplicationContext context) {
    return new AiwafFilter(engine, context);
}
```

The live mapping integration honors `@AiwafExempt`, `@AiwafExemptFrom`, `@AiwafOnly`, and `@AiwafRequireProtection` without a controller list. Do not register `AiwafInterceptor` and `AiwafFilter` as independent enforcement layers for the same request.

Committed, streaming, and asynchronous responses are recorded but are not replaced after bytes have been sent.

UUID protection uses Python-compatible weighted signals: malformed input, valid UUID requests that resolve to `404`, and successful requests that decay the score. UUID-valued query parameters and `{uuid}` route variables are supported. `SpringUuidModelLookup` optionally discovers UUID primary keys and unique UUID properties through the JPA metamodel; omit it when the application does not use JPA.

Generate a route manifest from the live Spring application rather than supplying controller methods manually:

```bash
java -cp "your-app.jar:aiwaf-java-1.4.0.jar" \
  com.aiwaf.cli.AiwafConsole init \
  --app com.example.Application \
  --output .aiwaf/paths.json
```

The command starts the application, reads every registered `RequestMappingHandlerMapping`, writes the shared `schema_version/framework/context_hash/routes` document, and then closes the temporary application context. At runtime, `AiwafEngine` loads this file automatically and appends its disabled middleware and rate-limit overrides after explicit `pathRules`, so explicit equal-prefix rules retain precedence. Set `pathManifestEnabled=false` or `aiwaf.path-manifest.enabled=false` to opt out.

### Jakarta Servlet

```java
AiwafConfig config = new AiwafConfig();
AiwafEngine engine = new AiwafEngine(config);

servletContext
    .addFilter("aiwaf", new AiwafServletFilter(engine))
    .addMappingForUrlPatterns(null, false, "/*");
```

The generic servlet filter uses request-time evaluation. Spring applications should prefer `AiwafFilter` for handler metadata and post-response scoring.

### Java configuration and models

| Area | Main `AiwafConfig` fields |
| --- | --- |
| Rate limiting | `rateLimitEnabled`, `rateLimitScope`, `rateLimitWindowSeconds`, `rateLimitMax`, `rateLimitFloodThreshold` |
| Headers | `headerValidationEnabled`, `requiredHeadersByMethod`, `minHeaderQualityScore`, size/count limits |
| Request | body, parameter, method, and content-type limits |
| Proxies | `trustedProxyCidrs`, `maxForwardedForEntries` |
| Exemptions | `exemptIps`, `exemptPaths`, `autoExemptPathPrefixes` |
| GeoIP | `geoBlockEnabled`, `geoAllowedCountries`, `geoBlockedCountries`, `geoIpDatabasePath`, cache limits |
| Route manifest | `pathManifestEnabled`, `pathManifestPath`, `pathRules` |
| UUID checks | `uuidTamperEnabled`, `uuidScoreWindowSeconds`, `uuidScoreBlockThreshold`, signal weights, `uuidParameterNames` |
| AI | `aiEnabled`, `aiModelPath`, `aiAnomalyScoreThreshold` |
| Storage | `storageBackend`, `storageFilePath`, `storageRedisUrl`, `storageKeyPrefix`, `storageRedisFailureMode`, Redis timeout and pool limits |
| Telemetry | `observabilityEnabled` |

GeoIP uses the bundled MMDB through the native Java reader; it does not require the external `mmdblookup` executable. A valid `X-Country` value is accepted only from a peer in `trustedProxyCidrs`; otherwise AIWAF resolves the client address itself. Default blocklist calls use Python-compatible reputation scoring and temporary escalation (15 minutes, 1 hour, then 24 hours). Use `BlocklistManager.blockPermanent(ip, reason)` for an explicit permanent administrative block.

`storageBackend="redis"` makes rate-limit buckets, honeypot timing, weighted UUID scores, recent-request anomaly state, blocklist data, exemptions, geo policy, and learned keywords visible to every application instance using the same key prefix. Redis enforcement updates use atomic server-side scripts. Memory, file, CSV, and SQLite engines keep request state isolated per `AiwafEngine`; constructing another engine no longer replaces an existing engine's stores. The static `RuntimeStorage` and `BlocklistManager` methods remain compatibility facades for command-line and administrative code.

The Java 1.3 Redis backend supports **standalone Redis only**. Sentinel and Cluster configurations are rejected during validation instead of silently operating with incomplete semantics. `fail_closed` is the default: an unavailable Redis instance prevents startup or raises a request-path storage failure, which results in the application denying availability rather than bypassing protection. `fail_open` permits startup and returns non-blocking enforcement fallbacks while Redis is unavailable; failed mutations return `false` at the `StorageBackend` boundary. Connection, socket, pool size, and pool wait limits are bounded and configurable. `RedisStorage.diagnostics()` reports operation/failure counts, latency, and live pool utilization.

Redis keys include a state-schema version, and each base key prefix has an atomic compatibility marker. Releases using the same schema can share state during a rolling upgrade. A runtime with an incompatible schema refuses the namespace—even in `fail_open` mode. Use a unique key prefix per application and environment; deploy an incompatible schema with a new prefix, migrate deliberately, then retire the old namespace. Serialized persistent values remain JSON within a schema version.

New Java models use the Python-aligned six-feature schema and are written as safe JSON containing the Python `aiwaf_rust.IsolationForest` state. Python can restore Java-generated forest state, and Java can load Python Rust-model artifacts. Existing signed Java binary artifacts, including nine-feature models, remain readable through the restricted legacy loader. Scikit-learn object artifacts are intentionally unsupported.

Java operational commands mirror the common Python workflow:

```bash
aiwaf-cli status
aiwaf-cli diagnose --model model.json
aiwaf-cli blocked
aiwaf-cli unblock 203.0.113.10
aiwaf-cli clear
aiwaf-cli train --events events.json --out model.json --backend java
aiwaf-cli replay --cases replay.json
aiwaf-cli model status --path model.json
aiwaf-cli logs --dir aiwaf_logs
```

Build Java from the repository root:

```bash
cd java
mvn test
mvn -Dgpg.skip=true package
```

The build produces the main, source, and Javadoc JARs under `java/target/`. The main JAR embeds the root license as `META-INF/LICENSE`. FastR training is optional and falls back to the Java trainer when R or `jsonlite` is unavailable.

### Python training replay protection

Python 1.1.6 with aiwaf-rust 0.2.5 records the evidence used by offline training
to block each IP. Rust computes SHA-256 receipts, locks concurrent trainers,
and persists `training_evidence.json` in the policy data directory. Flask,
FastAPI, and Django continue fitting models and learning keywords from history,
but block decisions use requests that have not already been handled for that IP.
Replaying old logs cannot extend an active block or recreate one after expiry
or manual removal. Fresh traffic must independently satisfy the blocking rules.

CLI listing filters expired blocks for display without rewriting its snapshot
back to the blocklist. This prevents concurrent training updates from being
erased by an older cleanup snapshot. Default first blocks last 900 seconds.

Keep the receipt file across restarts, log rotation, and blocklist clearing.
Receipts do not expire and grow with the distinct requests used in block
decisions. `AIWAF_TRAINING_STATE_DIR` can select a durable shared directory;
trainers sharing a blocklist must also share this directory. Missing native
capabilities or corrupt receipt state stop training visibly. A crash between
writing a block and committing its receipt can still require recovery; the two
writes are not a single transaction.

Receipts start with the first training run on this version. Requests processed
by older versions have no receipts, so previous enforcement cannot be inferred
retroactively. This change does not alter block durations or diagnose a block
disappearing before its expiry.

### JavaScript training replay protection

Node.js 1.1.4 with aiwaf-wasm 0.2.4 uses the same Rust receipt algorithm as
Python. The shared trainer applies this protection across JavaScript framework
adapters. The WASM `NativeTrainingEvidence` API hashes and filters evidence;
Node persists `node_training_evidence.json` in the blocklist CSV directory, or
the directory selected by `AIWAF_TRAINING_STATE_DIR`. Preserve this file across
expiry, unblocking, blocklist clearing, and log rotation. Sharing a blocklist
also requires sharing the evidence directory. Model fitting and keyword learning
still use full history; 404 and anomaly enforcement use fresh evidence only.
Log records without a valid stable timestamp are excluded from training.

The Node host serializes trainers with an exclusive lock directory. A crashed
trainer may leave `node_training_evidence.lock`; subsequent training fails
visibly rather than bypassing that lock. Remove that lock directory only after
confirming its trainer has stopped. Receipt snapshots are written and synced
before atomic replacement. As in Python, block writes and receipt commits are
separate operations, so a crash between them can still require recovery.

The CLI exits with failure for missing WASM receipt support, corrupt state, or
storage errors. Programmatic callers can await `require('aiwaf/train')()`.
Publish the matching WASM package before the Node package; `test:wasm-real`
checks receipt restoration against the installed dependency, not just a fixture.

Java's `train-model` operation fits and saves a model without adding blocklist
entries. Servlet and Spring use that shared Java core. Retraining alone therefore
cannot extend or recreate their runtime blocks; a regression test verifies this.

Blocklist expiry is also protected against concurrent updates. Node request
checks and listings do not delete expired snapshots. Explicit cleanup checks
expiry in the database DELETE, or inside the CSV mutation lock after reading
the current rows. CSV blocklist writers use the same lock and atomically replace
the file. A crashed writer can leave `<blocklist CSV path>.lock`; stop its owner
before removing that lock directory. Lock acquisition times out after 60 seconds
and reports an error rather than proceeding without protection.

Java CSV and file storage reload current data under a shared JVM and OS sidecar
lock for each operation. Reads filter expiry without rewriting disk; writes
preserve other instances' updates. OS locks release when a process exits; the
sidecar lock file stays in place and must not be removed while workers run.
File storage now incurs a reload for each operation. Restart all old workers
when upgrading so every writer participates in the locking protocol.

## Shared training schema

New training paths use these six features in this order:

1. `path_len`
2. `kw_hits`
3. `resp_time`
4. `status_idx`
5. `burst_count`
6. `total_404`

`resp_time` is expressed in seconds in feature vectors. Artifact summary fields explicitly named with `_ms` remain milliseconds.

The training lifecycle is:

1. Parse configured access logs or middleware-generated request logs.
2. Normalize request paths, status, timing, and per-IP state.
3. Extract the six features and suspicious route segments.
4. Train or update the Isolation Forest after configured volume thresholds are met.
5. Refresh learned keywords while excluding known legitimate and exempt routes.
6. Persist portable JSON forest state and runtime metadata; legacy language-specific artifacts remain read-only compatibility inputs.

Do not train directly on unreviewed sensitive request bodies or authentication headers. Treat models, logs, exported state, and MMDB files as deployment artifacts with controlled access.

## Deployment guidance

- Configure the application to trust forwarding headers only from known proxy CIDRs.
- Configure the outermost proxy to overwrite client-supplied forwarding headers.
- Use Redis, a database, or another shared backend when multiple workers must share rate-limit, blocklist, or exemption state.
- Keep health checks, static assets, and trusted webhook routes explicitly scoped instead of disabling protection globally.
- Tune anomaly detection with representative normal traffic before relying on it in production.
- Rotate logs and exclude secrets, authorization headers, cookies, and personal data.
- Sign model/configuration artifacts where the runtime supports HMAC verification.
- Run AIWAF alongside normal secure-development controls and an edge/reverse-proxy layer.

## Development and tests

### Whole monorepo

Create a Python environment, install all adapters, then install Node dependencies:

```bash
python -m pip install -e ".[django,flask,fastapi,rust]"
python -m pip install pytest pytest-cov coverage httpx
npm ci --prefix js
python aiwaf_test.py --coverage
```

Focused suites:

```bash
python aiwaf_test.py --python-only
python aiwaf_test.py --rust-only
python aiwaf_test.py --js-only
python aiwaf_test.py --wasm-only
python aiwaf_test.py --java-only
```

Full runs include the Java suite, which needs a JDK 17 or newer and Maven. Use `--no-java`
to skip it on a machine without them, or run it directly with `mvn -f java/pom.xml test`.

### Direct package commands

```bash
# Python
pytest

# Node.js
npm ci --prefix js
npm test --prefix js

# Rust core
cargo test --manifest-path rust/Cargo.toml --locked -p aiwaf_core

# Rust Python API, after installing the local wheel
pytest -q rust/tests/test_python_api.py

# Java
mvn -f java/pom.xml test
```

## Release automation

Release versions and tags must match the relevant manifest.

| Package | Manifest | Tag | Workflow |
| --- | --- | --- | --- |
| Python `aiwaf` | `pyproject.toml` | `python-v1.3.3` | `python-publish.yml` |
| npm `aiwaf` | `js/package.json` | `js-v1.3.3` | `npm-publish.yml` |
| PyPI `aiwaf-rust` | `rust/pyproject.toml` + `rust/Cargo.toml` | `rust-v0.2.17` | `rust-publish.yml` |
| npm `aiwaf-wasm` | `rust/crates/aiwaf_wasm/Cargo.toml` | `wasm-v0.2.13` | `wasm-publish.yml` |
| Maven `aiwaf-java` | `java/pom.xml` | `java-v1.4.0` | `java-publish.yml` |

Python and Rust publish to PyPI with trusted publishing. The npm workflows use npm trusted publishing and stage packages for approval. Configure the trusted publisher with organization `aiwaf-project`, repository `aiwaf`, the exact workflow filename, and the environment used by that workflow.

Java publishing uses the protected `maven-publish` environment plus `GPG_PRIVATE_KEY`, `GPG_PASSPHRASE`, `CENTRAL_TOKEN_USERNAME`, and `CENTRAL_TOKEN_PASSWORD`. The workflow verifies the tag, runs tests, signs the POM and all three JARs, uploads the deployment bundle, and waits for Maven Central publication.

Registry releases are immutable. Increment the relevant manifest version before publishing another release.

This release is Python `aiwaf 1.3.3` and npm `aiwaf 1.3.3`: npm `aiwaf init`, which loads the
app, no empty manifests from either CLI, handlers from source for `--routes` lists, Fastify
routes from a live app, and nested Pydantic body fields. It needs no new engine: Python 1.3.2,
npm 1.3.2, `aiwaf-rust 0.2.17` and `aiwaf-wasm 0.2.13` are published, Python 1.3.3 requires
`aiwaf-rust>=0.2.17`, and npm 1.3.3 requires `aiwaf-wasm ^0.2.13` (1.3.2 allowed 0.2.11, which
lacks the per-method lists and nested keys). Java 1.4.0 isn't on Maven Central yet.

## Security and limitations

- Deterministic controls can produce false positives when legitimate routes resemble common probes; use narrow route exemptions and regression tests.
- Language-native Isolation Forest implementations can produce different scores for identical feature rows.
- WASM object conversion has overhead and is not automatically faster than JavaScript for small calls.
- In-memory rate limits and runtime stores are local to one process.
- GeoIP data can be missing, stale, or imprecise and should not be the sole authorization control.
- Post-response blocking cannot replace data already committed or streamed to a client.
- No WAF can compensate for broken authorization or unsafe application logic.

Report vulnerabilities privately to the maintainers rather than opening a public exploit issue.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contributor setup and expectations, [`CHANGELOG.md`](CHANGELOG.md) for release history, and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for community standards.

## License

Original AIWAF project code in this checkout uses the unmodified [Elastic License 2.0](LICENSE), with separately licensed material identified in [NOTICE](NOTICE). AIWAF is the business offering these terms. ELv2 permits commercial use, modification, redistribution, and consulting, subject to its conditions. It restricts providing substantial AIWAF functionality as a hosted or managed service, bypassing license-key functionality, and removing required notices. ELv2-covered code is source available.

This migration applies to this checkout and future releases expressly distributed under ELv2. Previously distributed MIT copies retain their permissions; version numbers do not retroactively change their license. Do not republish an existing version with new licensing terms. Independently versioned packages record their license in their release artifacts.

Initial ELv2 release boundaries are Python/Node.js `1.1.2`, Rust/Python and WASM `0.2.2`, and Java `1.3.3`. Earlier MIT distributions retain their MIT permissions.

Retained MIT contributions keep their rights and notices in [licenses/MIT.txt](licenses/MIT.txt). The bundled IPinfo Lite database is provided by [IPinfo](https://ipinfo.io/) under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); ELv2 restrictions do not apply to that data. Package metadata uses an `AND` expression where separately licensed components are included; this does not offer a choice to license the entire package under MIT. Dependencies retain their respective terms.

Redistributions must include the applicable LICENSE, NOTICE, and licenses/ files. See [NOTICE](NOTICE) for historical rights, contributor notices, and attribution.
