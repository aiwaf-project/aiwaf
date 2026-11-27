# Changelog

## Python 1.3.4 / aiwaf-rust 0.2.18 / Node.js 1.3.4 / WASM 0.2.14 / Java 1.4.0 — prepared release

- **Release note:** publish aiwaf-rust 0.2.18 (`rust-v0.2.18`) before Python 1.3.4, which
  requires `aiwaf-rust>=0.2.18`, and aiwaf-wasm 0.2.14 (`wasm-v0.2.14`) before npm 1.3.4. npm
  1.3.4 declares `aiwaf-wasm ^0.2.13`, since the lockfile can't name an unpublished version: once
  0.2.14 is published, raise it to `^0.2.14` and refresh the lockfile before tagging `js-v1.3.4`.
  With 0.2.13 the per-method types below are ignored, and the agreed type is checked instead.
- **Fastify `$ref`s resolve in their own scope:** sibling plugins that reuse a schema `$id`
  for different schemas gave every route the last one (both routes the boolean, both complete).
  Each recorded route now keeps the plugin scope it was registered in (init's capture and the
  AIWAF plugin's `onRoute` hook both record `this`), and its `$ref`s resolve against that scope's
  `getSchemas()`, its own and its parents'. A route whose scope isn't known only takes an `$id`
  the source defines once; otherwise the reference stays unresolved and the route incomplete.
- **FastAPI optional body models:** `body: Optional[Filters] = None` saved the argument name
  `body` as a request key and lost the model's own fields; the model is read through
  `Optional` and `Annotated` now.
- **Route-local dependencies and hooks are followed:** a FastAPI route dependency that takes the
  request (`dependencies=[Depends(current_tenant)]`) is read from its source like a view, and
  what it reads (`tenant`) goes to the routes using it; one whose reads aren't all known keeps
  the `dependency` reason. A Fastify route's own `preHandler`, `onRequest`, `preParsing`,
  `preValidation` or `preSerialization` hooks, inline, in an array or by name, are read for that
  route alone; one that can't be read keeps the `middleware` reason.
- **Handlers a factory makes are linked:** a handler returned by a factory
  (`def make_export(fmt): def export(...)`, `function makeExport(format) { return async function
  exportReport(...) }`) is linked to its routes when the source defines its name once, even when
  several routes share it, so they're audited and complete. A handler nothing names, as a
  lambda, keeps `unread_view`.
- **Query and body types apart:** a name sent in both locations with different types (a
  query `limit` that's an integer, a JSON `limit` that's text) keeps each location's type in the
  new `inputs_types_by_location`, and the engines (aiwaf-rust, aiwaf-wasm, Java) check each value
  against its own location's type, then its method's, then the route's. FastAPI's query and
  body parameters, Fastify and Hapi `querystring`/`body` schemas, and Spring's `@RequestBody`
  fields (and `@RequestParam` next to a JSON body) give the locations. A Fastify schema naming
  a field in both parts kept the last part's type; it now merges.
- **Deeper nested keys:** closed JSON Schemas and Pydantic models give nested keys by path,
  `"customer.address": ["city", "zip"]` and `"lines[]": ["qty", "sku"]` for each item of a list,
  up to four levels, and the engines check them (six levels, 256 items). Open objects are left
  alone. An input used whole through a built-in (`lines.length`, `tags.map(...)`) is no longer
  read as a nested key.
- **Fastify routes before the plugin, by capture:** `init` adds an `onRoute` hook to each
  Fastify instance as the app creates it, before its own routes, so every route's options,
  schemas imported from other modules included, are read as registered. A schema the source
  passes but can't spell out, with no capture (a `--routes` list), leaves the route incomplete
  (`external_class`) rather than silently untyped.
- **Exact request keys:** the static reader takes a Pydantic field's alias
  (`Field(alias="eventId")`, `validation_alias`, string `AliasChoices`), and its attribute name
  only under `populate_by_name`, so `event_id` no longer joins `eventId` in a complete list. An
  alias generator, or an alias it can't read, leaves the model to the live app's list.
- **Reads before the handler:** FastAPI routes added with `include_router(router,
  dependencies=[...])` include what those dependencies read, as `tenant`; a Fastify request
  hook (`addHook('preHandler', ...)` and `onRequest`, `preParsing`, `preValidation`) adds its
  reads to every route, since the app doesn't say which routes a hook's scope holds.
- **Atomic manifest writes:** Python, npm and Java write the manifest to a temporary file beside
  it and replace the old one only after the write succeeds, keeping its permissions; a failed
  write (a full disk, a value JSON can't hold) leaves the old manifest intact.
- **Engine check:** `init` warns when the installed aiwaf-rust or aiwaf-wasm can't check an
  entry it wrote (per-method or per-location types, deeper nested keys), by the engine's new
  `method_types`, `location_types` and `nested_paths` features (`NativeCSVPipeline.supports`,
  and aiwaf-wasm's new `supports_feature`).
- **Per-method input types:** where methods on one path type a name differently (GET's text
  `mode`, POST's integer `mode`), the manifest keeps the type they agree on in `inputs` (text)
  and each method's own in the new `inputs_types_by_method` (`{"POST": {"mode": "int"}}`).
  aiwaf-rust, aiwaf-wasm and Java check a method's own type for its requests, so POST keeps its
  integer check while GET's text passes; HEAD reads GET's. Engines before aiwaf-rust 0.2.18 and
  aiwaf-wasm 0.2.14 ignore the entry. All three `init`s write it from the framework's types for
  each handler (FastAPI and Pydantic, Fastify and Hapi schemas, Nest pipes and DTOs, Spring).

- **Routes `init` missed:**
  - FastAPI routes added with `include_router()`, which FastAPI keeps as one entry and resolves
    while serving, and routes of apps attached with `mount()`, nested mounts included, all under
    their prefixes. Only top-level `@app` routes were saved.
  - An Express router mounted at several prefixes (`app.use(['/account', '/me'], router)`), in
    Express 4 (one pattern with `|`) and Express 5 (one matcher per path). The router was skipped.
  - Fastify routes registered before the AIWAF plugin, which its `onRoute` hook never sees: they
    now come from Fastify's own route table, with the plugin's records kept for the rest.
- **A failed audit no longer replaces a manifest:** a missing audit source (`--source` in Python
  and npm, `--audit-source` in Java) or a scan that fails now stops `init` with exit status 1,
  writing nothing. Python and npm warned and carried on; Java wrote the manifest first, so it
  replaced a working one with one missing its typed and per-method inputs, and exited 0. Java
  now audits before writing, as Python and npm do.
- **An audit that reads no files is no audit:** an audit source holding no Python, JavaScript or
  Java files left lists marked complete, as if every route had been checked for risky calls. It
  now counts as `--no-audit`, with a warning: types are saved and checked, no list is complete,
  and no `injection_audit` is saved.
- **npm audit crash on inputs named like object properties:** a handler (or a helper it
  calls) reading `body.constructor` or `body.toString` crashed the audit with "is not
  iterable". The audit used to log that as skipped and save no findings or inputs; now that a
  failed scan stops `init`, it surfaced, and those names are ordinary keys.
- **Complete only when the audit read the route:** a route whose handler code the audit never
  read was still marked complete when its framework's list was complete on its own (FastAPI's
  parameters, a declaration), as when `--source` pointed at a folder of unrelated files. It now
  keeps its inputs and types but isn't complete (`unread_view`), in Python, npm and Java. A
  framework list that isn't complete on its own, as a Fastify schema, is now kept for such a
  route too, where it was dropped. A handler counts as read when it's defined in the audited
  source, even when it reads nothing itself (a generic Django class-based view, a Nest
  controller method taking only a DTO): the audit report now carries the functions, classes
  and methods each file defines (`definitions`), which `aiwaf audit`'s own output leaves out.
- **Fastify routes before the plugin keep their schemas:** their schemas now come from the
  source, statically, from `app.post('/x', { schema }, handler)` and `app.route({ url, method,
  schema, handler })`, including a schema held in a constant in the same file. A route
  registered before the plugin lost its declared integer, which became `text`.
- **Fastify routes sharing a prefix:** reading a live Fastify app's route table dropped routes
  whose paths share a prefix (`/early` and `/early-order`), because Fastify prints those as a
  tree even with `commonPrefix: false`. Paths are now rebuilt from the tree.
- **Fastify `route({ url, handler })` handlers are audited:** the npm audit only linked route
  objects with Hapi's `path`, so these handlers' reads and risks weren't tied to their routes.
- **A type stays with the methods that declare it:** a GET whose query `ref` is any text and a
  POST whose body types `ref` as an integer on the same path saved `ref` as `int`, so a valid
  GET was a type mismatch. The engine keeps one type per name for every method, so a name read
  with different types by different methods is now `text` (the methods' name lists still
  differ). Handlers sharing a path combined their framework lists by letting the last one's type
  win (Python and npm), and a framework type overrode the code's reading on methods it wasn't
  declared for (all three).
- **Fastify routes with several methods on one path:** each method's handler and schema now
  come from the source on their own. The route table lists `/orders (GET, POST)` on one line,
  and both methods took GET's, so POST's inputs were missing from a list marked complete.
- **Pydantic names:** a field is listed under every key Pydantic accepts for it: its
  `validation_alias` (a string, each of `AliasChoices`, an `AliasPath`'s first key, as text),
  else its alias, and its own name too under `populate_by_name` or `validate_by_name`. A
  complete list missing an accepted name made strict inputs reject a valid request.
- **A text body read whole is no field (npm):** `request.body.length`, or a string method
  called on the body (`body.trim()`, `body.split(...)`), was saved as a JSON field named
  `length` or `trim` in a complete list. It's now a raw read (`raw_body`).
- **Fastify `$ref` schemas:** a schema referring to a shared one (`fastify.addSchema`) is
  resolved from the live app's `getSchemas()` and the source's literal `addSchema` calls, as
  `id#`, `id#/pointer` and local `#/definitions/x` references; it was an unknown schema
  (`external_class`).
- **Nested keys from Fastify, Hapi and other JSON Schemas:** an object property that lists its
  own properties and allows no others saves them as `inputs_nested`, as FastAPI's Pydantic
  models and the code reading already did. Keys are checked one level down, not inside lists.
- **Named Fastify handlers keep their method:** the audit links a named handler
  (`fastify.delete('/plans', removePlan)`) to its route by name, and took its method from the
  route's record of which method each named handler serves, which only Express routes had. A
  route list's named handler now records its methods too (Fastify, Koa, Hapi, Next and
  `--routes` lists), so a DELETE-only boolean `force`, or POST-only `dryRun` and `weeks`, keep
  their types and each method its own names.
- **Route-inventory tests** compare `init`'s routes with what each framework reports or serves:
  FastAPI's OpenAPI paths for each mounted app, a real request to every Express route, and
  Fastify's `hasRoute()`.

## Python 1.3.3 / Node.js 1.3.3 (released) / Java 1.4.0 (prepared)

- **Release note:** no engine changes. Python 1.3.2, npm 1.3.2, aiwaf-rust 0.2.17 and
  aiwaf-wasm 0.2.13 are published, so `python-v1.3.3` and `js-v1.3.3` can be tagged now.
  Python still requires `aiwaf-rust>=0.2.17`. npm now requires `aiwaf-wasm ^0.2.13` (was
  ^0.2.11, which let an install keep 0.2.11 and lose 1.3.2's per-method lists and nested keys),
  and the lockfile pins the published 0.2.13. Java 1.4.0 isn't on Maven Central yet; it also
  carries the Java `init` fixes below.
- **Declared types hold:** a type the framework validates now wins over the handler's own
  untyped read of the same name. A Fastify route whose JSON Schema types `quantity` as an
  integer saved it as `text` when the handler read `const { quantity } = request.body`, so its
  integer rule was never enforced. The same holds for Hapi schemas, Nest pipes and DTOs, and in
  Python for FastAPI and Pydantic types.
- **`--no-audit` keeps typed inputs:** with `--no-audit`, or when the source scan fails,
  `init` (Python, npm and Java) dropped every input, including framework-declared ones and
  `.aiwaf/inputs.json`. They are saved now, with their types, which are checked. No list is
  marked complete without the audit (the new reason `not_audited`): a complete list with no
  saved risks tells the engine the route reaches no risky call, so strict inputs and the
  relaxed SQL rules for typed routes still need the audit.
- **Koa and Nest JSON routes:** Koa handlers setting `ctx.body = { ... }` and Nest controller
  methods were classified as pages with an unknown category. Koa's assignment is now a JSON
  signal, controller method source is parsed, and Nest routes are JSON unless they use
  `@Render`, `@Redirect`, `@Sse`, or `@Res()` / `@Next()` without passthrough. A `--routes`
  entry can say `"response_type": "json"` itself.
- **Later casts give types (Python and npm):** a read kept in a variable and cast later, as
  Django's `page = request.GET.get("page")` then `int(page)`, or `size = int(size)` in a `try`,
  is typed from the cast; so are Express's `const { limit } = req.query` then `Number(limit)`
  and `parseInt(page || '1', 10)`. It stays text when the variable is also used another way, is
  cast to two types, or is cast only under an `if`.
- **Java `init`, the same fixes:** it no longer writes a manifest with no routes (`--source`,
  `--app` and `--class` alike), which replaced a working manifest with an empty one and still
  exited 0. Failures now exit non-zero: 1 when it finds no routes or can't load the class or
  folder, and 2 for a usage error; before, `--source` and `--class` errors and the usage text
  all exited 0. A class it can't find is named as such ("Controller class not found: ..."),
  where the message was the bare class name.
- **Java live `init` keeps per-method lists:** `--app` and `--class` saved one list for every
  method on a path, so strict inputs let a POST send `page`, a name only the GET handler reads;
  `--source` split them. Each live handler's framework names are now claimed by the methods it
  serves, giving the same `inputs_by_method`. A type Spring binds (`@RequestParam int page`)
  also holds over the code's untyped read of the same name.
- **Java later parse:** `String raw = request.getParameter("n")` followed by
  `Integer.parseInt(raw)` saves `n` as int, as Python and npm now do for later casts. It stays
  text when the local is used any other way (null and `isEmpty()` checks aside).
- **Java `init --app` uses a free port:** Spring Boot's web server started on the app's
  configured port, so `init` failed while the app, or anything else, held it. It now passes
  `--server.port=0` unless the arguments set a port. `SpringPathManifest.launchAndDiscover`
  reads the routes without writing a manifest.
- **npm `aiwaf init`:** loads the app and writes the manifest from the routes it registers, as
  Python's init does. It finds the app as package.json's main, or `app.js`, `server.js` or
  `index.js` (also under `src/` and `dist/`), or `--app <file>`, with `--export <name>` for a
  named export or a factory. Servers the app starts don't bind a port while it loads, which is
  also how an app that only calls `app.listen()` is found. It reads Express (with mounted
  routers), Fastify, Hapi, Nest and Koa with `@koa/router` or `koa-router`, from CommonJS, ES
  modules, or TypeScript Node can run. Before, `init` was an unknown command.
- **No empty manifests:** `aiwaf init` (Python and npm, including `aiwaf fastapi init`, the
  Flask CLI and Django's `manage.py aiwaf init`) and npm `aiwaf manifest` write nothing and exit
  with an error when they find no routes. npm `manifest` without `--routes` used to report
  success with 0 routes, which turned off route-specific protection without saying so and
  replaced any manifest already there. Without `--routes` it now loads the app as `init` does.
- **Route lists keep API detection:** a `--routes` JSON list carries no handlers, so routes such
  as `POST /orders` answering with `res.json()` were classified as pages unless under `/api/`.
  Their handlers now come from the source (`app.post('/orders', handler)`,
  `router.route('/x').get(...)`, including routers mounted under a prefix), giving the same API,
  form and request-body signals as the live app.
- **Fastify routes from a live app:** the manifest found none, because Fastify's default
  `printRoutes()` output is a prefix tree. It now reads one full path per line, with methods.
- **Handlers returning an object** (`async () => ({ ok: true })`, as Fastify, Hapi and Nest
  handlers do) count as JSON responses.
- **Python: nested Pydantic bodies:** a FastAPI body model field typed as another model, as
  `customer: Customer`, now saves that model's declared fields as `inputs_nested`
  (`{"customer": ["email", "name"]}`), so strict inputs checks keys inside it. FastAPI parses
  the body into the model, which drops undeclared keys, so these lists stand even where reading
  the code found only the whole field. A model that allows undeclared fields gets no list.

## Python 1.3.2 / aiwaf-rust 0.2.17 / Node.js 1.3.2 / WASM 0.2.13 / Java 1.4.0 — released

- **Release note:** Python 1.3.1, npm 1.3.1, aiwaf-rust 0.2.16 and aiwaf-wasm 0.2.12 are
  published. Publish aiwaf-rust 0.2.17 (`rust-v0.2.17`) before Python 1.3.2, which requires
  `aiwaf-rust>=0.2.17`, and aiwaf-wasm 0.2.13 (`wasm-v0.2.13`) before npm 1.3.2, then update
  `js/package-lock.json` to it (`^0.2.11` already allows it). With an older engine, per-method
  lists, nested keys and multipart names aren't checked, and setting `AIWAF_STRICT_INPUTS_EXEMPT`
  turns strict inputs off with a warning rather than checking the exempt paths. Java 1.4.0 isn't
  on Maven Central yet, so it carries the Java changes listed under 1.3.1 and 1.3.0 too.
- `init` accuracy, measured with a benchmark of seven apps with known inputs (Django + DRF,
  Flask, FastAPI, Express, NestJS, Fastify, Spring): name recall from 88.6% to 100%, routes
  called complete correctly from 42 to all 52 that can be, and none wrongly called complete
  (was 5). The fixes:
  - A path several handlers serve (GET and POST views, two controller methods) takes every
    handler's inputs, not only the last one's (Python and Java); a handler whose code isn't
    found makes the list incomplete (`unread_view`).
  - Express routers mounted with `app.use('/prefix', router)` are found, nested and with
    parameters, in Express 4 and 5.
  - A FastAPI `dict` or `Any` body, and `express.raw()` or `express.text()` before a handler,
    make the route incomplete instead of listing a field named `body` or `length`.
  - A query or body container passed to a project helper, as `parse_filters(request.args)` or
    `readAddress(req.body)`, is followed into it.
  - Path parameters aren't listed as query or body inputs; Flask's `request.files` names are;
    `request.get_json(silent=True) or {}` and loops over constant keys are read; Fastify's route
    options object isn't counted as middleware; and handlers that read nothing get an empty,
    complete list.
- Strict inputs check each method's own list. When different handlers serve a path's methods,
  `init` saves `inputs_by_method` (Python, Node.js and Java): each method takes the names its
  handler's code, its framework schema or its OpenAPI operation reads, plus every name nothing
  ties to one method (declarations, recordings). The engines take the request's method
  (`review_method` in aiwaf-rust and aiwaf-wasm, an overload in Java); HEAD uses GET's list, and
  an engine without it checks the whole list. In the init benchmark, names a method accepts
  without reading them went from 34 to 0, with no method's list missing a name it reads.
- Strict inputs check multipart form field names (a file part by its field name), and keys
  inside object-valued inputs when `init` knows them all (`inputs_nested`, from code that reads
  only constant keys inside them, or Java body classes' nested classes): `user.role` in a JSON
  object or a form's `user[role]`. `AIWAF_STRICT_INPUTS_EXEMPT` lists path prefixes it skips
  (`aiwaf.strict-inputs-exempt`, `security.strict_inputs_exempt`, Node.js
  `strictInputsExempt`). Shared cases cover all three engines.
- Node.js handlers passed by name (`app.get('/x', list)`) get their own method's list, from the
  methods Express routed them for; and names read off a raw body (`express.raw()`) aren't listed.
- Python reads request schemas from a view's own decorators' closures, as `@validate(Body)`:
  pydantic models, marshmallow schemas, DRF serializers and forms.
- The Java recorder records the application's own whole reads of the request
  (`getParameterMap`, `getParameterNames`, the body stream or reader) as `*`, telling them from
  Spring's binding reads by the calling code.

## Python 1.3.1 / aiwaf-rust 0.2.16 / Node.js 1.3.1 / WASM 0.2.12 — released

- **Release note:** Python 1.3.0, npm 1.3.0 and aiwaf-rust 0.2.15 are published; this release
  adds the route input work below. aiwaf-rust and aiwaf-wasm are unchanged, so Python 1.3.1
  still requires `aiwaf-rust>=0.2.15` and npm `aiwaf-wasm ^0.2.11`. Java 1.4.0 (not yet on
  Maven Central) also carries the Java changes listed under 1.3.0.
- **Strict inputs** (`AIWAF_STRICT_INPUTS`, off by default), in aiwaf-rust 0.2.16, aiwaf-wasm
  0.2.12 and Java: on a route whose saved input list is complete, a query or body name it never
  reads is `schema_unexpected_input`, logged in `AIWAF_ROUTE_SCHEMA_MODE=monitor` and blocked in
  `enforce`. CSRF tokens, `utm_*`, `fbclid`, `gclid`, `msclkid` and `_` are always allowed;
  `AIWAF_STRICT_INPUTS_ALLOW` adds names. Hosts pass it in the manifest the engine gets
  (`strict_inputs`), so older engines ignore it, and AIWAF warns and leaves it off with them.
  Shared cases: `strict_cases` in `tests/fixtures/route_schema_cases.json`.
- Recording turns on from AIWAF itself with `AIWAF_RECORD_INPUTS`: Flask's and FastAPI's
  `AIWAF(app)`, the Node.js middleware (from where it hands the request on, so its own reads
  aren't counted), Spring Boot (`aiwaf.record-inputs`, a filter registered last) and Django
  (aiwaf's app adds `DjangoInputRecorder` last to `MIDDLEWARE`).
- Route input lists (`init`) say why they're incomplete, follow imports, and take more sources,
  in Python, Node.js and Java:
  - `inputs_incomplete` lists each reason with its code, detail, file and line, and init's
    summary counts them: `helper`, `container_whole`, `dynamic_key`, `raw_body`, `class_view`,
    `dependency`, `external_class`, `model_fields`, `middleware` (Node.js) and `observed`.
  - Forms, serializers, models, zod schemas, helpers and constants are followed into the
    project's other files through imports (Python and Node.js), and the request into static
    methods of other classes (Java).
  - The framework is asked what it reads: Django generic views, DRF views, FastAPI's resolved
    parameters and dependencies, and Spring handler methods and body classes by reflection.
    Python reads the fields of classes the code reading couldn't from already-imported modules.
    Node.js reads Fastify route schemas, Hapi `validate` schemas (Joi), and NestJS `@Query()`
    and `@Body()` parameters, pipes and DTO classes (class-validator or `@nestjs/swagger`
    metadata) from the live app, and celebrate and express-validator middleware on Express
    routes. The Fastify plugin now records routes registered after it, with their schemas, as
    `fastify.aiwafRoutes`.
  - Node.js NestJS manifests are built from the controllers' metadata, with Nest's own route path
    rules (global prefix exclusions, `RouterModule` paths, URI versioning), and Express route
    extraction reads Express 5's `app.router` as well as Express 4's `app._router`; before, a
    live Nest 11 (Express 5) app produced no routes.
  - Express routes with middleware are classified by their last function, the handler, not the
    first middleware.
  - Node.js reads zod 3 and zod 4 schemas at runtime (Fastify route schemas, nestjs-zod DTOs),
    and `aiwaf.inputs()` takes a zod or Joi schema or a DTO class as well as a name map.
  - Node.js source reading understands validators written on the route: express-validator
    (`query`, `body`, `check`, `param`, `header`, `cookie`, `checkSchema`, `checkExact`) and
    celebrate. `checkExact`, or celebrate closing both query and body, makes the route complete.
  - Node.js dev dependencies: zod, celebrate, express-validator and nestjs-zod, so these readers
    are tested against the real libraries.
  - Node.js source reading also recognizes validators called through a namespace import
    (`ev.body('x')`, `ev.checkExact([...])`), an object on `this`, or a function's result.
  - Node.js reads a validation middleware's zod or Joi schema from its closure, through the
    inspector while the manifest is built, as for `validate(schema, 'query')`.
  - Django views read webargs arguments too (any webargs parser).
  - Node.js reads route middleware the project defines (`app.post('/x', checkBody, handler)`)
    from source like the handler, instead of giving it a `middleware` reason.
  - Python reads webargs `@use_args`/`@use_kwargs` and flask-smorest `@blp.arguments` on Flask
    views, from the live app (each decorator layer's schema, location and unknown setting) and
    from source (dict argmaps and marshmallow Schema classes, with `data_key`, `dump_only` and
    `Meta.unknown`). Query and body arguments that both RAISE on unknown names make the route
    complete. webargs and flask-smorest join the CI test installs.
  - Declarations: `@aiwaf.inputs(...)` (Python), `aiwaf.inputs({...})(handler)` (Node.js),
    `@AiwafInputs` (Java), or `.aiwaf/inputs.json`. OpenAPI specs with `AIWAF_OPENAPI_SPEC`.
  - Recorders for tests and staging, on only with `AIWAF_RECORD_INPUTS`: Django middleware,
    Flask, ASGI (FastAPI and Starlette), Express and a servlet filter. They record the names
    handlers read to `.aiwaf/observed_inputs.jsonl`, which init adds; a name the analysis
    missed makes the route incomplete.

## Python 1.3.0 / aiwaf-rust 0.2.15 / Node.js 1.3.0 / WASM 0.2.11 — released

- **Release note:** aiwaf-rust 0.2.14 and aiwaf-wasm 0.2.11 are published and carry the
  checks below. npm `aiwaf` requires `aiwaf-wasm ^0.2.11` (was ^0.2.9), with the lockfile
  pinning the published 0.2.11. Publish aiwaf-rust 0.2.15 (`rust-v0.2.15`) before tagging
  `python-v1.3.0`, which requires `aiwaf-rust>=0.2.15`; `js-v1.3.0` and `java-v1.4.0` don't
  depend on it.
- **aiwaf-rust 0.2.15** moves from PyO3 0.21 to 0.29.3. 0.21 isn't affected by
  GHSA-36hh-v3qg-5jq4 (the out-of-bounds read in list and tuple iterators' `nth` arrived in
  0.24, and aiwaf-rust never calls it on them), but scanners flag every PyO3 before 0.29. The
  Python API is unchanged; the training CSV reader holds its reader in a `Mutex`, because PyO3
  classes must now be thread-safe.
- The README describes attacks in words instead of quoting working payloads (reverse shells,
  JNDI lookups, webshell and PHP calls, XML entities and similar). npm refused every publish of
  aiwaf 1.3.0 with a bare 403 after these were added, and malware scanners flag them; aiwaf-wasm,
  whose registry entry doesn't carry the README text, published normally.
- Dependency alerts: Java pins `org.json` 20260719 (Jedis 7.5.3 brings 20251224, with
  CVE-2026-59171) and tests with AssertJ 3.27.7; Rust locks `rand` 0.8.6. The unused root
  `requirements.txt` and `tests/flask/requirements.txt`, frozen copies of a development
  environment that scanners read as production pins, are removed: installs use the ranges in
  `pyproject.toml`.
- Node.js loads without `sqlite3`'s native binary, which npm 12 doesn't build unless the app
  approves its install script: `require('aiwaf')` used to fail with knex's "npm install sqlite3"
  hint. The database now connects on first use, the stores keep their data in CSV files under
  `logs/storage`, and one warning names the fix (`npm approve-scripts sqlite3 && npm rebuild
  sqlite3`).
- Node.js finds `aiwaf-wasm` wherever npm installed it: the fallback loaders and the version
  check for the native pipeline used to look only in aiwaf's own `node_modules`, so they missed
  a copy hoisted to the app's `node_modules`.
- **Java 1.4.0** also brings `AIWAF_PROFILE` and the route checks for the new operations from
  the 1.2.6 release below: Java 1.3.9 wasn't published.
- Remote code execution (Python, Node.js and Java, regular and native): the payload check
  finds Log4Shell lookups (with nested `${lower:j}` forms), Shellshock, serialized Java, pickle,
  PHP, .NET and node-serialize objects, unsafe YAML tags, OGNL and SpEL that reach a process,
  Spring4Shell's `class.module.classLoader`, template sandbox escapes, reverse shells and PHP
  code, in every request value, path segment and header, including `User-Agent` and `Accept`.
  It also finds deserialization gadgets (Fastjson and Jackson `@type`, XStream, `XMLDecoder`),
  EL, Thymeleaf, Ruby ERB, Twig, Smarty, Velocity and Mako escapes, JSP, ASP and PHP webshell
  code, uploads named as server scripts (`filename="shell.php"`), and commands only an attack
  sends (`; cat /etc/passwd`, `${IFS}`, `certutil -urlcache`, `regsvr32 /i:`, PowerShell
  download cradles). Each value is checked again with escapes, HTML entities, fullwidth and
  invisible characters undone, and as base64. It follows `AIWAF_SQL_INJECTION_MODE`. Java's
  rules are generated from aiwaf-rust's by `scripts/sync_rce_rules.py`, and a test fails when
  they differ. The shared cases in `tests/fixtures/rce_cases.json` include ordinary text that
  must never match.
- Remote code execution, harder to slip past: each field of a query string or form body is
  checked on its own, so a serialized object in one field (`data=rO0AB...`, a PHP `O:8:...`
  object, `%AC%ED%00%05...`) is found; raw binary Java, pickle, .NET and Ruby objects are
  found in Python and Node.js bodies, which were read as text with those bytes lost; gzip
  bodies are read inflated, and one that inflates past
  `AIWAF_PAYLOAD_MAX_BYTES` is refused with `413`; and `__proto__` and
  `constructor.prototype` keys are blocked as `rce_prototype_pollution` in every engine.
- Command injection: `rce_command_injection` blocks probe and timing commands chained onto the
  end of a value, as scanners send them (`x;id`, `|whoami`, `&& uname -a`, `; sleep 5`,
  ``test`id` ``, `$(sleep 5)`), on every route. Ordinary text such as `name; id` and Markdown's
  `` `id` `` passes; the shared cases include both.
- WebSocket messages are checked like request bodies: `WebSocketGuard` ASGI middleware for
  FastAPI, Starlette and Django Channels, `inspect_message` in Python; `guardWebSocketServer`
  (ws), `guardSocketIo` and `inspectMessage` in Node.js; `AiwafMessageCore.check` in Java. A
  message with a finding closes the socket with 1008. Flask-SocketIO: `guard_socketio` and
  `@aiwaf_message()`. Java: `AiwafWebSocketHandler` for Spring WebSocket and `AiwafEndpoint` for
  Jakarta WebSocket (`spring-websocket` and the Jakarta WebSocket API are `provided`).
- Route schemas from `init` (Python, Node.js and Java): the audit saves what each route reads
  from the query and body, with types (`int`, `float`, `bool`, `uuid`, `text`) from conversions,
  type hints, Pydantic, Django forms, DRF serializers, zod, Spring parameters and DTOs, and
  whether that list is complete. Requests are reviewed against it: a typed input sent with
  another type is `schema_type_mismatch`, and an `sql_function`, `sql_comment` or
  `sql_breakout_numeric` finding that comes only from inputs the schema clears (typed inputs,
  text that reaches no operation, inputs the route never reads) is cleared. Quote breakouts and
  remote code execution rules still block everywhere. Findings from the URL path are reviewed
  by the route's parameters. `AIWAF_ROUTE_SCHEMA_MODE` is `monitor` by default, which only logs;
  `enforce` acts on it. The native Python and Node.js pipelines review in the engine too, and the
  Node.js native `inspectPayload()` reviews its findings. The request is followed into helpers in
  the same file or class, and keys held in constants are known, in all three audits.
- Multipart bodies are found about ten times faster (a 60 KB upload in about 18 µs, from 189 µs).
- Multipart forms are checked (Python, Node.js and Java, regular and native), by their first
  `AIWAF_PAYLOAD_MAX_BYTES`: part headers (`filename="shell.php"`), field names, text fields
  as any other value, and uploaded files for binary serialized objects and PHP, JSP and ASP
  code. Large uploads still reach the app whole. Node.js reads the start of an unparsed
  multipart body and puts it back; Java reads the container's parts, and a servlet with no
  multipart config still gets the raw upload unread. Checked in embedded Tomcat 11, Jetty 12 and
  Undertow 2.3 in the tests.
- Deflate bodies are read inflated like gzip ones, and Node.js reads `br`. Python and Node.js
  refuse an inspectable body in an encoding they can't read with `413`, as the native
  pipeline did. Java inflates gzip and deflate bodies for the check when
  `allowCompressedRequestBodies` is on.
- Base64 is unwrapped up to three layers deep, including base64 of gzip or deflate.
- Every header is checked: the check no longer stops after 64. More than 256 headers, or 256
  cookies, is refused with `413` instead of being checked in part.
- Faster payload checks (aiwaf-rust, aiwaf-wasm and Java): a value with none of the characters
  any remote code execution rule needs skips those rules, after a single byte scan; a whole
  JSON or XML document is checked for them only when it has a marker of an attack in its shape,
  such as `@type`; and a plain word, number, email or identifier skips the SQL tokenizer and is
  checked only for SQL-only names such as `tempdb`. In release-build timings, a typical JSON
  body takes about 25 µs to check, from 37 µs, and a typical query string about 11 µs, from
  19 µs; values with spaces, quotes or operators still get the full tokenizer.
- Injection audit: more sinks that run code. Python `importlib.import_module` and `__import__`,
  `os.exec*` and `os.spawn*`, Jinja `from_string`, and `jsonpickle`, `cloudpickle`, `joblib`,
  `torch.load` and `pandas.read_pickle`; Node.js dynamic `require()` and `import()`, string
  timers, `new vm.Script`, eval workers, `execFile` and `fork`, and `v8.deserialize`; Java JNDI
  `lookup` (not `java:` names), `Class.forName`, Groovy, MVEL, OGNL, EL expressions, Velocity,
  XStream and `SerializationUtils.deserialize`.


## Python 1.2.6 / aiwaf-rust 0.2.13 / Node.js 1.2.5 / WASM 0.2.10 — released

- `AIWAF_PROFILE` (Python, Node.js and Java): `balanced` keeps every default; `strict` halves
  the rate and flood limits and turns on UUID scoring, header validation and the method policy
  where the runtime has them as settings; `monitor` is a dry run that logs "AIWAF monitor: would
  block ..." and blocks nothing, adding no IP to the blocklist. Settings given explicitly win.
  Monitor runs the regular middleware: the automatic native pipeline stays off, and a native
  pipeline chosen explicitly refuses to start.
- Route checks for the new operations (aiwaf-rust 0.2.13, aiwaf-wasm 0.2.10 and Java): routes
  whose input reaches HTML output, a redirect, an LDAP filter, an XPath query, an XML parser, a
  regular expression or an object key are checked for markup and script URLs, off-site redirect
  tricks (`//host`, `user@host`, `javascript:`), filter and quote breaks, `<!DOCTYPE`, nested
  quantifiers and `__proto__`. Plain absolute URLs pass the redirect rule, since the site's own
  host isn't known there.

## Python 1.2.5 / Node.js 1.2.4 / Java 1.3.8 — released

- **Release note:** these need no new engine: Python requires `aiwaf-rust>=0.2.12` and npm
  `aiwaf` `aiwaf-wasm ^0.2.9`, both already published. npm `aiwaf` 1.2.3 was staged but never
  released, and npm doesn't reuse a version, so npm `aiwaf` 1.2.4 ships the Node.js changes of
  this section and the next.
- Python requires `aiwaf-rust>=0.2.12` (was 0.2.11), and npm `aiwaf` `aiwaf-wasm ^0.2.9`
  (was ^0.2.8), so the features of the previous section that needed the newer engines (the
  Django logger between other middleware, Node.js user-agent lists) work after an install
  instead of falling back to the regular middleware. Node.js picks the native pipeline by
  default with aiwaf-wasm 0.2.9 or newer.
- Injection audit (Python, Node.js and Java): input is now followed through a helper's return value,
  into a list, set or map filled in a loop (`append`, `add`, `push` and similar), and into
  functions imported from other project files, through relative imports and `require()`. A
  finding in another file names the handler's file in `entry_file`, which `--manifest` and
  `init` use to link it to the route. Route values already parsed as numbers or UUIDs no longer
  count as input: Python parameters typed `int`, `float`, `bool`, `UUID` and similar (also
  inside `Optional`, `Annotated` or a union with `None`), Flask `<int:…>`, `<float:…>` and
  `<uuid:…>` converters, and NestJS `ParseIntPipe`, `ParseFloatPipe`, `ParseBoolPipe` and
  `ParseUUIDPipe`. Java follows calls into other scanned classes through field, parameter and
  local variable types, static calls, and an interface or base class with one implementation, and
  no longer counts Spring parameters typed as numbers, booleans, dates or UUIDs (also inside
  `Optional`, `List` or `Set`, or as arrays) as input. Java findings gained `entryFile()`; the
  old seven-field `Finding` constructor still works.
- Injection audit: method calls on instances of the project's classes are followed into the
  class, in all three languages: values built with `new`/a constructor, typed variables,
  parameters and fields (NestJS constructor injection, FastAPI `Depends(Service)`, Spring fields),
  and, in Java, other calls' return types and `var`. Java follows every implementation of an
  interface, up to five. Node.js follows `tsconfig.json` or `jsconfig.json` `paths` and `baseUrl`
  (with `extends`) and packages in the scanned folder, by their `package.json` `exports`, `module`
  or `main`. A call that runs one of the project's own methods is no longer also guessed to be a
  database or file call from its name, as `this.users.find(id)` was.
- Injection audit, measured: `scripts/audit_benchmark.py` scores an audit report on an OWASP
  Benchmark checkout. Before this release the Java audit scored +3.0 and the Python audit +4.0
  on OWASP's scale (0 is guessing); both now score +100 on every injection category, with no
  false alarms (1,572 Java and 677 Python cases). The benchmarks are small synthetic cases, so
  this isn't proof of the same on real applications.
- Injection audit, more operations (Python, Node.js and Java): `xss`, `redirect`, `ldap`,
  `xpath`, `xxe` (only with external entities turned on) and `redos`, and `prototype`
  pollution in Node.js. Header injection isn't reported: Node.js, Werkzeug, Django and servlet
  containers reject newlines in header values.
- Injection audit, fewer false alarms: checks that return or throw when a value fails
  (`isalnum()`, regex tests, allowlists, `url_has_allowed_host_and_scheme`, `'..'` checks)
  make it safe; encoders are safe only for what they encode for (`html.escape` and
  `encodeForHTML` for HTML, `shlex.quote` for shells, `path.basename` for paths), where some
  stopped every finding before; conditions and switches on values known before running, dict
  and object keys, and list positions are followed. A call into the project's own method isn't
  also guessed to be a database call from its name.
- Injection audit, more input: GraphQL resolvers (Graphene, Strawberry, Ariadne, Apollo-style
  resolver maps, NestJS `@Args`, TypeGraphQL, Spring `@Argument`, DGS `@InputArgument`), tRPC
  `input`, Next.js server actions, Fetch API requests (`request.json()`, `formData()`,
  `searchParams`), Spring WebFlux, and WebSocket messages (ws, Socket.IO, FastAPI, Channels,
  Flask-SocketIO, `@MessageMapping`, `@OnMessage`). Input is followed through callbacks
  (`.then()`, `.map()`, Java lambdas), objects that keep it in fields, values middleware keeps
  on the request or `flask.g`, request and session attributes, `with open(...)`, `pathlib`
  paths, `StringIO`, Base64 and `file:` URIs.
- Injection audit, CI: `--format sarif` (SARIF 2.1.0 with stable fingerprints),
  `--baseline report.json` to report only new findings, and `aiwaf: ignore` or
  `aiwaf: ignore[sql]` comments. A helper called with the same input from many routes is
  walked once: a 400-route test project with 512 shared database calls each takes 0.7 s in
  Node.js, 1.1 s in Python (12.7 s before) and 1.7 s in Java.
- Fix (Node.js audit): a project with very many findings no longer overflows the call stack.
- Tests: `python aiwaf_test.py` runs the Java suite with Maven too, and `--java-only` runs it
  alone. `--no-java` skips it on machines without a JDK 17+ and Maven. The monorepo CI workflow
  sets up Java 17 and runs on changes under `java/`.

## Python 1.2.4 / aiwaf-rust 0.2.12 / Node.js 1.2.3 / WASM 0.2.9 — prepared release

- **Release note:** Python 1.2.3, aiwaf-rust 0.2.11 and aiwaf-wasm 0.2.8 are published; npm
  `aiwaf` 1.2.3 isn't yet, so it ships the Node.js changes of both sections. Python 1.2.4 still
  needs only aiwaf-rust 0.2.11, and npm `aiwaf` only aiwaf-wasm 0.2.8: the features below that
  need the newer engines fall back to the regular middleware without them.
- **Fix (FastAPI, regular and native):** load the trained JSON model configured by
  `ai_anomaly.model_path` (or `AIWAF_MODEL_PATH`) at middleware startup. Missing or invalid
  artifacts log a warning and leave model scoring disabled. Route discovery now uses the
  application router, including mounted and parameterized paths, for keyword protections
  and regular anomaly history checks.
- Django: the regular logger middleware may be listed between other AIWAF middleware and still
  run beside the native pipeline (with aiwaf-rust 0.2.12, whose request context now names the
  stage that blocked). It logs what it logged beside the regular chain: not the blocks of
  stages listed before it, and, for a block decided after the response by such a stage, the
  response it saw. Native UUID blocks are now returned from `process_view`, where the regular
  UUID check returns them, so other middleware's request hooks run first, as before, and a
  logger listed anywhere records them.
- FastAPI: beside the native pipeline, requests are logged by a plain ASGI version of the
  regular logging middleware, which writes the same files and lines without Starlette's
  BaseHTTPMiddleware. In local benchmarks, AIWAF's own cost on a native FastAPI request with
  request logging fell from about 0.38 ms to 0.22 ms (0.12 ms without logging), against about
  1.7 ms for the regular middleware.
- **Fix (Node.js, regular and native):** restore default login prefixes for the relaxed
  login form timing threshold. Explicit `AIWAF_LOGIN_PATH_PREFIXES: []` still disables them.
- Node.js: `AIWAF_SUSPICIOUS_USER_AGENTS` and `AIWAF_LEGITIMATE_BOTS` are matched in JavaScript,
  by the adapter, and the engine takes the verdict (with aiwaf-wasm 0.2.9; with 0.2.8, an app
  with its own lists keeps the regular middleware). Patterns with any
  flags, and JavaScript-only syntax such as lookbehind, now carry over.

## Python 1.2.3 / aiwaf-rust 0.2.11 / Node.js 1.2.3 / WASM 0.2.8 / Java 1.3.7 — prepared release

- **Upgrade order:** publish aiwaf-rust 0.2.11 and aiwaf-wasm 0.2.8 before Python 1.2.3 and
  Node.js 1.2.3, which require them. Node.js runs the native pipeline automatically only with
  aiwaf-wasm 0.2.8 or later; with an older one it keeps the regular middleware and says why.

- Flask and FastAPI: when the regular middleware runs as the native pipeline (or with
  `from_settings`), honeypot timing is kept in the process's memory, as their regular honeypot
  middleware keeps it, instead of in a CSV file rewritten on every GET. In local benchmarks,
  AIWAF's own cost on a Flask request fell from about 0.26 ms to 0.18 ms. A native profile that
  sets `honeypot_backend` itself is unchanged.
- Python's CSV stores no longer check that every CSV file exists before each read: the check
  runs again when the data directory changes, and at least once a second. With a new client on
  every request, a native Flask request fell from about 0.73 ms to 0.59 ms, and the regular
  middleware got faster too.
- **Fix (Node.js):** ten store methods (geo, exemptions, dynamic keywords, model and request log)
  returned their database query without awaiting it, so a database error skipped the CSV
  fallback and escaped. They now await it and fall back as intended.
- Node.js's request log without a database no longer reads the whole log for each logged
  request (to find the next id) or each recent-history check: it reads only the end of the
  file. On a 50,000-row log, a logged request's write fell from about 44 ms to 0.2 ms.
- Node.js builds route plans once per path and route markers, and compiles wildcard path rules,
  route patterns and the UUID route prefix once, instead of on every request. The UUID route
  prefix is now matched as literal text: a `.` in it no longer matches any character.
- Node.js's native pipeline reports a request log it can't write once, not on every request.
- **Java:** CSV and file storage read their file again only when it changed (by inode, change
  time, modification time and size), instead of on every read, and reads take a shared lock,
  so readers in different processes don't wait for each other. A blocklist read on a
  5,000-entry CSV store fell from about 1.9 ms to 45 µs; changes from other processes and
  outside edits are still seen. A store read that fails is logged once and read as nothing
  found, as in the native pipelines, instead of failing the request.
- Python's native Django pipeline does less work per request. In local benchmarks AIWAF's own
  cost per request fell from about 0.20 ms to 0.13 ms, against about 3.1 ms for the regular
  middleware.
  - With Django's `LocMemCache` (exactly that class, not a subclass), honeypot timing and rate
    counts use the engine's own memory instead of a Django cache call per request. Both keep
    values in the process only; a process that builds more than one native middleware now
    counts per middleware rather than per process. Since the cache isn't called, a
    `LocMemCache` with ignored errors no longer keeps the regular middleware. Shared caches
    such as Redis or Memcached are still used as before.
  - Route plans from path rules and view decorators are computed once per path and view, in a
    bounded cache; route checks still inspect every request.
  - URL resolution is cached per URLconf, active language and path, in a bounded cache.
- **Fix (Node.js):** an app with geo blocking that ran as the native pipeline automatically
  stopped blocking by country, because nothing told the native pipeline which country a
  request came from. It now looks countries up where the regular geo check finds them: the
  MMDB database, then the `x-country-code` header, cached per address. `AIWAF_GEO_MMDB_PATH`,
  `AIWAF_GEO_CACHE_SECONDS` and `AIWAF_GEO_CACHE_PREFIX` carry over. An app's `geoResolver` is
  used as the regular check uses it, and may be asynchronous: the request waits for it.
- **Fix (Django):** the regular anomaly middleware saves each request to `RequestLog`, which
  `aiwaf train` reads when there is no access log, and drops its model until there are
  `AIWAF_MIN_AI_LOGS` logs. As the native pipeline, neither happened: requests weren't saved,
  and the model was used at once. The native pipeline now saves the same rows when
  `AIWAF_ACCESS_LOG` names no log with requests in it to train from, and the model is carried
  over only when the regular middleware would use it. Saving a row is a database write per
  request: in a local load test on SQLite, native Django handled about 340 requests a second
  per worker with it, against about 100 for the regular middleware.
- **Fix (Django and Node.js):** the regular header check also ran aiwaf-rust's or aiwaf-wasm's
  check, which capped the User-Agent at 500 characters whatever `AIWAF_MAX_USER_AGENT_LENGTH`
  allowed. It now uses the app's cap, with aiwaf-rust 0.2.11 or aiwaf-wasm 0.2.8; older builds
  keep 500.
- Python: if the native pipeline is chosen automatically and can't start, the regular
  middleware runs instead, with a warning on `aiwaf.native`, as in Node.js. Before, the error
  stopped Django, Flask or FastAPI from starting. A native pipeline set up explicitly still
  raises its error.
- Python and Node.js say why an app runs the regular middleware and not the native pipeline:
  the settings that don't carry over, or the aiwaf-rust or aiwaf-wasm that is missing or too
  old. It's logged once, at info level on `aiwaf.native` (Python) or with `console.info`
  (Node.js), and not for an app that turned the native pipeline off itself.
- Python: more settings no longer keep an app on the regular middleware.
  - Settings only `aiwaf train` and the other commands read, such as `AIWAF_AI_CONTAMINATION`,
    `AIWAF_MIN_TRAIN_LOGS` and the `AIWAF_RUST_*` and `AIWAF_PYTHON_*` training options.
  - `AIWAF_GEOIP_DB_PATH`, `AIWAF_GEO_CACHE_SECONDS` and `AIWAF_GEO_CACHE_PREFIX`, which the
    native adapters read as the regular geo middleware does. Django's native middleware read a
    fixed cache prefix and lifetime before.
  - Django: `AIWAF_ACCESS_LOG`, `AIWAF_MIN_AI_LOGS`, `AIWAF_DISABLE_AI` and the `AIWAF_MODEL_*`
    settings, which decide the model the regular middleware uses; that model is handed over.
    Also `AIWAF_LOG_FORMAT` and `AIWAF_LOG_LEVEL`, which Django's middleware doesn't read.
  - Flask: `AIWAF_ACCESS_LOG`. With `AIWAF_LOG_FORMAT` set (JSON, CSV or combined), the
    regular logging middleware writes `access.log`, `error.log` and `aiwaf.log` beside the
    native pipeline, in that format, instead of the native request log.
  - FastAPI: `logging_middleware.log_dir`.
  - `AIWAF_RUST_REQUEST_ENGINE`, since the native pipeline always checks requests in Rust.
- More settings carry over to the native pipeline, so they no longer keep an app on the
  regular middleware:
  - Django: `AIWAF_MAX_HEADER_BYTES`, `AIWAF_MAX_HEADER_COUNT`, `AIWAF_MAX_USER_AGENT_LENGTH`
    and `AIWAF_MAX_ACCEPT_LENGTH` (with aiwaf-rust 0.2.11). Flask's regular middleware doesn't
    read them, so they don't change Flask's native pipeline either.
  - Django and Flask: `AIWAF_WINDOW_SECONDS`, the anomaly window, and
    `AIWAF_ALLOWED_PATH_KEYWORDS`. The native keyword checks now skip the same words the
    regular ones do: the app's route words, its allowed and exempt keywords, and the regular
    defaults. Before, the native pipeline used a shorter built-in list.
  - Django and Flask: extended block info (`AIWAF_BLOCKLIST_STORE_EXTENDED_INFO` and its header
    limits and redactions; Flask's `AIWAF_CAPTURE_EXTENDED_REQUEST_INFO` and
    `AIWAF_EXTENDED_REQUEST_INFO_*`). A native block records the request's headers in the
    regular blocklist, as the regular middleware does.
  - FastAPI: `logging_middleware.log_format`. With JSON or CSV, the regular logging middleware
    writes `access.log` beside the native pipeline, as Flask now does.
  - Node.js: `AIWAF_LOGIN_PATH_PREFIXES`, `AIWAF_MAX_HEADER_BYTES`, `AIWAF_MAX_HEADER_COUNT`,
    `AIWAF_MAX_USER_AGENT_LENGTH`, `AIWAF_MAX_ACCEPT_LENGTH` and `AIWAF_BLOCKED_USER_AGENTS`
    (with aiwaf-wasm 0.2.8). The native header check now also blocks the regular default
    scanners (sqlmap, nikto, acunetix, nmap and masscan) and counts header bytes as Node.js
    sends the names.
- **Fix (Node.js):** the native header check followed the Python rules, so it answered
  differently from the regular Node.js middleware even with default options. The regular check
  applies the header combinations whatever is required, treats an empty User-Agent as
  suspicious, lets a legitimate bot skip the rest, skips static files, counts the protocol as a
  header, and then runs aiwaf-wasm's own check; with `AIWAF_HEADER_VALIDATION` off it still runs
  aiwaf-wasm's User-Agent check. The native pipeline now does exactly the same (with aiwaf-wasm
  0.2.8). In side-by-side tests the two pipelines answer the same across ten header cases in
  four configurations, and at the exact header byte and count limits.
- Node.js: the rest of the regular options carry over to the native pipeline (with aiwaf-wasm
  0.2.8), so none of them keeps an app on the regular middleware:
  - `AIWAF_EXEMPT_KEYWORDS` and `AIWAF_ALLOWED_PATH_KEYWORDS`, checked as the regular keyword
    check does: an allowed fragment in the path skips keyword blocks, and the first matching
    keyword blocks unless it's exempt.
  - `AIWAF_POST_ONLY_SUFFIXES`, with the method policy, and `AIWAF_REQUIRED_HEADERS` per method.
  - Models kept in the model store (`AIWAF_MODEL_STORAGE` "db" or "cache", and its fallback and
    cache settings), read when the native pipeline starts. A stored model in the older format
    keeps the regular middleware, with a warning.
  - Request logging (`AIWAF_MIDDLEWARE_LOGGING` with `AIWAF_MIDDLEWARE_LOG_PATH`, `_LOG_CSV`,
    `_LOG_CSV_PATH` and `_LOG_DB`): the regular request logger runs beside the native pipeline
    and writes the same JSONL, CSV and database logs, blocks included.
  - `AIWAF_SUSPICIOUS_USER_AGENTS` and `AIWAF_LEGITIMATE_BOTS`, as pattern strings or RegExps
    with no flag but `i`. A RegExp with other flags keeps the regular middleware.
  - Extended block info (`AIWAF_CAPTURE_EXTENDED_REQUEST_INFO`,
    `AIWAF_EXTENDED_INFO_MAX_HEADERS`, `AIWAF_EXTENDED_INFO_MAX_VALUE_LENGTH`): a native block
    records the same request info in the regular blocklist.
  - `AIWAF_MIN_AI_LOGS`: the model is used only when there are that many recent logs at
    startup, as in the regular anomaly check.
  - The native pipeline's `countryLookup` option may now return a promise.
- Node.js: the automatic switch to the native pipeline now needs aiwaf-wasm 0.2.8, which can
  answer exactly as the regular middleware. With an older aiwaf-wasm the app keeps the regular
  middleware, with a warning to upgrade, instead of running native with the older header rules.
  A native pipeline set up explicitly still runs on an older aiwaf-wasm when it uses none of
  the new options.
- Django: with `AIWAF_MIDDLEWARE_LOGGING`, the regular logger middleware runs beside the native
  pipeline and writes the same CSV log and `RequestLog` rows, when it's listed before or after
  every other AIWAF middleware. Listed between them, the app keeps the regular middleware.
- **Fix (Python):** side-by-side tests now send the same requests through the regular
  middleware and the native pipeline in Django, Flask and FastAPI, and compare every response:
  header checks in several configurations, keywords and SQL injection, rate limits, honeypot
  timing, UUID tampering and path exemptions. They found two differences, now fixed (with
  aiwaf-rust 0.2.11):
  - Django's and FastAPI's regular header checks skip static files (`.css`, `.js`, `/static/`
    and the like); the native pipeline checked them.
  - FastAPI's regular header check also flags a bare `Mozilla/5.0` User-Agent.
  With an older aiwaf-rust, an app with the header check keeps the regular middleware.
- **Fix:** side-by-side tests now also compare keyword learning, the anomaly model, geo blocking
  and request logs, and found these differences, now fixed:
  - Keyword learning from 404 probes followed one native rule; each regular middleware has its
    own. The native pipeline now learns by Django's, Flask's, FastAPI's or Node.js's rule
    (with aiwaf-rust 0.2.11 and aiwaf-wasm 0.2.8), never learning the words that rule skips.
    With an older engine, an app with the anomaly stage keeps the regular middleware.
  - Django's and Flask's regular keyword checks skip paths under the app's own route prefixes;
    FastAPI's skips its default legitimate words. The native pipeline now does too.
  - Node.js: a keyword learned by this process applies to the next request, as the regular
    middleware's in-memory counts make it, instead of after the store write and its cache.
  - Node.js: anomaly detection runs the regular check itself, after the other stages, as the
    regular middleware does: its features, model (any format, from the file or the model
    store), log minimum and request-log history. Before, the native pipeline judged recent
    behavior from its own history and blocked scans the regular middleware let through.
  - Django: geo blocking applies only with `AIWAF_GEO_BLOCK_ENABLED` and an allow or block list
    of the app's own, as in the regular middleware.
  - FastAPI: the native pipeline set up storage again when the app built its middleware,
    emptying in-memory stores of anything added after `AIWAF(app)`.
  - FastAPI: requests are logged by the regular logging middleware beside the native pipeline,
    so requests AIWAF blocked aren't logged, as before.
- **Fix (Flask):** when the anomaly model blocked a request after its view ran, the regular
  middleware returned a `(body, 403)` tuple from `after_request`, which Flask can't send, so the
  request failed with a 500. It now answers 403.
- Python: `AIWAF_REQUIRED_HEADERS` given per method (a dict with `DEFAULT`) carries over to the
  native pipeline in Django and Flask, with aiwaf-rust 0.2.11.
- aiwaf-rust 0.2.11 and aiwaf-wasm 0.2.8: the header check takes the app's caps and blocked
  user agents (`header_max_bytes`, `header_max_count`, `header_max_user_agent`,
  `header_max_accept`, `blocked_user_agents`). aiwaf-wasm also takes `login_prefixes`,
  `header_bytes_without_prefix`, `exempt_keywords`, `allowed_path_keywords`,
  `post_only_suffixes`, `required_headers_by_method` and `header_policy` ("python", "node" or
  "node_ua", with `header_skip_extensions`, `header_skip_paths` and `header_wasm_check`).
  aiwaf-rust also takes `required_headers_by_method`, `header_skip_extensions`,
  `header_skip_paths` and `extra_suspicious_user_agents`. `supports()` reports each of them. The native pipeline also no longer creates `rate_buckets.csv`
  unless rate limits are counted in it (`rate_backend` "csv").
- In local benchmarks, AIWAF's own cost on a FastAPI request is about 0.13 ms as the native
  pipeline, against about 1.7 ms for the regular middleware. With 1, 4 and 8 Django worker
  processes sharing one data directory and database, the native pipeline handled about 2,100
  requests a second per worker at every size, without errors.

## Python 1.2.2 / aiwaf-rust 0.2.10 / Node.js 1.2.2 — prepared release

- **Fix (aiwaf-rust 0.2.10):** the native pipeline's startup check of its request log read the
  whole file into memory, and kept it cached for the life of the worker, just to compare the
  header row. Logs only grow, so a long-running site could need gigabytes per worker, or be
  killed on a host with a memory limit. It now reads only the header. A request log that can't
  be written, as on a full disk, no longer turns every response into a 500: the failure is
  logged once on `aiwaf.native` and requests carry on.
- **Fix:** Python's native pipeline handles store failures like Node.js's: a failed blocklist,
  exemption, keyword or country read is logged and read as not blocked, not exempt or nothing
  learned, so a database outage doesn't turn every request into an error. A failed exemption
  read now matches the regular middleware, which fails open. A block or keyword that can't be
  written is logged instead of failing a response that was already decided, and the worker
  still remembers its own block.
- **Fix:** the switch to the native pipeline checks that its data directory can be created and
  written. With model storage, the regular middleware writes no files, so a site whose working
  directory isn't writable (the default `aiwaf_data` is relative to it) failed to start after
  the switch. It now stays on the regular middleware and says why.
- **Fix (Node.js):** when `aiwaf(options)` chose the native pipeline automatically and it then
  couldn't start (an unreadable model file, say), the first request awaited the failed startup,
  and Express 4 left that as an unhandled rejection, which ends the Node.js process. The
  automatic switch now falls back to the regular middleware with a warning. A native pipeline
  chosen explicitly passes its startup error to `next(error)` for each request instead of
  crashing, and the engine's startup errors are now `Error`s with a message instead of strings.
- **Fix (Node.js, regular middleware):** an error inside `aiwaf(options)`, such as a store
  failing during a database outage, became an unhandled rejection, which ends the Node.js
  process, and left the request hanging. The middleware now passes it to `next(error)`.
  Blocklist and recent-request reads fail open with one warning per kind, as exemption checks
  already did; with the default `AIWAF_RATE_CACHE_FAILURE_MODE = 'closed'` the rate limiter
  still answers 503 when its storage fails, and `'open'` lets requests through. A block that
  can't be written is reported and the request is still denied.
- **Fix (Node.js, regular middleware):** a catch-all `*` route, as in `app.all('*', ...)`, made
  the anomaly stage build an invalid pattern and throw on every request, which ended the
  process. Route paths are now matched as Express matches them: `:name` is one segment
  (route parameters were never matched before, so `/users/:id` didn't recognise `/users/5`),
  `*` matches anything, other characters are literal, and arrays and RegExp routes work.
- Request handling was compared with the regular middleware in Django, Flask and FastAPI,
  with hundreds of malformed requests (invalid encodings, NUL bytes, oversized headers and
  query strings, broken bodies) and failing stores, lookups and disks, and in Express over raw
  sockets; the native pipeline now errors in no case where the regular middleware doesn't.

- **Fix:** a site whose rate limits put a limit above the flood threshold failed to start on
  1.2.0 and 1.2.1 when the regular middleware switched to the native pipeline, with
  `ValueError: Invalid RATE_LIMIT for path rule`. A common case was a path rule raising `MAX`
  above the default flood threshold of 40, such as `{"PREFIX": "/", "RATE_LIMIT": {"MAX": 100}}`.
  The regular rate limiter flood-blocks above the flood threshold before such a limit is reached,
  so the native pipelines in Python and Node.js now treat that limit as the threshold, blocking
  at the same point, for path rules and for the profile. The settings converters report it as
  carried, so it no longer keeps an app on the regular middleware. A rule window under one
  second or a negative limit is still refused at startup.
- **Fix:** other settings the regular middleware accepts could also stop a site that switched
  to the native pipeline. The settings converters now check every value against what the
  native engines take, and the switch only happens when all of them start natively:
  - Values are normalized as the regular middleware reads them: `AIWAF_SQL_INJECTION_MODE`,
    `AIWAF_INJECTION_RISK_MODE` and the rate key mode in any case (`"BLOCK"`, `"IP"`); in
    Node.js, a fractional `MAX_REQ` or header score, a negative minimum form time, and
    comma-separated keyword or country lists.
  - Values native can't represent are reported and keep the regular middleware: windows under
    one second (or not positive in Node.js), negative or zero limits, a UUID threshold of 0,
    negative weights or times, an out-of-range `AIWAF_PAYLOAD_MAX_BYTES`, an invalid
    path-rule `RATE_LIMIT`, and a path rule that isn't a dictionary.
  - Django: with the honeypot middleware listed, a `DummyCache` or a cache set to ignore errors
    keeps the regular middleware, which runs with either; native honeypot timing can't.
  - Node.js: options the converter can't read keep the regular middleware instead of failing.
  Tests now try hundreds of edge values and combinations and require every one reported
  without differences to start natively.

## Python 1.2.1 / Node.js 1.2.1 — prepared release

- Flask's `AIWAF_USE_CSV` carries over to the native pipeline. It was reported as a setting
  with no native equivalent, so a Flask app that set it stayed on the regular middleware.
  The native pipeline uses Flask's own stores, which follow it; without store support, only
  `AIWAF_USE_CSV = False` (database storage) is reported as a difference.
- npm `aiwaf` declares `aiwaf-wasm ^0.2.7`. The published 1.2.0 still declared `^0.2.6`,
  which resolves to 0.2.7 on a fresh install but lets a lockfile keep 0.2.6.

## Python 1.2.0 / aiwaf-rust 0.2.9 / Node.js 1.2.0 / WASM 0.2.7 — prepared release

- The native pipelines read their stores less. Python reads only what the request's stages
  use (nothing on routes where path rules turn them off, and only the exemption for an
  exempt request), and caches blocklist and exemption answers per IP and path for
  `store_cache_seconds` (default 5), as it already cached keywords and countries. Node.js
  does the same with `storeCacheSeconds`, and its anomaly stage now sees stored blocks. A block
  or exemption made elsewhere can take that long to apply; this worker's own blocks apply
  at once. `store_cache_seconds: 0` turns the caching off. In a local benchmark with
  repeat clients, AIWAF's own cost per request fell from about 1.1 ms to 0.2 ms on Django
  with the ORM, against about 3.2 ms for the regular middleware.
- **Behavior change:** Node.js's `aiwaf(options)` runs the native pipeline by default when
  the installed `aiwaf-wasm` is 0.2.7 or newer and every option carries over exactly.
  Otherwise it runs the regular middleware as before. It logs a line when it picks the
  native pipeline, and `AIWAF_NATIVE_PIPELINE: false` (or the environment variable) keeps
  the regular middleware. For this, the native pipeline now resolves client addresses
  through `AIWAF_TRUSTED_PROXIES` as the regular middleware does (`trustedProxies`), and
  loads a model saved by `aiwaf train` (`modelFile`). The sample model shipped in the
  package isn't used natively.
- **Behavior change:** Python's regular setup runs as the native pipeline by default when
  the installed `aiwaf-rust` has every native feature (0.2.9 or newer) and every
  setting carries over exactly. In Django the regular classes in `MIDDLEWARE` run as one
  native middleware at the place of the last of them; in Flask and FastAPI, `AIWAF(app)`
  installs the native pipeline with the stages its plan would register. Request logs go
  to `aiwaf_native_access.csv` in the regular log directory, which the Flask and FastAPI
  trainers read, and a loaded aiwaf-rust model carries over. AIWAF logs a line when it
  switches. `AIWAF_NATIVE_PIPELINE = False` (Django settings, Flask config, or the
  environment variable) or `native_pipeline=False` (Flask and FastAPI `AIWAF`) keeps the
  regular middleware. With aiwaf-rust 0.2.8 nothing switches.
- Python's `aiwaf init` now prints the native setup for the detected framework, with the
  conversion report for the project's settings and whether it switches automatically;
  `--no-native` skips it. Django's native middleware now applies `AIWAF_SETTINGS`, the
  legacy nested settings, before reading the flat ones, as the regular middleware does.
  FastAPI's converter now carries `storage.data_dir` (`AIWAF_DATA_DIR`), and logs where
  the trainer reads instead of reporting logging as changed.
- Node.js's `aiwaf.native()` no longer rejects uploads. Unparsed bodies are handled as
  in Python: multipart, binary and other bodies that aren't JSON, form or XML pass
  through unread, so upload middleware after AIWAF gets the whole stream. An unparsed
  JSON, form or XML body is read up to `max_bytes`, inspected and put back on the stream,
  so AIWAF can be mounted before the app's body parsers. Previously any unparsed body
  was refused with 413. This works with the published `aiwaf-wasm` 0.2.6.
- Rust-owned pipelines can share rate limits across workers. Redis (or another shared
  counter) records each request atomically and returns the window's count; aiwaf-rust and
  aiwaf-wasm still decide throttle and flood from it, including path-rule limits. Python
  takes `rate_backend: "redis"` with `rate_redis_url`, `"django_cache"` in Django, or
  `rate_counter=` with `rate_backend: "counter"`. Node.js's `aiwaf.native()` takes
  `rateCounter`, with `aiwaf.redisRateCounter(client)` for node-redis or ioredis, and
  `rateFailureMode`. Both use the same script and keys. A failing counter answers 503, as
  the regular limiter does. `from_settings` carries each framework's Redis rate settings and
  Node's `cache` option over. It needs `aiwaf-rust` 0.2.9 and `aiwaf-wasm` 0.2.7; older
  builds refuse a counter at startup, and `from_settings` keeps per-worker counts and says so.
- Node.js's `aiwaf.native()` can use Node's regular stores instead of its own CSV files.
  `stores: 'regular'` reads blocks, IP and path exemptions, learned keywords (past
  `dynamicTopN`, as in the regular middleware) and blocked countries from the regular
  blocklist, exemption, keyword and geo stores, from the database when configured, and
  writes new blocks and learned keywords back. `stores` also takes any object with the six
  store methods, sync or async. `from_settings` turns this on. Facts are read before the
  engine decides, with keywords and countries cached for `storeCacheSeconds`; a failing
  store read is reported and fails open, and writes run after the response. It needs
  `aiwaf-wasm` 0.2.7. With 0.2.6, `stores` fails at startup and `from_settings` keeps the
  native CSV files with a warning.
- The Python Rust-owned pipeline can use the app's own stores instead of its CSV files.
  `"store": "django"`, `"flask"` or `"fastapi"` uses that framework's regular stores, following
  its storage settings, so blocks, exemptions and learned keywords are shared with the regular
  middleware and its management commands. Through them Flask's `*.css`-style path exemptions
  and FastAPI's private-range IP exemptions work natively too. `"from_settings"` turns this on.
  Elsewhere `CSVPipeline(stores=...)` takes any object with the six store methods. The
  adapter reads the facts a decision needs before it, caching keywords and countries for
  `store_cache_seconds`, and writes the engine's blocks and keywords after it; the decision
  itself stays in aiwaf-rust. It needs `aiwaf-rust` 0.2.9. With 0.2.8, `"from_settings"`
  keeps the native CSV files and says so, and an explicit store fails at startup.
- UUID checks in the Rust-owned pipelines match the regular middleware. Node.js's
  `aiwaf.native()` takes `uuidResolver(uid, req)`, sync or async: a well-formed UUID under
  `uuid_prefix` whose record doesn't exist is blocked with `uuid_not_found`, and a failing
  resolver lets the request through. Django's native middleware now checks the route's
  `uuid` URL parameter, as the regular UUID middleware does, instead of only `?uuid=`. The
  Node.js settings converter carries `uuidResolver` and `uuidRoutePrefix` over. Both need
  `aiwaf-rust` 0.2.9 and `aiwaf-wasm` 0.2.7; older builds refuse `uuidResolver` at startup,
  and Django logs once that route UUIDs aren't checked.
- The Rust-owned pipelines can start from the settings the regular middleware already
  uses: `AIWAF_CSV_PIPELINE = "from_settings"` in Django, `NativeAIWAF(app, "from_settings")`
  in Flask, `NativeAIWAFMiddleware(app, config="from_settings")` in FastAPI (from its
  `AIWAFConfig`), and `aiwaf({ ..., AIWAF_NATIVE_PIPELINE: 'from_settings' })` in Node.js.
  Settings are read with the regular middleware's own defaults, which differ from the
  native ones in places (Django's login exemptions, Node's 60-second rate windows and
  disabled header validation), and a startup report lists every setting that didn't carry
  over exactly and why. `native_config()`, `native_config_fastapi()` and
  `aiwaf.nativeOptionsFromSettings()` return the converted options and report on their own.
- The Rust-owned request pipelines support `sql_mode: "monitor"`. SQL findings, including
  over-size bodies, are logged without blocking, as in the regular middleware: on the
  `aiwaf.payload` logger in Python, and as `{event: 'aiwaf.payload', rule, mode}` through
  the new `logger` option in Node.js, where `inspectPayload` and `inspectBody` return `null`
  for a monitored finding. It needs `aiwaf-rust` 0.2.9 and `aiwaf-wasm` 0.2.7; older
  builds fail at startup with an upgrade message.
- The Rust-owned request pipelines now support path rules and the route manifest. In
  Python (`NativeAIWAF`, `NativeAIWAFMiddleware`, `AIWAF_CSV_PIPELINE`) the new adapter keys
  are `path_rules`, `manifest_path` and `injection_risk_mode`; in Node.js (`aiwaf.native()`)
  they're `pathRules`, `manifestPath` and `injectionRiskMode`. `DISABLE` turns stages off for
  a prefix, `RATE_LIMIT` replaces the rate limits there, and the route-specific checks saved
  by `init` now run in Python's native pipeline too, as they already did in Node's. Python's
  native pipeline now reads `.aiwaf/paths.json` by default, as the regular middleware does,
  so an existing manifest with saved risks turns those checks on. `RATE_LIMIT` and Python's
  route checks need `aiwaf-rust` 0.2.9 and `aiwaf-wasm` 0.2.7; with the published
  0.2.8 and 0.2.6 they fail at startup with an upgrade message, and everything else works.

## Python 1.1.9 / aiwaf-rust 0.2.8 / Node.js 1.1.6 / WASM 0.2.6 / Java 1.3.6 — prepared release

- **Security:** Python and Node.js no longer let a client choose its own address with
  `X-Forwarded-For`, which could evade blocks and rate limits or match `AIWAF_EXEMPT_IPS`.
  Forwarded headers now count only from a trusted peer and are walked from the right.
  **Upgrade check:** proxies connecting from public addresses must be added to the new
  `AIWAF_TRUSTED_PROXIES` setting, or all visitors will appear as the proxy. Proxies on
  loopback and private networks are trusted by default. AIWAF logs a warning naming an
  untrusted peer that sends these headers, and behind Cloudflare
  `AIWAF_TRUSTED_PROXIES=default,cloudflare` is enough.
  `CF-Connecting-IP` and `Forwarded`-style fallbacks are no longer read for the client IP.
- **Security:** Java stops reading `X-Forwarded-For` at a malformed entry instead of
  skipping it, so entries left of it can't be taken as the client address. It then
  falls back to `X-Real-IP` or the peer address.
- **Security:** Node.js dependency updates for advisories published since 1.1.5:
  `body-parser` 1.20.9 and `express` 4.22.3 (denial of service through trailing bytes
  after a compressed body), `qs` 6.17.1, and `glob` 14.0.6. `npm audit --omit=dev`
  reports no vulnerabilities. Releases now fail on any known vulnerability in packages
  users install, and on high-severity ones in test-only packages; Jest's `deepmerge`
  has a low-severity advisory with no fixed version yet.
- Python, Node.js and Java: when a peer that isn't trusted sends `X-Forwarded-For` or
  `X-Real-IP`, AIWAF logs one warning per process naming that peer, so a proxy or CDN
  missing from `AIWAF_TRUSTED_PROXIES` shows up in the logs. `AIWAF_TRUSTED_PROXIES` also
  accepts two names: `default` for the runtime's built-in ranges and `cloudflare` for
  Cloudflare's published ranges. Behind Cloudflare, use `default,cloudflare`. The
  Cloudflare ranges are opt-in, since anyone can send requests through Cloudflare.
- Python and Node.js resolve the client address in `aiwaf-rust` and `aiwaf-wasm`
  (`TrustedProxySet`), with trusted networks parsed once per setting. A request with
  31 forged Cloudflare hops costs about 6 µs in Python instead of 360 µs. The
  Python and JavaScript walks remain only as fallbacks for older native packages,
  and parity tests check that both give the same answers. Node.js now writes IPv6
  addresses in canonical form, as Python does, for example `2001:db8::1` for
  `2001:DB8::1`, and an IPv4-mapped address such as `::ffff:a00:1` as IPv4.
- Java parses the trusted proxy list once into masks and caches it by its contents, instead
  of parsing every range on each lookup. With the Cloudflare list, 16 forged
  `X-Forwarded-For` hops cost about 6 µs instead of 350 µs. A plain address entry now
  matches the same address in any form, such as `2001:DB8::1` and `2001:db8::1`, and
  a prefix length must be digits only (`10.0.0.0/+8` is rejected).
- Java: trusted proxies can now be set with the `AIWAF_TRUSTED_PROXIES` environment
  variable, the `aiwaf.trusted-proxies` Spring property, or `security.trusted_proxies`
  in the JSON config file, each replacing the default. The file key takes a list or a
  comma-separated string, `save()` writes it, and loading fails on an invalid entry.
  The default is unchanged: loopback only.
- Python and Rust log parsers no longer drop or misread some valid lines
  ([#34](https://github.com/aiwaf-project/aiwaf/issues/34)). They accept `-` as the
  response size, as in `304` responses and AIWAF's own Flask and FastAPI logs, accept
  whole-number `response-time=` values, and read the trailing `5ms` in AIWAF's logs as
  0.005 seconds. Response times now come only from `response-time=` or a trailing `ms`
  value: a referer or user agent that looks like a number, such as `"1.5"`, is no longer
  read as one. Ships in `aiwaf-rust` 0.2.8, which the Python package already requires.
- Fix the middleware CSV log path so only a trailing `.log` becomes `.csv`
  (Django and Flask middleware plus the `aiwaf_logging` and `debug_csv`
  commands). A `.log` elsewhere in the path, such as in a folder name, and a
  non-`.log` extension now keep the configured path; anyone who relied on the
  old rewrite will see CSV logs move to the configured folder and may need to
  move existing files.
- Write Flask and FastAPI combined access log timestamps in UTC to match their
  `+0000` offset. On non-UTC servers, entries written before and after this fix
  differ by the server's UTC offset; existing log files are not converted.
- Java: read the first client field in access logs, retaining IPv6, IPv4-mapped
  IPv6, and hostname fields verbatim, matching the Python parser. Later IPv4
  fields no longer replace the client address during ingestion.

- Add `audit` to the Python, Node.js and Java command lines. It parses source without
  running it and reports where request input can reach dangerous operations. It's
  report-only and doesn't affect request handling.
- **Security:** SQL injection checks now use a tokenizer instead of four signature rules.
  It flags a value only when it breaks out of a SQL value and continues as SQL, which
  catches attacks with no quote such as `1 OR 1=1` and `1 AND SLEEP(5)`, and MySQL's
  executed `/*!...*/` comments, which bypassed the signature rules. It no longer blocks
  ordinary text such as `"Item" #3` or `Drop table tennis; update your set list`.
  **Upgrade check:** rule names change from `sql_union_select`, `sql_boolean_tautology`,
  `sql_quote_comment` and `sql_stacked_statement` to `sql_breakout_numeric`,
  `sql_breakout_single_quote` and `sql_breakout_double_quote`. Text such as
  `union select password` with no value to close is no longer blocked.
- **Requires `aiwaf-rust` 0.2.8.** Python's SQL check and route checks now run in Rust, with
  no Python copy: one call per request does the walking, decoding, JSON parsing, size limits
  and tokenizer. `aiwaf-rust` adds `policy_sql_inspect`, `RouteRiskChecker`,
  `policy_route_rule` and `policy_sql_tokenizer`. `policy_sql_signature` and
  `policy_sql_signatures` remain for older releases and return tokenizer results. A JSON body
  is checked by its values instead of its raw text, and a literal `+` is no longer lost to
  form decoding. Shared case files run against every copy in CI.
- **Node.js requires `aiwaf-wasm` 0.2.6 for request checks.** The SQL check and route checks
  run in `aiwaf-wasm`, from the same Rust code as Python, with no JavaScript copy.
  `aiwaf-wasm` adds `sql_inspect`, `RouteRiskChecker`, `route_rule` and `sql_tokenizer`.
  Tests and CI build `aiwaf-wasm` from this repository with
  `scripts/build_wasm_from_source.py`.
- **Security:** SQL detection was measured against the OWASP Core Rule Set and libinjection
  test data, and the gaps found were closed. libinjection detection rose from 69.8% to
  94.9%, and its false-positive list from 9 to 1 flagged value. Payloads now caught include
  `WHERE` straight after the quote, `foo) UNION ALL SELECT`, `-6863 union select`,
  `; DROP TABLE`, `-1 + (SELECT ...)`, MSSQL `WAITFOR` and `EXEC` without a semicolon,
  blind payloads such as `' and 526=527`, names found only in SQL such as `pg_sleep(` and
  `information_schema`, `%u` escapes and typographic quotes, and values in XML bodies.
  Text such as `7" #2 CIRCULAR NEEDLE`, `"IS" STAMP` and `BASE; SET SWAROVSKI` passes.
  `scripts/sql_benchmark.py` runs both test sets in CI with minimum results.
- **Security:** detection of OWASP CRS paranoia level 1 and 2 SQL injection tests rose from
  35% to 71% and from 17% to 42%, and libinjection detection to 98.9%, with no more benign
  values flagged. Newly caught: a value that starts with `AND 2771=1709)`, whole statements
  such as `SELECT * FROM users` and `DROP TABLE users;--`, `NOT LIKE` after the quote,
  `x'|email--`, `a'or'1'='1` without spaces, comments glued to keywords such as
  `SELECT/**/password` and `/*!50000`, PostgreSQL casts such as `::int`, calls such as
  `group_concat(`, `char(113)` and `select(user())`, MSSQL names such as `xp_dirtree`,
  `tempdb` and `declare @x`, Oracle packages such as `utl_http`, and MongoDB operators such
  as `[$ne]`. Code such as `await version()` and `delete Math.pow;` passes. Rust and Java
  have the same rules. Also caught: payloads ending in a null byte, as sqlmap sends,
  `ORDER BY 3631` after a breakout, column aliases such as `) AS t FROM`, `admin';`, `'x'!='y`,
  `in (select * from`, `union select col from`, `@x:=(select`, `ALTER TABLE t ADD`, MySQL's
  `1.e(` number trick, `IF((1=2),...)`, `IIF(`, `ELT(`, `CASE WHEN 1=2 THEN`, SQLite's
  `randomblob(`, PostgreSQL's `lo_import(` and `pg_sleep`, and `foo + (SELECT ... WHERE 1=1)`.
  Sizes such as `18';` and phrases such as `5 as a gift from` pass.
  Code such as `await sleep(10)`, `time.sleep(1)` and `sleep(1000)`, Markdown such as
  `` `'limit'` ``, and prose mentioning `sys.tables` now pass, while Oracle's
  `USER_LOCK.SLEEP(5)` is caught.
- **Security:** the SQL check now also reads the URL path, one segment at a time, as in
  `/post/1 OR 1=1`, and the headers: cookie names and values, the path and query string of
  `Referer`, `X-Original-URL` and `X-Rewrite-URL`, and every other header as one value, such as
  `User-Agent`, `X-Forwarded-For`, `Accept-Language` and apps' own `X-` headers. Headers holding
  credentials, framing or browser-set values (`Authorization`, `Accept`, `Content-*`, `Sec-*`,
  `If-*`, `Host`, CSRF tokens, tracing IDs) aren't checked. This applies in Python, Node.js,
  Java and both native pipelines. `aiwaf-rust` adds `policy_sql_inspect_headers` and
  `policy_sql_inspect_path`, and `aiwaf-wasm` adds `sql_inspect_headers` and
  `sql_inspect_path`. Node.js and Python now pass every header and Rust chooses which to
  check. Shared cases are in `tests/fixtures/sql_header_cases.json` and
  `tests/fixtures/sql_path_cases.json`. Python now also
  inspects XML bodies (`application/xml`, `text/xml`, `+xml`). Java's request walker now
  matches the Rust one: it decodes `%uXXXX` escapes, reads XML bodies by their values, and
  checks the same headers.
- Fuzzing for the SQL tokenizer, request walker and route checks, with cargo-fuzz targets
  in `rust/fuzz` run in CI, and random-input tests in `cargo test`. They found that `` in
  the Rust regexes was slow on long non-ASCII input; it's now ASCII-only, up to eight times
  faster on those inputs.
- `init` audits source and saves route-specific checks into the manifest. Requests to those
  routes are checked for patterns that matter to the operation their input reaches, and
  matches are blocked. `AIWAF_INJECTION_RISK_MODE` is `block` by default, and `log` or
  `off` change that. `init --no-audit` and `audit --manifest` are also available.
- Java can now read source with the JDK Compiler Tree API. `aiwaf-cli init --source`
  builds a route manifest without starting the application, including for plain Servlet
  applications, and reads method bodies for API, form and login signals. Source analysis
  needs a JDK.
- Java access-log ingestion now takes the first field of each line as the client, as Python
  does ([#45](https://github.com/aiwaf-project/aiwaf/pull/45)). Logs that put something before
  the client field, such as a syslog header, now record that prefix as the client; strip it
  before ingesting, or log the client first.
- Recognize PATCH HTTP method in Python and Rust access log parsers.
- Rust and WASM: `validate_headers_with_config` now enforces every required header. It
  previously checked only `User-Agent` and `Accept`, and only `HTTP_`-style names. The
  Django, FastAPI, Flask and Node.js middleware already enforced these headers themselves.
- CI builds `aiwaf-rust` from this repository for Python tests instead of installing it from PyPI.

## Python 1.1.8 / Rust-Python 0.2.7 / Node.js 1.1.5 / WASM 0.2.5 / Java 1.3.5 — released 2026-10-14

- Update package versions and matching native dependency requirements for the next release.
- Remove benchmark/profiling utilities and consolidate release documentation into README/CHANGELOG.
- Remove the deprecated Rust floating-point module import.
- Use blocklist/allowlist in Python, JavaScript, Rust/WASM, Java, CLI help and
  website documentation. Retain prior import, method and command aliases.
- Preserve existing CSV filenames, Django database/model identity, stored legacy
  reason tags and native wire compatibility. Accept older configuration exports
  without discarding blocks or exemptions.

## Python 1.1.7 / aiwaf-rust 0.2.6 — native honeypot cache

- Add opt-in Redis and supplied-cache honeypot timing to Flask and FastAPI's
  native pipelines. FastAPI runs cache-backed native checks in its worker thread
  pool. CSV remains the default in both adapters; other policies are unchanged.

- Default Django native honeypot timing to Django's configured cache, with an
  optional `honeypot_cache_alias`. Explicit CSV and native memory remain available.
- Delegate timestamp I/O to Django while Rust owns timing, TTL validation,
  exemptions, and block decisions. Other native policy stores retain CSV behavior.
- Return generic HTTP 503 for cache failures or invalid state. Reject DummyCache
  and configured ignored cache errors; warn when LocMemCache isolates workers.
- Leave expiry cleanup to cache TTL so a stale POST cannot delete a concurrent
  GET refresh. Until the next GET or cache expiry, repeated expired-page POSTs
  continue to return 409. Existing timestamps are not migrated between backends.

## Node.js 1.1.4 / Java 1.3.4 — blocklist expiry concurrency fixes

- Make Node blocklist request checks read-only. Database cleanup deletes only
  rows still expired when the delete executes; CSV cleanup checks the latest
  rows under the same cross-process lock used by blocklist writes.
- Reload Java CSV and file storage under a shared JVM and OS file lock for each
  operation. Listing filters expiry without rewriting disk; mutations preserve
  updates from other instances and processes and report persistence failures.
- Verify refreshed blocks survive cleanup and concurrent writers retain all
  unrelated blocks. These fixes are separate from offline training replay.

## Python 1.1.6 / aiwaf-rust 0.2.5 — training replay protection

- Make Flask CLI blocklist listing read-only. Expiry filtering no longer writes
  an older snapshot over blocks concurrently added or refreshed by a trainer.
- Display the canonical block date in CLI output instead of reporting Unknown
  when CSV entries use `added_date` rather than the old `timestamp` field.

- Persist Rust-owned enforcement receipts independently of blocklist expiry and
  manual unblocking. Reusing handled requests cannot extend or recreate a block.
- Apply 404 thresholds and anomaly enforcement to fresh evidence in Flask,
  FastAPI, and Django, while retaining full history for model fitting and learning.
- Serialize trainers with a cross-process file lock and atomically persist
  SHA-256 receipts. Preserve receipts across log rotation and application restarts.
- Require the matching native package; missing capabilities and corrupt receipt
  state fail visibly instead of silently allowing replay.

## Node.js dependency security updates

- Node.js 1.1.4 / WASM 0.2.4 share Rust-owned offline enforcement receipts with
  Python. Historical requests cannot extend or recreate handled blocks; fresh
  404 and anomaly evidence must independently satisfy enforcement thresholds.
- Persist Node receipts across restarts and serialize offline trainers. Report
  missing native APIs, invalid state, and storage failures with a CLI failure.
- Keep Java model training free of blocklist side effects, verified by a new
  runtime-state regression test. File storage concurrency is fixed separately.

- Upgrade Jest, Fastify, and the NestJS test dependencies to patched releases.
- Replace the direct deprecated glob 8 dependency with maintained glob 13.
- Remove broad transitive overrides that forced incompatible parser versions;
  apply targeted fixes for qs and the NYC loader's YAML dependency instead.
- Audit development dependencies as well as production dependencies in npm
  release validation.

## Python 1.1.4 

- Detect the single installed framework for bare CLI commands; require an
  explicit choice when detection is ambiguous. Add the `fastapi` CLI alias.
- Reuse existing project and user CSV directories in the Flask management CLI.
- Standardize framework CSV initialization on shared policy files. Import
  legacy runtime CSV data at startup, preserving expiry times and metadata,
  retaining original sources and backups, and preventing migration replay.
- Preserve keyword frequencies and runtime history in auxiliary metadata.
- Use UTF-8 for Python fallback CSV I/O to match Rust reads and writes.
- Retain `aiwaf-rust>=0.2.3`; no new native API is required for these changes.

## Next releases — ELv2 migration

- Python and Node.js 1.1.2, Rust/Python and WASM 0.2.2, and Java 1.3.3 adopt
  standard Elastic License 2.0 for original project code, with retained MIT
  contributions and separately licensed IPinfo Lite data identified in NOTICE.
- Preserve the earlier MIT permissions and notices; do not retroactively
  replace licenses on previously distributed releases.
- Include LICENSE, NOTICE, and licenses/ in release packaging, and align
  package license metadata and WASM generation with the new terms.
- Synchronize the Python runtime version with its package metadata. The
  internal Rust core crate advances to 0.2.1.

See the [README license section](README.md#license) for version boundaries
and the release automation section for validation requirements.

## Java 1.3.0

- Hardened Redis operation with fail-open/fail-closed policies, bounded
  connection and pool waits, latency/pool diagnostics, non-blocking key scans,
  and state-schema compatibility guards for rolling upgrades. Java 1.3
  explicitly supports standalone Redis and rejects Sentinel/Cluster modes.

## Java 1.2.0

- Added scoped per-engine storage contexts and a Redis backend with atomic,
  distributed rate-limit, honeypot, UUID-score, and anomaly-history state.
- Aligned path, IP, geo-only, and route-level logging exemption behavior with
  Python.
- Added Python-style JSON configuration loading, environment overrides, deep
  merge, validation, saving, and broader Spring property coverage.
- Aligned Java defaults for private-IP exemptions and anomaly detection with
  Python.
- Added Spring Boot servlet auto-configuration with live MVC route, annotation,
  and JPA UUID-field discovery.
- Load generated path manifests at runtime while preserving explicit path-rule
  precedence.
- Replaced the external `mmdblookup` process with the native Java MaxMind DB
  reader and bounded country caching.
- Added Python-compatible reputation weights, metadata, and progressive
  15-minute, 1-hour, and 24-hour default blocks; explicit permanent blocking
  remains available.
- Added Spring property and environment configuration for the new runtime
  behavior.

## 1.0.8

- Require `aiwaf-rust>=0.2.1` for the optional Rust extra.
- Normalize HTTP headers before Rust validation so valid FastAPI requests are not blocked.
- Skip the Rust header validator when FastAPI's Rust path is disabled.

## 1.0.7

- Added reputation-based IP blocking with weighted offenses, progressive
  temporary blocks, expiration handling, and richer stored metadata.
- Added automatic legacy blocklist compatibility and backend-aware migration
  commands for Django, Flask, and FastAPI.
- Added legacy CSV schema detection and safe conversion of imported permanent
  blocks.
- Added request payload-field inference to generated path manifests.
- Enabled installed `aiwaf_rust` capabilities automatically, with Python
  fallback when the extension or a specific capability is unavailable.
