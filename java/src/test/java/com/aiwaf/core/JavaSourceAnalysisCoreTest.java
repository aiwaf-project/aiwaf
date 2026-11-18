package com.aiwaf.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.Arguments;
import java.util.stream.Stream;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

class JavaSourceAnalysisCoreTest {
    @TempDir
    Path directory;

    private JavaSourceAnalysisCore.Report analyze(String source) throws IOException {
        Files.writeString(directory.resolve("App.java"), source);
        return JavaSourceAnalysisCore.analyze(directory, false);
    }

    private static String spring(String body) {
        return """
                @RestController
                class App {
                    @GetMapping("/x")
                    String view(@RequestParam String value) {
                        %s
                        return "ok";
                    }
                }
                """.formatted(body);
    }

    private List<String> sinks(String body) throws IOException {
        return analyze(spring(body)).findings().stream().map(JavaSourceAnalysisCore.Finding::sink).toList();
    }

    @Test
    void springInputReachingSqlIsReported() throws IOException {
        JavaSourceAnalysisCore.Finding finding = analyze(spring("jdbcTemplate.queryForList(\"SELECT \" + value);")).findings().get(0);
        assertEquals("sql", finding.sink());
        assertEquals("jdbcTemplate.queryForList", finding.call());
        assertEquals("@RequestParam value", finding.source());
        assertEquals("App.view", finding.function());
        assertEquals(5, finding.line());
    }

    @Test
    void servletInputReachingSqlIsReported() throws IOException {
        JavaSourceAnalysisCore.Report report = analyze("""
                class App extends HttpServlet {
                    protected void doGet(HttpServletRequest req, HttpServletResponse res) {
                        String id = req.getParameter("id");
                        statement.executeQuery("SELECT " + id);
                    }
                }
                """);
        assertEquals("req.getParameter(\"id\")", report.findings().get(0).source());
        assertEquals("sql", report.findings().get(0).sink());
    }

    @ParameterizedTest
    @CsvSource(delimiter = '|', textBlock = """
            statement.executeQuery("SELECT " + value);          | sql
            conn.prepareStatement("SELECT " + value);           | sql
            entityManager.createNativeQuery(value);             | sql
            entityManager.createQuery("from T where " + value); | sql
            jdbcTemplate.update(String.format("x %s", value));  | sql
            new BasicQuery(value);                              | nosql
            Document.parse(value);                              | nosql
            Runtime.getRuntime().exec(value);                   | command
            new ProcessBuilder("sh", "-c", value);              | command
            new File(value);                                    | path
            new FileInputStream(value);                         | path
            Files.readString(Path.of(value));                   | path
            Paths.get("/data", value);                          | path
            Files.readString(base.resolve(value));              | path
            new File("/data", value);                           | path
            restTemplate.getForObject(value, String.class);     | ssrf
            new URL(value);                                     | ssrf
            URI.create(value);                                  | ssrf
            engine.eval(value);                                 | code
            parser.parseExpression(value);                      | code
            yaml.load(value);                                   | deserialization
            new ObjectInputStream(value);                       | deserialization
            """)
    void eachSinkTypeIsReported(String statement, String sink) throws IOException {
        assertTrue(sinks(statement).contains(sink), () -> statement + " should report " + sink);
    }

    @ParameterizedTest
    @ValueSource(strings = {
        "PreparedStatement p = conn.prepareStatement(\"SELECT * FROM t WHERE id = ?\"); p.setString(1, value);",
        "jdbcTemplate.query(\"SELECT * FROM t WHERE id = ?\", mapper, value);",
        "executor.execute(value);",
        "logger.info(value);",
        "value.isEmpty();"
    })
    void safeUsesAreNotReported(String statement) throws IOException {
        assertEquals(List.of(), sinks(statement));
    }

    @ParameterizedTest
    @ValueSource(strings = {"Integer.parseInt(value)", "Long.valueOf(value)", "UUID.fromString(value)", "FilenameUtils.getName(value)"})
    void sanitizersStopTracking(String cleaner) throws IOException {
        assertEquals(List.of(), sinks("var clean = " + cleaner + "; statement.executeQuery(\"x\" + clean);"));
    }

    @Test
    void inputIsTrackedThroughAssignmentsAndStringBuilders() throws IOException {
        assertEquals(List.of("sql"), sinks("""
                String first = value.trim();
                StringBuilder sql = new StringBuilder("SELECT ");
                sql.append(first);
                statement.executeQuery(sql.toString());
                """));
    }

    @Test
    void overwritingAValueStopsTrackingIt() throws IOException {
        assertEquals(List.of(), sinks("String a = value; a = \"safe\"; Runtime.getRuntime().exec(a);"));
    }

    @Test
    void overwritingInABranchKeepsTrackingIt() throws IOException {
        assertEquals(List.of("command"), sinks("String a = value; if (a.isEmpty()) { a = \"safe\"; } Runtime.getRuntime().exec(a);"));
    }

    @Test
    void inputIsFollowedIntoHelpers() throws IOException {
        JavaSourceAnalysisCore.Report report = analyze("""
                @RestController
                class App {
                    @GetMapping("/x")
                    String view(@RequestParam String value) { outer(value); return ""; }
                    private void outer(String text) { run("SELECT " + text); }
                    private void run(String sql) { statement.execute(sql); }
                }
                """);
        assertEquals("App.view -> outer -> run", report.findings().get(0).function());
    }

    @Test
    void aHelpersReturnValueCarriesItsInput() throws IOException {
        JavaSourceAnalysisCore.Report report = analyze("""
                @RestController
                class App {
                    @GetMapping("/u") String v(@RequestParam String n) { jdbcTemplate.queryForList(build(n)); return "ok"; }
                    @GetMapping("/s") String s(@RequestParam String n) { jdbcTemplate.queryForList("x" + clean(n)); return "ok"; }
                    private String build(String n) { Runnable r = () -> { return; }; return "SELECT " + n; }
                    private int clean(String n) { return Integer.parseInt(n); }
                }
                """);
        assertEquals(List.of("App.v"), report.findings().stream().map(JavaSourceAnalysisCore.Finding::function).toList());
    }

    @Test
    void aCollectionBuiltInALoopCarriesItsInput() throws IOException {
        assertEquals(List.of("sql"), analyze("""
                @RestController
                class App {
                    @GetMapping("/u") String v(@RequestParam List<String> ids) {
                        List<String> parts = new ArrayList<>();
                        for (String id : ids) { parts.add(id); }
                        jdbcTemplate.queryForList("SELECT " + String.join(",", parts));
                        return "ok";
                    }
                }
                """).findings().stream().map(JavaSourceAnalysisCore.Finding::sink).toList());
    }

    @Test
    void valuesSpringConvertsToNumbersOrUuidsAreNotInput() throws IOException {
        JavaSourceAnalysisCore.Report report = analyze("""
                @RestController
                class App {
                    @GetMapping("/a/{id}") String a(@PathVariable Long id) { jdbcTemplate.queryForList("SELECT " + id); return ""; }
                    @GetMapping("/b") String b(@RequestParam int page, @RequestParam UUID token, @RequestParam Optional<Integer> n,
                                               @RequestParam List<Long> ids, @RequestParam long[] more) {
                        jdbcTemplate.queryForList("SELECT " + page + token + n + ids + more);
                        return "";
                    }
                    @GetMapping("/c") String c(@RequestParam String name, @RequestParam Optional<String> other) {
                        jdbcTemplate.queryForList("SELECT " + name + other);
                        return "";
                    }
                }
                """);
        assertEquals(List.of("App.c"), report.findings().stream().map(JavaSourceAnalysisCore.Finding::function).toList());
    }

    @Test
    void callsIntoOtherClassesAreFollowedAndNameTheHandlersFile() throws IOException {
        Files.createDirectories(directory.resolve("svc"));
        Files.writeString(directory.resolve("svc/UserService.java"), """
                class UserService {
                    void search(String term) { jdbcTemplate.queryForList("SELECT * FROM t WHERE n = '" + term + "'"); }
                    String harmless(String term) { return term; }
                }
                """);
        Files.writeString(directory.resolve("svc/Queries.java"), "class Queries { static void run(String q) { statement.executeQuery(q); } }");
        Files.writeString(directory.resolve("svc/Repo.java"), """
                interface Repo { void find(String q); }
                class RepoImpl implements Repo { public void find(String q) { jdbcTemplate.queryForList("SELECT " + q); } }
                """);
        JavaSourceAnalysisCore.Report report = analyze("""
                @RestController
                class App {
                    private final UserService users;
                    @Autowired private Repo repo;
                    @GetMapping("/search") String s(@RequestParam String q) { users.search(q); return ""; }
                    @GetMapping("/fields") String f(@RequestParam String q) { this.users.search(q); return ""; }
                    @GetMapping("/harmless") String h(@RequestParam String q) { users.harmless(q); return ""; }
                    @GetMapping("/static") String t(@RequestParam String q) { Queries.run(q); return ""; }
                    @GetMapping("/repo") String r(@RequestParam String q) { repo.find(q); return ""; }
                    @GetMapping("/local") String l(@RequestParam String q) { UserService other = make(); other.search(q); return ""; }
                }
                """);
        assertEquals(List.of("App.t -> run", "App.r -> find", "App.f -> search", "App.l -> search", "App.s -> search"),
                report.findings().stream().map(JavaSourceAnalysisCore.Finding::function).toList());
        for (JavaSourceAnalysisCore.Finding finding : report.findings()) {
            assertTrue(finding.entryFile().endsWith("/App.java"), finding.toString());
            assertFalse(finding.file().endsWith("/App.java"), finding.toString());
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> first = ((List<Map<String, Object>>) report.toMap().get("findings")).get(0);
        assertTrue(String.valueOf(first.get("entry_file")).endsWith("/App.java"));
    }

    @Test
    void callsOnReturnedValuesAndEveryImplementationAreFollowed() throws IOException {
        JavaSourceAnalysisCore.Report report = analyze("""
                @RestController
                class App {
                    private final Services services;
                    private final Store store;
                    @GetMapping("/a") String a(@RequestParam String q) { services.users().search(q); return ""; }
                    @GetMapping("/b") String b(@RequestParam String q) { var u = services.users(); u.search(q); return ""; }
                    @GetMapping("/c") String c(@RequestParam String q) { var u = new UserDao(); u.search(q); return ""; }
                    @GetMapping("/d") String d(@RequestParam String q) { store.save(q); return ""; }
                    @GetMapping("/e") String e(@RequestParam String q) { ((UserDao) any()).search(q); return ""; }
                    @GetMapping("/f") String f(@RequestParam String q) { new UserDao().execute(q); return ""; }
                }
                class Services { UserDao users() { return new UserDao(); } }
                class UserDao {
                    void search(String q) { jdbcTemplate.queryForList("SELECT " + q); }
                    void execute(String q) { log.info(q); }
                }
                interface Store { void save(String v); }
                class SqlStore implements Store { public void save(String v) { statement.executeUpdate("INSERT " + v); } }
                class FileStore implements Store { public void save(String v) { new File(v); } }
                class MemoryStore implements Store { public void save(String v) { map.put(v, v); } }
                """);
        assertEquals(List.of("App.a -> search", "App.b -> search", "App.c -> search", "App.e -> search", "App.d -> save", "App.d -> save"),
                report.findings().stream().map(JavaSourceAnalysisCore.Finding::function).toList());
        assertEquals(List.of("sql", "sql", "sql", "sql", "sql", "path"),
                report.findings().stream().map(JavaSourceAnalysisCore.Finding::sink).toList());
    }

    @Test
    void interfacesWithManyImplementationsAreNotFollowed() throws IOException {
        StringBuilder source = new StringBuilder("""
                @RestController
                class App {
                    private final Store store;
                    @GetMapping("/d") String d(@RequestParam String q) { store.save(q); return ""; }
                }
                interface Store { void save(String v); }
                """);
        for (int index = 0; index < 6; index++) {
            source.append("class Store").append(index).append(" implements Store { public void save(String v) { new File(v); } }\n");
        }
        assertEquals(0, analyze(source.toString()).findings().size());
    }

    static Stream<Arguments> flowCases() {
        return Stream.of(
                Arguments.of("response writer", List.of("xss"), "response.getWriter().println(value);"),
                Arguments.of("encoded for HTML", List.of(), "response.getWriter().println(ESAPI.encoder().encodeForHTML(value));"),
                Arguments.of("encoded for HTML, then used in SQL", List.of("sql"), "String v = ESAPI.encoder().encodeForHTML(value); jdbcTemplate.update(\"DELETE \" + v);"),
                Arguments.of("redirect", List.of("redirect"), "response.sendRedirect(value);"),
                Arguments.of("LDAP search", List.of("ldap"), "ctx.search(\"ou=users\", \"(uid=\" + value + \")\", controls);"),
                Arguments.of("LDAP search encoded", List.of(), "ctx.search(\"ou=users\", \"(uid=\" + ESAPI.encoder().encodeForLDAP(value) + \")\", controls);"),
                Arguments.of("XPath", List.of("xpath"), "javax.xml.xpath.XPath xp = factory.newXPath(); xp.evaluate(\"/u[@id='\" + value + \"']\", doc);"),
                Arguments.of("regular expression", List.of("redos"), "java.util.regex.Pattern.compile(value);"),
                Arguments.of("regular expression quoted", List.of(), "java.util.regex.Pattern.compile(java.util.regex.Pattern.quote(value));"),
                Arguments.of("XML parser", List.of("xxe"), "builder.parse(new java.io.ByteArrayInputStream(value.getBytes()));"),
                Arguments.of("XML parser hardened", List.of(), "factory.setFeature(\"http://apache.org/xml/features/disallow-doctype-decl\", true); builder.parse(new java.io.ByteArrayInputStream(value.getBytes()));"),
                Arguments.of("regex check that returns", List.of(), "if (!value.matches(\"[a-z]+\")) return \"\"; Runtime.getRuntime().exec(value);"),
                Arguments.of("allowlist branch", List.of(), "if (ALLOWED.contains(value)) { jdbcTemplate.update(\"ORDER BY \" + value); }"),
                Arguments.of("dot-dot check makes a path safe, not SQL", List.of("sql"), "if (value.contains(\"..\")) throw new IllegalArgumentException(); new java.io.File(value); jdbcTemplate.update(value);"),
                Arguments.of("constant condition", List.of(), "String bar; int num = 106; if ((7 * 18) + num > 200) bar = \"safe\"; else bar = value; jdbcTemplate.update(bar);"),
                Arguments.of("switch on a known value", List.of(), "String bar; switch (\"ABC\".charAt(1)) { case 'A': bar = value; break; case 'B': bar = \"bob\"; break; default: bar = value; } jdbcTemplate.update(bar);"),
                Arguments.of("list position", List.of(), "java.util.List<String> l = new java.util.ArrayList<>(); l.add(\"safe\"); l.add(value); l.add(\"more\"); l.remove(0); jdbcTemplate.update(l.get(1));"),
                Arguments.of("map key", List.of(), "java.util.Map<String, String> m = new java.util.HashMap<>(); m.put(\"a\", \"safe\"); m.put(\"b\", value); jdbcTemplate.update(m.get(\"a\"));"),
                Arguments.of("map key read back", List.of("sql"), "java.util.Map<String, String> m = new java.util.HashMap<>(); m.put(\"a\", \"safe\"); m.put(\"b\", value); jdbcTemplate.update(m.get(\"b\"));"),
                Arguments.of("lambda on input", List.of("command"), "java.util.Optional.of(value).ifPresent(v -> Runtime.getRuntime().exec(v));"));
    }

    // Each case: what it shows, the operations reported, and the code. Safe look-alikes report nothing.
    @ParameterizedTest(name = "{0}")
    @MethodSource("flowCases")
    void inputFlowsAndTheirSafeLookAlikes(String name, List<String> expected, String body) throws IOException {
        String source = """
                @RestController
                class App {
                    @GetMapping("/x")
                    String view(@RequestParam String value, HttpServletResponse response) throws Exception {
                        %s
                        return "ok";
                    }
                }
                """.formatted(body);
        assertEquals(expected, analyze(source).findings().stream().map(JavaSourceAnalysisCore.Finding::sink).distinct().sorted().toList());
    }

    @Test
    void frameworkInputAndStoredAttributesAreFollowed() throws IOException {
        JavaSourceAnalysisCore.Report report = analyze("""
                @Controller
                class Gql {
                    @QueryMapping String user(@Argument String id) { jdbcTemplate.queryForList("SELECT " + id); return ""; }
                    @QueryMapping String count(@Argument int n) { jdbcTemplate.queryForList("SELECT " + n); return ""; }
                    @DgsQuery String dgs(@InputArgument String q) { jdbcTemplate.queryForList("SELECT " + q); return ""; }
                }
                @Controller
                class Chat {
                    @MessageMapping("/chat") void send(String message, Principal user) { Runtime.getRuntime().exec(message); }
                    @MessageMapping("/room/{id}") void room(@DestinationVariable String id) { jdbcTemplate.update("DELETE " + id); }
                }
                @ServerEndpoint("/ws")
                class Socket { @OnMessage public void onMessage(String text, Session session) { new java.io.File(text); } }
                class Handlers {
                    Mono<ServerResponse> find(ServerRequest request) {
                        String q = request.queryParam("q").orElse("");
                        jdbcTemplate.queryForList("SELECT " + q);
                        return request.bodyToMono(String.class).map(body -> { Runtime.getRuntime().exec(body); return body; }).then();
                    }
                }
                class Later extends HttpServlet {
                    protected void doGet(HttpServletRequest req, HttpServletResponse res) {
                        String t = (String) req.getAttribute("term");
                        jdbcTemplate.update("DELETE " + t);
                    }
                }
                class Filter extends HttpServlet {
                    protected void doGet(HttpServletRequest req, HttpServletResponse res) { req.setAttribute("term", req.getParameter("q")); }
                }
                """);
        assertEquals(List.of("Chat.room", "Chat.send", "Gql.dgs", "Gql.user", "Handlers.find", "Handlers.find", "Later.doGet", "Socket.onMessage"),
                report.findings().stream().map(JavaSourceAnalysisCore.Finding::function).sorted().toList());
    }

    @Test
    void ignoreCommentsBaselinesAndSarif() throws IOException {
        Path file = directory.resolve("App.java");
        Files.writeString(file, """
                @RestController
                class App {
                    @GetMapping("/x") String v(@RequestParam String a, @RequestParam String b, @RequestParam String c) throws Exception {
                        Runtime.getRuntime().exec(a);
                        Runtime.getRuntime().exec(b); // aiwaf: ignore[command]
                        // aiwaf: ignore
                        jdbcTemplate.update("DELETE " + c);
                        jdbcTemplate.update("DELETE " + a); // aiwaf: ignore[xss]
                        return "";
                    }
                }
                """);
        JavaSourceAnalysisCore.Report report = JavaSourceAnalysisCore.analyze(directory, false);
        assertEquals(List.of(4L, 8L), report.findings().stream().map(JavaSourceAnalysisCore.Finding::line).toList());
        assertEquals(2, report.suppressed());
        assertEquals(2, report.toMap().get("suppressed"));
        assertTrue(report.toText().contains("Left out 2 marked with an aiwaf: ignore comment."));

        Map<String, Object> baseline = report.toMap();
        Files.writeString(file, Files.readString(file).replace("return \"\";", "new java.io.File(b);\n        return \"\";"));
        JavaSourceAnalysisCore.Report later = JavaSourceAnalysisCore.applyBaseline(JavaSourceAnalysisCore.analyze(directory, false), baseline);
        assertEquals(List.of("path"), later.findings().stream().map(JavaSourceAnalysisCore.Finding::sink).toList());
        assertEquals(2, later.baselined());
        assertTrue(later.toText().contains("Left out 2 already in the baseline."));

        Map<String, Object> sarif = JavaSourceAnalysisCore.toSarif(report);
        assertEquals("2.1.0", sarif.get("version"));
        @SuppressWarnings("unchecked")
        Map<String, Object> run = ((List<Map<String, Object>>) sarif.get("runs")).get(0);
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> results = (List<Map<String, Object>>) run.get("results");
        assertEquals(List.of("aiwaf.command", "aiwaf.sql"), results.stream().map(item -> item.get("ruleId")).toList());
        assertEquals("error", results.get(0).get("level"));
    }

    @Test
    void recursiveHelpersFinish() throws IOException {
        assertEquals(1, analyze("""
                class App {
                    String view(@RequestParam String value) { loop(value); return ""; }
                    void loop(String v) { loop(v); Runtime.getRuntime().exec(v); }
                }
                """).findings().size());
    }

    @Test
    void methodsWithoutRequestInputAreIgnored() throws IOException {
        assertEquals(0, analyze("class App { void task(String v) { Runtime.getRuntime().exec(v); } }").findings().size());
    }

    @Test
    void jaxRsParametersAreInput() throws IOException {
        assertEquals(1, analyze("class App { String find(@QueryParam(\"a\") String a) { return em.createNativeQuery(a).toString(); } }").findings().size());
    }

    @Test
    void springRoutesAreFoundWithoutLaunchingTheApp() throws IOException {
        JavaSourceAnalysisCore.Report report = analyze("""
                @RestController
                @RequestMapping("/api/items")
                class App {
                    @PostMapping("/{id}")
                    String save(@RequestBody Item item) { return ""; }
                    @RequestMapping(path = "/legacy", method = RequestMethod.PUT)
                    String legacy() { return ""; }
                    void notARoute() {}
                }
                """);
        assertEquals(List.of("/api/items/legacy", "/api/items/{id}"), report.routes().stream().map(JavaSourceAnalysisCore.Route::path).toList());
        JavaSourceAnalysisCore.Route save = report.routes().get(1);
        assertEquals(List.of("POST"), save.methods());
        assertEquals("App.save", save.view());
        assertTrue(save.api().isApi());
        assertTrue(save.api().requestBody());
        assertEquals(List.of("PUT"), report.routes().get(0).methods());
    }

    @Test
    void servletRoutesAreFoundFromSource() throws IOException {
        JavaSourceAnalysisCore.Report report = analyze("""
                @WebServlet(urlPatterns = {"/orders", "/orders/all"})
                class App extends HttpServlet {
                    protected void doGet(HttpServletRequest req, HttpServletResponse res) { res.setContentType("application/json"); }
                    protected void doPost(HttpServletRequest req, HttpServletResponse res) { req.getInputStream(); }
                }
                """);
        assertEquals(List.of("/orders", "/orders/all"), report.routes().stream().map(JavaSourceAnalysisCore.Route::path).toList());
        JavaSourceAnalysisCore.Route route = report.routes().get(0);
        assertEquals(List.of("GET", "POST"), route.methods().stream().sorted().toList());
        assertTrue(route.api().signals().contains("response:json"));
        assertTrue(route.api().requestBody());
    }

    @Test
    void methodBodiesRevealLoginAndFormRoutes() throws IOException {
        JavaSourceAnalysisCore.Report report = analyze("""
                @Controller
                class App {
                    @PostMapping("/session")
                    String signIn(HttpServletRequest request) {
                        authManager.authenticate(token);
                        request.getParameter("user");
                        return "home";
                    }
                }
                """);
        JavaSourceAnalysisCore.Route route = report.routes().get(0);
        assertEquals(true, route.auth().get("is_auth"));
        assertTrue(route.api().formConfidence() > 0.0);
    }

    @Test
    void sourceRoutesBuildASharedFormatManifest() throws IOException {
        JavaSourceAnalysisCore.Report report = analyze("""
                @Controller
                class App {
                    @PostMapping("/session")
                    String signIn() { authManager.authenticate(token); return "home"; }
                    @RestController
                    static class Api {
                        @GetMapping("/data")
                        String data() { return ""; }
                    }
                }
                """);
        Map<String, Object> manifest = PathManifestCore.buildManifestFromSource("java-source", report.routes());
        assertEquals("java-source", manifest.get("framework"));
        @SuppressWarnings("unchecked")
        Map<String, Map<String, Object>> routes = (Map<String, Map<String, Object>>) manifest.get("routes");
        assertEquals("auth", routes.get("/session").get("category"));
        assertEquals("api", routes.get("/data").get("category"));
        assertEquals("App.signIn", routes.get("/session").get("view"));

        Path written = PathManifestCore.writeManifest(manifest, directory.resolve("paths.json"));
        assertFalse(PathManifestCore.compileManifestToPathRules(PathManifestCore.loadManifest(written)).isEmpty());
    }

    @Test
    void parseErrorsAreReportedAndOtherFilesStillScan() throws IOException {
        Files.writeString(directory.resolve("Bad.java"), "class {");
        Files.writeString(directory.resolve("Good.java"), spring("Runtime.getRuntime().exec(value);"));
        JavaSourceAnalysisCore.Report report = JavaSourceAnalysisCore.analyze(directory, false);
        assertEquals(1, report.filesScanned());
        assertEquals(1, report.errors().size());
        assertEquals("command", report.findings().get(0).sink());
    }

    @Test
    void testsAndBuildOutputAreSkippedUnlessAsked() throws IOException {
        for (String name : List.of("src/test/java/A.java", "target/B.java", "src/main/CTest.java")) {
            Path file = directory.resolve(name);
            Files.createDirectories(file.getParent());
            Files.writeString(file, spring("Runtime.getRuntime().exec(value);"));
        }
        assertEquals(0, JavaSourceAnalysisCore.analyze(directory, false).filesScanned());
        assertEquals(2, JavaSourceAnalysisCore.analyze(directory, true).filesScanned());
    }

    @Test
    void oneFileCanBeScanned() throws IOException {
        Path file = directory.resolve("One.java");
        Files.writeString(file, spring("new File(value);"));
        assertEquals(1, JavaSourceAnalysisCore.analyze(file, false).findings().size());
    }

    @Test
    void reportsUseTheSharedFormatAndAreReadable() throws IOException {
        JavaSourceAnalysisCore.Report report = analyze(spring("new File(value);"));
        Map<String, Object> map = report.toMap();
        assertEquals(List.of("schema_version", "language", "files_scanned", "findings", "errors"), List.copyOf(map.keySet()));
        assertEquals("java", map.get("language"));
        @SuppressWarnings("unchecked")
        Map<String, Object> finding = ((List<Map<String, Object>>) map.get("findings")).get(0);
        assertEquals(List.of("file", "line", "function", "sink", "call", "source", "confidence"), List.copyOf(finding.keySet()));
        assertTrue(report.toText().contains("@RequestParam value reaches new File in App.view"));
        assertTrue(report.toText().contains("found 1 possible"));
    }

    @Test
    void aCleanReportSaysItIsNotProof() throws IOException {
        JavaSourceAnalysisCore.Report report = analyze("class App {}");
        assertTrue(report.toText().contains("doesn't prove the code is safe"));
        assertFalse(report.toText().contains("App.java:"));
    }
}
