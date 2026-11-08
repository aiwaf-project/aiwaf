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
