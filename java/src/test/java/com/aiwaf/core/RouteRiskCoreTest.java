package com.aiwaf.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

class RouteRiskCoreTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String VIEW = """
            @RestController
            class App {
                @GetMapping("/search")
                String view(@RequestParam("q") String value) throws Exception {
                    Runtime.getRuntime().exec(value);
                    return "ok";
                }
            }
            """;

    @TempDir
    Path directory;

    private final List<String> logged = new ArrayList<>();
    private final Handler handler = new Handler() {
        @Override
        public void publish(LogRecord record) {
            logged.add(record.getMessage());
        }

        @Override
        public void flush() {}

        @Override
        public void close() {}
    };

    @BeforeEach
    void setUp() {
        RouteRiskCore.clearCache();
        Logger.getLogger("com.aiwaf.injection_risk").addHandler(handler);
    }

    @AfterEach
    void tearDown() {
        Logger.getLogger("com.aiwaf.injection_risk").removeHandler(handler);
        RouteRiskCore.clearCache();
    }

    private static Map<String, Object> risk(String sink, String input) {
        Map<String, Object> risk = new HashMap<>();
        risk.put("sink", sink);
        risk.put("input", input);
        risk.put("confidence", "medium");
        return risk;
    }

    private static Map<String, Object> manifest(String route, Map<String, Object> risk) {
        Map<String, Object> entry = new HashMap<>();
        entry.put("view", "App.view");
        entry.put("injection_risks", List.of(risk));
        Map<String, Object> routes = new HashMap<>();
        routes.put(route, entry);
        routes.put("/health", new HashMap<>(Map.of("view", "App.health")));
        Map<String, Object> manifest = new HashMap<>();
        manifest.put("routes", routes);
        return manifest;
    }

    private static List<Map<String, String>> inspect(Map<String, Object> manifest, String path, Map<String, String> query, String body) {
        return new RouteRiskCore.Risks(manifest).inspect(path, query, body);
    }

    private static JsonNode cases() throws IOException {
        return MAPPER.readTree(Path.of("..", "tests", "fixtures", "route_risk_cases.json").toFile());
    }

    @Test
    void sharedRuleCases() throws IOException {
        for (JsonNode item : cases().get("rules")) {
            JsonNode expected = item.get("rule");
            assertEquals(expected.isNull() ? null : expected.asText(),
                    RouteRiskCore.ruleFor(item.get("sink").asText(), item.get("value").asText()),
                    () -> item.toString());
        }
    }

    @Test
    void sharedRouteCases() throws IOException {
        for (JsonNode item : cases().get("routes")) {
            RouteRiskCore.CompiledRoute route = RouteRiskCore.compileRoute(item.get("route").asText());
            Matcher match = route.pattern().matcher(item.get("path").asText());
            if (item.get("values").isNull()) {
                assertFalse(match.matches(), item::toString);
            } else {
                assertTrue(match.matches(), item::toString);
                Map<String, String> found = new HashMap<>();
                for (int index = 0; index < route.names().size(); index++) {
                    if (!route.names().get(index).isEmpty()) found.put(route.names().get(index), match.group(index + 1));
                }
                assertEquals(MAPPER.convertValue(item.get("values"), Map.class), found, item::toString);
            }
        }
    }

    @Test
    void onlyTheNamedInputIsChecked() {
        Map<String, Object> saved = manifest("/search", risk("path", "file"));
        assertEquals(List.of(), inspect(saved, "/search", Map.of("other", "../x"), ""));
        assertEquals(List.of(Map.of("route", "/search", "sink", "path", "input", "file", "rule", "path_traversal", "confidence", "medium")),
                inspect(saved, "/search", Map.of("file", "../x"), ""));
    }

    @Test
    void anUnknownInputChecksEverything() {
        Map<String, Object> saved = manifest("/search", risk("command", ""));
        assertEquals("command_substitution", inspect(saved, "/search", Map.of(), "{\"deep\": {\"cmd\": \"$(id)\"}}").get(0).get("rule"));
        assertEquals("command_substitution", inspect(saved, "/search", Map.of("a", "$(id)"), null).get(0).get("rule"));
    }

    @Test
    void nestedValuesCountAsTheFieldThatHoldsThem() {
        Map<String, Object> saved = manifest("/search", risk("nosql", "user"));
        assertEquals("nosql_operator", inspect(saved, "/search", Map.of(), "{\"user\": {\"$ne\": 1}}").get(0).get("rule"));
        assertEquals("nosql_operator", inspect(saved, "/search", Map.of("user[$ne]", "1"), "").get(0).get("rule"));
        assertEquals("nosql_operator", inspect(saved, "/search", Map.of(), "user%5B%24ne%5D=1").get(0).get("rule"));
    }

    @Test
    void onlyTheOperationsRulesApplyAndRouteValuesAreInputs() {
        assertEquals(List.of(), inspect(manifest("/search", risk("path", "q")), "/search", Map.of("q", "$(id)"), ""));
        assertEquals("path_traversal", inspect(manifest("/files/{name:path}", risk("path", "name")), "/files/../x", Map.of(), "").get(0).get("rule"));
    }

    @Test
    void routesWithoutSavedRisksGetNothing() {
        Map<String, Object> saved = manifest("/search", risk("path", ""));
        assertEquals(List.of(), inspect(saved, "/health", Map.of("x", "../y"), ""));
        assertEquals(List.of(), inspect(saved, "/other", Map.of("x", "../y"), ""));
        assertTrue(new RouteRiskCore.Risks(null).isEmpty());
        assertTrue(new RouteRiskCore.Risks(Map.of("routes", "bad")).isEmpty());
    }

    @Test
    void exactRoutesWinAndRepeatsReportOnce() {
        Map<String, Object> routes = new HashMap<>();
        routes.put("/items/{id}", Map.of("injection_risks", List.of(risk("path", ""))));
        routes.put("/items/new", Map.of("injection_risks", List.of(risk("command", ""))));
        assertEquals("/items/new", inspect(Map.of("routes", routes), "/items/new", Map.of("x", "$(id)"), "").get(0).get("route"));
        assertEquals(1, inspect(manifest("/search", risk("path", "")), "/search", Map.of("a", "../x", "b", "../y"), "").size());
    }

    @Test
    void requestInputsReadEachFormatAndAreBounded() {
        assertTrue(RouteRiskCore.requestInputs(null, "a=b%20c").contains(new RouteRiskCore.Input("a", "b c")));
        assertTrue(RouteRiskCore.requestInputs(null, "plain").contains(new RouteRiskCore.Input("", "plain")));
        assertTrue(RouteRiskCore.requestInputs(null, "{broken").contains(new RouteRiskCore.Input("", "{broken")));
        assertTrue(RouteRiskCore.requestInputs(null, "[\"b\"]").contains(new RouteRiskCore.Input("", "b")));
        assertEquals(List.of(), RouteRiskCore.requestInputs(null, null));
        assertTrue(RouteRiskCore.requestInputs(null, "k=v&".repeat(5000)).size() <= 512);
        String deep = "{\"a\":".repeat(20) + "\"x\"" + "}".repeat(20);
        assertFalse(RouteRiskCore.requestInputs(null, deep).contains(new RouteRiskCore.Input("a", "x")));
    }

    private AiwafConfig config(Map<String, Object> saved, String mode) throws IOException {
        Path file = directory.resolve("paths.json");
        Files.writeString(file, MAPPER.writeValueAsString(saved));
        AiwafConfig config = new AiwafConfig();
        config.pathManifestPath = file.toString();
        config.injectionRiskMode = mode;
        return config;
    }

    private static AiwafRequest request(String path, Map<String, String> query) {
        return new AiwafRequest("GET", path, "203.0.113.5", "", Map.of(), query, 0L, Set.of());
    }

    @Test
    void logModeLogsNamesAndNeverValues() throws IOException {
        RouteRiskCore.observe(request("/search", Map.of("evil_name", "../topsecret")), config(manifest("/search", risk("path", "")), "log"));
        assertEquals(1, logged.size());
        assertEquals(Map.of("event", "aiwaf.injection_risk", "mode", "log", "method", "GET", "route", "/search",
                "sink", "path", "input", "", "rule", "path_traversal", "confidence", "medium"), MAPPER.readValue(logged.get(0), Map.class));
        assertFalse(logged.get(0).contains("topsecret"));
        assertFalse(logged.get(0).contains("evil_name"));
    }

    @ParameterizedTest
    @ValueSource(strings = {"off", "", "block", "monitor"})
    void onlyLogModeTurnsThisOn(String mode) throws IOException {
        RouteRiskCore.observe(request("/search", Map.of("x", "../y")), config(manifest("/search", risk("path", "")), mode));
        assertEquals(List.of(), logged);
    }

    @Test
    void missingManifestsAndBadRequestsAreIgnored() throws IOException {
        AiwafConfig config = new AiwafConfig();
        config.injectionRiskMode = "log";
        config.pathManifestPath = directory.resolve("missing.json").toString();
        RouteRiskCore.observe(request("/search", Map.of("x", "../y")), config);
        RouteRiskCore.observe(request("/search", Map.of("x", "../y")), null);
        assertEquals(List.of(), logged);
        AiwafConfig working = config(manifest("/search", risk("path", "")), "log");
        RouteRiskCore.observe(null, working);
        working.pathManifestEnabled = false;
        RouteRiskCore.observe(request("/search", Map.of("x", "../y")), working);
        assertEquals(List.of(), logged);
    }

    @Test
    void theEngineLogsWithoutChangingTheDecision() throws IOException {
        AiwafConfig on = config(manifest("/search", risk("command", "q")), "log");
        AiwafDecision decision = new AiwafEngine(on).evaluate(request("/search", Map.of("q", "$(id)")));
        assertTrue(logged.stream().anyMatch(line -> line.contains("command_substitution")));
        logged.clear();
        RouteRiskCore.clearCache();
        AiwafConfig off = config(manifest("/search", risk("command", "q")), "off");
        assertEquals(decision.allowed(), new AiwafEngine(off).evaluate(request("/search", Map.of("q", "$(id)"))).allowed());
        assertEquals(List.of(), logged);
    }

    @ParameterizedTest
    @CsvSource(delimiter = '|', nullValues = "null", textBlock = """
            @RequestParam q            | q
            @PathVariable id           | id
            req.getParameter("id")     | id
            @RequestBody               | ''
            req.getQueryString()       | ''
            null                       | ''
            """)
    void inputFromSource(String source, String expected) {
        assertEquals(expected, RouteRiskCore.inputFromSource(source));
    }

    private JavaSourceAnalysisCore.Report audit(String relative, String source) throws IOException {
        Path file = directory.resolve("src").resolve(relative);
        Files.createDirectories(file.getParent());
        Files.writeString(file, source);
        return JavaSourceAnalysisCore.analyze(directory.resolve("src"), false);
    }

    private static Map<String, Object> routes(String path, String view) {
        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("view", view);
        Map<String, Object> routes = new LinkedHashMap<>();
        routes.put(path, entry);
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("routes", routes);
        return manifest;
    }

    @Test
    void findingsNameTheParameter() throws IOException {
        JavaSourceAnalysisCore.Report report = audit("App.java", VIEW);
        assertEquals("@RequestParam q", report.findings().get(0).source());
        report = audit("App.java", spring("@RequestBody Item item", "Runtime.getRuntime().exec(item);"));
        assertEquals("@RequestBody", report.findings().get(0).source());
        report = audit("App.java", spring("HttpServletRequest req", "Runtime.getRuntime().exec(req.getParameter(\"a\"));"));
        assertEquals("req.getParameter(\"a\")", report.findings().get(0).source());
    }

    private static String spring(String parameter, String body) {
        return "class App { String view(" + parameter + ") throws Exception { " + body + " return \"\"; } }";
    }

    @ParameterizedTest
    @ValueSource(strings = {"App.view", "com.acme.App.view"})
    void findingsLinkToRoutesByView(String view) throws IOException {
        JavaSourceAnalysisCore.Report report = audit("com/acme/App.java", VIEW);
        Map<String, Object> saved = routes("/search", view);
        assertEquals(new RouteRiskCore.Counts(1, 1, 0), RouteRiskCore.applyReport(report, saved));
        assertEquals(List.of(Map.of("sink", "command", "input", "q", "confidence", "high")),
                ((Map<?, ?>) ((Map<?, ?>) saved.get("routes")).get("/search")).get("injection_risks"));
    }

    @Test
    void sameNamedClassesElsewhereDoNotLink() throws IOException {
        JavaSourceAnalysisCore.Report report = audit("com/acme/App.java", VIEW);
        assertEquals(new RouteRiskCore.Counts(0, 0, 1), RouteRiskCore.applyReport(report, routes("/search", "org.other.App.view")));
        assertEquals(new RouteRiskCore.Counts(0, 0, 1), RouteRiskCore.applyReport(report, routes("/search", "App.other")));
    }

    @Test
    void servletsAndNestedClassesLink() throws IOException {
        JavaSourceAnalysisCore.Report report = audit("App.java", """
                @WebServlet("/orders")
                class App extends HttpServlet {
                    protected void doGet(HttpServletRequest req, HttpServletResponse res) throws Exception {
                        new File(req.getParameter("name"));
                    }
                }
                """);
        assertEquals(1, RouteRiskCore.applyReport(report, routes("/orders", "App")).linked());
        assertEquals(1, RouteRiskCore.applyReport(report, routes("/orders", "App.doGet")).linked());
        report = audit("App.java", "class Outer { static class App { void go(@RequestParam String v) { new File(v); } } }");
        assertEquals(1, RouteRiskCore.applyReport(report, routes("/x", "Outer$App.go")).linked());
    }

    @Test
    void sourceManifestsLinkEndToEnd() throws IOException {
        JavaSourceAnalysisCore.Report report = audit("App.java", VIEW);
        Path file = directory.resolve("paths.json");
        PathManifestCore.writeManifest(PathManifestCore.buildManifestFromSource("java-source", report.routes()), file);
        RouteRiskCore.Counts counts = RouteRiskCore.updateManifest(report, file);
        assertEquals(new RouteRiskCore.Counts(1, 1, 0), counts);
        assertTrue(RouteRiskCore.formatCounts(counts, "x").contains("Saved 1 possible injection paths to 1 routes"));
        assertTrue(RouteRiskCore.formatCounts(new RouteRiskCore.Counts(0, 0, 2), "x").contains("couldn't be tied"));
        assertEquals(1, new RouteRiskCore.Risks(PathManifestCore.loadManifest(file)).inspect("/search", Map.of("q", "$(id)"), "").size());
    }

    @Test
    void aCleanAuditRemovesOldRisksAndNothingElse() throws IOException {
        Map<String, Object> saved = routes("/x", "App.go");
        @SuppressWarnings("unchecked")
        Map<String, Object> entry = (Map<String, Object>) ((Map<String, Object>) saved.get("routes")).get("/x");
        entry.put("injection_risks", List.of(risk("sql", "")));
        RouteRiskCore.applyReport(audit("App.java", "class App {}"), saved);
        assertNull(entry.get("injection_risks"));
        assertEquals("App.go", entry.get("view"));
        assertEquals(1, ((Map<?, ?>) saved.get("injection_audit")).get("files_scanned"));
    }

    @Test
    void savedRisksNeverChangePathRules() throws IOException {
        Map<String, Object> saved = routes("/search", "App.view");
        @SuppressWarnings("unchecked")
        Map<String, Object> entry = (Map<String, Object>) ((Map<String, Object>) saved.get("routes")).get("/search");
        entry.put("protections", Map.of("rate_limit", Map.of("requests", 5)));
        int before = PathManifestCore.compileManifestToPathRules(saved).size();
        RouteRiskCore.applyReport(audit("App.java", VIEW), saved);
        assertTrue(entry.containsKey("injection_risks"));
        assertEquals(before, PathManifestCore.compileManifestToPathRules(saved).size());
    }

    @Test
    void updatingNeedsAManifest() throws IOException {
        JavaSourceAnalysisCore.Report report = audit("App.java", VIEW);
        assertThrows(IllegalArgumentException.class, () -> RouteRiskCore.updateManifest(report, directory.resolve("missing.json")));
        assertThrows(IllegalArgumentException.class, () -> RouteRiskCore.applyReport(report, new HashMap<>()));
    }
}
