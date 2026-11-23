package com.aiwaf.core;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The shared cases in tests/fixtures/route_schema_cases.json, which aiwaf-rust and aiwaf-wasm run too. */
class RouteSchemaCoreTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    @TempDir
    Path directory;
    private final List<String> logged = new ArrayList<>();
    private final Handler handler = new Handler() {
        @Override public void publish(LogRecord record) { logged.add(record.getMessage()); }
        @Override public void flush() {}
        @Override public void close() {}
    };

    @BeforeEach
    void listen() {
        RouteSchemaCore.clearCache();
        Logger.getLogger("com.aiwaf.route_schema").addHandler(handler);
    }

    @AfterEach
    void stop() {
        Logger.getLogger("com.aiwaf.route_schema").removeHandler(handler);
        RouteSchemaCore.clearCache();
    }

    private static JsonNode fixture() throws Exception {
        return MAPPER.readTree(Path.of("..", "tests", "fixtures", "route_schema_cases.json").toFile());
    }

    private static Map<String, String> query(String text) {
        Map<String, String> out = new LinkedHashMap<>();
        for (String pair : text.split("&")) {
            if (pair.isEmpty()) continue;
            int equals = pair.indexOf('=');
            out.put(URLDecoder.decode(equals < 0 ? pair : pair.substring(0, equals), StandardCharsets.UTF_8),
                    URLDecoder.decode(equals < 0 ? "" : pair.substring(equals + 1), StandardCharsets.UTF_8));
        }
        return out;
    }

    @Test
    @SuppressWarnings("unchecked")
    void sharedCases() throws Exception {
        JsonNode cases = fixture();
        RouteSchemaCore.Schemas schemas = new RouteSchemaCore.Schemas(MAPPER.convertValue(cases.get("manifest"), Map.class));
        for (JsonNode item : cases.get("cases")) {
            Map<String, String> query = query(item.get("query").asText());
            String body = item.get("body").asText();
            String found = SqlInjectionCore.inspectQueryAndBody(query, body, 65536);
            RouteSchemaCore.Review review = schemas.review(item.get("path").asText(), query, body, 65536, found);
            String outcome = review != null ? review.action() : found != null ? "block" : "pass";
            assertEquals(item.get("expect").asText(), outcome, item.toString());
            if (outcome.equals("mismatch")) assertEquals(item.get("input").asText(), review.input(), item.toString());
        }
        for (JsonNode item : cases.get("path_cases")) {
            String path = item.get("path").asText();
            String found = SqlInjectionCore.inspectPath(path, 65536);
            RouteSchemaCore.Review review = found == null ? null : schemas.reviewPath(path, 65536, found);
            String outcome = review != null ? review.action() : found != null ? "block" : "pass";
            assertEquals(item.get("expect").asText(), outcome, item.toString());
            if (outcome.equals("mismatch")) assertEquals(item.get("input").asText(), review.input(), item.toString());
        }
        assertNull(RouteSchemaCore.segmentType("<slug>"));
        assertEquals("int", RouteSchemaCore.segmentType("{id:int}"));
        assertTrue(RouteSchemaCore.fits("int", MAPPER.readTree("null")));
        assertFalse(RouteSchemaCore.fits("int", MAPPER.readTree("{\"a\":1}")));
        assertTrue(RouteSchemaCore.fits("text", MAPPER.readTree("{\"a\":1}")));
    }

    @Test
    @SuppressWarnings("unchecked")
    void sharedStrictCases() throws Exception {
        JsonNode cases = fixture();
        Map<String, Object> manifest = new LinkedHashMap<>(MAPPER.convertValue(cases.get("manifest"), Map.class));
        manifest.put("strict_inputs", Map.of("allow", MAPPER.convertValue(cases.get("strict_cases").get("allow"), List.class)));
        RouteSchemaCore.Schemas schemas = new RouteSchemaCore.Schemas(manifest);
        assertTrue(schemas.strict());
        assertFalse(new RouteSchemaCore.Schemas(MAPPER.convertValue(cases.get("manifest"), Map.class)).strict());
        for (JsonNode item : cases.get("strict_cases").get("cases")) {
            Map<String, String> query = query(item.get("query").asText());
            String body = item.get("body").asText();
            String found = SqlInjectionCore.inspectQueryAndBody(query, body, 65536);
            RouteSchemaCore.Review review = schemas.review(item.get("path").asText(), query, body, 65536, found);
            String outcome = review != null ? review.action() : found != null ? "block" : "pass";
            assertEquals(item.get("expect").asText(), outcome, item.toString());
            if (item.has("input")) assertEquals(item.get("input").asText(), review.input(), item.toString());
        }
        Map<String, Object> plainAllow = new LinkedHashMap<>(manifest);
        plainAllow.put("strict_inputs", Map.of("allow", List.of(5)));
        String long_ = "x".repeat(100);
        RouteSchemaCore.Review cut = new RouteSchemaCore.Schemas(plainAllow).review("/search", Map.of(long_, "1"), "", 65536, null);
        assertEquals(64, cut.input().length());
    }

    @Test
    void strictInputsLogOrBlockInTheEngine() throws Exception {
        AiwafConfig strict = config("enforce");
        strict.strictInputs = true;
        String extra = "{\"body\": \"hi\", \"post_id\": 7, \"role\": \"admin\"}";
        assertEquals("schema_unexpected_input", new AiwafEngine(strict).evaluate(post("/comments", extra)).reason());
        strict.strictInputsAllow = List.of("role");
        assertTrue(new AiwafEngine(strict).evaluate(post("/comments", extra)).allowed());
        assertTrue(new AiwafEngine(config("enforce")).evaluate(post("/comments", extra)).allowed());
        AiwafConfig monitor = config("monitor");
        monitor.strictInputs = true;
        monitor.strictInputsAllow = null;
        logged.clear();
        assertTrue(new AiwafEngine(monitor).evaluate(post("/comments", extra)).allowed());
        assertTrue(logged.stream().anyMatch(line -> line.contains("\"action\":\"unexpected\"") && line.contains("\"input\":\"role\"")), logged.toString());
    }

    private AiwafConfig config(String mode) throws Exception {
        Path file = directory.resolve("paths.json");
        Files.writeString(file, MAPPER.writeValueAsString(fixture().get("manifest")));
        AiwafConfig config = new AiwafConfig();
        config.privateIpsExempted = false;
        config.pathManifestPath = file.toString();
        config.routeSchemaMode = mode;
        return config;
    }

    private static AiwafRequest post(String path, String body) {
        Map<String, String> headers = Map.of("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept", "text/html",
                "Accept-Language", "en-US", "Accept-Encoding", "gzip", "Connection", "keep-alive", "Content-Type", "application/json");
        return new AiwafRequest("POST", path, "198.51.100.91", "US", headers, Map.of(), System.currentTimeMillis(), Set.of(), body);
    }

    @Test
    void theEngineClearsProseInEnforceModeAndOnlyLogsInMonitorMode() throws Exception {
        String prose = "{\"body\": \"use sleep(5) in the query\", \"post_id\": 7}";
        AiwafDecision enforced = new AiwafEngine(config("enforce")).evaluate(post("/comments", prose));
        assertTrue(enforced.allowed(), enforced.reason());
        assertEquals("schema_type_mismatch", new AiwafEngine(config("enforce")).evaluate(post("/comments", "{\"post_id\": \"7 OR 1=1\"}")).reason());
        assertFalse(new AiwafEngine(config("enforce")).evaluate(post("/comments", "{\"body\": \"x' OR '1'='1\"}")).allowed());
        AiwafDecision monitored = new AiwafEngine(config(new AiwafConfig().routeSchemaMode)).evaluate(post("/comments", prose));
        assertEquals("sql_function", monitored.reason());
        Map<?, ?> event = MAPPER.readValue(logged.get(logged.size() - 1), Map.class);
        assertEquals(Map.of("event", "aiwaf.route_schema", "mode", "monitor", "method", "POST", "route", "/comments",
                "action", "relax", "rule", "sql_function"), event);
        assertFalse(String.join("", logged).contains("sleep"));
        assertEquals("sql_function", RouteSchemaCore.apply(post("/comments", prose), config("off"), "sql_function"));
        AiwafConfig disabled = config("enforce");
        disabled.pathManifestEnabled = false;
        assertEquals("sql_function", RouteSchemaCore.apply(post("/comments", prose), disabled, "sql_function"));
        assertNull(RouteSchemaCore.apply(post("/comments", "{}"), config("enforce"), (String) null));
        // A path's finding is reviewed by the route's parameters; a header's stands.
        assertTrue(new AiwafEngine(config("enforce")).evaluate(post("/posts/use-sleep(5)-in-queries", null)).allowed());
        assertEquals("sql_function", RouteSchemaCore.apply(post("/comments", "{}"), config("enforce"), new SqlInjectionCore.Found("sql_function", "headers")));
    }
}
