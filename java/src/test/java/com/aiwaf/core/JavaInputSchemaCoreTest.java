package com.aiwaf.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** What init saves as each Spring or servlet handler's inputs. */
class JavaInputSchemaCoreTest {
    @TempDir
    Path directory;

    private Map<String, JavaInputSchemaCore.Inputs> inputs(String source) throws Exception {
        Files.writeString(directory.resolve("App.java"), source);
        Map<String, JavaInputSchemaCore.Inputs> out = new HashMap<>();
        for (JavaInputSchemaCore.Inputs entry : JavaSourceAnalysisCore.analyze(directory, false).inputs()) out.put(entry.function(), entry);
        return out;
    }

    private static final String HEADER = """
            import java.util.*;
            import org.springframework.web.bind.annotation.*;
            import jakarta.servlet.http.*;
            """;

    @Test
    void springParametersAndBodiesGiveTypes() throws Exception {
        Map<String, JavaInputSchemaCore.Inputs> found = inputs(HEADER + """
                record Comment(String body, long postId, @com.fasterxml.jackson.annotation.JsonProperty("tag_ids") List<Integer> tags, UUID owner) {}
                class Base { boolean draft; }
                class Reply extends Base { String text; Double score; static int ignored; }
                @RestController
                class App {
                    @GetMapping("/search")
                    String search(@RequestParam("q") String query, @RequestParam int page, Optional<Boolean> flag, @PathVariable long id,
                                  @RequestHeader("x") String header, org.springframework.data.domain.Pageable pageable) { return query; }
                    @PostMapping("/comments")
                    String comment(@RequestBody Comment comment, java.security.Principal user) { return ""; }
                    @PostMapping("/replies")
                    String reply(@ModelAttribute Reply reply) { return ""; }
                }
                """);
        JavaInputSchemaCore.Inputs search = found.get("App.search");
        assertEquals(Map.of("q", "text", "page", "int", "flag", "bool", "size", "int", "sort", "text"), search.inputs());
        assertTrue(search.complete());
        assertEquals(Map.of("body", "text", "postId", "int", "tag_ids", "int", "owner", "uuid"), found.get("App.comment").inputs());
        assertTrue(found.get("App.comment").complete());
        assertEquals(Map.of("draft", "bool", "text", "text", "score", "float"), found.get("App.reply").inputs());
    }

    @Test
    void servletReadsAndAnythingNotUnderstood() throws Exception {
        Map<String, JavaInputSchemaCore.Inputs> found = inputs(HEADER + """
                class Loose { @com.fasterxml.jackson.annotation.JsonAnySetter void any(String k, Object v) {} }
                @RestController
                class App {
                    @GetMapping("/a") String a(HttpServletRequest request) {
                        return request.getParameter("q") + Integer.parseInt(request.getParameter("n")) + request.getHeader("x");
                    }
                    @GetMapping("/map") String map(@RequestParam Map<String, String> all) { return ""; }
                    @PostMapping("/body") String body(@RequestBody Map<String, Object> all) { return ""; }
                    @PostMapping("/loose") String loose(@RequestBody Loose all) { return ""; }
                    @PostMapping("/unknown") String unknown(@RequestBody Elsewhere all) { return ""; }
                    @GetMapping("/raw") String raw(HttpServletRequest request) { return request.getQueryString(); }
                    @GetMapping("/passed") String passed(HttpServletRequest request) { return Helpers.helper(request); }
                    @GetMapping("/dynamic") String dynamic(HttpServletRequest request, String key) { return request.getParameter(key); }
                    String helper(HttpServletRequest request) { return ""; }
                }
                class Servlet extends HttpServlet {
                    protected void doGet(HttpServletRequest req, HttpServletResponse resp) { req.getParameter("id"); }
                }
                @RestController
                class Bound {
                    @InitBinder void bind(Object binder) {}
                    @GetMapping("/b") String b(@RequestParam String q) { return q; }
                }
                """);
        assertEquals(Map.of("q", "text", "n", "int"), found.get("App.a").inputs());
        assertTrue(found.get("App.a").complete());
        for (String name : new String[] {"App.map", "App.body", "App.loose", "App.unknown", "App.raw", "App.passed", "App.dynamic", "Bound.b"}) {
            assertFalse(found.get(name).complete(), name);
        }
        assertEquals(Map.of("id", "text"), found.get("Servlet.doGet").inputs());
        assertFalse(found.containsKey("App.helper"));
    }

    @Test
    void aRawJsonBodyParsedWithJacksonIsFollowedAndAnyOtherUseStaysRaw() throws Exception {
        Map<String, JavaInputSchemaCore.Inputs> found = inputs(HEADER + """
                class Tree extends HttpServlet {
                    private final com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
                    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws Exception {
                        com.fasterxml.jackson.databind.JsonNode node = mapper.readTree(req.getInputStream());
                        if (node.has("note")) System.out.println(node.get("note").asText());
                        save(node.path("sku").asText(), node.get("qty").asInt(), node.get("rush").asBoolean());
                    }
                }
                class Typed extends HttpServlet {
                    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws Exception {
                        Order order = mapper.readValue(req.getReader(), Order.class);
                    }
                }
                class Order { String sku; int qty; Line line; }
                class Line { String code; }
                class Whole extends HttpServlet {
                    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws Exception {
                        com.fasterxml.jackson.databind.JsonNode node = mapper.readTree(req.getInputStream());
                        store(node);
                    }
                }
                class Dynamic extends HttpServlet {
                    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws Exception {
                        com.fasterxml.jackson.databind.JsonNode node = mapper.readTree(req.getInputStream());
                        node.get(req.getHeader("X-Field"));
                    }
                }
                class Stream extends HttpServlet {
                    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws Exception {
                        req.getInputStream().readAllBytes();
                    }
                }
                class Missing extends HttpServlet {
                    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws Exception {
                        mapper.readValue(req.getInputStream(), Elsewhere.class);
                    }
                }
                """);
        assertEquals(Map.of("note", "text", "sku", "text", "qty", "int", "rush", "bool"), found.get("Tree.doPost").inputs());
        assertTrue(found.get("Tree.doPost").complete());
        assertEquals(Map.of("sku", "text", "qty", "int", "line", "text"), found.get("Typed.doPost").inputs());
        assertTrue(found.get("Typed.doPost").complete());
        assertEquals(Map.of("line", java.util.List.of("code")), found.get("Typed.doPost").nested());
        // The parsed tree used whole, read with a key that isn't constant, or the stream read itself: raw.
        for (String name : new String[] {"Whole.doPost", "Dynamic.doPost", "Stream.doPost"}) {
            assertFalse(found.get(name).complete(), name);
            assertEquals("raw_body", found.get(name).reasons().get(0).code(), name);
        }
        assertEquals("external_class", found.get("Missing.doPost").reasons().get(0).code());
    }

    @Test
    void theRequestIsFollowedIntoTheClassesOwnMethodsAndConstantKeysAreKnown() throws Exception {
        Map<String, JavaInputSchemaCore.Inputs> found = inputs(HEADER + """
                @RestController
                class App {
                    private static final String PAGE = "page";
                    @GetMapping("/a") String a(HttpServletRequest request) { return render(request); }
                    @GetMapping("/b") String b(HttpServletRequest request) { return this.render(request); }
                    @GetMapping("/c") String c(HttpServletRequest request) { return Other.render(request); }
                    @GetMapping("/d") String d(HttpServletRequest request) { return loop(request); }
                    @GetMapping("/e") String e(HttpServletRequest request) { return request.getParameter(MISSING); }
                    private String render(HttpServletRequest r) { return r.getParameter("q") + Integer.parseInt(r.getParameter(PAGE)); }
                    private String loop(HttpServletRequest r) { return loop(r); }
                }
                """);
        assertEquals(Map.of("q", "text", "page", "int"), found.get("App.a").inputs());
        assertTrue(found.get("App.a").complete());
        assertTrue(found.get("App.b").complete());
        assertFalse(found.get("App.c").complete());
        assertFalse(found.get("App.d").complete());  // Recursion isn't followed.
        assertFalse(found.get("App.e").complete());
    }

    @Test
    void initSavesInputsIntoTheManifest() throws Exception {
        Files.writeString(directory.resolve("App.java"), HEADER + """
                @RestController
                class App { @GetMapping("/search") String search(@RequestParam String q, @RequestParam int page) { return q; } }
                """);
        Map<String, Object> routes = new HashMap<>(Map.of("/search", new HashMap<>(Map.of("view", "App.search")), "/other", new HashMap<>(Map.of("view", "x"))));
        Map<String, Object> manifest = new HashMap<>(Map.of("routes", routes));
        RouteRiskCore.Counts counts = RouteRiskCore.applyReport(JavaSourceAnalysisCore.analyze(directory, false), manifest);
        assertEquals(1, counts.inputs());
        Map<?, ?> search = (Map<?, ?>) routes.get("/search");
        assertEquals(Map.of("q", "text", "page", "int"), search.get("inputs"));
        assertEquals(true, search.get("inputs_complete"));
        assertFalse(((Map<?, ?>) routes.get("/other")).containsKey("inputs"));
        assertTrue(JavaSourceAnalysisCore.analyze(directory, false).toMap().containsKey("inputs"));
    }

    @Test
    void eachGapSaysWhyWithItsLine() throws Exception {
        Map<String, JavaInputSchemaCore.Inputs> found = inputs(HEADER + """
                class Paging {
                    static int page(HttpServletRequest request) { return Integer.parseInt(request.getParameter("page")); }
                }
                @RestController
                class App {
                    @GetMapping("/helper")
                    String helper(HttpServletRequest request) { return Paging.page(request) + other(request); }
                    @GetMapping("/whole")
                    String whole(HttpServletRequest request, @RequestParam Map<String, String> all) { return request.getParameterMap().toString(); }
                    @GetMapping("/dynamic")
                    String dynamic(HttpServletRequest request, String key) { return request.getParameter(key); }
                    @PostMapping("/raw")
                    String raw(HttpServletRequest request, org.springframework.web.multipart.MultipartFile file) throws Exception {
                        return request.getReader().readLine() + request.getAttribute("x");
                    }
                    @PostMapping("/body")
                    String body(@RequestBody Missing missing, @RequestBody Loose loose, @RequestBody Map<String, Object> map,
                                @RequestParam Weird weird, org.springframework.web.context.request.WebRequest web) { return ""; }
                    @GetMapping("/stored")
                    Object stored(HttpServletRequest request) { return request; }
                    String other(HttpServletRequest request) { return request.getParameter("o"); }
                }
                class Loose { @com.fasterxml.jackson.annotation.JsonAnySetter void any(String k, Object v) {} }
                class Weird {}
                @RestController
                class Shared {
                    @ModelAttribute void before() {}
                    @GetMapping("/shared") String shared(@RequestParam String q) { return q; }
                }
                """);
        assertEquals(Map.of("page", "int", "o", "text"), found.get("App.helper").inputs());
        assertTrue(found.get("App.helper").complete());
        assertEquals(java.util.List.of("container_whole", "container_whole"), codes(found.get("App.whole")));
        assertEquals(java.util.List.of("dynamic_key"), codes(found.get("App.dynamic")));
        assertEquals(java.util.List.of("raw_body", "raw_body", "helper"), codes(found.get("App.raw")));
        assertEquals(java.util.List.of("external_class", "model_fields", "container_whole", "external_class", "helper"),
                codes(found.get("App.body")));
        assertEquals(java.util.List.of("helper"), codes(found.get("App.stored")));
        assertEquals(java.util.List.of("class_view"), codes(found.get("Shared.shared")));
        assertTrue(found.get("App.dynamic").reasons().get(0).line() > 0);
        Map<String, Object> map = JavaInputSchemaCore.asMap(new JavaInputSchemaCore.Reason("helper", "x", 0));
        assertEquals(Map.of("code", "helper", "detail", "x"), map);
        assertEquals(java.util.List.of(), new JavaInputSchemaCore.Inputs("f", "a", Map.of(), true).reasons());
    }

    private static java.util.List<String> codes(JavaInputSchemaCore.Inputs entry) {
        return entry.reasons().stream().map(JavaInputSchemaCore.Reason::code).toList();
    }

    @Test
    void aReadKeptInALocalTakesTheTypeItIsParsedTo() throws Exception {
        Map<String, JavaInputSchemaCore.Inputs> found = inputs(HEADER + """
                class App extends HttpServlet {
                    protected void doGet(HttpServletRequest request, HttpServletResponse response) {
                        String raw = request.getParameter("limit");
                        if (raw == null || raw.isBlank()) return;
                        int limit = Integer.parseInt(raw);
                        String flag = request.getParameter("flag");
                        boolean on = Boolean.parseBoolean(flag);
                        String name = request.getParameter("name");
                        log(name);
                        int n = Integer.parseInt(name);
                        String both = request.getParameter("both");
                        long a = Long.parseLong(both);
                        double b = Double.parseDouble(both);
                    }
                    void log(String text) {}
                }
                """);
        assertEquals(Map.of("limit", "int", "flag", "bool", "name", "text", "both", "text"), found.get("App.doGet").inputs());
    }
}
