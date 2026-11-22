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
}
