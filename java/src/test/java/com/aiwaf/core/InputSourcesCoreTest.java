package com.aiwaf.core;

import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;

import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.TreeMap;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Inputs from the framework, declarations, OpenAPI and recordings, combined into each route. */
class InputSourcesCoreTest {
    private static final ObjectMapper JSON = new ObjectMapper();

    @TempDir
    Path directory;

    record Comment(String body, long postId, @JsonProperty("tag_ids") List<Integer> tags, UUID owner) {}

    static class Base {
        boolean draft;
    }

    static class Reply extends Base {
        String text;
        Double score;
        @JsonProperty("when") String at;
        @JsonIgnore String hidden;
        static int ignored;
        transient int skipped;
    }

    static class Loose {
        @JsonAnySetter
        void any(String key, Object value) {}
    }

    static class LooseField {
        @JsonAnySetter Map<String, Object> rest;
    }

    enum Color { RED }

    /** Matched by simple name, as Spring Data's is. */
    interface Pageable {}

    static class Handlers {
        public String search(@RequestParam("q") String query, @RequestParam("p") int page, @RequestParam(name = "flag") Optional<Boolean> flag,
                             @RequestParam("color") Color color, @PathVariable long id, @RequestHeader("x") String header,
                             java.security.Principal user, Pageable pageable, @RequestParam("n") int[] numbers, long unnamed) { return ""; }

        public String comment(@RequestBody Comment comment, @ModelAttribute Reply reply) { return ""; }

        public String gaps(@RequestParam Map<String, String> all, @RequestParam Object odd, @RequestBody Loose loose,
                           @RequestBody LooseField field, @RequestBody Map<String, Object> map,
                           @RequestPart("f") String part, java.io.InputStream stream) { return ""; }

        public String request(jakarta.servlet.http.HttpServletRequest request) { return ""; }

        @AiwafInputs(value = {"q", "page:int", " :x"}, complete = false)
        public String declared() { return ""; }
    }

    @AiwafInputs({"id:uuid"})
    static class Declared {
        public String handler() { return ""; }
    }

    private static Method method(Class<?> type, String name) {
        for (Method method : type.getDeclaredMethods()) if (method.getName().equals(name)) return method;
        throw new AssertionError(name);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, String> inputs(Map<String, Object> found) {
        return (Map<String, String>) found.get("inputs");
    }

    @SuppressWarnings("unchecked")
    private static List<String> codes(Object reasons) {
        return ((List<Map<String, Object>>) reasons).stream().map(reason -> String.valueOf(reason.get("code"))).toList();
    }

    @Test
    void kindsAndCanonicalPaths() {
        assertEquals(List.of("int", "int", "float", "float", "bool", "uuid", "text"),
                List.of("int", "Integer", "number", "decimal", "boolean", "uuid", "other").stream().map(InputSourcesCore::kind).toList());
        assertEquals("text", InputSourcesCore.kind(null));
        assertEquals("/posts/{}", InputSourcesCore.canonical("/posts/{id}"));
        assertEquals("/posts/{}", InputSourcesCore.canonical("posts/<int:id>/"));
        assertEquals("/files/{}", InputSourcesCore.canonical("/files/**?x=1"));
        assertEquals("/", InputSourcesCore.canonical(null));
    }

    @Test
    void reflectionReadsParametersAndBodies() {
        Map<String, Object> search = InputSourcesCore.frameworkInputs(method(Handlers.class, "search"));
        assertEquals(Map.of("q", "text", "p", "int", "flag", "bool", "color", "text", "page", "int", "size", "int", "sort", "text",
                "n", "int"), inputs(search));
        // Without -parameters, an unannotated parameter's name isn't known.
        assertEquals(List.of("dynamic_key"), codes(search.get("reasons")));
        assertTrue(Boolean.TRUE.equals(search.get("standalone")));
        Map<String, Object> comment = InputSourcesCore.frameworkInputs(method(Handlers.class, "comment"));
        assertEquals(Map.of("body", "text", "postId", "int", "tag_ids", "int", "owner", "uuid", "text", "text", "score", "float",
                "when", "text", "draft", "bool"), inputs(comment));
        Map<String, Object> gaps = InputSourcesCore.frameworkInputs(method(Handlers.class, "gaps"));
        assertEquals(List.of("container_whole", "external_class", "model_fields", "model_fields", "container_whole", "raw_body", "raw_body"),
                codes(gaps.get("reasons")));
        assertFalse(Boolean.TRUE.equals(InputSourcesCore.frameworkInputs(method(Handlers.class, "request")).get("standalone")));
        assertNull(InputSourcesCore.scalar(Map.class));
        assertEquals("text", InputSourcesCore.scalar(Color.class));
    }

    @Test
    void annotationsDeclareHandlersAndClasses() {
        assertEquals(Map.of("inputs", Map.of("q", "text", "page", "int"), "complete", false),
                InputSourcesCore.declaredOn(method(Handlers.class, "declared")));
        assertEquals(Map.of("inputs", Map.of("id", "uuid"), "complete", true), InputSourcesCore.declaredOn(method(Declared.class, "handler")));
        assertNull(InputSourcesCore.declaredOn(method(Handlers.class, "request")));
    }

    @Test
    void routesCombineHandlersThatShareAPath() throws Exception {
        List<PathManifestCore.RouteInfo> routes = new ArrayList<>();
        routes.add(new PathManifestCore.RouteInfo("/c", List.of("POST"), Handlers.class, method(Handlers.class, "comment")));
        routes.add(new PathManifestCore.RouteInfo("/c", List.of("GET"), Handlers.class, method(Handlers.class, "request")));
        routes.add(new PathManifestCore.RouteInfo("/d", List.of("GET"), Declared.class, method(Declared.class, "handler")));
        routes.add(null);
        InputSourcesCore.Sources sources = InputSourcesCore.forRoutes(routes, InputSourcesCore.Sources.none());
        assertFalse(Boolean.TRUE.equals(sources.framework().get("/c").get("standalone")));
        assertEquals("uuid", ((Map<?, ?>) sources.annotations().get("/d").get("inputs")).get("id"));
    }

    private static final String SPEC = """
            {"openapi": "3.0.3", "paths": {
              "/items/{id}": {
                "parameters": [{"$ref": "#/components/parameters/Verbose"}],
                "get": {"parameters": [
                  {"name": "id", "in": "path", "schema": {"type": "integer"}},
                  {"name": "page", "in": "query", "schema": {"type": "integer"}},
                  {"name": "tags", "in": "query", "schema": {"type": "array", "items": {"type": "string", "format": "uuid"}}}]},
                "put": {"requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Item"}}}}},
                "summary": "not an operation"},
              "/open": {"post": {
                "parameters": [{"name": "filter", "in": "query", "schema": {"type": "object"}}, {"$ref": "#/missing"}],
                "requestBody": {"content": {
                  "application/json": {"schema": {"type": "object", "additionalProperties": true, "properties": {"a": {"type": ["number", "null"]}}}},
                  "text/plain": {"schema": {"type": "string"}}}}}},
              "/union": {"post": {"requestBody": {"content": {
                "application/x-www-form-urlencoded": {"schema": {"oneOf": [{"type": "object", "properties": {"x": {"type": "boolean"}}}, {"$ref": "#/components/schemas/Item"}]}},
                "application/merge-patch+json": {"schema": {"type": "array"}}}}}},
              "/broken": {"post": {"requestBody": {"content": {"application/json": {"schema": {"$ref": "other.json#/X"}}}}}}},
             "components": {
              "parameters": {"Verbose": {"name": "verbose", "in": "query", "schema": {"type": "boolean"}}},
              "schemas": {"Item": {"allOf": [{"type": "object", "properties": {"name": {"type": "string"}}}, {"properties": {"qty": {"type": "integer"}}}]}}}}
            """;

    @Test
    void openapiOperationsListQueryAndBodyInputs() throws Exception {
        Map<String, Map<String, Map<String, Object>>> found = InputSourcesCore.openapiInputs(JSON.readTree(SPEC));
        assertEquals(Map.of("verbose", "bool", "page", "int", "tags", "uuid"), found.get("/items/{}").get("GET").get("inputs"));
        assertEquals(true, found.get("/items/{}").get("PUT").get("complete"));
        assertEquals(Map.of("verbose", "bool", "name", "text", "qty", "int"), found.get("/items/{}").get("PUT").get("inputs"));
        assertEquals(Map.of("a", "float"), found.get("/open").get("POST").get("inputs"));
        assertEquals(List.of("container_whole", "external_class", "container_whole", "raw_body"), codes(found.get("/open").get("POST").get("reasons")));
        assertEquals(Map.of("x", "bool", "name", "text", "qty", "int"), found.get("/union").get("POST").get("inputs"));
        assertEquals(List.of("container_whole"), codes(found.get("/union").get("POST").get("reasons")));
        assertEquals(List.of("external_class"), codes(found.get("/broken").get("POST").get("reasons")));
    }

    @Test
    void filesLoad() throws Exception {
        Path spec = Files.writeString(directory.resolve("openapi.json"), SPEC);
        assertTrue(InputSourcesCore.loadSpec(spec).has("paths"));
        assertThrows(java.io.IOException.class, () -> InputSourcesCore.loadSpec(Files.writeString(directory.resolve("bad.json"), "{}")));
        assertThrows(java.io.IOException.class, () -> InputSourcesCore.loadSpec(directory.resolve("spec.yaml")));
        Path seen = Files.writeString(directory.resolve("seen.jsonl"), String.join("\n",
                "{\"route\": \"/search/\", \"key\": \"q\"}", "{\"route\": \"/search\", \"key\": \"*\"}",
                "{\"route\": \"/posts/{id}\", \"key\": \"page\", \"type\": \"int\"}", "not json", "[1]", "{\"route\": \"/x\"}"));
        Map<String, Map<String, String>> observed = InputSourcesCore.loadObservations(seen);
        assertEquals(Map.of("q", "text", "*", "*"), observed.get("/search"));
        assertEquals(Map.of("page", "int"), observed.get("/posts/{}"));
        Path declared = Files.writeString(directory.resolve("inputs.json"),
                "{\"routes\": {\"/a/{id}\": {\"inputs\": {\"n\": \"int\"}}, \"/b\": {\"complete\": false}, \"/c\": \"skipped\"}}");
        Map<String, Map<String, Object>> declarations = InputSourcesCore.loadDeclarations(declared);
        assertEquals(Map.of("inputs", Map.of("n", "int"), "complete", true), declarations.get("/a/{}"));
        assertEquals(false, declarations.get("/b").get("complete"));
        assertEquals(2, declarations.size());
        assertTrue(InputSourcesCore.loadDeclarations(Files.writeString(directory.resolve("null.json"), "3")).isEmpty());
    }

    @Test
    void sourcesComeFromTheEnvironment() throws Exception {
        assertTrue(InputSourcesCore.sourcesFromEnvironment(directory, Map.of()).declarations().isEmpty());
        Files.createDirectories(directory.resolve(".aiwaf"));
        Files.writeString(directory.resolve(".aiwaf/inputs.json"), "{\"/a\": {\"inputs\": {\"q\": \"text\"}}}");
        Files.writeString(directory.resolve(".aiwaf/observed_inputs.jsonl"), "{\"route\": \"/a\", \"key\": \"q\"}");
        Path spec = Files.writeString(directory.resolve("spec.json"), SPEC);
        InputSourcesCore.Sources sources = InputSourcesCore.sourcesFromEnvironment(directory, Map.of("AIWAF_OPENAPI_SPEC", spec.toString()));
        assertFalse(sources.declarations().isEmpty());
        assertFalse(sources.observations().isEmpty());
        assertFalse(sources.openapi().isEmpty());
        InputSourcesCore.Sources named = InputSourcesCore.sourcesFromEnvironment(directory, Map.of(
                "AIWAF_INPUTS_FILE", directory.resolve("none.json").toString(), "AIWAF_OBSERVED_INPUTS", directory.resolve("none").toString()));
        assertTrue(named.declarations().isEmpty() && named.observations().isEmpty());
    }

    private static Map<String, Object> route(Object... pairs) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (int i = 0; i < pairs.length; i += 2) out.put((String) pairs[i], pairs[i + 1]);
        return out;
    }

    @Test
    void finalizeCombinesEverySource() {
        Map<String, Object> routes = new TreeMap<>();
        routes.put("/read", route("inputs", Map.of("q", "text"), "inputs_complete", true));
        routes.put("/framework", route());
        routes.put("/covered", route("inputs", Map.of(), "inputs_complete", false,
                "inputs_incomplete", List.of(Map.of("code", "external_class", "detail", "x"))));
        routes.put("/annotated", route("inputs", Map.of("q", "text"), "inputs_complete", false,
                "inputs_incomplete", List.of(Map.of("code", "helper", "detail", "x"))));
        routes.put("/filed/{id}", route("methods", List.of("GET")));
        routes.put("/spec", route("methods", List.of("GET", "POST"), "inputs", Map.of(), "inputs_complete", false,
                "inputs_incomplete", List.of(Map.of("code", "helper", "detail", "x"))));
        routes.put("/partial", route("methods", List.of("GET", "POST")));
        routes.put("/any", route());
        routes.put("/seen", route("inputs", Map.of("q", "text"), "inputs_complete", true));
        routes.put("/whole", route("inputs", Map.of(), "inputs_complete", true));
        routes.put("/only-seen", route());
        routes.put("/untouched", route("view", "x"));
        routes.put("/odd", "not a map");
        Map<String, Map<String, Object>> framework = Map.of(
                "/framework", Map.of("inputs", Map.of("title", "text"), "covers", List.of(), "reasons", List.of(), "standalone", true),
                "/covered", Map.of("inputs", Map.of(), "covers", List.of("external_class"), "reasons", List.of(), "standalone", false));
        Map<String, Map<String, Object>> annotations = Map.of("/annotated", Map.of("inputs", Map.of("q", "int"), "complete", true));
        Map<String, Map<String, Map<String, Object>>> openapi = Map.of(
                "/spec", Map.of("GET", Map.of("inputs", Map.of("a", "int"), "complete", true),
                        "POST", Map.of("inputs", Map.of("b", "text"), "complete", true)),
                "/partial", Map.of("GET", Map.of("inputs", Map.of("a", "int"), "complete", true)),
                "/any", Map.of("GET", Map.of("inputs", Map.of(), "complete", true)));
        InputSourcesCore.Sources sources = new InputSourcesCore.Sources(framework, annotations,
                Map.of("/filed/{}", Map.of("inputs", Map.of("n", "int"), "complete", true)), openapi,
                Map.of("/seen", Map.of("q", "text", "extra", "text"), "/whole", Map.of("*", "*"), "/only-seen", Map.of("k", "text")));
        InputSourcesCore.Counts counts = InputSourcesCore.finalize(routes, sources);
        assertEquals(true, ((Map<?, ?>) routes.get("/framework")).get("inputs_complete"));
        assertEquals(true, ((Map<?, ?>) routes.get("/covered")).get("inputs_complete"));
        assertEquals(Map.of("q", "int"), ((Map<?, ?>) routes.get("/annotated")).get("inputs"));
        assertEquals(Map.of("n", "int"), ((Map<?, ?>) routes.get("/filed/{id}")).get("inputs"));
        assertEquals(true, ((Map<?, ?>) routes.get("/spec")).get("inputs_complete"));
        assertEquals(Map.of("a", "int"), ((Map<?, ?>) routes.get("/partial")).get("inputs"));
        assertEquals(List.of("observed"), codes(((Map<?, ?>) routes.get("/seen")).get("inputs_incomplete")));
        assertEquals(List.of("container_whole"), codes(((Map<?, ?>) routes.get("/whole")).get("inputs_incomplete")));
        assertEquals(Map.of("k", "text"), ((Map<?, ?>) routes.get("/only-seen")).get("inputs"));
        assertFalse(((Map<?, ?>) routes.get("/untouched")).containsKey("inputs"));
        assertEquals(Map.of("container_whole", 1, "observed", 1), counts.reasons());
        assertEquals(2, counts.incomplete());
        assertEquals("complete inputs for " + counts.complete() + " of " + (counts.complete() + 2)
                + " routes; incomplete because of container_whole (1), observed (1)", InputSourcesCore.describe(counts));
        assertEquals("", InputSourcesCore.describe(new InputSourcesCore.Counts(0, 0, Map.of())));
    }

    public record Line(String sku, int quantity) {}

    public static class Orders {
        @org.springframework.web.bind.annotation.GetMapping
        public String list(@org.springframework.web.bind.annotation.RequestParam("page") int page) { return ""; }

        @org.springframework.web.bind.annotation.PostMapping
        public String create(@org.springframework.web.bind.annotation.RequestBody Line line) { return ""; }
    }

    private static Map<String, Object> ordersRoute() {
        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("methods", List.of("GET", "POST"));
        entry.put("inputs", new TreeMap<>(Map.of("page", "text")));
        entry.put("inputs_complete", true);
        // The GET handler's code read page; the framework binds it there as an int.
        entry.put("_input_methods", new TreeMap<>(Map.of("GET", List.of("page"))));
        Map<String, Object> routes = new TreeMap<>();
        routes.put("/orders", entry);
        return routes;
    }

    @Test
    void frameworkNamesAreClaimedByTheirMethodsAndTheirTypesHold() throws Exception {
        List<PathManifestCore.RouteInfo> live = List.of(
                new PathManifestCore.RouteInfo("/orders", List.of("GET"), Orders.class, method(Orders.class, "list")),
                new PathManifestCore.RouteInfo("/orders", List.of("POST"), Orders.class, method(Orders.class, "create")));
        InputSourcesCore.Sources sources = InputSourcesCore.forRoutes(live, InputSourcesCore.Sources.none());
        Map<String, Object> routes = ordersRoute();
        InputSourcesCore.finalize(routes, sources);
        Map<?, ?> orders = (Map<?, ?>) routes.get("/orders");
        // The code read page untyped; the framework binds it as an int.
        assertEquals(Map.of("page", "int", "sku", "text", "quantity", "int"), orders.get("inputs"));
        assertEquals(Map.of("GET", List.of("page"), "POST", List.of("quantity", "sku")), orders.get("inputs_by_method"));

        Map<String, Object> unread = ordersRoute();
        ((Map<String, Object>) unread.get("/orders")).remove("inputs");
        ((Map<String, Object>) unread.get("/orders")).remove("inputs_complete");
        unread.put("/plain", new LinkedHashMap<>(Map.of("methods", List.of("GET"))));
        InputSourcesCore.finalizeWithoutCode(unread, sources);
        Map<?, ?> types = (Map<?, ?>) unread.get("/orders");
        assertEquals(Map.of("page", "int", "sku", "text", "quantity", "int"), types.get("inputs"));
        assertEquals(false, types.get("inputs_complete"));
        assertEquals(List.of("not_audited"), codes(types.get("inputs_incomplete")));
        assertNull(types.get("inputs_by_method"));
        assertNull(((Map<?, ?>) unread.get("/plain")).get("inputs"));
    }

    public static class Split {
        @org.springframework.web.bind.annotation.PostMapping
        public String create(@org.springframework.web.bind.annotation.RequestParam("ref") int ref) { return ""; }
    }

    @Test
    void aFrameworkTypeHoldsOnlyForTheMethodsItIsDeclaredFor() throws Exception {
        List<PathManifestCore.RouteInfo> live = List.of(
                new PathManifestCore.RouteInfo("/orders", List.of("POST"), Split.class, method(Split.class, "create")));
        InputSourcesCore.Sources sources = InputSourcesCore.forRoutes(live, InputSourcesCore.Sources.none());
        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("methods", List.of("GET", "POST"));
        entry.put("inputs", new TreeMap<>(Map.of("ref", "text")));
        entry.put("inputs_complete", true);
        // GET's code reads ref untyped; only POST binds it as an int.
        entry.put("_input_methods", new TreeMap<>(Map.of("GET", List.of("ref"))));
        Map<String, Object> routes = new TreeMap<>(Map.of("/orders", entry));
        InputSourcesCore.finalize(routes, sources);
        assertEquals(Map.of("ref", "text"), ((Map<?, ?>) routes.get("/orders")).get("inputs"));
    }

    public static class Plans {
        @org.springframework.web.bind.annotation.GetMapping
        public String list(@org.springframework.web.bind.annotation.RequestParam("mode") String mode) { return mode; }

        @org.springframework.web.bind.annotation.PostMapping
        public String create(@org.springframework.web.bind.annotation.RequestParam("mode") int mode) { return ""; }
    }

    @Test
    void eachMethodKeepsItsOwnTypeWhereMethodsDiffer() throws Exception {
        List<PathManifestCore.RouteInfo> live = List.of(
                new PathManifestCore.RouteInfo("/plans", List.of("GET"), Plans.class, method(Plans.class, "list")),
                new PathManifestCore.RouteInfo("/plans", List.of("POST"), Plans.class, method(Plans.class, "create")));
        InputSourcesCore.Sources sources = InputSourcesCore.forRoutes(live, InputSourcesCore.Sources.none());
        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("methods", List.of("GET", "POST"));
        Map<String, Object> routes = new TreeMap<>(Map.of("/plans", entry));
        InputSourcesCore.finalize(routes, sources);
        assertEquals(Map.of("mode", "text"), entry.get("inputs"));
        assertEquals(Map.of("POST", Map.of("mode", "int")), entry.get("inputs_types_by_method"));
    }

    public record Filters(String limit, int since) {}

    public static class Reports {
        @org.springframework.web.bind.annotation.PostMapping
        public String run(@org.springframework.web.bind.annotation.RequestParam("limit") int limit,
                          @org.springframework.web.bind.annotation.RequestBody Filters filters) { return ""; }

        @org.springframework.web.bind.annotation.PostMapping
        public String form(@org.springframework.web.bind.annotation.RequestParam("limit") int limit) { return ""; }
    }

    @Test
    void aQueryAndABodyKeepTheirOwnTypesForOneName() throws Exception {
        Map<String, Object> run = InputSourcesCore.frameworkInputs(method(Reports.class, "run"));
        assertEquals(Map.of("query", Map.of("limit", "int"), "body", Map.of("limit", "text", "since", "int")), run.get("locations"));
        // Without a JSON body, a @RequestParam may come from a form body: no location is claimed.
        assertNull(InputSourcesCore.frameworkInputs(method(Reports.class, "form")).get("locations"));
        InputSourcesCore.Sources sources = InputSourcesCore.forRoutes(
                List.of(new PathManifestCore.RouteInfo("/reports", List.of("POST"), Reports.class, method(Reports.class, "run"))),
                InputSourcesCore.Sources.none());
        Map<String, Object> entry = new LinkedHashMap<>(Map.of("methods", List.of("POST")));
        Map<String, Object> routes = new TreeMap<>(Map.of("/reports", entry));
        InputSourcesCore.finalize(routes, sources);
        assertEquals(Map.of("limit", "text", "since", "int"), entry.get("inputs"));
        assertEquals(Map.of("POST", Map.of("query", Map.of("limit", "int"))), entry.get("inputs_types_by_location"));
    }
}
