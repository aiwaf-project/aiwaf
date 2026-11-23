package com.aiwaf.core;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.TextNode;

import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;
import java.util.regex.Pattern;

/**
 * Reviews requests against the inputs {@code init} saved for their route: the Java form of
 * aiwaf-rust's {@code RouteRisks::review}, checked by the same cases in
 * {@code tests/fixtures/route_schema_cases.json}.
 *
 * <p>A typed input sent with a value that isn't of its type is a mismatch. When the general check's
 * finding is one of {@link #RELAXED_RULES} and comes from the query or body, the request is checked
 * again without the inputs the schema clears: typed inputs of their type, text that reaches no
 * operation, and inputs the route never reads. Those still get every other rule. Nothing is
 * cleared when the route's inputs aren't known in full.
 */
public final class RouteSchemaCore {
    private static final Logger LOGGER = Logger.getLogger("com.aiwaf.route_schema");
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final Map<String, Schemas> CACHE = new ConcurrentHashMap<>();
    private static final int MAX_ITEMS = 512;
    /** The general check's rules a cleared input only logs: the ones that fire on prose and code snippets. */
    public static final Set<String> RELAXED_RULES = Set.of("sql_function", "sql_comment", "sql_breakout_numeric");
    public static final Set<String> INPUT_TYPES = Set.of("int", "float", "bool", "uuid", "text");
    private static final Pattern INT = Pattern.compile("^[+-]?[0-9]{1,20}$");
    private static final Pattern FLOAT = Pattern.compile("(?i)^[+-]?(?:[0-9]{1,40}\\.?[0-9]{0,40}|\\.[0-9]{1,40})(?:e[+-]?[0-9]{1,4})?$");
    private static final Pattern UUID = Pattern.compile(
            "(?i)^(?:urn:uuid:)?\\{?[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}\\}?$");
    private static final Set<String> BOOLS = Set.of("true", "false", "1", "0", "on", "off", "yes", "no", "t", "f", "y", "n");

    /**
     * Names strict inputs always allows: CSRF tokens, read by middleware rather than the handler,
     * and tracking and cache-busting parameters. A trailing * matches any rest. As aiwaf-rust.
     */
    public static final List<String> STRICT_ALLOWED = List.of(
            "csrfmiddlewaretoken", "csrf_token", "_csrf", "_token", "authenticity_token", "utm_*", "fbclid", "gclid", "msclkid", "_");
    private static final int MAX_LOGGED_NAME = 64;

    private static String logged(String name) {
        return name.codePoints().limit(MAX_LOGGED_NAME).collect(StringBuilder::new, StringBuilder::appendCodePoint, StringBuilder::append).toString();
    }

    static boolean allowed(List<String> patterns, String name) {
        for (String pattern : patterns) {
            if (pattern.endsWith("*") ? name.startsWith(pattern.substring(0, pattern.length() - 1)) : pattern.equals(name)) return true;
        }
        return false;
    }

    /** action is "mismatch" (with input and type), "unexpected" (with input) or "relax" (with rule). */
    public record Review(String action, String route, String input, String type, String rule) {}

    /**
     * byMethod: the names each method reads, when init could tell them apart
     * (inputs_by_method); a method without its own list reads the route's whole list.
     */
    private record Route(String path, RouteRiskCore.CompiledRoute compiled, Map<String, String> inputs, boolean complete,
                         Set<String> reached, boolean unnamedRisk, List<String> pathTypes, Map<String, Set<String>> byMethod,
                         Map<String, Set<String>> nested) {
        boolean reads(String method, String name) {
            String upper = method == null ? "" : method.toUpperCase(Locale.ROOT);
            Set<String> own = byMethod.get(upper);
            if (own == null && upper.equals("HEAD")) own = byMethod.get("GET");
            return own != null ? own.contains(name) : inputs.containsKey(name);
        }
    }

    /** The type a route segment's own syntax gives, as in <int:id> or {id:uuid}, or null. */
    static String segmentType(String segment) {
        String converter = null;
        if (segment.startsWith("<") && segment.contains(":")) converter = segment.substring(1, segment.indexOf(':'));
        else if (segment.startsWith("{") && segment.endsWith("}") && segment.contains(":")) {
            converter = segment.substring(segment.indexOf(':') + 1, segment.length() - 1);
        }
        return converter != null && Set.of("int", "float", "uuid").contains(converter) ? converter : null;
    }

    private record Field(String name, JsonNode value, String raw) {}

    /**
     * A multipart form's parts as fields, found by its opening boundary line: text parts with
     * their values, file parts with null. Null when the body isn't one.
     */
    static List<Field> multipartFields(String text) {
        if (text == null || !text.startsWith("--")) return null;
        int lineEnd = text.indexOf("\r\n");
        if (lineEnd < 3 || lineEnd > 202) return null;
        String delimiter = "\r\n" + text.substring(0, lineEnd);
        List<Field> fields = new ArrayList<>();
        int at = lineEnd + 2;
        while (at < text.length() && fields.size() < MAX_ITEMS) {
            int end = text.indexOf(delimiter, at);
            if (end < 0) end = text.length();
            String chunk = text.substring(at, end);
            int split = chunk.indexOf("\r\n\r\n");
            String headers = split < 0 ? chunk : chunk.substring(0, split);
            String name = MultipartPreview.param(headers, "name");
            boolean file = MultipartPreview.param(headers, "filename") != null || MultipartPreview.param(headers, "filename*") != null;
            if (name != null && !name.isEmpty()) {
                String content = split < 0 ? "" : chunk.substring(split + 4);
                fields.add(new Field(inputName(name), file ? com.fasterxml.jackson.databind.node.NullNode.getInstance() : TextNode.valueOf(content), name));
            }
            if (end + delimiter.length() >= text.length() || text.startsWith("--", end + delimiter.length())) break;
            at = end + delimiter.length() + 2;
        }
        return fields;
    }

    /** A form name's key inside its input, as role in user[role]; null for a plain name. */
    static String[] bracketKey(String raw) {
        String key = decode(raw.split("=", 2)[0]);
        int open = key.indexOf('[');
        int close = key.indexOf(']', open + 1);
        if (open <= 0 || close <= open + 1) return null;
        return new String[] {key.substring(0, open), key.substring(open + 1, close)};
    }

    private RouteSchemaCore() {}

    static boolean fits(String kind, JsonNode value) {
        if (value == null || value.isNull()) return true;
        if (value.isTextual()) {
            String text = value.textValue().trim();
            if (text.isEmpty()) return true;
            return switch (kind) {
                case "int" -> INT.matcher(text).matches();
                case "float" -> FLOAT.matcher(text).matches();
                case "bool" -> BOOLS.contains(text.toLowerCase(Locale.ROOT));
                case "uuid" -> UUID.matcher(text).matches();
                default -> true;
            };
        }
        if (value.isArray()) {
            int count = 0;
            for (JsonNode item : value) {
                if (++count > MAX_ITEMS) break;
                if (item.isObject() || !fits(kind, item)) return false;
            }
            return true;
        }
        if (value.isObject()) return kind.equals("text");
        return true;  // Numbers and booleans can't carry text.
    }

    private static String inputName(String name) {
        int bracket = name.indexOf('[');
        return bracket >= 0 ? name.substring(0, bracket) : name;
    }

    private static String decode(String text) {
        try {
            return URLDecoder.decode(text, StandardCharsets.UTF_8);
        } catch (IllegalArgumentException ex) {
            return text;
        }
    }

    /** Saved inputs for one manifest, compiled for lookup by request path. */
    public static final class Schemas {
        private final List<Route> routes = new ArrayList<>();
        /** Strict inputs' allowed names, when the manifest's strict_inputs turns it on, and the path prefixes it skips. */
        private List<String> strict;
        private final List<String> exempt = new ArrayList<>();

        public Schemas(Map<String, Object> manifest) {
            if (manifest != null && manifest.get("strict_inputs") instanceof Map<?, ?> settings) {
                strict = new ArrayList<>(STRICT_ALLOWED);
                if (settings.get("allow") instanceof List<?> names) {
                    for (Object name : names) if (name instanceof String text) strict.add(text);
                }
                if (settings.get("exempt") instanceof List<?> prefixes) {
                    for (Object prefix : prefixes) {
                        if (prefix instanceof String text && !text.replaceAll("\\*+$", "").isEmpty()) exempt.add(text.replaceAll("\\*+$", ""));
                    }
                }
            }
            if (manifest != null && manifest.get("routes") instanceof Map<?, ?> all) {
                for (Map.Entry<?, ?> route : all.entrySet()) {
                    if (!(route.getValue() instanceof Map<?, ?> entry)) continue;
                    List<?> risks = entry.get("injection_risks") instanceof List<?> saved ? saved : List.of();
                    Map<String, String> inputs = null;
                    if (entry.get("inputs") instanceof Map<?, ?> saved) {
                        inputs = new HashMap<>();
                        for (Map.Entry<?, ?> input : saved.entrySet()) {
                            String kind = String.valueOf(input.getValue());
                            inputs.put(String.valueOf(input.getKey()), INPUT_TYPES.contains(kind) ? kind : "text");
                        }
                    }
                    if (risks.isEmpty() && inputs == null) continue;
                    Set<String> reached = new java.util.HashSet<>();
                    boolean unnamed = false;
                    for (Object item : risks) {
                        if (!(item instanceof Map<?, ?> risk)) continue;
                        String input = risk.get("input") == null ? "" : String.valueOf(risk.get("input"));
                        if (input.isEmpty()) unnamed = true;
                        else reached.add(input);
                    }
                    String path = String.valueOf(route.getKey());
                    List<String> pathTypes = new ArrayList<>();
                    for (String segment : path.split("/")) {
                        if (!segment.isEmpty() && "<{:*".indexOf(segment.charAt(0)) >= 0) pathTypes.add(segmentType(segment));
                    }
                    Map<String, Set<String>> byMethod = new HashMap<>();
                    if (entry.get("inputs_by_method") instanceof Map<?, ?> methods) {
                        for (Map.Entry<?, ?> method : methods.entrySet()) {
                            if (!(method.getValue() instanceof List<?> names)) continue;
                            Set<String> listed = new java.util.HashSet<>();
                            for (Object name : names) if (name instanceof String text) listed.add(text);
                            byMethod.put(String.valueOf(method.getKey()).toUpperCase(Locale.ROOT), listed);
                        }
                    }
                    Map<String, Set<String>> nested = new HashMap<>();
                    if (entry.get("inputs_nested") instanceof Map<?, ?> keys) {
                        for (Map.Entry<?, ?> key : keys.entrySet()) {
                            if (!(key.getValue() instanceof List<?> names)) continue;
                            Set<String> listed = new java.util.HashSet<>();
                            for (Object name : names) if (name instanceof String text) listed.add(text);
                            nested.put(String.valueOf(key.getKey()), listed);
                        }
                    }
                    routes.add(new Route(path, RouteRiskCore.compileRoute(path), inputs,
                            inputs != null && Boolean.TRUE.equals(entry.get("inputs_complete")), reached, unnamed, pathTypes, byMethod, nested));
                }
            }
            // Exact routes win over routes with parameters, as in RouteRiskCore.
            routes.sort(Comparator.<Route>comparingInt(route -> route.compiled().names().size())
                    .thenComparingInt(route -> -route.path().length()));
        }

        public boolean strict() {
            return strict != null;
        }

        public boolean hasInputs() {
            return routes.stream().anyMatch(route -> route.inputs() != null);
        }

        /** The review of one request, after the general check found {@code found} (or null); null keeps its answer. */
        public Review review(String path, Map<String, String> query, String body, int budget, String found) {
            return review("", path, query, body, budget, found);
        }

        /** As review, for a request with this method: strict inputs check that method's own list when saved. */
        public Review review(String method, String path, Map<String, String> query, String body, int budget, String found) {
            Route route = routes.stream().filter(item -> item.compiled().pattern().matcher(path == null || path.isEmpty() ? "/" : path).matches())
                    .findFirst().orElse(null);
            if (route == null || route.inputs() == null) return null;
            List<Field> queryFields = new ArrayList<>();
            if (query != null) {
                for (Map.Entry<String, String> entry : query.entrySet()) {
                    if (queryFields.size() >= MAX_ITEMS || entry.getKey() == null) break;
                    queryFields.add(new Field(inputName(entry.getKey()), TextNode.valueOf(entry.getValue() == null ? "" : entry.getValue()), entry.getKey()));
                }
            }
            // The body as a form or JSON object; null when it's neither and can't be reviewed.
            List<Field> bodyFields = null;
            boolean multipart = false;
            ObjectNode json = null;
            String text = body == null ? "" : body;
            String start = text.stripLeading();
            if (text.isBlank()) {
                bodyFields = List.of();
            } else if (start.startsWith("{")) {
                try {
                    if (JSON.readTree(text) instanceof ObjectNode object) {
                        json = object;
                        bodyFields = new ArrayList<>();
                        for (Map.Entry<String, JsonNode> entry : object.properties()) bodyFields.add(new Field(entry.getKey(), entry.getValue(), entry.getKey()));
                    }
                } catch (Exception ex) {
                    bodyFields = null;
                }
            } else if (multipartFields(text) != null) {
                bodyFields = multipartFields(text);
                multipart = true;
            } else if (text.contains("=") && !start.startsWith("[") && !start.startsWith("<") && !start.startsWith("-")) {
                bodyFields = new ArrayList<>();
                for (String pair : text.split("&")) {
                    if (pair.isEmpty() || bodyFields.size() >= MAX_ITEMS) continue;
                    int equals = pair.indexOf('=');
                    String key = decode(equals >= 0 ? pair.substring(0, equals) : pair);
                    bodyFields.add(new Field(inputName(key), TextNode.valueOf(decode(equals >= 0 ? pair.substring(equals + 1) : "")), pair));
                }
            }
            List<Field> all = new ArrayList<>(queryFields);
            if (bodyFields != null) all.addAll(bodyFields);
            for (Field field : all) {
                String kind = route.inputs().get(field.name());
                if (kind != null && !kind.equals("text") && !fits(kind, field.value())) {
                    return new Review("mismatch", route.path(), field.name(), kind, null);
                }
            }
            // Strict inputs: on a route whose list is complete, a name it never reads. Only the
            // query and a form or JSON object body are named; another body's names aren't known.
            String requestPath = path == null || path.isEmpty() ? "/" : path;
            if (strict != null && route.complete() && exempt.stream().noneMatch(requestPath::startsWith)) {
                for (Field field : all) {
                    if (!route.reads(method, field.name()) && !allowed(strict, field.name())) {
                        return new Review("unexpected", route.path(), logged(field.name()), null, null);
                    }
                }
                // Keys inside an object-valued input whose keys init knows, as user.role.
                if (json != null) {
                    for (Map.Entry<String, JsonNode> entry : json.properties()) {
                        Set<String> keys = route.nested().get(entry.getKey());
                        if (keys == null || !(entry.getValue() instanceof ObjectNode inner)) continue;
                        for (Map.Entry<String, JsonNode> sub : inner.properties()) {
                            String key = sub.getKey();
                            String full = entry.getKey() + "." + key;
                            if (!keys.contains(key) && !allowed(strict, full)) return new Review("unexpected", route.path(), logged(full), null, null);
                        }
                    }
                }
                List<Field> pairs = new ArrayList<>(queryFields);
                if (bodyFields != null && json == null) pairs.addAll(bodyFields);
                for (Field field : pairs) {
                    String[] split = bracketKey(field.raw());
                    if (split == null) continue;
                    Set<String> keys = route.nested().get(split[0]);
                    String full = split[0] + "." + split[1];
                    if (keys != null && !keys.contains(split[1]) && !allowed(strict, full)) return new Review("unexpected", route.path(), logged(full), null, null);
                }
            }
            // A multipart body can't be rebuilt without the parts the schema clears.
            if (found == null || !RELAXED_RULES.contains(found) || !route.complete() || route.unnamedRisk() || bodyFields == null || multipart) return null;
            // A finding from headers or the path isn't the query's or body's to clear.
            if (SqlInjectionCore.inspectQueryAndBody(query, body, budget) == null) return null;
            Map<String, String> keptQuery = new LinkedHashMap<>();
            for (Field field : queryFields) {
                Boolean cleared = cleared(route, field, budget);
                if (cleared == null) return null;
                if (!cleared) keptQuery.put(field.raw(), query.get(field.raw()));
            }
            String keptBody;
            if (json != null) {
                ObjectNode kept = JSON.createObjectNode();
                for (Field field : bodyFields) {
                    Boolean cleared = cleared(route, field, budget);
                    if (cleared == null) return null;
                    if (!cleared) kept.set(field.raw(), field.value());
                }
                keptBody = kept.isEmpty() ? "" : kept.toString();
            } else {
                List<String> kept = new ArrayList<>();
                for (Field field : bodyFields) {
                    Boolean cleared = cleared(route, field, budget);
                    if (cleared == null) return null;
                    if (!cleared) kept.add(field.raw());
                }
                keptBody = String.join("&", kept);
            }
            return SqlInjectionCore.inspectQueryAndBody(keptQuery, keptBody, budget) == null
                    ? new Review("relax", route.path(), null, null, found) : null;
        }

        /**
         * Reviews a finding from the URL path: the route's static segments are code, so only its
         * parameters can carry input. Typed ones must fit their type; text ones that reach no
         * operation are cleared, still getting the rules not in RELAXED_RULES.
         */
        public Review reviewPath(String path, int budget, String found) {
            String requested = path == null || path.isEmpty() ? "/" : path;
            Route route = routes.stream().filter(item -> item.compiled().pattern().matcher(requested).matches()).findFirst().orElse(null);
            if (route == null || route.inputs() == null) return null;
            java.util.regex.Matcher match = route.compiled().pattern().matcher(requested);
            if (!match.matches()) return null;
            boolean relaxable = found != null && RELAXED_RULES.contains(found) && !route.unnamedRisk();
            StringBuilder rebuilt = new StringBuilder();
            int last = 0;
            List<String> names = route.compiled().names();
            for (int index = 0; index < names.size(); index++) {
                String value = match.group(index + 1);
                if (value == null) continue;
                String name = names.get(index);
                String kind = index < route.pathTypes().size() && route.pathTypes().get(index) != null
                        ? route.pathTypes().get(index) : route.inputs().getOrDefault(name, "text");
                String text = decode(value);
                if (!kind.equals("text")) {
                    if (!fits(kind, TextNode.valueOf(text))) return new Review("mismatch", route.path(), name, kind, null);
                } else if (name.isEmpty() || route.reached().contains(name)) {
                    continue;
                } else {
                    String rule = SqlInjectionCore.inspectQueryAndBody(null, text, budget);
                    if (rule != null && !RELAXED_RULES.contains(rule)) return null;
                }
                rebuilt.append(requested, last, match.start(index + 1)).append('1');
                last = match.end(index + 1);
            }
            rebuilt.append(requested.substring(last));
            if (!relaxable) return null;
            return SqlInjectionCore.inspectPath(rebuilt.toString(), budget) == null ? new Review("relax", route.path(), null, null, found) : null;
        }

        /** True when the schema clears a field, false when it keeps every rule, null when it carries an attack. */
        private static Boolean cleared(Route route, Field field, int budget) {
            String kind = route.inputs().get(field.name());
            if (kind != null && !kind.equals("text")) return true;
            if (kind != null && route.reached().contains(field.name())) return false;
            String text = field.value().isTextual() ? field.value().textValue() : field.value().toString();
            String rule = SqlInjectionCore.inspectQueryAndBody(null, text, budget);
            return rule != null && !RELAXED_RULES.contains(rule) ? null : true;
        }
    }

    static void clearCache() {
        CACHE.clear();
    }

    /**
     * The general check's finding (or null) after this request's route schema reviews it, and
     * logs what the review found. Monitor, the default, only logs; enforce returns
     * schema_type_mismatch for a mismatch, schema_unexpected_input for an unexpected input (with
     * config.strictInputs) and null for a cleared finding. A failing review keeps it.
     */
    public static String apply(AiwafRequest request, AiwafConfig config, String finding) {
        return apply(request, config, finding == null ? null : new SqlInjectionCore.Found(finding, "payload"));
    }

    /** As apply, for a finding with where it came from: a header's always stands, a path's is reviewed by its parameters. */
    public static String apply(AiwafRequest request, AiwafConfig config, SqlInjectionCore.Found found) {
        String finding = found == null ? null : found.rule();
        if (found != null && found.source().equals("headers")) return finding;
        try {
            String mode = config == null || config.routeSchemaMode == null ? "off" : config.routeSchemaMode.trim().toLowerCase(Locale.ROOT);
            if (!config.pathManifestEnabled || !(mode.equals("enforce") || mode.equals("monitor"))) return finding;
            String location = config.pathManifestPath == null || config.pathManifestPath.isBlank()
                    ? PathManifestCore.DEFAULT_MANIFEST_PATH : config.pathManifestPath;
            List<String> allow = config.strictInputs ? List.copyOf(config.strictInputsAllow == null ? List.of() : config.strictInputsAllow) : null;
            List<String> exempt = List.copyOf(config.strictInputsExempt == null ? List.of() : config.strictInputsExempt);
            Schemas schemas = CACHE.computeIfAbsent(location + "\u0000" + allow + "\u0000" + exempt, key -> {
                Map<String, Object> manifest = PathManifestCore.loadManifest(Path.of(location));
                if (allow == null || manifest == null) return new Schemas(manifest);
                Map<String, Object> strict = new LinkedHashMap<>(manifest);
                strict.put("strict_inputs", Map.of("allow", allow, "exempt", exempt));
                return new Schemas(strict);
            });
            if (!schemas.hasInputs()) return finding;
            String body = config.requestBodyInspectionEnabled ? request.bodyPreview() : null;
            String rule = "payload_inspection_limit".equals(finding) ? null : finding;
            Review review = found != null && found.source().equals("path")
                    ? (rule == null ? null : schemas.reviewPath(request.path(), config.requestBodyInspectionBytes, rule))
                    : schemas.review(request.method(), request.path(), request.query(), body, config.requestBodyInspectionBytes, rule);
            if (review == null) return finding;
            Map<String, String> line = new LinkedHashMap<>();
            line.put("event", "aiwaf.route_schema");
            line.put("mode", mode);
            line.put("method", String.valueOf(request.method()));
            line.put("route", review.route());
            line.put("action", review.action());
            if (review.action().equals("mismatch")) {
                line.put("input", review.input());
                line.put("type", review.type());
            } else if (review.action().equals("unexpected")) {
                line.put("input", review.input());
            } else {
                line.put("rule", review.rule());
            }
            LOGGER.warning(JSON.writeValueAsString(line));
            if (!mode.equals("enforce")) return finding;
            if (review.action().equals("mismatch")) return "schema_type_mismatch";
            return review.action().equals("unexpected") ? "schema_unexpected_input" : null;
        } catch (Exception ex) {
            return finding;
        }
    }
}
