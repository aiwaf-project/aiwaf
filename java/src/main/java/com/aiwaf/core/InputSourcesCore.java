package com.aiwaf.core;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.lang.annotation.Annotation;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.Parameter;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.RecordComponent;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Where a route's inputs come from besides reading its code, and how they're combined: the same
 * rules as Python's {@code input_sources} and Node's {@code inputSources.js}.
 *
 * <ol>
 *   <li>The framework: each Spring handler method's parameters and body classes, read by
 *       reflection, including body classes from libraries the source scan can't see.</li>
 *   <li>Declarations: {@link AiwafInputs} on a handler or its class, or {@code .aiwaf/inputs.json}.</li>
 *   <li>An OpenAPI spec ({@code AIWAF_OPENAPI_SPEC}).</li>
 *   <li>Inputs recorded while tests or staging traffic ran (see
 *       {@code com.aiwaf.servlet.AiwafInputRecorderFilter}). These only add names, and a name the code
 *       read that the analysis didn't find makes the route incomplete.</li>
 * </ol>
 *
 * <p>A declaration or a closed OpenAPI operation is authoritative: the route is complete with what
 * it lists plus anything the code was seen reading.
 */
public final class InputSourcesCore {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final Set<String> KINDS = Set.of("int", "float", "bool", "uuid", "text");
    private static final int MAX_REASONS = 20;
    public static final String WHOLE = "*";

    private InputSourcesCore() {}

    /** Every source init combines with what reading the code found. Empty maps where there's none. */
    public record Sources(Map<String, Map<String, Object>> framework, Map<String, Map<String, Object>> annotations,
                          Map<String, Map<String, Object>> declarations, Map<String, Map<String, Map<String, Object>>> openapi,
                          Map<String, Map<String, String>> observations) {
        public static Sources none() {
            return new Sources(Map.of(), Map.of(), Map.of(), Map.of(), Map.of());
        }

        /** The same sources with the framework's and @AiwafInputs declarations, by manifest path. */
        public Sources withRoutes(Map<String, Map<String, Object>> framework, Map<String, Map<String, Object>> annotations) {
            return new Sources(framework, annotations, declarations, openapi, observations);
        }
    }

    static String kind(String value) {
        String name = value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
        if (KINDS.contains(name)) return name;
        if (name.equals("integer") || name.equals("long")) return "int";
        if (name.equals("number") || name.equals("double") || name.equals("decimal")) return "float";
        if (name.equals("boolean")) return "bool";
        return "text";
    }

    private static void merge(Map<String, String> inputs, String name, String kind) {
        inputs.merge(name, kind, (old, now) -> old.equals(now) ? old : "text");
    }

    private static Map<String, Object> reason(String code, String detail) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("code", code);
        out.put("detail", detail);
        return out;
    }

    private static final Pattern PARAMETER = Pattern.compile("^(?:<[^>]+>|\\{[^}]+\\}|:[A-Za-z_]\\w*\\??|\\*\\w*|\\*\\*)$");

    /** A route template with every parameter written {}, so /a/{id}, /a/:id and /a/<int:id> match. */
    public static String canonical(String route) {
        String text = route == null ? "" : route.split("\\?", 2)[0].trim();
        List<String> parts = new ArrayList<>();
        for (String part : text.split("/")) {
            if (!part.isEmpty()) parts.add(PARAMETER.matcher(part).matches() ? "{}" : part);
        }
        return "/" + String.join("/", parts);
    }

    // 1. The framework -------------------------------------------------------------------------

    private static final Map<String, String> TYPES = Map.ofEntries(
            Map.entry("int", "int"), Map.entry("Integer", "int"), Map.entry("long", "int"), Map.entry("Long", "int"),
            Map.entry("short", "int"), Map.entry("Short", "int"), Map.entry("byte", "int"), Map.entry("Byte", "int"),
            Map.entry("BigInteger", "int"), Map.entry("double", "float"), Map.entry("Double", "float"), Map.entry("float", "float"),
            Map.entry("Float", "float"), Map.entry("BigDecimal", "float"), Map.entry("boolean", "bool"), Map.entry("Boolean", "bool"),
            Map.entry("UUID", "uuid"), Map.entry("String", "text"), Map.entry("CharSequence", "text"), Map.entry("char", "text"),
            Map.entry("Character", "text"), Map.entry("LocalDate", "text"), Map.entry("LocalDateTime", "text"),
            Map.entry("Instant", "text"), Map.entry("OffsetDateTime", "text"), Map.entry("ZonedDateTime", "text"));
    private static final Set<String> REQUEST_TYPES = Set.of(
            "HttpServletRequest", "ServletRequest", "WebRequest", "NativeWebRequest", "MultipartHttpServletRequest");
    private static final Set<String> NOT_INPUTS = Set.of(
            "HttpServletResponse", "ServletResponse", "Model", "ModelMap", "BindingResult", "Errors", "Principal", "Authentication",
            "Locale", "HttpSession", "SessionStatus", "RedirectAttributes", "UriComponentsBuilder", "HttpHeaders", "TimeZone",
            "ZoneId", "HttpMethod");
    private static final Set<String> NOT_INPUT_ANNOTATIONS = Set.of(
            "RequestHeader", "CookieValue", "AuthenticationPrincipal", "RequestAttribute", "SessionAttribute", "Value", "PathVariable");
    private static final Set<String> WHOLE_TYPES = Set.of(
            "Map", "MultiValueMap", "HashMap", "LinkedHashMap", "JsonNode", "ObjectNode", "Object", "String");
    private static final Set<String> RAW_TYPES = Set.of("MultipartFile", "HttpEntity", "RequestEntity", "InputStream", "Reader");

    /** int, float, bool, uuid or text for a Java type, through collections, Optional and arrays; null for others. */
    static String scalar(Type type) {
        if (type instanceof Class<?> cls) {
            if (cls.isArray()) return scalar(cls.getComponentType());
            if (cls.isEnum()) return "text";
            return TYPES.get(cls.getSimpleName());
        }
        if (type instanceof ParameterizedType parameterized && parameterized.getRawType() instanceof Class<?> raw
                && (Collection.class.isAssignableFrom(raw) || raw == Optional.class)
                && parameterized.getActualTypeArguments().length == 1) {
            return scalar(parameterized.getActualTypeArguments()[0]);
        }
        return null;
    }

    private static Optional<Annotation> annotation(Annotation[] annotations, String simpleName) {
        return Arrays.stream(annotations).filter(item -> item.annotationType().getSimpleName().equals(simpleName)).findFirst();
    }

    /** The first non-empty value or name of an annotation, as @RequestParam("q") or @JsonProperty("tag_ids"). */
    static String annotationName(Annotation annotation) {
        for (String attribute : List.of("value", "name")) {
            try {
                Object value = annotation.annotationType().getMethod(attribute).invoke(annotation);
                if (value instanceof String text && !text.isEmpty()) return text;
                if (value instanceof String[] texts && texts.length > 0 && !texts[0].isEmpty()) return texts[0];
            } catch (ReflectiveOperationException ignored) {
                // Not an attribute of this annotation.
            }
        }
        return null;
    }

    /**
     * A body or form class's bound fields and types, or null when they can't all be known; why
     * holds (code, detail) then. Records by their components, other classes by their fields up
     * through their superclasses, with @JsonProperty names and @JsonIgnore fields left out.
     */
    static Map<String, String> beanFields(Class<?> type, String[] why) {
        String name = type.getSimpleName();
        if (WHOLE_TYPES.contains(name) || Map.class.isAssignableFrom(type) || type.isInterface() || type.isPrimitive()) {
            why[0] = "container_whole";
            why[1] = name + " takes the whole body";
            return null;
        }
        Map<String, String> fields = new LinkedHashMap<>();
        if (type.isRecord()) {
            for (RecordComponent component : type.getRecordComponents()) {
                Annotation[] marks = component.getAccessor().getAnnotations();
                Optional<Annotation> property = annotation(component.getAnnotations(), "JsonProperty").or(() -> annotation(marks, "JsonProperty"));
                String field = property.map(InputSourcesCore::annotationName).orElse(component.getName());
                String kind = scalar(component.getGenericType());
                merge(fields, field, kind == null ? "text" : kind);
            }
            return fields;
        }
        for (Class<?> current = type; current != null && current != Object.class; current = current.getSuperclass()) {
            for (Method method : current.getDeclaredMethods()) {
                if (annotation(method.getAnnotations(), "JsonAnySetter").isPresent()) {
                    why[0] = "model_fields";
                    why[1] = name + " takes any property (@JsonAnySetter)";
                    return null;
                }
            }
            for (Field field : current.getDeclaredFields()) {
                if (Modifier.isStatic(field.getModifiers()) || Modifier.isTransient(field.getModifiers()) || field.isSynthetic()) continue;
                Annotation[] marks = field.getAnnotations();
                if (annotation(marks, "JsonIgnore").isPresent()) continue;
                if (annotation(marks, "JsonAnySetter").isPresent()) {
                    why[0] = "model_fields";
                    why[1] = name + " takes any property (@JsonAnySetter)";
                    return null;
                }
                String key = annotation(marks, "JsonProperty").map(InputSourcesCore::annotationName).orElse(field.getName());
                String kind = scalar(field.getGenericType());
                merge(fields, key == null ? field.getName() : key, kind == null ? "text" : kind);
            }
        }
        return fields;
    }

    /**
     * What a Spring handler method reads by way of the framework: {inputs, covers, reasons,
     * standalone}. standalone means the list needs no source scan: the handler takes no request.
     */
    public static Map<String, Object> frameworkInputs(Method method) {
        Map<String, String> found = new LinkedHashMap<>();
        List<Map<String, Object>> reasons = new ArrayList<>();
        boolean standalone = true;
        for (Parameter parameter : method.getParameters()) {
            Annotation[] marks = parameter.getAnnotations();
            Set<String> names = new HashSet<>();
            for (Annotation mark : marks) names.add(mark.annotationType().getSimpleName());
            Class<?> type = parameter.getType();
            String typeName = type.getSimpleName();
            if (REQUEST_TYPES.contains(typeName)) {
                standalone = false;  // The source scan reads what's taken from the request.
                continue;
            }
            if (NOT_INPUTS.contains(typeName) || names.stream().anyMatch(NOT_INPUT_ANNOTATIONS::contains)) continue;
            if (typeName.equals("Pageable")) {
                found.put("page", "int");
                found.put("size", "int");
                found.put("sort", "text");
                continue;
            }
            if (RAW_TYPES.contains(typeName) || names.contains("RequestPart")) {
                reasons.add(reason("raw_body", "parameter " + parameter.getName() + " reads the raw body or a multipart part (" + typeName + ")"));
                continue;
            }
            String kind = scalar(parameter.getParameterizedType());
            if (names.contains("RequestParam") || (kind != null && !names.contains("RequestBody") && !names.contains("ModelAttribute"))) {
                if (kind == null) {
                    if (Map.class.isAssignableFrom(type)) reasons.add(reason("container_whole", "a parameter binds every parameter (" + typeName + ")"));
                    else reasons.add(reason("external_class", "a parameter has a type that isn't known (" + typeName + ")"));
                    continue;
                }
                String named = annotation(marks, "RequestParam").map(InputSourcesCore::annotationName).orElse(null);
                if (named == null && !parameter.isNamePresent()) {
                    reasons.add(reason("dynamic_key", "a parameter's name isn't compiled in; compile with -parameters or name it in @RequestParam"));
                    continue;
                }
                merge(found, named != null ? named : parameter.getName(), kind);
                continue;
            }
            String[] why = new String[2];
            Map<String, String> fields = beanFields(type, why);
            if (fields == null) reasons.add(reason(why[0], why[1]));
            else fields.forEach((name, fieldKind) -> merge(found, name, fieldKind));
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("inputs", found);
        out.put("covers", List.of("external_class", "model_fields"));
        out.put("reasons", reasons);
        out.put("standalone", standalone);
        return out;
    }

    // 2. Declarations ----------------------------------------------------------------------------

    /** The @AiwafInputs on a handler, or on its class, as {inputs, complete}; null when there's none. */
    public static Map<String, Object> declaredOn(Method method) {
        AiwafInputs declared = method.getAnnotation(AiwafInputs.class);
        if (declared == null) declared = method.getDeclaringClass().getAnnotation(AiwafInputs.class);
        if (declared == null) return null;
        Map<String, String> inputs = new LinkedHashMap<>();
        for (String item : declared.value()) {
            String[] parts = item.split(":", 2);
            if (!parts[0].isBlank()) inputs.put(parts[0].trim(), kind(parts.length > 1 ? parts[1] : "text"));
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("inputs", inputs);
        out.put("complete", declared.complete());
        return out;
    }

    /** .aiwaf/inputs.json: {"/route": {"inputs": {...}, "complete": true}}, optionally under "routes". */
    public static Map<String, Map<String, Object>> loadDeclarations(Path file) throws IOException {
        JsonNode loaded = JSON.readTree(Files.readString(file));
        JsonNode routes = loaded != null && loaded.has("routes") ? loaded.get("routes") : loaded;
        Map<String, Map<String, Object>> declared = new TreeMap<>();
        if (routes == null || !routes.isObject()) return declared;
        for (Iterator<Map.Entry<String, JsonNode>> it = routes.fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> route = it.next();
            JsonNode spec = route.getValue();
            if (!spec.isObject()) continue;
            Map<String, String> inputs = new LinkedHashMap<>();
            JsonNode given = spec.path("inputs");
            for (Iterator<Map.Entry<String, JsonNode>> fields = given.fields(); fields.hasNext(); ) {
                Map.Entry<String, JsonNode> field = fields.next();
                inputs.put(field.getKey(), kind(field.getValue().asText()));
            }
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("inputs", inputs);
            out.put("complete", !spec.has("complete") || spec.get("complete").asBoolean(true));
            declared.put(canonical(route.getKey()), out);
        }
        return declared;
    }

    // 3. OpenAPI ---------------------------------------------------------------------------------

    static String openapiKind(JsonNode schema) {
        JsonNode type = schema.path("type");
        String kind = type.isTextual() ? type.asText() : null;
        if (type.isArray()) {
            List<String> kinds = new ArrayList<>();
            type.forEach(item -> { if (!item.asText().equals("null")) kinds.add(item.asText()); });
            kind = kinds.size() == 1 ? kinds.get(0) : null;
        }
        if ("integer".equals(kind)) return "int";
        if ("number".equals(kind)) return "float";
        if ("boolean".equals(kind)) return "bool";
        if ("string".equals(kind) && "uuid".equals(schema.path("format").asText())) return "uuid";
        if ("array".equals(kind) && schema.path("items").isObject()) return openapiKind(schema.get("items"));
        return "text";
    }

    /** Each operation's query and body inputs in an OpenAPI 3 spec, by canonical path and method. */
    public static Map<String, Map<String, Map<String, Object>>> openapiInputs(JsonNode spec) {
        Map<String, Map<String, Map<String, Object>>> result = new TreeMap<>();
        for (Iterator<Map.Entry<String, JsonNode>> paths = spec.path("paths").fields(); paths.hasNext(); ) {
            Map.Entry<String, JsonNode> path = paths.next();
            JsonNode item = resolve(spec, path.getValue());
            if (item == null) continue;
            for (Iterator<Map.Entry<String, JsonNode>> operations = item.fields(); operations.hasNext(); ) {
                Map.Entry<String, JsonNode> entry = operations.next();
                String method = entry.getKey().toUpperCase(Locale.ROOT);
                if (!Set.of("GET", "POST", "PUT", "PATCH", "DELETE").contains(method) || !entry.getValue().isObject()) continue;
                JsonNode operation = entry.getValue();
                Map<String, String> found = new LinkedHashMap<>();
                List<Map<String, Object>> reasons = new ArrayList<>();
                List<JsonNode> parameters = new ArrayList<>();
                item.path("parameters").forEach(parameters::add);
                operation.path("parameters").forEach(parameters::add);
                for (JsonNode raw : parameters) {
                    JsonNode parameter = resolve(spec, raw);
                    if (parameter == null) {
                        reasons.add(reason("external_class", "a parameter reference can't be resolved"));
                        continue;
                    }
                    if (!"query".equals(parameter.path("in").asText())) continue;
                    JsonNode schema = Optional.ofNullable(resolve(spec, parameter.path("schema"))).orElse(JSON.createObjectNode());
                    if ("object".equals(schema.path("type").asText()) && parameter.path("explode").asBoolean(true) && !schema.has("properties")) {
                        reasons.add(reason("container_whole", "query parameter " + parameter.path("name").asText() + " is a free-form object"));
                        continue;
                    }
                    found.put(parameter.path("name").asText(), openapiKind(schema));
                }
                JsonNode body = resolve(spec, operation.path("requestBody"));
                if (body != null) {
                    for (Iterator<Map.Entry<String, JsonNode>> contents = body.path("content").fields(); contents.hasNext(); ) {
                        Map.Entry<String, JsonNode> content = contents.next();
                        String media = content.getKey().split(";")[0].trim();
                        if (Set.of("application/json", "application/x-www-form-urlencoded", "multipart/form-data").contains(media)
                                || media.endsWith("+json")) {
                            properties(spec, content.getValue().path("schema"), reasons, 0).forEach((name, kind) -> merge(found, name, kind));
                        } else {
                            reasons.add(reason("raw_body", "the body is " + content.getKey()));
                        }
                    }
                }
                Map<String, Object> out = new LinkedHashMap<>();
                out.put("inputs", found);
                out.put("complete", reasons.isEmpty());
                out.put("reasons", reasons);
                result.computeIfAbsent(canonical(path.getKey()), key -> new TreeMap<>()).put(method, out);
            }
        }
        return result;
    }

    private static JsonNode resolve(JsonNode spec, JsonNode node) {
        int depth = 0;
        while (node != null && node.isObject() && node.has("$ref") && depth < 16) {
            String ref = node.get("$ref").asText();
            if (!ref.startsWith("#/")) return null;
            JsonNode target = spec;
            for (String part : ref.substring(2).split("/")) {
                target = target == null ? null : target.get(part.replace("~1", "/").replace("~0", "~"));
            }
            node = target;
            depth++;
        }
        return node != null && node.isObject() ? node : null;
    }

    private static Map<String, String> properties(JsonNode spec, JsonNode input, List<Map<String, Object>> reasons, int depth) {
        JsonNode schema = resolve(spec, input);
        if (schema == null || depth > 8) {
            reasons.add(reason("external_class", "the body schema can't be resolved"));
            return Map.of();
        }
        Map<String, String> found = new LinkedHashMap<>();
        boolean combined = false;
        for (String key : List.of("allOf", "oneOf", "anyOf")) {
            if (schema.path(key).isArray()) {
                combined = true;
                schema.get(key).forEach(part -> found.putAll(properties(spec, part, reasons, depth + 1)));
            }
        }
        if (combined && !schema.has("properties")) return found;
        if (!schema.path("type").asText("object").equals("object")) {
            reasons.add(reason("container_whole", "the body isn't an object with named properties"));
            return found;
        }
        JsonNode extra = schema.get("additionalProperties");
        if (extra != null && !extra.isNull() && !(extra.isBoolean() && !extra.asBoolean())) {
            reasons.add(reason("container_whole", "the body schema allows properties it doesn't name"));
        }
        for (Iterator<Map.Entry<String, JsonNode>> it = schema.path("properties").fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> property = it.next();
            found.put(property.getKey(), openapiKind(Optional.ofNullable(resolve(spec, property.getValue())).orElse(JSON.createObjectNode())));
        }
        return found;
    }

    /** An OpenAPI spec from a JSON file. */
    public static JsonNode loadSpec(Path file) throws IOException {
        if (file.toString().matches("(?i).*\\.ya?ml$")) {
            throw new IOException(file + " is YAML; give init a JSON spec (springdoc serves /v3/api-docs as JSON)");
        }
        JsonNode spec = JSON.readTree(Files.readString(file));
        if (spec == null || !spec.has("paths")) throw new IOException(file + " isn't an OpenAPI spec: it has no paths");
        return spec;
    }

    // 4. Recorded inputs -------------------------------------------------------------------------

    /** Recorded reads, by canonical route to {name: type}. */
    public static Map<String, Map<String, String>> loadObservations(Path file) throws IOException {
        Map<String, Map<String, String>> found = new TreeMap<>();
        for (String line : Files.readAllLines(file)) {
            JsonNode item;
            try {
                item = JSON.readTree(line);
            } catch (IOException ignored) {
                continue;
            }
            if (item == null || !item.isObject() || item.path("route").asText().isEmpty() || item.path("key").asText().isEmpty()) continue;
            Map<String, String> route = found.computeIfAbsent(canonical(item.get("route").asText()), key -> new LinkedHashMap<>());
            String key = item.get("key").asText();
            if (key.equals(WHOLE)) route.put(WHOLE, WHOLE);
            else merge(route, key, kind(item.path("type").asText("text")));
        }
        return found;
    }

    /**
     * The declaration file, OpenAPI spec and recorded inputs init should read, where present:
     * AIWAF_INPUTS_FILE (default .aiwaf/inputs.json), AIWAF_OPENAPI_SPEC (no default) and
     * AIWAF_OBSERVED_INPUTS (default .aiwaf/observed_inputs.jsonl, which the recorder writes).
     */
    public static Sources sourcesFromEnvironment(Path root, Map<String, String> env) throws IOException {
        Path declarations = env.containsKey("AIWAF_INPUTS_FILE") ? Path.of(env.get("AIWAF_INPUTS_FILE")) : root.resolve(".aiwaf/inputs.json");
        Path observed = env.containsKey("AIWAF_OBSERVED_INPUTS") ? Path.of(env.get("AIWAF_OBSERVED_INPUTS"))
                : root.resolve(".aiwaf/observed_inputs.jsonl");
        String spec = env.get("AIWAF_OPENAPI_SPEC");
        return new Sources(Map.of(), Map.of(),
                Files.isRegularFile(declarations) ? loadDeclarations(declarations) : Map.of(),
                spec != null && !spec.isBlank() ? openapiInputs(loadSpec(Path.of(spec))) : Map.of(),
                Files.isRegularFile(observed) ? loadObservations(observed) : Map.of());
    }

    /**
     * The framework's and @AiwafInputs declarations for discovered routes, by manifest path, added to
     * base. Handlers that share a path are combined: their inputs and reasons together, standalone
     * only when every one is.
     */
    @SuppressWarnings("unchecked")
    public static Sources forRoutes(Collection<PathManifestCore.RouteInfo> routes, Sources base) {
        Map<String, Map<String, Object>> framework = new TreeMap<>();
        Map<String, Map<String, Object>> annotations = new TreeMap<>();
        for (PathManifestCore.RouteInfo route : routes) {
            if (route == null || route.method() == null) continue;
            String path = PathManifestCore.normalizePath(route.path());
            Map<String, Object> found = frameworkInputs(route.method());
            framework.merge(path, found, (old, now) -> {
                Map<String, Object> combined = new LinkedHashMap<>(old);
                Map<String, String> inputs = new LinkedHashMap<>((Map<String, String>) old.get("inputs"));
                ((Map<String, String>) now.get("inputs")).forEach((name, kind) -> merge(inputs, name, kind));
                List<Object> reasons = new ArrayList<>((List<Object>) old.get("reasons"));
                reasons.addAll((List<Object>) now.get("reasons"));
                combined.put("inputs", inputs);
                combined.put("reasons", reasons);
                combined.put("standalone", Boolean.TRUE.equals(old.get("standalone")) && Boolean.TRUE.equals(now.get("standalone")));
                return combined;
            });
            Map<String, Object> declared = declaredOn(route.method());
            if (declared != null) annotations.putIfAbsent(path, declared);
        }
        return base.withRoutes(framework, annotations);
    }

    // Combining ----------------------------------------------------------------------------------

    /** Counts of routes with complete and incomplete inputs, and of each reason. */
    public record Counts(int complete, int incomplete, Map<String, Integer> reasons) {}

    /** Combine every source into each route's inputs, inputs_complete and inputs_incomplete. */
    @SuppressWarnings("unchecked")
    private static final Pattern PATH_PARAMETER = Pattern.compile("<(?:[^:>]+:)?([^>]+)>|\\{([^}:]+)(?::[^}]*)?\\}|:([A-Za-z_]\\w*)\\??");

    /** The parameter names in a route, as id in /a/{id}, /a/<int:id> or /a/:id. */
    static List<String> pathParameters(String route) {
        List<String> names = new ArrayList<>();
        Matcher matcher = PATH_PARAMETER.matcher(route == null ? "" : route);
        while (matcher.find()) {
            for (int group = 1; group <= matcher.groupCount(); group++) {
                if (matcher.group(group) != null) {
                    names.add(matcher.group(group));
                    break;
                }
            }
        }
        return names;
    }

    /**
     * Each method's names, when the route's complete list is read by different methods' code: a
     * method takes the names its own code reads, and every name no method's code claims. Null
     * when every method reads them all.
     */
    static Map<String, Object> byMethod(Map<String, String> found, Object methods, Map<String, Set<String>> claims, Set<String> shared) {
        Set<String> all = new TreeSet<>();
        if (methods instanceof Collection<?> list) list.forEach(method -> all.add(String.valueOf(method).toUpperCase(Locale.ROOT)));
        if (all.size() < 2 || claims.isEmpty()) return null;
        Set<String> claimed = new TreeSet<>();
        claims.values().forEach(claimed::addAll);
        Map<String, Object> result = new TreeMap<>();
        boolean narrower = false;
        for (String method : all) {
            List<String> names = found.keySet().stream()
                    .filter(name -> claims.getOrDefault(method, Set.of()).contains(name) || shared.contains(name) || !claimed.contains(name)).toList();
            narrower |= names.size() < found.size();
            result.put(method, names);
        }
        return narrower ? result : null;
    }

    public static Counts finalize(Map<String, Object> routes, Sources sources) {
        int complete = 0;
        int incomplete = 0;
        Map<String, Integer> counted = new TreeMap<>();
        for (Map.Entry<String, Object> route : routes.entrySet()) {
            if (!(route.getValue() instanceof Map<?, ?> raw)) continue;
            Map<String, Object> data = (Map<String, Object>) raw;
            // Which method reads what: the code reading's and OpenAPI's, per method. Names from
            // anywhere else are shared by every method.
            Map<String, Set<String>> claims = new TreeMap<>();
            if (data.remove("_input_methods") instanceof Map<?, ?> saved) {
                saved.forEach((method, names) -> {
                    if (names instanceof Collection<?> list) list.forEach(name -> claims.computeIfAbsent(String.valueOf(method), key -> new TreeSet<>()).add(String.valueOf(name)));
                });
            }
            Set<String> shared = new TreeSet<>();
            Map<String, Object> nested = data.remove("_nested") instanceof Map<?, ?> savedNested ? (Map<String, Object>) savedNested : Map.of();
            Set<String> flat = new TreeSet<>();
            if (data.remove("_flat") instanceof Collection<?> savedFlat) savedFlat.forEach(name -> flat.add(String.valueOf(name)));
            boolean read = data.containsKey("inputs_complete");
            Map<String, String> found = new TreeMap<>();
            if (data.get("inputs") instanceof Map<?, ?> saved) saved.forEach((name, kind) -> found.put(String.valueOf(name), String.valueOf(kind)));
            List<Map<String, Object>> reasons = new ArrayList<>();
            if (data.get("inputs_incomplete") instanceof List<?> saved) {
                for (Object item : saved) if (item instanceof Map<?, ?> map) reasons.add((Map<String, Object>) map);
            }
            Map<String, Object> framework = sources.framework().get(route.getKey());
            if (framework != null) {
                Collection<Object> covers = (Collection<Object>) framework.getOrDefault("covers", List.of());
                reasons.removeIf(item -> covers.contains(item.get("code")));
                reasons.addAll((Collection<Map<String, Object>>) framework.getOrDefault("reasons", List.of()));
                ((Map<String, String>) framework.getOrDefault("inputs", Map.of())).forEach((name, kind) -> {
                    merge(found, name, kind);
                    shared.add(name);
                });
                read = read || Boolean.TRUE.equals(framework.get("standalone"));
            }
            String key = canonical(route.getKey());
            boolean authoritative = false;
            for (Map<String, Object> declared : Arrays.asList(sources.annotations().get(route.getKey()), sources.declarations().get(key))) {
                if (declared == null) continue;
                found.putAll((Map<String, String>) declared.getOrDefault("inputs", Map.of()));
                shared.addAll(((Map<String, String>) declared.getOrDefault("inputs", Map.of())).keySet());
                flat.addAll(((Map<String, String>) declared.getOrDefault("inputs", Map.of())).keySet());
                authoritative = !Boolean.FALSE.equals(declared.get("complete")) || authoritative;
                read = true;
            }
            Map<String, Map<String, Object>> operations = sources.openapi().get(key);
            if (operations != null) {
                List<String> methods = data.get("methods") instanceof List<?> listed && !listed.isEmpty()
                        ? listed.stream().map(String::valueOf).toList() : List.copyOf(operations.keySet());
                List<Map<String, Object>> matched = methods.stream().filter(operations::containsKey).map(operations::get).toList();
                for (String method : methods) {
                    if (!operations.containsKey(method)) continue;
                    ((Map<String, String>) operations.get(method).get("inputs")).forEach((name, kind) -> {
                        merge(found, name, kind);
                        claims.computeIfAbsent(method.toUpperCase(Locale.ROOT), item -> new TreeSet<>()).add(name);
                    });
                }
                if (!matched.isEmpty() && matched.size() == methods.size() && matched.stream().allMatch(op -> Boolean.TRUE.equals(op.get("complete")))) {
                    authoritative = true;
                }
                read = true;
            }
            if (authoritative) reasons.clear();
            for (Map.Entry<String, String> seen : sources.observations().getOrDefault(key, Map.of()).entrySet()) {
                if (seen.getKey().equals(WHOLE)) {
                    reasons.add(reason("container_whole", "the code was seen reading a query or body container whole"));
                    continue;
                }
                if (!found.containsKey(seen.getKey()) && read) {
                    reasons.add(reason("observed", seen.getKey() + " was read while recording, but the analysis didn't find it"));
                }
                merge(found, seen.getKey(), seen.getValue());
                shared.add(seen.getKey());
                read = true;
            }
            if (!read) continue;
            // A route's own parameters come from the path, not the query or body.
            pathParameters(route.getKey()).forEach(found::remove);
            data.put("inputs", found);
            data.put("inputs_complete", reasons.isEmpty());
            // Keys inside an input, from the body class's nested classes, when every handler reading it has them.
            Map<String, Object> inner = new TreeMap<>();
            nested.forEach((name, keys) -> {
                if (found.containsKey(name) && !flat.contains(name)) inner.put(name, keys);
            });
            if (!inner.isEmpty() && reasons.isEmpty()) data.put("inputs_nested", inner);
            else data.remove("inputs_nested");
            Map<String, Object> byMethod = reasons.isEmpty() ? byMethod(found, data.get("methods"), claims, shared) : null;
            if (byMethod != null) data.put("inputs_by_method", byMethod);
            else data.remove("inputs_by_method");
            if (reasons.isEmpty()) {
                data.remove("inputs_incomplete");
                complete++;
            } else {
                data.put("inputs_incomplete", List.copyOf(reasons.subList(0, Math.min(reasons.size(), MAX_REASONS))));
                incomplete++;
                for (Map<String, Object> item : reasons) counted.merge(String.valueOf(item.get("code")), 1, Integer::sum);
            }
        }
        return new Counts(complete, incomplete, counted);
    }

    /** One line for init's summary: how many routes have complete inputs, and the top reasons. */
    public static String describe(Counts counts) {
        int total = counts.complete() + counts.incomplete();
        if (total == 0) return "";
        StringBuilder text = new StringBuilder("complete inputs for " + counts.complete() + " of " + total + " routes");
        List<Map.Entry<String, Integer>> top = new ArrayList<>(counts.reasons().entrySet());
        top.sort(Map.Entry.<String, Integer>comparingByValue().reversed().thenComparing(Map.Entry.comparingByKey()));
        if (!top.isEmpty()) {
            text.append("; incomplete because of ");
            text.append(String.join(", ", top.stream().limit(4).map(entry -> entry.getKey() + " (" + entry.getValue() + ")").toList()));
        }
        return text.toString();
    }
}
