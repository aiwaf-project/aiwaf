package com.aiwaf.core;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.springframework.core.annotation.AnnotatedElementUtils;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.util.ReflectionUtils;

import java.io.IOException;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Collection;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/** Route discovery and Python-compatible path-manifest generation. */
public final class PathManifestCore {
    public static final String SCHEMA_VERSION = "1.0";
    public static final String DEFAULT_MANIFEST_PATH = ".aiwaf/paths.json";
    private static final Set<String> HTTP_METHODS = Set.of("GET", "POST", "PUT", "PATCH", "DELETE");
    private static final Set<String> MIDDLEWARE_NAMES = Set.of(
            "geo_block", "ip_keyword_block", "rate_limit", "ai_anomaly", "honeypot",
            "uuid_tamper", "header_validation", "logging"
    );
    private static final long MAX_MANIFEST_BYTES = 10L * 1024L * 1024L;
    private static final ObjectMapper MAPPER = new ObjectMapper()
            .enable(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS);

    private PathManifestCore() {}

    /**
     * Backward-compatible entry API. The returned entries retain the original Java type while
     * the written document uses the shared Python/JavaScript manifest schema.
     */
    public static List<PathManifestEntry> generateManifest(List<RouteInfo> routes, String outputPath) {
        List<RouteInfo> safeRoutes = routes == null ? List.of() : routes;
        List<PathManifestEntry> entries = new ArrayList<>();
        for (RouteInfo route : safeRoutes) {
            ApiDetection api = ApiDetectionCore.detectApiEndpoint(route.method(), route.controllerClass(), route.path());
            Map<String, Object> auth = AuthDetectionCore.detectAuthEndpoint(route.path(), route.method(), route.controllerClass());
            entries.add(new PathManifestEntry(normalizePath(route.path()), normalizedMethods(route.httpMethods()), api, auth));
        }
        if (outputPath != null && !outputPath.isBlank()) {
            writeManifest(buildManifest("spring", safeRoutes), Path.of(outputPath));
        }
        return List.copyOf(entries);
    }

    public static Map<String, Object> buildManifest(String framework, Collection<RouteInfo> routes) {
        String frameworkName = framework == null || framework.isBlank() ? "spring" : framework;
        Map<String, Map<String, Object>> routeMap = new TreeMap<>();
        if (routes != null) {
            routes.stream()
                    .filter(route -> route != null && !isInternalAiwafPath(route.path()))
                    .sorted(Comparator.comparing(route -> normalizePath(route.path())))
                    .forEach(route -> mergeRoute(routeMap, route));
        }

        Map<String, Object> context = new TreeMap<>();
        context.put("app_context", Map.of());
        context.put("framework", frameworkName);
        context.put("routes", routeMap);

        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("schema_version", SCHEMA_VERSION);
        manifest.put("framework", frameworkName);
        manifest.put("context_hash", sha256(stableJson(context)));
        manifest.put("generated_at", Instant.now().truncatedTo(ChronoUnit.SECONDS).toString());
        manifest.put("routes", routeMap);
        return manifest;
    }

    public static Path writeManifest(Map<String, Object> manifest, Path outputPath) {
        if (outputPath == null) throw new IllegalArgumentException("outputPath must not be null");
        try {
            Path absolute = outputPath.toAbsolutePath().normalize();
            if (absolute.getParent() != null) Files.createDirectories(absolute.getParent());
            String json = MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(manifest) + "\n";
            Files.writeString(absolute, json, StandardCharsets.UTF_8);
            return absolute;
        } catch (IOException ex) {
            throw new IllegalStateException("Failed to write manifest to " + outputPath + ": " + ex.getMessage(), ex);
        }
    }

    /** Load a generated manifest. Missing, oversized, malformed, or incompatible files are ignored safely. */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> loadManifest(Path manifestPath) {
        if (manifestPath == null || !Files.isRegularFile(manifestPath)) return Map.of();
        try {
            if (Files.size(manifestPath) > MAX_MANIFEST_BYTES) return Map.of();
            Object raw = MAPPER.readValue(manifestPath.toFile(), Object.class);
            if (!(raw instanceof Map<?, ?> source)) return Map.of();
            Map<String, Object> manifest = stringObjectMap(source);
            Object schema = manifest.get("schema_version");
            if (schema != null && !SCHEMA_VERSION.equals(String.valueOf(schema))) return Map.of();
            return manifest;
        } catch (IOException | RuntimeException ignored) {
            return Map.of();
        }
    }

    /** Compile the shared manifest protections into the same path rules used by the Java engine. */
    public static List<AiwafConfig.PathRule> compileManifestToPathRules(Map<String, Object> manifest) {
        if (manifest == null || !(manifest.get("routes") instanceof Map<?, ?> routes)) return List.of();
        List<AiwafConfig.PathRule> rules = new ArrayList<>();
        for (Map.Entry<?, ?> route : routes.entrySet()) {
            if (route.getKey() == null || !(route.getValue() instanceof Map<?, ?> rawEntry)) continue;
            Map<String, Object> entry = stringObjectMap(rawEntry);
            Map<String, Object> protections = entry.get("protections") instanceof Map<?, ?> raw
                    ? stringObjectMap(raw) : Map.of();
            Set<String> disabled = new HashSet<>();
            for (Map.Entry<String, Object> protection : protections.entrySet()) {
                String name = protection.getKey().trim().toLowerCase(Locale.ROOT);
                if (!MIDDLEWARE_NAMES.contains(name)) continue;
                Object value = protection.getValue();
                if (Boolean.FALSE.equals(value)) {
                    disabled.add(name);
                } else if (value instanceof Map<?, ?> options
                        && Boolean.FALSE.equals(stringObjectMap(options).get("enabled"))) {
                    disabled.add(name);
                }
            }

            Map<String, Object> rate = protectionMap(protections.get("rate_limit"));
            if (rate.isEmpty()) rate = protectionMap(protections.get("api_rate_limit"));
            Integer max = firstInteger(rate, "MAX", "max", "requests");
            Integer window = firstInteger(rate, "WINDOW", "window", "window_seconds");
            Integer flood = firstInteger(rate, "FLOOD", "flood");
            if (disabled.isEmpty() && max == null && window == null && flood == null) continue;

            Map<String, Map<String, Integer>> overrides = new HashMap<>();
            Map<String, Integer> rateOverrides = new HashMap<>();
            if (max != null) rateOverrides.put("max", max);
            if (window != null) rateOverrides.put("window", window);
            if (flood != null) rateOverrides.put("flood", flood);
            if (!rateOverrides.isEmpty()) overrides.put("rate_limit", rateOverrides);
            rules.add(new AiwafConfig.PathRule(
                    ExemptionsCore.normalizePath(String.valueOf(route.getKey()), true),
                    false, max, window, flood, disabled, overrides
            ));
        }
        return List.copyOf(rules);
    }

    /** Append manifest-derived rules after explicit rules so equal-prefix explicit settings win. */
    public static synchronized void applyManifest(AiwafConfig config) {
        if (config == null || !config.pathManifestEnabled || config.pathManifestApplied) return;
        config.pathManifestApplied = true;
        String configured = config.pathManifestPath;
        if (configured == null || configured.isBlank()) configured = DEFAULT_MANIFEST_PATH;
        try {
            config.pathRules.addAll(compileManifestToPathRules(loadManifest(Path.of(configured))));
        } catch (RuntimeException ignored) {
            // A malformed optional path must not prevent the application from starting.
        }
    }

    /** Discover concrete mappings from controller classes or instances without inventing routes. */
    public static List<RouteInfo> discoverControllerRoutes(Object... controllers) {
        List<RouteInfo> routes = new ArrayList<>();
        if (controllers == null) return routes;
        for (Object controller : controllers) {
            if (controller == null) continue;
            Class<?> type = controller instanceof Class<?> cls ? cls : controller.getClass();
            String[] classPaths = extractPaths(AnnotatedElementUtils.findMergedAnnotation(type, RequestMapping.class));
            if (classPaths.length == 0) classPaths = new String[]{""};
            for (Method method : ReflectionUtils.getUniqueDeclaredMethods(type)) {
                Mapping mapping = extractMethodMapping(method);
                if (mapping == null) continue;
                String[] methodPaths = mapping.paths().length == 0 ? new String[]{""} : mapping.paths();
                List<String> methods = normalizedMethods(requestMethods(mapping.methods()));
                if (methods.isEmpty()) methods = HTTP_METHODS.stream().sorted().toList();
                for (String classPath : classPaths) {
                    for (String methodPath : methodPaths) {
                        routes.add(new RouteInfo(joinPaths(classPath, methodPath), methods, type, method));
                    }
                }
            }
        }
        routes.sort(Comparator.comparing(RouteInfo::path).thenComparing(route -> route.method().getName()));
        return routes;
    }

    public static String normalizePath(String path) {
        if (path == null || path.isBlank()) return "/";
        String normalized = path.trim().replaceAll("/+", "/");
        if (!normalized.startsWith("/")) normalized = "/" + normalized;
        if (normalized.length() > 1 && normalized.endsWith("/")) {
            normalized = normalized.substring(0, normalized.length() - 1);
        }
        return normalized.toLowerCase(Locale.ROOT);
    }

    public static boolean isInternalAiwafPath(String path) {
        String normalized = normalizePath(path);
        return normalized.equals("/aiwaf") || normalized.startsWith("/aiwaf/")
                || normalized.equals("/.aiwaf") || normalized.startsWith("/.aiwaf/");
    }

    private static void mergeRoute(Map<String, Map<String, Object>> routes, RouteInfo route) {
        String path = normalizePath(route.path());
        Map<String, Object> fresh = routeEntry(route, path);
        Map<String, Object> existing = routes.get(path);
        if (existing == null) {
            routes.put(path, fresh);
            return;
        }
        TreeSet<String> methods = new TreeSet<>();
        addStrings(methods, existing.get("methods"));
        addStrings(methods, fresh.get("methods"));
        existing.put("methods", List.copyOf(methods));
    }

    private static Map<String, Object> routeEntry(RouteInfo route, String path) {
        List<String> methods = normalizedMethods(route.httpMethods());
        ApiDetection api = ApiDetectionCore.detectApiEndpoint(route.method(), route.controllerClass(), path);
        Map<String, Object> auth = AuthDetectionCore.detectAuthEndpoint(path, route.method(), route.controllerClass());

        String lower = path.toLowerCase(Locale.ROOT);
        boolean authAction = Boolean.TRUE.equals(auth.get("is_auth"));
        boolean authRequired = startsWithAny(lower,
                "/portal/", "/dashboard/", "/account/", "/accounts/", "/profile/", "/settings/");
        String category = "unknown";
        String responseType = api.responseType() == null || api.responseType().isBlank() ? "html" : api.responseType();

        Map<String, Object> protections = new LinkedHashMap<>();
        protections.put("rate_limit", mapOf("requests", 60, "window_seconds", 60));
        protections.put("header_validation", mapOf("enabled", true));
        protections.put("ip_keyword_block", mapOf("enabled", true));
        protections.put("ai_anomaly", mapOf("enabled", true));

        if (isStaticPath(lower)) {
            category = "static";
            protections.put("header_validation", mapOf("enabled", false));
            protections.put("ai_anomaly", mapOf("enabled", false));
            protections.put("honeypot", mapOf("enabled", false));
        } else if (lower.equals("/admin") || lower.startsWith("/admin/")) {
            category = "admin";
            authRequired = true;
            protections.put("rate_limit", mapOf("requests", 30, "window_seconds", 60));
        } else if (authAction || containsAny(lower, "/login", "/signin", "/auth/login")) {
            category = "auth";
            protections.put("rate_limit", mapOf("requests", 30, "window_seconds", 60));
            protections.put("honeypot", mapOf("enabled", true));
        } else if ("form".equals(api.payloadType()) || api.formConfidence() > 0.0) {
            category = "form";
            responseType = api.responseType() == null || api.responseType().isBlank() ? "mixed" : api.responseType();
            protections.put("rate_limit", mapOf("requests", 30, "window_seconds", 60));
            protections.put("payload_validation", mapOf("max_body_bytes", 1_048_576));
            protections.put("honeypot", mapOf("enabled", true));
        } else if (api.isApi() || api.confidence() > 0.0 || lower.startsWith("/api/")) {
            category = "api";
            responseType = "json";
            protections.put("rate_limit", mapOf("requests", 120, "window_seconds", 60));
            protections.put("api_rate_limit", mapOf("requests", 120, "window_seconds", 60));
            protections.put("payload_validation", mapOf("max_body_bytes", 1_048_576, "max_json_depth", 8));
            protections.put("content_type_validation", mapOf("require_valid_content_type", true));
            protections.put("honeypot", mapOf("enabled", false));
        } else if (containsAny(lower, "/upload", "/uploads", "/files")) {
            category = "upload";
            protections.put("rate_limit", mapOf("requests", 20, "window_seconds", 60));
            protections.put("payload_validation", mapOf("max_body_bytes", 1_048_576));
        } else if (authRequired) {
            category = "app";
            protections.put("rate_limit", mapOf("requests", 90, "window_seconds", 60));
        }
        if (methods.contains("POST") && !Set.of("api", "upload").contains(category)) {
            protections.put("rate_limit", mapOf("requests", 30, "window_seconds", 60));
        }

        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("methods", methods);
        entry.put("view", route.controllerClass() == null || route.method() == null
                ? "" : route.controllerClass().getName() + "." + route.method().getName());
        entry.put("name", route.method() == null ? "" : route.method().getName());
        entry.put("category", category);
        entry.put("response_type", responseType);
        entry.put("auth_required", authRequired);
        entry.put("protections", protections);
        if (authAction) {
            entry.put("auth_action", String.valueOf(auth.getOrDefault("auth_type", "login")));
            entry.put("auth_confidence", 0.9);
            entry.put("auth_signals", List.of("path_or_handler:auth"));
        }
        if (api.confidence() > 0.0) {
            entry.put("api_confidence", api.confidence());
            entry.put("api_signals", api.signals());
        }
        if (api.payloadType() != null && !api.payloadType().isBlank()) entry.put("payload_type", api.payloadType());
        if (api.formConfidence() > 0.0) {
            entry.put("form_confidence", api.formConfidence());
            entry.put("form_signals", api.formSignals());
        }
        entry.put("request_body", api.requestBody());
        return entry;
    }

    private static Mapping extractMethodMapping(Method method) {
        GetMapping get = AnnotatedElementUtils.findMergedAnnotation(method, GetMapping.class);
        if (get != null) return new Mapping(pathsFrom(get.value(), get.path()), new RequestMethod[]{RequestMethod.GET});
        PostMapping post = AnnotatedElementUtils.findMergedAnnotation(method, PostMapping.class);
        if (post != null) return new Mapping(pathsFrom(post.value(), post.path()), new RequestMethod[]{RequestMethod.POST});
        PutMapping put = AnnotatedElementUtils.findMergedAnnotation(method, PutMapping.class);
        if (put != null) return new Mapping(pathsFrom(put.value(), put.path()), new RequestMethod[]{RequestMethod.PUT});
        PatchMapping patch = AnnotatedElementUtils.findMergedAnnotation(method, PatchMapping.class);
        if (patch != null) return new Mapping(pathsFrom(patch.value(), patch.path()), new RequestMethod[]{RequestMethod.PATCH});
        DeleteMapping delete = AnnotatedElementUtils.findMergedAnnotation(method, DeleteMapping.class);
        if (delete != null) return new Mapping(pathsFrom(delete.value(), delete.path()), new RequestMethod[]{RequestMethod.DELETE});
        RequestMapping request = AnnotatedElementUtils.findMergedAnnotation(method, RequestMapping.class);
        if (request != null) return new Mapping(extractPaths(request), request.method());
        return null;
    }

    private static String[] extractPaths(RequestMapping mapping) {
        return mapping == null ? new String[0] : pathsFrom(mapping.value(), mapping.path());
    }

    private static String[] pathsFrom(String[] value, String[] path) {
        if (path != null && path.length > 0) return path;
        return value == null ? new String[0] : value;
    }

    private static List<String> requestMethods(RequestMethod[] methods) {
        if (methods == null) return List.of();
        List<String> out = new ArrayList<>();
        for (RequestMethod method : methods) if (method != null) out.add(method.name());
        return out;
    }

    private static List<String> normalizedMethods(Collection<String> methods) {
        if (methods == null) return List.of();
        return methods.stream()
                .filter(value -> value != null && HTTP_METHODS.contains(value.toUpperCase(Locale.ROOT)))
                .map(value -> value.toUpperCase(Locale.ROOT))
                .distinct()
                .sorted()
                .toList();
    }

    private static String joinPaths(String left, String right) {
        String a = left == null ? "" : left.trim();
        String b = right == null ? "" : right.trim();
        if (a.isEmpty() && b.isEmpty()) return "/";
        if (a.isEmpty()) return normalizePath(b);
        if (b.isEmpty()) return normalizePath(a);
        return normalizePath(a + "/" + b);
    }

    private static void addStrings(Set<String> target, Object value) {
        if (!(value instanceof Collection<?> collection)) return;
        for (Object item : collection) if (item != null) target.add(String.valueOf(item));
    }

    private static boolean isStaticPath(String path) {
        if (startsWithAny(path, "/static/", "/media/", "/assets/")) return true;
        return Set.of(".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".woff", ".woff2")
                .stream().anyMatch(path::endsWith);
    }

    private static boolean startsWithAny(String value, String... prefixes) {
        for (String prefix : prefixes) if (value.startsWith(prefix)) return true;
        return false;
    }

    private static boolean containsAny(String value, String... tokens) {
        for (String token : tokens) if (value.contains(token)) return true;
        return false;
    }

    private static Map<String, Object> mapOf(Object... values) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (int i = 0; i + 1 < values.length; i += 2) out.put(String.valueOf(values[i]), values[i + 1]);
        return out;
    }

    private static Map<String, Object> protectionMap(Object value) {
        return value instanceof Map<?, ?> raw ? stringObjectMap(raw) : Map.of();
    }

    private static Map<String, Object> stringObjectMap(Map<?, ?> raw) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> item : raw.entrySet()) {
            if (item.getKey() != null) out.put(String.valueOf(item.getKey()), item.getValue());
        }
        return out;
    }

    private static Integer firstInteger(Map<String, Object> values, String... keys) {
        for (String key : keys) {
            Object value = values.get(key);
            if (value instanceof Number number) return number.intValue();
            if (value != null) {
                try {
                    return Integer.parseInt(String.valueOf(value));
                } catch (NumberFormatException ignored) {
                    // Try the next compatible spelling.
                }
            }
        }
        return null;
    }

    private static byte[] stableJson(Object value) {
        try {
            return MAPPER.writeValueAsBytes(value);
        } catch (IOException ex) {
            throw new IllegalStateException("Unable to serialize manifest context", ex);
        }
    }

    private static String sha256(byte[] value) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(value));
        } catch (Exception ex) {
            throw new IllegalStateException("SHA-256 is unavailable", ex);
        }
    }

    private record Mapping(String[] paths, RequestMethod[] methods) {}

    public record RouteInfo(String path, List<String> httpMethods, Class<?> controllerClass, Method method) {}
}
