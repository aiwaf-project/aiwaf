package com.aiwaf.core;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Uses audit findings on live requests, in log-only mode.
 *
 * <p>{@code audit --manifest} saves which operation each route's input reaches. When the injection
 * risk mode is {@code log}, requests to those routes are checked for patterns that matter to that
 * operation, and matches are logged. This never blocks or changes a request, and log events carry
 * names from the manifest and rule names only, never request values.
 */
public final class RouteRiskCore {
    private static final Logger LOGGER = Logger.getLogger("com.aiwaf.injection_risk");
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final int MAX_VALUE_CHARS = 4096;
    private static final int MAX_ITEMS = 512;
    private static final int MAX_DEPTH = 8;
    private static final String NOSQL = "where|ne|eq|gt|gte|lt|lte|in|nin|regex|exists|or|and|nor|not|expr";

    private record Rule(String name, Pattern pattern) {}

    private static Rule rule(String name, String pattern) {
        return new Rule(name, Pattern.compile(pattern, Pattern.CASE_INSENSITIVE));
    }

    // Keep these in step with route_risks.py and routeRisks.js. The shared cases in
    // tests/fixtures/route_risk_cases.json run against all three.
    private static final Map<String, List<Rule>> SINK_RULES = sinkRules();

    private static Map<String, List<Rule>> sinkRules() {
        Map<String, List<Rule>> rules = new LinkedHashMap<>();
        List<Rule> sql = new ArrayList<>();
        SqlInjectionCore.RULES.forEach((name, pattern) -> sql.add(new Rule(name, pattern)));
        rules.put("sql", sql);
        rules.put("command", List.of(
                rule("command_chain", "[;&|]\\s*(?:cat|curl|wget|nc|ncat|bash|sh|zsh|cmd|powershell|rm|id|whoami|uname|ping)\\b"),
                rule("command_substitution", "\\$\\([^)]{0,200}\\)|`[^`]{1,200}`")));
        rules.put("path", List.of(
                rule("path_traversal", "(?:^|[/\\\\])\\.\\.(?:[/\\\\]|$)"),
                rule("path_sensitive_location", "^(?:/etc/|/proc/|/root/|/var/log/|[a-z]:[/\\\\]|\\\\\\\\)"),
                rule("path_null_byte", "\\x00")));
        rules.put("ssrf", List.of(
                rule("ssrf_scheme", "^\\s*(?:file|gopher|dict|ftp|ldap|tftp|netdoc)://"),
                rule("ssrf_internal_host", "^\\s*(?:https?:)?//(?:[^/@?#\\s]{0,128}@)?(?:localhost|127\\.\\d|0\\.0\\.0\\.0|10\\.\\d|192\\.168\\.|169\\.254\\.|172\\.(?:1[6-9]|2\\d|3[01])\\.|\\[::1\\]|metadata\\.google\\.internal)")));
        rules.put("nosql", List.of(
                rule("nosql_operator", "^\\$(?:" + NOSQL + ")$|\\[\\$(?:" + NOSQL + ")\\]|\\{\\s*\"?\\$(?:" + NOSQL + ")\\b")));
        rules.put("code", List.of(
                rule("code_expression", "__import__|\\b(?:eval|exec|compile)\\s*\\(|\\brequire\\s*\\(|\\bprocess\\.(?:env|exit|mainModule)|java\\.lang\\.|Runtime\\.getRuntime|\\bT\\s*\\(\\s*java")));
        rules.put("template", List.of(
                rule("template_expression", "\\{\\{.{0,200}\\}\\}|\\{%.{0,200}%\\}|\\$\\{.{0,200}\\}|<%.{0,200}%>|#\\{.{0,200}\\}")));
        rules.put("deserialization", List.of(new Rule("deserialization_marker", Pattern.compile(
                "^rO0AB|^aced0005|^gASV|!!python/|!!javax\\.|^O:\\d+:\"|__reduce__|_\\$\\$ND_FUNC\\$\\$_"))));
        return Map.copyOf(rules);
    }

    private static final Pattern ROUTE_SEGMENT = Pattern.compile("^(?:<(?:[^:>]+:)?([^>]+)>|\\{\\*?([^:}]+)(?::[^}]*)?\\}|:([^/?*]+)[?*]?)$");
    private static final Pattern WILDCARD_SEGMENT = Pattern.compile("^(?:<path:[^>]+>|\\{\\*?[^:}]+:(?:path|\\.\\*|\\.\\+)\\}|\\{\\*[^}]+\\}|:[^/?*]+\\*|\\*\\*)$");

    private static final Map<String, Risks> CACHE = new ConcurrentHashMap<>();

    private RouteRiskCore() {}

    public record Input(String name, String value) {}

    public record CompiledRoute(Pattern pattern, List<String> names) {}

    public record Counts(int routes, int linked, int unlinked) {}

    public static String ruleFor(String sink, String value) {
        String text = value == null ? "" : value.length() > MAX_VALUE_CHARS ? value.substring(0, MAX_VALUE_CHARS) : value;
        List<Rule> rules = SINK_RULES.getOrDefault(sink, List.of());
        String found = firstRule(rules, text);
        if (found == null && text.indexOf('%') >= 0) {
            try {
                found = firstRule(rules, URLDecoder.decode(text, StandardCharsets.UTF_8));
            } catch (IllegalArgumentException ignored) {
                return null;
            }
        }
        return found;
    }

    private static String firstRule(List<Rule> rules, String text) {
        for (Rule rule : rules) {
            if (rule.pattern().matcher(text).find()) return rule.name();
        }
        return null;
    }

    /** Compiles a manifest route from any supported framework's syntax. */
    public static CompiledRoute compileRoute(String path) {
        StringBuilder pattern = new StringBuilder("^");
        List<String> names = new ArrayList<>();
        for (String segment : String.valueOf(path).split("/")) {
            if (segment.isEmpty()) continue;
            Matcher match = ROUTE_SEGMENT.matcher(segment);
            boolean named = match.matches();
            String name = "";
            if (named) {
                for (int group = 1; group <= match.groupCount(); group++) {
                    if (match.group(group) != null) name = match.group(group);
                }
            }
            if (WILDCARD_SEGMENT.matcher(segment).matches()) {
                names.add(name);
                pattern.append("/(.*)");
            } else if (segment.equals("*")) {
                names.add("");
                pattern.append("/([^/]+)");
            } else if (named) {
                names.add(name);
                pattern.append("/([^/]+)");
            } else {
                pattern.append('/').append(Pattern.quote(segment));
            }
        }
        return new CompiledRoute(Pattern.compile(pattern.append("/?$").toString()), List.copyOf(names));
    }

    private static String inputName(String name) {
        int bracket = name.indexOf('[');
        return bracket >= 0 ? name.substring(0, bracket) : name;
    }

    public static List<Input> requestInputs(Map<String, String> query, String body) {
        List<Input> items = new ArrayList<>();
        if (query != null) {
            for (Map.Entry<String, String> entry : query.entrySet()) {
                if (items.size() >= MAX_ITEMS || entry.getKey() == null) break;
                String name = inputName(entry.getKey());
                items.add(new Input(name, entry.getKey()));
                if (entry.getValue() != null) items.add(new Input(name, entry.getValue()));
            }
        }
        if (body != null && !body.isEmpty()) {
            String stripped = body.stripLeading();
            if (stripped.startsWith("{") || stripped.startsWith("[")) {
                try {
                    walk(JSON.readTree(stripped), "", "", items, 0);
                } catch (IOException ex) {
                    items.add(new Input("", body));
                }
            } else if (body.contains("=")) {
                for (String pair : body.split("&")) {
                    if (items.size() >= MAX_ITEMS) break;
                    int equals = pair.indexOf('=');
                    String key = decode(equals >= 0 ? pair.substring(0, equals) : pair);
                    items.add(new Input(inputName(key), key));
                    items.add(new Input(inputName(key), decode(equals >= 0 ? pair.substring(equals + 1) : "")));
                }
            } else {
                items.add(new Input("", body));
            }
        }
        return items.size() > MAX_ITEMS ? List.copyOf(items.subList(0, MAX_ITEMS)) : items;
    }

    private static String decode(String text) {
        try {
            return URLDecoder.decode(text, StandardCharsets.UTF_8);
        } catch (IllegalArgumentException ex) {
            return text;
        }
    }

    // Nested values also count as the top-level field that holds them, as in {"user": {"$ne": 1}}.
    private static void walk(JsonNode value, String name, String root, List<Input> items, int depth) {
        if (items.size() >= MAX_ITEMS || depth > MAX_DEPTH || value == null || value.isNull()) return;
        if (value.isObject()) {
            for (Map.Entry<String, JsonNode> field : value.properties()) {
                String key = field.getKey();
                String top = root.isEmpty() ? key : root;
                items.add(new Input(key, key));
                if (!top.equals(key)) items.add(new Input(top, key));
                walk(field.getValue(), key, top, items, depth + 1);
            }
        } else if (value.isArray()) {
            for (JsonNode item : value) walk(item, name, root, items, depth + 1);
        } else {
            items.add(new Input(name, value.asText()));
            if (!root.isEmpty() && !root.equals(name)) items.add(new Input(root, value.asText()));
        }
    }

    private record RiskRoute(String path, CompiledRoute compiled, List<Map<?, ?>> risks) {}

    /** Saved risks for one manifest, compiled for lookup by request path. */
    public static final class Risks {
        private final List<RiskRoute> routes = new ArrayList<>();

        public Risks(Map<String, Object> manifest) {
            if (manifest != null && manifest.get("routes") instanceof Map<?, ?> all) {
                for (Map.Entry<?, ?> route : all.entrySet()) {
                    if (route.getValue() instanceof Map<?, ?> entry && entry.get("injection_risks") instanceof List<?> saved && !saved.isEmpty()) {
                        List<Map<?, ?>> risks = new ArrayList<>();
                        for (Object item : saved) {
                            if (item instanceof Map<?, ?> risk) risks.add(risk);
                        }
                        String path = String.valueOf(route.getKey());
                        routes.add(new RiskRoute(path, compileRoute(path), risks));
                    }
                }
            }
            // Exact routes win over routes with parameters.
            routes.sort(Comparator.<RiskRoute>comparingInt(route -> route.compiled().names().size())
                    .thenComparingInt(route -> -route.path().length()));
        }

        public boolean isEmpty() {
            return routes.isEmpty();
        }

        public List<Map<String, String>> inspect(String path, Map<String, String> query, String body) {
            for (RiskRoute route : routes) {
                Matcher match = route.compiled().pattern().matcher(path == null ? "/" : path);
                if (match.matches()) return inspectRoute(route, match, query, body);
            }
            return List.of();
        }

        private static String text(Map<?, ?> risk, String key) {
            Object value = risk.get(key);
            return value == null ? "" : String.valueOf(value);
        }

        private static List<Map<String, String>> inspectRoute(RiskRoute route, Matcher match, Map<String, String> query, String body) {
            List<Input> items = new ArrayList<>(requestInputs(query, body));
            List<String> names = route.compiled().names();
            for (int index = 0; index < names.size(); index++) {
                String value = match.group(index + 1);
                if (value != null && !value.isEmpty()) items.add(new Input(names.get(index), value));
            }
            List<Map<String, String>> events = new ArrayList<>();
            Set<String> seen = new HashSet<>();
            for (Map<?, ?> risk : route.risks()) {
                String sink = text(risk, "sink");
                String wanted = text(risk, "input");
                for (Input item : items) {
                    if (!wanted.isEmpty() && !wanted.equals(item.name())) continue;
                    String rule = ruleFor(sink, item.value());
                    if (rule != null && seen.add(sink + '\0' + wanted + '\0' + rule)) {
                        // The input name comes from the manifest, because request names are untrusted.
                        Map<String, String> event = new LinkedHashMap<>();
                        event.put("route", route.path());
                        event.put("sink", sink);
                        event.put("input", wanted);
                        event.put("rule", rule);
                        event.put("confidence", text(risk, "confidence"));
                        events.add(event);
                    }
                }
            }
            return events;
        }
    }

    static void clearCache() {
        CACHE.clear();
    }

    /** Logs matches for this request when log mode applies. This must never affect the request. */
    public static void observe(AiwafRequest request, AiwafConfig config) {
        try {
            if (config == null || !config.pathManifestEnabled
                    || !"log".equalsIgnoreCase(String.valueOf(config.injectionRiskMode).trim())) {
                return;
            }
            String location = config.pathManifestPath == null || config.pathManifestPath.isBlank()
                    ? PathManifestCore.DEFAULT_MANIFEST_PATH : config.pathManifestPath;
            Risks risks = CACHE.computeIfAbsent(location, key -> new Risks(PathManifestCore.loadManifest(Path.of(key))));
            if (risks.isEmpty()) return;
            for (Map<String, String> event : risks.inspect(request.path(), request.query(), request.bodyPreview())) {
                Map<String, String> line = new LinkedHashMap<>();
                line.put("event", "aiwaf.injection_risk");
                line.put("mode", "log");
                line.put("method", String.valueOf(request.method()));
                line.putAll(event);
                LOGGER.warning(JSON.writeValueAsString(line));
            }
        } catch (IOException | RuntimeException ignored) {
            // Nothing to do: this is a diagnostic, not a protection.
        }
    }

    public static String inputFromSource(String source) {
        String text = source == null ? "" : source;
        Matcher annotated = Pattern.compile("^@\\w+ (.+)$").matcher(text);
        if (annotated.matches()) return annotated.group(1);
        Matcher literal = Pattern.compile("\\(\"(.+?)\"\\)$").matcher(text);
        return literal.find() ? literal.group(1) : "";
    }

    static boolean viewMatches(String rawView, String entry, String file) {
        String view = rawView == null ? "" : rawView.replace('$', '.');
        if (view.isEmpty() || entry == null || entry.isEmpty()) return false;
        int dot = entry.indexOf('.');
        String owner = dot >= 0 ? entry.substring(0, dot) : entry;
        boolean named = view.equals(entry) || view.endsWith("." + entry) || view.equals(owner) || view.endsWith("." + owner);
        if (!named) return false;
        // Reflection gives package names. Same-named classes elsewhere are different routes.
        StringBuilder packagePath = new StringBuilder();
        for (String part : view.split("\\.")) {
            if (part.isEmpty() || !Character.isLowerCase(part.charAt(0)) || view.endsWith("." + part)) break;
            packagePath.append(part).append('/');
        }
        String normalized = "/" + String.valueOf(file).replace('\\', '/');
        return packagePath.isEmpty() || normalized.contains("/" + packagePath);
    }

    /** Replaces saved risks in a manifest with this report's findings, and counts the results. */
    @SuppressWarnings("unchecked")
    public static Counts applyReport(JavaSourceAnalysisCore.Report report, Map<String, Object> manifest) {
        if (!(manifest.get("routes") instanceof Map<?, ?> rawRoutes)) {
            throw new IllegalArgumentException("The manifest has no routes");
        }
        Map<String, Object> routes = (Map<String, Object>) rawRoutes;
        for (Object entry : routes.values()) {
            if (entry instanceof Map<?, ?> data) ((Map<String, Object>) data).remove("injection_risks");
        }
        int linked = 0;
        int unlinked = 0;
        Map<String, Map<String, Map<String, String>>> collected = new TreeMap<>();
        for (JavaSourceAnalysisCore.Finding finding : report.findings()) {
            String entry = finding.function().split(" -> ")[0];
            boolean matched = false;
            for (Map.Entry<String, Object> route : routes.entrySet()) {
                if (route.getValue() instanceof Map<?, ?> data
                        && viewMatches(String.valueOf(data.get("view")), entry, finding.file())) {
                    matched = true;
                    String input = inputFromSource(finding.source());
                    Map<String, String> risk = new LinkedHashMap<>();
                    risk.put("sink", finding.sink());
                    risk.put("input", input);
                    risk.put("confidence", finding.confidence());
                    collected.computeIfAbsent(route.getKey(), key -> new TreeMap<>()).put(finding.sink() + '\0' + input, risk);
                }
            }
            if (matched) linked++;
            else unlinked++;
        }
        collected.forEach((path, risks) ->
                ((Map<String, Object>) routes.get(path)).put("injection_risks", List.copyOf(risks.values())));
        Map<String, Object> audit = new LinkedHashMap<>();
        audit.put("generated_at", Instant.now().truncatedTo(ChronoUnit.SECONDS).toString());
        audit.put("files_scanned", report.filesScanned());
        audit.put("linked_findings", linked);
        audit.put("unlinked_findings", unlinked);
        manifest.put("injection_audit", audit);
        return new Counts(collected.size(), linked, unlinked);
    }

    /** Saves a report's findings into an existing manifest file. */
    @SuppressWarnings("unchecked")
    public static Counts updateManifest(JavaSourceAnalysisCore.Report report, Path manifestPath) {
        Map<String, Object> loaded = PathManifestCore.loadManifest(manifestPath);
        if (loaded.isEmpty()) {
            throw new IllegalArgumentException("Manifest not found or unreadable: " + manifestPath + ". Generate it first with init.");
        }
        Map<String, Object> manifest = JSON.convertValue(loaded, LinkedHashMap.class);
        Counts counts = applyReport(report, manifest);
        PathManifestCore.writeManifest(manifest, manifestPath);
        return counts;
    }

    public static String formatCounts(Counts counts, String manifestPath) {
        String text = String.format(Locale.ROOT, "Saved %d possible injection paths to %d routes in %s.",
                counts.linked(), counts.routes(), manifestPath);
        return counts.unlinked() == 0 ? text
                : text + " " + counts.unlinked() + " couldn't be tied to a route, so they get no route-specific checks.";
    }
}
