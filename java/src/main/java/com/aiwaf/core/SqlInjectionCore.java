package com.aiwaf.core;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Bounded SQL injection checks with the SQL tokenizer. Findings never contain request values.
 *
 * <p>This mirrors the Rust request walker ({@code aiwaf_core::payload}): values are decoded twice,
 * including IIS {@code %uXXXX} escapes, JSON and XML documents are checked by their values, and
 * cookies, the Referer's query string and the User-Agent are checked as well.
 */
public final class SqlInjectionCore {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final Pattern PERCENT_U = Pattern.compile("%[uU]([0-9a-fA-F]{4})");
    private static final Pattern XML_VALUES = Pattern.compile("=\\s*\"([^\"]*)\"|=\\s*'([^']*)'|>([^<]+)<");
    // Headers that hold credentials, framing or browser-set values. Any other header is checked.
    private static final Set<String> UNCHECKED_HEADERS = Set.of(
            "accept", "accept-encoding", "authorization", "proxy-authorization", "cache-control", "connection",
            "content-encoding", "content-length", "content-md5", "content-type", "host", "keep-alive", "pragma",
            "te", "trailer", "transfer-encoding", "upgrade", "upgrade-insecure-requests", "dnt", "priority",
            "x-csrf-token", "x-csrftoken", "x-xsrf-token", "x-api-key", "x-amz-security-token", "x-amz-date",
            "x-amz-content-sha256", "x-hub-signature", "x-hub-signature-256", "x-request-id", "x-correlation-id",
            "traceparent", "tracestate", "baggage");
    private record Item(Object value, int depth) {}
    private SqlInjectionCore() {}

    public static String inspect(AiwafRequest request, AiwafConfig config) {
        Found found = find(request, config);
        return found == null ? null : found.rule();
    }

    /** A finding and where it came from: "payload" (the query or body), "headers" or "path". */
    public record Found(String rule, String source) {}

    /** The check's finding, with where it came from, or null. */
    public static Found find(AiwafRequest request, AiwafConfig config) {
        String body = config.requestBodyInspectionEnabled ? request.bodyPreview() : null;
        String rule = inspectQueryAndBody(request.query(), body, config.requestBodyInspectionBytes);
        if (rule != null) return new Found(rule, "payload");
        rule = inspectHeaders(request.headers(), config.requestBodyInspectionBytes);
        if (rule != null) return new Found(rule, "headers");
        rule = inspectPath(request.path(), config.requestBodyInspectionBytes);
        return rule == null ? null : new Found(rule, "path");
    }

    /** The check of a request's query and body alone, without its headers and path. */
    static String inspectQueryAndBody(Map<String, String> query, String body, int budget) {
        List<Item> items = new ArrayList<>();
        int bytes = 0;
        if (query != null) {
            for (var entry : query.entrySet()) {
                // A prototype key, as in __proto__[shell]=/bin/sh.
                if (RceCore.pollutingKey(entry.getKey())) return "rce_prototype_pollution";
                for (String value : new String[]{entry.getKey(), entry.getValue()}) {
                    if (value == null) continue;
                    bytes += value.getBytes(StandardCharsets.UTF_8).length;
                    items.add(new Item(value, 0));
                }
            }
        }
        if (MultipartPreview.is(body)) {
            // A multipart form, part by part, within the budget: its parts' shape here, and
            // each text field's value below, as any other value.
            String rule = null;
            int room = Math.max(0, budget - bytes);
            for (MultipartPreview.PartView part : MultipartPreview.parts(body.length() > room ? body.substring(0, room) : body)) {
                if (RceCore.pollutingKey(part.name())) return "rce_prototype_pollution";
                rule = RceCore.classify(part.headers());
                if (rule == null && part.file()) rule = RceCore.classifyUpload(part.content());
                if (rule != null) return rule;
                if (!part.file()) items.add(new Item(new String(part.content(), StandardCharsets.UTF_8), 1));
            }
            body = null;
        } else if (body != null) {
            bytes += body.getBytes(StandardCharsets.UTF_8).length;
            items.add(new Item(body, 0));
        }
        if (bytes > budget) return "payload_inspection_limit";
        return scan(items, body);
    }

    /**
     * Checks a URL path one segment at a time, as in /post/1 OR 1=1. A query string is ignored.
     * The whole path is also checked when a comment spans segments, as in /get/sleep(/*x*&#47;12).
     */
    static String inspectPath(String path, int budget) {
        if (path == null || path.isEmpty()) return null;
        String bare = path.split("[?#]", 2)[0];
        if (bare.getBytes(StandardCharsets.UTF_8).length > budget) return "payload_inspection_limit";
        String[] segments = bare.split("/", 257);
        for (int i = 0; i < Math.min(segments.length, 256); i++) {
            String segment = segments[i];
            if (segment.isEmpty()) continue;
            String rule = checkValue(segment, budget);
            int equals = segment.indexOf('=');
            // A matrix parameter, as in /cart;jsessionid=1, is checked by its value.
            if (rule == null && equals >= 0) rule = checkValue(segment.substring(equals + 1), budget);
            if (rule != null) return rule;
        }
        String decoded = decodeOnce(decodeOnce(bare, true), true);
        if (decoded.contains("/*") || decoded.contains("*/")) {
            return SqlTokenizerCore.classify(decoded.replaceFirst("^/+", ""));
        }
        return null;
    }

    private static boolean headerChecked(String name) {
        return !(UNCHECKED_HEADERS.contains(name) || name.startsWith("sec-") || name.startsWith("if-")
                || name.startsWith("content-") || name.startsWith("access-control-") || name.startsWith(":"));
    }

    /**
     * Checks request headers, as aiwaf-rust does: cookie names and values, the query string and
     * path of the Referer, X-Original-URL and X-Rewrite-URL, and any other header as one value,
     * except those holding credentials, framing or browser-set values. Names match in any case.
     */
    static String inspectHeaders(Map<String, String> headers, int budget) {
        if (headers == null) return null;
        // Every header is checked, so an attack can't be pushed past the ones that are. A request
        // with more headers, or more header bytes, than can be checked is refused, as in aiwaf-rust.
        int seen = 0;
        long total = 0;
        for (var entry : headers.entrySet()) {
            total += (entry.getKey() == null ? 0 : entry.getKey().length()) + (entry.getValue() == null ? 0 : entry.getValue().length());
            if (++seen > 256 || total > Math.max(budget, 65536)) return "payload_inspection_limit";
            if (entry.getKey() == null || entry.getValue() == null || entry.getValue().isBlank()) continue;
            String name = entry.getKey().trim().toLowerCase(Locale.ROOT);
            if (name.startsWith("http_")) name = name.substring(5);
            name = name.replace('_', '-');
            if (!headerChecked(name)) {
                // Headers skipped for SQL still carry Log4Shell and Shellshock, as in User-Agent.
                String rce = RceCore.classify(entry.getValue());
                if (rce == null) rce = RceCore.classify(decodeOnce(entry.getValue(), true));
                if (rce != null) return rce;
                continue;
            }
            String value = entry.getValue();
            String rule = switch (name) {
                case "cookie" -> inspectCookies(value, budget);
                case "referer", "x-original-url", "x-rewrite-url" -> {
                    int start = value.indexOf('?');
                    String address = start < 0 ? value : value.substring(0, start);
                    String query = start < 0 ? "" : value.substring(start + 1);
                    int fragment = query.indexOf('#');
                    String found = checkText(fragment < 0 ? query : query.substring(0, fragment), budget);
                    // The path after the host, as in https://example.com/post/1 OR 1=1.
                    int scheme = address.indexOf("://");
                    String path = address;
                    if (scheme >= 0) {
                        int slash = address.indexOf('/', scheme + 3);
                        path = slash < 0 ? "" : address.substring(slash);
                    }
                    yield found != null ? found : inspectPath(path, budget);
                }
                default -> checkValue(value, budget);
            };
            if (rule != null) return rule;
        }
        return null;
    }

    private static String inspectCookies(String header, int budget) {
        String[] pairs = header.split(";", 258);
        if (pairs.length > 256) return "payload_inspection_limit";
        for (int i = 0; i < pairs.length; i++) {
            String pair = pairs[i].trim();
            int equals = pair.indexOf('=');
            String name = equals < 0 ? "" : pair.substring(0, equals).trim();
            String value = equals < 0 ? pair : pair.substring(equals + 1).trim();
            String rule = SqlTokenizerCore.classify(name);
            if (rule == null) rule = checkText(value, budget);
            if (rule != null) return rule;
        }
        return null;
    }

    // Checks a cookie value or query string the way a raw body is checked, split into form fields.
    private static String checkText(String value, int budget) {
        if (value.isEmpty()) return null;
        if (value.getBytes(StandardCharsets.UTF_8).length > budget) return "payload_inspection_limit";
        return scan(List.of(new Item(value, 0)), value);
    }

    // Checks one value, such as a header or a path segment, without splitting it into form
    // fields: in 0 <= a and a <= b, the = isn't a field separator.
    private static String checkValue(String value, int budget) {
        if (value.isEmpty()) return null;
        if (value.getBytes(StandardCharsets.UTF_8).length > budget) return "payload_inspection_limit";
        return scan(List.of(new Item(value, 0)), null);
    }

    private static String decodeOnce(String value, boolean plusIsSpace) {
        // IIS-style %uXXXX escapes, as in %u2019 OR 1=1.
        Matcher matcher = PERCENT_U.matcher(value);
        StringBuilder unicode = new StringBuilder();
        while (matcher.find()) {
            matcher.appendReplacement(unicode, Matcher.quoteReplacement(
                    new String(Character.toChars(Integer.parseInt(matcher.group(1), 16)))));
        }
        matcher.appendTail(unicode);
        String text = plusIsSpace ? unicode.toString() : unicode.toString().replace("+", "%2B");
        try {
            return URLDecoder.decode(text, StandardCharsets.UTF_8);
        } catch (IllegalArgumentException ignored) {
            return unicode.toString();
        }
    }

    private static List<String> xmlValues(String text) {
        List<String> values = new ArrayList<>();
        Matcher matcher = XML_VALUES.matcher(text);
        while (matcher.find() && values.size() < 512) {
            String value = matcher.group(1) != null ? matcher.group(1) : matcher.group(2) != null ? matcher.group(2) : matcher.group(3);
            if (!value.trim().isEmpty()) values.add(value.trim());
        }
        return values;
    }

    // `raw` is the one value split into form fields: a raw body or header. Everything else is one value.
    private static String scan(List<Item> items, String raw) {
        ArrayDeque<Item> pending = new ArrayDeque<>(items);
        int nodes = 0;
        while (!pending.isEmpty()) {
            Item item = pending.removeLast();
            if (++nodes > 4096 || item.depth() > 16) return "payload_inspection_limit";
            if (item.value() instanceof JsonNode node) {
                if (node.isObject()) {
                    // {"__proto__": {...}} or {"constructor": {"prototype": {...}}}.
                    if (node.has("__proto__") || (node.path("constructor").isObject() && node.path("constructor").has("prototype"))) {
                        return "rce_prototype_pollution";
                    }
                    var fields = node.fields();
                    while (fields.hasNext()) {
                        var entry = fields.next();
                        pending.add(new Item(entry.getKey(), item.depth() + 1));
                        pending.add(new Item(entry.getValue(), item.depth() + 1));
                        if (pending.size() > 4096) return "payload_inspection_limit";
                    }
                } else if (node.isArray()) {
                    for (JsonNode child : node) {
                        pending.add(new Item(child, item.depth() + 1));
                        if (pending.size() > 4096) return "payload_inspection_limit";
                    }
                } else if (node.isTextual()) pending.add(new Item(node.textValue(), item.depth() + 1));
                continue;
            }
            String original = String.valueOf(item.value());
            String value = decodeOnce(decodeOnce(original, true), true);
            // Remote code execution is checked in the whole document first: a gadget such as
            // {"@type": "com.sun.rowset.JdbcRowSetImpl"} is in its shape, not in any one value.
            String rce = RceCore.classifyDocument(original);
            if (rce == null && !value.equals(original)) rce = RceCore.classifyDocument(value);
            if (rce != null) return rce;
            // A JSON document is checked by its values, so its own quotes aren't read as a
            // breakout. Parse it before decoding when that works, so a literal + survives.
            boolean parsed = false;
            for (String candidate : new String[]{original, value}) {
                String trimmed = candidate.stripLeading();
                if (parsed || !(trimmed.startsWith("{") || trimmed.startsWith("["))) continue;
                try {
                    pending.add(new Item(JSON.readTree(candidate), item.depth() + 1));
                    parsed = true;
                } catch (Exception ignored) { /* Not JSON: checked as text below. */ }
            }
            if (!parsed && value.stripLeading().startsWith("<")) {
                // An XML body is checked by its attribute values and text, as in
                // <a probe="1234 OR 1=1">, so its own quotes aren't read as a breakout.
                List<String> values = xmlValues(value);
                for (String xml : values) pending.add(new Item(xml, item.depth() + 1));
                parsed = !values.isEmpty();
            }
            if (parsed) continue;
            // Form decoding turns a literal + into a space, as in '+(SELECT 1)+', so also check the
            // value with only its %XX escapes decoded.
            String kept = decodeOnce(decodeOnce(original, false), false);
            // Only the raw body is split into form fields. A query value or JSON value, as in
            // "--warp=e=mc^2", is one value.
            java.util.function.UnaryOperator<String> check =
                    item.value() == raw ? SqlTokenizerCore::classifyFields : SqlTokenizerCore::classify;
            // Remote code execution first: those rules match only attack text.
            // The raw body's form fields are also checked one by one, as in data=rO0AB....
            String rule = item.value() == raw ? RceCore.classifyFields(original) : null;
            if (rule == null) rule = RceCore.classify(value);
            if (rule == null && !kept.equals(value)) rule = RceCore.classify(kept);
            if (rule == null) rule = check.apply(value);
            if (rule == null && !kept.equals(value)) rule = check.apply(kept);
            if (rule != null) return rule;
        }
        return null;
    }
}
