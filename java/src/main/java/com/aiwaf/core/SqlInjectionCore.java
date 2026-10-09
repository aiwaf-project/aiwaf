package com.aiwaf.core;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.Map;
import java.util.regex.Pattern;

/** Bounded deterministic SQL signatures. Findings never contain request values. */
public final class SqlInjectionCore {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final Pattern COMMENTS = Pattern.compile("/\\*[\\s\\S]*?\\*/");
    private static final Map<String, Pattern> RULES = Map.of(
        "sql_union_select", Pattern.compile("\\bunion\\s+(?:all\\s+)?select\\b", Pattern.CASE_INSENSITIVE),
        "sql_boolean_tautology", Pattern.compile("['\"`]\\s*(?:or|and)\\s+(?:true\\b|\\d+\\s*=\\s*\\d+|['\"][^'\"\\r\\n]{0,80}['\"]\\s*=\\s*['\"])", Pattern.CASE_INSENSITIVE),
        "sql_quote_comment", Pattern.compile("['\"`]\\s*(?:--|#)"),
        "sql_stacked_statement", Pattern.compile(";\\s*(?:drop\\s+table|delete\\s+from|insert\\s+into|update\\s+\\w+\\s+set)\\b", Pattern.CASE_INSENSITIVE)
    );
    private record Item(Object value, int depth) {}
    private SqlInjectionCore() {}

    public static String inspect(AiwafRequest request, AiwafConfig config) {
        ArrayDeque<Item> pending = new ArrayDeque<>();
        int bytes = 0;
        if (request.query() != null) {
            for (var entry : request.query().entrySet()) {
                for (String value : new String[]{entry.getKey(), entry.getValue()}) {
                    if (value == null) continue;
                    bytes += value.getBytes(StandardCharsets.UTF_8).length;
                    pending.add(new Item(value, 0));
                }
            }
        }
        if (config.requestBodyInspectionEnabled && request.bodyPreview() != null) {
            bytes += request.bodyPreview().getBytes(StandardCharsets.UTF_8).length;
            pending.add(new Item(request.bodyPreview(), 0));
        }
        if (bytes > config.requestBodyInspectionBytes) return "payload_inspection_limit";
        int nodes = 0;
        while (!pending.isEmpty()) {
            Item item = pending.removeLast();
            if (++nodes > 4096 || item.depth() > 16) return "payload_inspection_limit";
            if (item.value() instanceof JsonNode node) {
                if (node.isObject()) {
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
            String value = String.valueOf(item.value());
            for (int i = 0; i < 2; i++) {
                try { value = URLDecoder.decode(value, StandardCharsets.UTF_8); }
                catch (IllegalArgumentException ignored) { break; }
            }
            String trimmed = value.stripLeading();
            if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
                try { pending.add(new Item(JSON.readTree(value), item.depth() + 1)); }
                catch (Exception ignored) { /* Non-JSON strings still receive signature checks. */ }
            }
            value = COMMENTS.matcher(value).replaceAll(" ");
            for (var rule : RULES.entrySet()) if (rule.getValue().matcher(value).find()) return rule.getKey();
        }
        return null;
    }
}
