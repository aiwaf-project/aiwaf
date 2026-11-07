package com.aiwaf.core;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;

/** Bounded SQL injection checks with the SQL tokenizer. Findings never contain request values. */
public final class SqlInjectionCore {
    private static final ObjectMapper JSON = new ObjectMapper();
    private record Item(Object value, int depth) {}
    private SqlInjectionCore() {}

    public static String inspect(AiwafRequest request, AiwafConfig config) {
        return scan(request, config, SqlTokenizerCore::classifyFields);
    }

    private static String scan(AiwafRequest request, AiwafConfig config, java.util.function.UnaryOperator<String> detect) {
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
            String original = value;
            for (int i = 0; i < 2; i++) {
                try { value = URLDecoder.decode(value, StandardCharsets.UTF_8); }
                catch (IllegalArgumentException ignored) { break; }
            }
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
            if (parsed) continue;
            // Form decoding turns a literal + into a space, as in '+(SELECT 1)+', so also check the
            // value with only its %XX escapes decoded.
            String kept = original;
            for (int i = 0; i < 2; i++) {
                try { kept = URLDecoder.decode(kept.replace("+", "%2B"), StandardCharsets.UTF_8); }
                catch (IllegalArgumentException ignored) { break; }
            }
            // Only the raw body is split into form fields. A query value or JSON value, as in
            // "--warp=e=mc^2", is one value.
            java.util.function.UnaryOperator<String> check =
                    item.value() == request.bodyPreview() ? detect : SqlTokenizerCore::classify;
            String rule = check.apply(value);
            if (rule == null && !kept.equals(value)) rule = check.apply(kept);
            if (rule != null) return rule;
        }
        return null;
    }
}
