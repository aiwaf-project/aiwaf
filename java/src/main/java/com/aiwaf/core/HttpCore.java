package com.aiwaf.core;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

public final class HttpCore {
    private HttpCore() {}

    public static Map<String, String> normalizeHeaders(Map<String, ?> headers) {
        Map<String, String> out = new HashMap<>();
        if (headers == null) {
            return out;
        }
        for (Map.Entry<String, ?> entry : headers.entrySet()) {
            String key = entry.getKey();
            if (key == null) {
                continue;
            }
            String name = key.trim().toLowerCase(Locale.ROOT).replace("_", "-");
            if (name.isEmpty()) {
                continue;
            }
            Object value = entry.getValue();
            out.put(name, value == null ? "" : String.valueOf(value));
        }
        return out;
    }

    public static Map<String, String> normalizeWsgiEnviron(Map<String, ?> environ) {
        Map<String, String> out = new HashMap<>();
        if (environ == null) {
            return out;
        }
        for (Map.Entry<String, ?> entry : environ.entrySet()) {
            String k = entry.getKey();
            if (k == null) {
                continue;
            }
            String upper = k.toUpperCase(Locale.ROOT);
            String name = null;
            if (upper.startsWith("HTTP_")) {
                name = upper.substring(5).toLowerCase(Locale.ROOT).replace("_", "-");
            } else if ("CONTENT_TYPE".equals(upper) || "CONTENT_LENGTH".equals(upper)) {
                name = upper.toLowerCase(Locale.ROOT).replace("_", "-");
            } else if ("SERVER_PROTOCOL".equals(upper)) {
                name = "server-protocol";
            }
            if (name == null || name.isBlank()) {
                continue;
            }
            Object value = entry.getValue();
            out.put(name, value == null ? "" : String.valueOf(value));
        }
        return out;
    }
}
