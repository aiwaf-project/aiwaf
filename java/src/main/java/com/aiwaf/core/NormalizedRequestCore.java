package com.aiwaf.core;

import java.util.Map;

public record NormalizedRequestCore(
        String ip,
        String method,
        String path,
        Map<String, String> headers,
        String query,
        String body,
        boolean knownPath,
        boolean exemptPath
) {
    public NormalizedRequestCore(String ip, String method, String path, Map<String, String> headers) {
        this(ip, method, path, headers, "", "", false, false);
    }
}
