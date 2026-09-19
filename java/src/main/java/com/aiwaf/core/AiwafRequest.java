package com.aiwaf.core;

import java.util.Map;
import java.util.Set;

public record AiwafRequest(
        String method,
        String path,
        String ip,
        String country,
        Map<String, String> headers,
        Map<String, String> query,
        long nowEpochMillis,
        Set<String> disabledMiddlewares,
        String bodyPreview
) {
    public AiwafRequest(
            String method, String path, String ip, String country,
            Map<String, String> headers, Map<String, String> query,
            long nowEpochMillis, Set<String> disabledMiddlewares
    ) {
        this(method, path, ip, country, headers, query, nowEpochMillis, disabledMiddlewares, "");
    }

    public AiwafRequest withDisabledMiddlewares(Set<String> disabled) {
        return new AiwafRequest(method, path, ip, country, headers, query, nowEpochMillis, disabled, bodyPreview);
    }
}
