package com.aiwaf.core;

import java.time.LocalDateTime;

public record NormalizedEvent(
        String ip,
        String method,
        String path,
        int statusCode,
        double responseTimeMs,
        LocalDateTime timestamp,
        String userAgent,
        String query,
        String body,
        boolean knownPath,
        boolean exemptPath
) {}
