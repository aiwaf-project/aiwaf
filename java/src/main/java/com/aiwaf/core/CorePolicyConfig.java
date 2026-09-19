package com.aiwaf.core;

import java.util.Set;

public record CorePolicyConfig(
        HeaderValidationConfig headerValidation,
        AnomalyConfig anomaly,
        RateLimitPolicyCore.RateLimitConfig rateLimit
) {
    public CorePolicyConfig() {
        this(new HeaderValidationConfig(), new AnomalyConfig(), new RateLimitPolicyCore.RateLimitConfig());
    }

    public void validate() {
        if (headerValidation.minPresenceScore() < 0.0 || headerValidation.minPresenceScore() > 1.0) {
            throw new IllegalArgumentException("header_validation.min_presence_score must be between 0 and 1");
        }
        if (headerValidation.minQualityScore() < 0.0) {
            throw new IllegalArgumentException("header_validation.min_quality_score must be non-negative");
        }
        if (rateLimit.maxRequestsPerWindow() <= 0) {
            throw new IllegalArgumentException("rate_limit.max_requests_per_window must be positive");
        }
        if (rateLimit.windowSeconds() <= 0) {
            throw new IllegalArgumentException("rate_limit.window_seconds must be positive");
        }
        if (rateLimit.floodBurstThreshold() <= 0) {
            throw new IllegalArgumentException("rate_limit.flood_burst_threshold must be positive");
        }
    }

    public record HeaderValidationConfig(
            Set<String> requiredHeaders,
            double minPresenceScore,
            double minQualityScore
    ) {
        public HeaderValidationConfig() {
            this(Set.of("user-agent", "accept"), 0.5, 3.0);
        }
    }

    public record AnomalyConfig(
            Set<String> suspiciousKeywords,
            int pathLengthLimit,
            Set<String> methodAllowlist
    ) {
        public AnomalyConfig() {
            this(
                    Set.of(".php", "xmlrpc", "wp-", ".env", ".git", ".bak", "conflg", "shell", "filemanager"),
                    512,
                    Set.of("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")
            );
        }
    }
}
