package com.aiwaf.core;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DetectorCoreParityTest {

    @Test
    void anomaly_detects_keywords_and_method() {
        NormalizedRequestCore req = new NormalizedRequestCore(
                "1.2.3.4",
                "TRACE",
                "/wp-admin/../../etc/passwd",
                Map.of("user-agent", "ua", "accept", "*/*")
        );
        AnomalyCore.CoreAnalysisDecision decision = AnomalyCore.scoreRequestAnomaly(req, new CorePolicyConfig().anomaly());
        assertFalse(decision.allow());
        assertTrue(decision.score() > 0.5);
        assertTrue(decision.reasons().contains("method_not_allowlisted"));
        assertTrue(decision.reasons().contains("suspicious_keywords"));
    }

    @Test
    void rate_limit_and_flood_detection() {
        RateLimitPolicyCore.RateLimiter limiter = new RateLimitPolicyCore.RateLimiter(
                new RateLimitPolicyCore.RateLimitConfig(2, 60, 2)
        );
        RateLimitPolicyCore.RequestKey req = new RateLimitPolicyCore.RequestKey("9.9.9.9", "/");

        RateLimitPolicyCore.RegisterResult r1 = limiter.register(req, 0.0);
        RateLimitPolicyCore.RegisterResult r2 = limiter.register(req, 1.0);
        RateLimitPolicyCore.RegisterResult r3 = limiter.register(req, 2.0);

        assertTrue(r1.ok());
        assertTrue(r2.ok());
        assertTrue(r2.metadata().get("flood") == 1);
        assertFalse(r3.ok());
        assertTrue(r3.metadata().get("limited") == 1);
        assertTrue(r3.metadata().get("path_flood") == 1);
    }

    @Test
    void analyze_request_uses_dynamic_keywords() {
        NormalizedRequestCore req = new NormalizedRequestCore(
                "4.4.4.4",
                "GET",
                "/private/shell-upload",
                Map.of("user-agent", "ua", "accept", "*/*")
        );
        ApiPolicyCore.AnalysisDecision decision = ApiPolicyCore.analyzeRequest(
                req,
                List.of("shell-upload"),
                null,
                List.of(),
                List.of(),
                List.of()
        );
        assertTrue(decision.reasons().contains("suspicious_keywords"));
    }

    @Test
    void behavior_deviation_penalty() {
        Map<String, Object> baseline = Map.of(
                "method_ratio", Map.of("GET", 0.98, "POST", 0.02),
                "hot_paths", Map.of(
                        "/path-0", 0, "/path-1", 1, "/path-2", 2, "/path-3", 3,
                        "/path-4", 4, "/path-5", 5, "/path-6", 6, "/path-7", 7
                )
        );
        NormalizedRequestCore req = new NormalizedRequestCore(
                "4.4.4.5",
                "POST",
                "/never-seen-path",
                Map.of("user-agent", "ua", "accept", "*/*")
        );
        ApiPolicyCore.AnalysisDecision decision = ApiPolicyCore.analyzeRequest(
                req,
                List.of(),
                baseline,
                List.of(),
                List.of(),
                List.of()
        );
        assertTrue(decision.reasons().contains("behavior_deviation"));
    }

    @Test
    void keyword_filtering_parity_rules() {
        List<String> filtered = AnomalyCore.filterSuspiciousKeywords(
                "/admin/dashboard",
                java.util.Set.of("admin", "shell", ".env"),
                true,
                java.util.Set.of(".env"),
                java.util.Set.of("admin", "dashboard"),
                java.util.Set.of("assets", "static")
        );
        assertFalse(filtered.contains("admin"));
        assertFalse(filtered.contains(".env"));
        assertTrue(filtered.contains("shell"));
    }

    @Test
    void analyze_request_respects_safe_prefix_filter() {
        NormalizedRequestCore req = new NormalizedRequestCore(
                "6.6.6.6",
                "GET",
                "/assets/wp-admin-shell",
                Map.of("user-agent", "Mozilla/5.0 Chrome/120", "accept", "text/html,application/xml"),
                "",
                "",
                false,
                false
        );
        ApiPolicyCore.AnalysisDecision decision = ApiPolicyCore.analyzeRequest(
                req,
                List.of(),
                null,
                List.of(),
                List.of(),
                List.of("assets")
        );
        assertFalse(decision.reasons().contains("suspicious_keywords"));
    }

    @Test
    void config_validation_rejects_invalid_min_score() {
        CorePolicyConfig cfg = new CorePolicyConfig(
                new CorePolicyConfig.HeaderValidationConfig(
                        new CorePolicyConfig().headerValidation().requiredHeaders(),
                        1.5,
                        new CorePolicyConfig().headerValidation().minQualityScore()
                ),
                new CorePolicyConfig().anomaly(),
                new CorePolicyConfig().rateLimit()
        );
        boolean threw = false;
        try {
            cfg.validate();
        } catch (IllegalArgumentException ex) {
            threw = true;
        }
        assertTrue(threw);
    }
}
