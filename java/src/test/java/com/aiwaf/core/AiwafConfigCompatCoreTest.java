package com.aiwaf.core;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AiwafConfigCompatCoreTest {

    @Test
    void python_style_structured_config_maps_to_aiwaf_config() {
        Map<String, Object> cfg = new java.util.HashMap<>();
        cfg.put("storage", Map.of("backend", "memory", "file_path", "aiwaf_data.json"));
        cfg.put("header_validation", Map.of("enabled", true, "quality_threshold", 3, "exempt_paths", List.of("/health")));
        cfg.put("rate_limiting", Map.of("enabled", true, "window_seconds", 10, "max_requests", 20, "flood_threshold", 40));
        cfg.put("honeypot", Map.of("enabled", true, "min_form_time", 1.0));
        cfg.put("ip_keyword_block", Map.of("enabled", true, "malicious_keywords", List.of(".env", ".git", "phpmyadmin", "xmlrpc")));
        cfg.put("geo_block", Map.of("enabled", false, "block_countries", List.of("CN", "RU")));
        cfg.put("ai_anomaly", Map.of("enabled", false));
        cfg.put("uuid_tamper", Map.of("enabled", true));
        cfg.put("exemptions", Map.of("private_ips_exempted", true, "auto_exempt_patterns", List.of("127.0.0.1")));
        cfg.put("legitimate_route_hints", List.of("/payments/invoices", "AdminDashboard"));
        cfg.put("path_rules", List.of(
                Map.of("PREFIX", "/health", "DISABLE", List.of("header_validation", "rate_limit", "ai_anomaly")),
                Map.of("PREFIX", "/api/protected", "RATE_LIMIT", Map.of("WINDOW", 10, "MAX", 10))
        ));
        AiwafConfig out = AiwafConfigCompatCore.fromStructured(cfg);
        assertEquals("memory", out.storageBackend);
        assertEquals("aiwaf_data.json", out.storageFilePath);
        assertTrue(out.headerValidationEnabled);
        assertEquals(10, out.rateLimitWindowSeconds);
        assertEquals(20, out.rateLimitMax);
        assertEquals(40, out.rateLimitFloodThreshold);
        assertTrue(out.honeypotEnabled);
        assertFalse(out.geoBlockEnabled);
        assertFalse(out.aiEnabled);
        assertTrue(out.uuidTamperEnabled);
        assertTrue(out.exemptIps.contains("127.0.0.1"));
        assertTrue(out.legitimatePathKeywords.contains("payments"));
        assertTrue(out.legitimatePathKeywords.contains("dashboard"));
        assertEquals(2, out.pathRules.size());
    }

    @Test
    void env_overrides_apply() {
        AiwafConfig out = AiwafConfigCompatCore.fromStructuredAndEnv(
                Map.of("rate_limiting", Map.of("window_seconds", 10)),
                Map.of(
                        "AIWAF_RATE_WINDOW", "15",
                        "AIWAF_RATE_MAX", "99",
                        "AIWAF_GEO_BLOCK_ENABLED", "true",
                        "AIWAF_GEO_BLOCK_COUNTRIES", "CN,RU"
                )
        );
        assertEquals(15, out.rateLimitWindowSeconds);
        assertEquals(99, out.rateLimitMax);
        assertTrue(out.geoBlockEnabled);
        assertTrue(out.geoBlockedCountries.contains("CN"));
        assertTrue(out.geoBlockedCountries.contains("RU"));
    }
}
