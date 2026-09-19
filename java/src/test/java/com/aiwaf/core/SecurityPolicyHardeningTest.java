package com.aiwaf.core;

import com.aiwaf.runtime.CidrUtil;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SecurityPolicyHardeningTest {

    @Test
    void saturated_rate_limiter_denies_new_keys_without_evicting_active_bucket() {
        AiwafConfig config = baseConfig();
        config.maxRuntimeStateEntries = 100;
        config.rateLimitMax = 1;
        AiwafEngine engine = new AiwafEngine(config);

        assertTrue(engine.evaluate(request("/protected", "198.51.100.1", Map.of())).allowed());
        for (int i = 2; i <= 100; i++) {
            assertTrue(engine.evaluate(request("/p/" + i, "198.51.100." + i, Map.of())).allowed());
        }

        assertEquals(429, engine.evaluate(request("/overflow", "203.0.113.1", Map.of())).statusCode());
        assertEquals(429, engine.evaluate(request("/protected", "198.51.100.1", Map.of())).statusCode());
    }

    @Test
    void route_media_type_policy_rejects_unexpected_content_type() {
        AiwafConfig config = baseConfig();
        config.rateLimitEnabled = false;
        config.allowedContentTypesByPathPrefix.put("/api/", Set.of("application/json"));
        AiwafEngine engine = new AiwafEngine(config);

        assertEquals(415, engine.evaluate(request("/api/items", "198.51.100.2",
                Map.of("Content-Type", "application/xml"))).statusCode());
        assertTrue(engine.evaluate(request("/api/items", "198.51.100.2",
                Map.of("Content-Type", "application/json; charset=utf-8"))).allowed());
    }

    @Test
    void rejects_duplicate_parameter_marker() {
        AiwafConfig config = baseConfig();
        config.rateLimitEnabled = false;
        AiwafEngine engine = new AiwafEngine(config);

        AiwafDecision decision = engine.evaluate(request("/api/items", "198.51.100.3",
                Map.of("AIWAF-Internal-Duplicate-Parameters", "true")));
        assertFalse(decision.allowed());
        assertEquals(400, decision.statusCode());
    }

    @Test
    void cidr_matching_supports_ipv4_and_ipv6_without_hostname_resolution() {
        assertTrue(CidrUtil.contains("192.0.2.0/24", "192.0.2.44"));
        assertFalse(CidrUtil.contains("192.0.2.0/24", "192.0.3.44"));
        assertTrue(CidrUtil.contains("2001:db8::/32", "2001:db8:1::7"));
        assertFalse(CidrUtil.contains("2001:db8::/32", "2001:db9::7"));
        assertFalse(CidrUtil.isLiteral("example.com"));
    }

    private static AiwafConfig baseConfig() {
        AiwafConfig config = new AiwafConfig();
        config.headerValidationEnabled = false;
        config.privateIpsExempted = false;
        config.honeypotEnabled = false;
        config.ipKeywordBlockEnabled = false;
        return config;
    }

    private static AiwafRequest request(String path, String ip, Map<String, String> headers) {
        return new AiwafRequest(
                "POST", path, ip, "", headers, Map.of(), System.currentTimeMillis(), Set.of());
    }
}
