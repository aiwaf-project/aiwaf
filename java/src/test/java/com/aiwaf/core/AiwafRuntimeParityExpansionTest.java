package com.aiwaf.core;

import com.aiwaf.runtime.RuntimeStorage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AiwafRuntimeParityExpansionTest {

    @Test
    void malformed_uuid_honors_path_and_ip_exemptions() {
        AiwafConfig config = minimalConfig();
        config.uuidTamperEnabled = true;
        config.exemptPaths.add("/uuid-exempt");
        config.exemptIps.add("198.51.100.22");
        AiwafEngine engine = new AiwafEngine(config);

        assertTrue(engine.evaluate(request("/uuid-exempt", "198.51.100.21", Map.of("uuid", "broken"), "US")).allowed());
        assertTrue(engine.evaluate(request("/uuid", "198.51.100.22", Map.of("uuid", "broken"), "US")).allowed());

        AiwafDecision blocked = engine.evaluate(request("/uuid", "198.51.100.23", Map.of("uuid", "broken"), "US"));
        assertFalse(blocked.allowed());
        assertEquals(403, blocked.statusCode());
    }

    @Test
    void runtime_and_pattern_ip_exemptions_apply_to_all_request_checks() {
        AiwafConfig config = minimalConfig();
        config.rateLimitEnabled = true;
        config.rateLimitMax = 1;
        config.rateLimitFloodThreshold = 10;
        config.exemptIpPatterns.add("198.51.100.*");
        AiwafEngine engine = new AiwafEngine(config);
        engine.runtimeStorage().exemptionStore().addIp("203.0.113.8", "trusted runtime client");

        assertTrue(engine.evaluate(request("/limited", "198.51.100.90", Map.of(), "US")).allowed());
        assertTrue(engine.evaluate(request("/limited", "198.51.100.90", Map.of(), "US")).allowed());
        assertTrue(engine.evaluate(request("/limited", "203.0.113.8", Map.of(), "US")).allowed());
        assertTrue(engine.evaluate(request("/limited", "203.0.113.8", Map.of(), "US")).allowed());
    }

    @Test
    void geo_exempt_paths_skip_only_geo_middleware() {
        AiwafConfig config = minimalConfig();
        config.geoBlockEnabled = true;
        config.geoBlockedCountries.add("US");
        config.geoExemptPaths.add("/geo-exempt");
        config.rateLimitEnabled = true;
        config.rateLimitMax = 1;
        config.rateLimitFloodThreshold = 10;
        AiwafEngine engine = new AiwafEngine(config);

        assertTrue(engine.evaluate(request("/geo-exempt", "198.51.100.31", Map.of(), "US")).allowed());
        assertEquals(429, engine.evaluate(request("/geo-exempt", "198.51.100.31", Map.of(), "US")).statusCode());
        assertEquals(403, engine.evaluate(request("/geo-blocked", "198.51.100.32", Map.of(), "US")).statusCode());
    }

    @Test
    void logging_honors_request_path_rule_and_path_exemptions() {
        AiwafConfig config = minimalConfig();
        config.pathRules.add(new AiwafConfig.PathRule(
                "/private", false, null, null, null, Set.of("logging")));
        config.exemptPaths.add("/health-custom");
        AiwafEngine engine = new AiwafEngine(config);

        assertFalse(engine.shouldApplyMiddleware(request("/private/x", "198.51.100.40", Map.of(), "US"), "logging"));
        assertFalse(engine.shouldApplyMiddleware(request("/health-custom", "198.51.100.40", Map.of(), "US"), "logging"));
        AiwafRequest routeDisabled = new AiwafRequest(
                "GET", "/normal", "198.51.100.40", "US", headers(), Map.of(),
                System.currentTimeMillis(), Set.of("logging"));
        assertFalse(engine.shouldApplyMiddleware(routeDisabled, "logging"));
        assertTrue(engine.shouldApplyMiddleware(request("/normal", "198.51.100.40", Map.of(), "US"), "logging"));
    }

    @Test
    void logging_core_does_not_write_route_disabled_requests(@TempDir Path tempDir) {
        AiwafConfig config = minimalConfig();
        config.logDir = tempDir.resolve("logs").toString();
        config.pathRules.add(new AiwafConfig.PathRule(
                "/private", false, null, null, null, Set.of("logging")));
        AiwafRequest privateRequest = request("/private/x", "198.51.100.41", Map.of(), "US");

        AiwafLoggingCore.log(config, privateRequest, AiwafDecision.allow(), 200, 1, 0);
        assertFalse(Files.exists(tempDir.resolve("logs/access.log")));

        AiwafLoggingCore.log(
                config, request("/normal", "198.51.100.41", Map.of(), "US"),
                AiwafDecision.allow(), 200, 1, 0);
        assertTrue(Files.exists(tempDir.resolve("logs/access.log")));
    }

    @Test
    void constructing_another_engine_does_not_replace_the_first_engine_storage() {
        AiwafConfig firstConfig = minimalConfig();
        firstConfig.geoBlockEnabled = true;
        AiwafEngine first = new AiwafEngine(firstConfig);
        first.runtimeStorage().geoBlockStore().addCountry("US");

        AiwafConfig secondConfig = minimalConfig();
        secondConfig.geoBlockEnabled = true;
        AiwafEngine second = new AiwafEngine(secondConfig);

        assertEquals(403, first.evaluate(request("/geo", "198.51.100.51", Map.of(), "US")).statusCode());
        assertTrue(second.evaluate(request("/geo", "198.51.100.52", Map.of(), "US")).allowed());
    }

    @Test
    void engines_can_share_one_context_for_distributed_style_enforcement() {
        AiwafConfig config = minimalConfig();
        config.rateLimitEnabled = true;
        config.rateLimitMax = 1;
        config.rateLimitFloodThreshold = 10;
        RuntimeStorage.Context shared = RuntimeStorage.create("memory", null, null, null);
        AiwafEngine first = new AiwafEngine(config, null, shared);
        AiwafEngine second = new AiwafEngine(config, null, shared);

        assertTrue(first.evaluate(request("/shared", "198.51.100.61", Map.of(), "US")).allowed());
        assertEquals(429, second.evaluate(request("/shared", "198.51.100.61", Map.of(), "US")).statusCode());
    }

    private static AiwafConfig minimalConfig() {
        AiwafConfig config = new AiwafConfig();
        config.privateIpsExempted = false;
        config.pathManifestEnabled = false;
        config.headerValidationEnabled = false;
        config.rateLimitEnabled = false;
        config.honeypotEnabled = false;
        config.ipKeywordBlockEnabled = false;
        config.aiEnabled = false;
        return config;
    }

    private static AiwafRequest request(String path, String ip, Map<String, String> query, String country) {
        return new AiwafRequest(
                "GET", path, ip, country, headers(), query, System.currentTimeMillis(), Set.of());
    }

    private static Map<String, String> headers() {
        return Map.of(
                "User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept", "text/html,application/xhtml+xml",
                "Accept-Language", "en-US,en;q=0.9"
        );
    }
}
