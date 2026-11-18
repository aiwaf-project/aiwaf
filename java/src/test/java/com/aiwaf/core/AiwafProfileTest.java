package com.aiwaf.core;

import com.aiwaf.runtime.BlocklistManager;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AiwafProfileTest {
    @TempDir
    Path directory;

    private static AiwafRequest req(String path, String ip) {
        Map<String, String> headers = new HashMap<>();
        headers.put("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)");
        headers.put("Accept", "text/html,application/xhtml+xml");
        headers.put("Accept-Language", "en-US,en;q=0.9");
        headers.put("Accept-Encoding", "gzip, deflate, br");
        headers.put("Connection", "keep-alive");
        return new AiwafRequest("GET", path, ip, "US", headers, Map.of(), System.currentTimeMillis(), Set.of());
    }

    @Test
    void strictHalvesRateLimitsAndExplicitSettingsWin() {
        AiwafConfig strict = AiwafConfigCompatCore.fromStructured(Map.of("profile", "strict"));
        assertEquals(List.of(10, 20), List.of(strict.rateLimitMax, strict.rateLimitFloodThreshold));
        assertTrue(strict.headerValidationEnabled && strict.methodValidationEnabled && strict.uuidScoreEnabled);
        AiwafConfig explicit = AiwafConfigCompatCore.fromStructuredAndEnv(
                Map.of("rate_limiting", Map.of("max_requests", 7)), Map.of("AIWAF_PROFILE", "strict"));
        assertEquals(7, explicit.rateLimitMax);
        assertEquals(20, explicit.rateLimitFloodThreshold);
        AiwafConfig balanced = AiwafConfigCompatCore.fromStructured(Map.of());
        assertEquals(List.of(20, 40), List.of(balanced.rateLimitMax, balanced.rateLimitFloodThreshold));
        AiwafConfig monitor = new AiwafConfig().applyProfile("MONITOR");
        assertEquals(List.of("monitor", "monitor", "log"), List.of(monitor.profile, monitor.sqlInjectionMode, monitor.injectionRiskMode));
        assertThrows(IllegalArgumentException.class, () -> new AiwafConfig().applyProfile("loose"));
    }

    @Test
    void configFilesNameTheProfile() throws Exception {
        Path file = directory.resolve("aiwaf.json");
        Files.writeString(file, "{\"profile\": \"strict\"}");
        assertEquals(10, AiwafConfigFileCore.load(file, Map.of()).rateLimitMax);
        assertEquals("monitor", AiwafConfigFileCore.load(directory.resolve("missing.json"), Map.of("AIWAF_PROFILE", "monitor")).profile);
    }

    @Test
    void monitorAllowsWhatItWouldBlockAndLogsIt() {
        List<String> messages = new java.util.ArrayList<>();
        Handler handler = new Handler() {
            @Override public void publish(LogRecord record) { messages.add(record.getMessage()); }
            @Override public void flush() {}
            @Override public void close() {}
        };
        Logger logger = Logger.getLogger("com.aiwaf.monitor");
        logger.addHandler(handler);
        try {
            for (String profile : List.of("balanced", "monitor")) {
                AiwafConfig config = new AiwafConfig().applyProfile(profile);
                config.privateIpsExempted = false;
                config.rateLimitMax = 1;
                config.rateLimitFloodThreshold = 2;
                config.blockIpOnFloodBreach = true;
                AiwafEngine engine = new AiwafEngine(config);
                String ip = profile.equals("monitor") ? "198.51.100.71" : "198.51.100.70";
                List<Boolean> allowed = new java.util.ArrayList<>();
                for (int index = 0; index < 4; index++) {
                    allowed.add(engine.evaluate(req("/p", ip)).allowed());
                }
                if (profile.equals("monitor")) {
                    assertEquals(List.of(true, true, true, true), allowed);
                    assertFalse(BlocklistManager.isBlocked(engine.runtimeStorage(), ip));
                } else {
                    assertTrue(allowed.contains(false));
                }
            }
        } finally {
            logger.removeHandler(handler);
        }
        assertTrue(messages.stream().anyMatch(message -> message.startsWith("AIWAF monitor: would block GET /p from 198.51.100.71")), messages.toString());
        assertTrue(messages.stream().anyMatch(message -> message.startsWith("AIWAF monitor: would add 198.51.100.71 to the blocklist")), messages.toString());
    }
}
