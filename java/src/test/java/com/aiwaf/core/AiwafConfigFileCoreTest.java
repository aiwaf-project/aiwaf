package com.aiwaf.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AiwafConfigFileCoreTest {
    @TempDir
    Path tempDir;

    @Test
    void json_load_env_override_and_save_round_trip() throws Exception {
        Path source = tempDir.resolve("aiwaf.json");
        Files.writeString(source, """
                {
                  "storage": {"backend": "memory", "key_prefix": "tenant-a"},
                  "rate_limiting": {"max_requests": 31, "window_seconds": 12, "flood_threshold": 50},
                  "ai_anomaly": {"enabled": false},
                  "exemptions": {"private_ips_exempted": false}
                }
                """);

        AiwafConfig config = AiwafConfigFileCore.load(source, Map.of("AIWAF_RATE_MAX", "44"));
        assertEquals(44, config.rateLimitMax);
        assertEquals(12, config.rateLimitWindowSeconds);
        assertFalse(config.aiEnabled);
        assertFalse(config.privateIpsExempted);
        assertEquals("tenant-a", config.storageKeyPrefix);

        Path saved = tempDir.resolve("nested/saved.json");
        AiwafConfigFileCore.save(config, saved);
        AiwafConfig reloaded = AiwafConfigFileCore.load(saved, Map.of());
        assertEquals(44, reloaded.rateLimitMax);
        assertEquals("tenant-a", reloaded.storageKeyPrefix);
    }

    @Test
    void deep_merge_preserves_unmodified_nested_values() {
        Map<String, Object> merged = AiwafConfigFileCore.deepMerge(
                Map.of("rate_limiting", Map.of("max_requests", 20, "window_seconds", 10)),
                Map.of("rate_limiting", Map.of("max_requests", 50))
        );
        Map<?, ?> rate = (Map<?, ?>) merged.get("rate_limiting");
        assertEquals(50, rate.get("max_requests"));
        assertEquals(10, rate.get("window_seconds"));
    }

    @Test
    void validation_rejects_missing_redis_url_and_bad_thresholds() {
        AiwafConfig config = new AiwafConfig();
        config.storageBackend = "redis";
        config.storageRedisUrl = null;
        config.rateLimitMax = 50;
        config.rateLimitFloodThreshold = 40;

        List<String> errors = AiwafConfigFileCore.validate(config);
        assertTrue(errors.stream().anyMatch(value -> value.contains("redis_url")));
        assertTrue(errors.stream().anyMatch(value -> value.contains("flood_threshold")));
        assertThrows(IllegalArgumentException.class, () -> {
            Path path = tempDir.resolve("invalid.json");
            Files.writeString(path, "{\"storage\":{\"backend\":\"redis\"}}");
            AiwafConfigFileCore.load(path, Map.of());
        });
    }
}
