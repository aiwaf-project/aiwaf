package com.aiwaf.spring;

import com.aiwaf.core.AiwafConfig;
import com.aiwaf.core.AiwafConfigCompatCore;
import org.springframework.core.env.Environment;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

/** Maps conventional Spring properties and AIWAF environment variables onto the core config. */
public final class SpringAiwafConfig {
    private SpringAiwafConfig() {}

    public static AiwafConfig fromEnvironment(Environment environment) {
        AiwafConfig config = new AiwafConfig();
        AiwafConfigCompatCore.applyEnv(config, System.getenv());
        if (environment == null) return config;

        config.headerValidationEnabled = bool(environment, "aiwaf.header-validation.enabled", config.headerValidationEnabled);
        config.minHeaderQualityScore = integer(environment, "aiwaf.header-validation.quality-threshold", config.minHeaderQualityScore);
        config.rateLimitEnabled = bool(environment, "aiwaf.rate-limit.enabled", config.rateLimitEnabled);
        config.rateLimitMax = integer(environment, "aiwaf.rate-limit.max", config.rateLimitMax);
        config.rateLimitWindowSeconds = integer(environment, "aiwaf.rate-limit.window-seconds", config.rateLimitWindowSeconds);
        config.rateLimitFloodThreshold = integer(environment, "aiwaf.rate-limit.flood-threshold", config.rateLimitFloodThreshold);
        config.blockIpOnRateLimitBreach = bool(environment, "aiwaf.rate-limit.block-ip", config.blockIpOnRateLimitBreach);
        config.blockIpOnFloodBreach = bool(environment, "aiwaf.rate-limit.block-ip-on-flood", config.blockIpOnFloodBreach);
        String scope = environment.getProperty("aiwaf.rate-limit.scope");
        if (scope != null) {
            try {
                config.rateLimitScope = AiwafConfig.RateLimitScope.valueOf(
                        scope.trim().replace('-', '_').toUpperCase(Locale.ROOT));
            } catch (IllegalArgumentException ignored) {
                // Retain the safe default.
            }
        }

        config.honeypotEnabled = bool(environment, "aiwaf.honeypot.enabled", config.honeypotEnabled);
        config.minFormTimeSeconds = decimal(environment, "aiwaf.honeypot.min-form-time", config.minFormTimeSeconds);
        config.maxFormPageTimeSeconds = decimal(environment, "aiwaf.honeypot.max-page-time", config.maxFormPageTimeSeconds);
        config.uuidTamperEnabled = bool(environment, "aiwaf.uuid.enabled", config.uuidTamperEnabled);
        config.uuidScoreEnabled = bool(environment, "aiwaf.uuid.score-enabled", config.uuidScoreEnabled);
        config.uuidScoreBlockThreshold = integer(environment, "aiwaf.uuid.block-threshold", config.uuidScoreBlockThreshold);

        config.geoBlockEnabled = bool(environment, "aiwaf.geo.enabled", config.geoBlockEnabled);
        config.geoBlockedCountries.addAll(csvUpper(environment.getProperty("aiwaf.geo.blocked-countries")));
        config.geoAllowedCountries.addAll(csvUpper(environment.getProperty("aiwaf.geo.allowed-countries")));
        config.geoIpDatabasePath = text(environment, "aiwaf.geo.database-path", config.geoIpDatabasePath);
        config.geoCacheSeconds = integer(environment, "aiwaf.geo.cache-seconds", config.geoCacheSeconds);

        config.aiEnabled = bool(environment, "aiwaf.ai.enabled", config.aiEnabled);
        config.aiModelPath = text(environment, "aiwaf.ai.model-path", config.aiModelPath);
        config.aiAnomalyScoreThreshold = decimal(environment, "aiwaf.ai.threshold", config.aiAnomalyScoreThreshold);
        config.pathManifestEnabled = bool(environment, "aiwaf.path-manifest.enabled", config.pathManifestEnabled);
        config.pathManifestPath = text(environment, "aiwaf.path-manifest.path", config.pathManifestPath);
        config.storageBackend = text(environment, "aiwaf.storage.backend", config.storageBackend);
        config.storageFilePath = text(environment, "aiwaf.storage.file-path", config.storageFilePath);
        config.loggingEnabled = bool(environment, "aiwaf.logging.enabled", config.loggingEnabled);
        config.logDir = text(environment, "aiwaf.logging.directory", config.logDir);
        config.logFormat = text(environment, "aiwaf.logging.format", config.logFormat);
        config.privateIpsExempted = bool(environment, "aiwaf.exemptions.private-ips", config.privateIpsExempted);
        config.exemptIps.addAll(csv(environment.getProperty("aiwaf.exemptions.ips")));
        return config;
    }

    private static boolean bool(Environment env, String key, boolean fallback) {
        return env.getProperty(key, Boolean.class, fallback);
    }

    private static int integer(Environment env, String key, int fallback) {
        return env.getProperty(key, Integer.class, fallback);
    }

    private static double decimal(Environment env, String key, double fallback) {
        return env.getProperty(key, Double.class, fallback);
    }

    private static String text(Environment env, String key, String fallback) {
        String value = env.getProperty(key);
        return value == null || value.isBlank() ? fallback : value.trim();
    }

    private static Set<String> csv(String value) {
        if (value == null || value.isBlank()) return Set.of();
        Set<String> out = new HashSet<>();
        Arrays.stream(value.split(",")).map(String::trim).filter(v -> !v.isEmpty()).forEach(out::add);
        return out;
    }

    private static Set<String> csvUpper(String value) {
        Set<String> out = new HashSet<>();
        for (String item : csv(value)) out.add(item.toUpperCase(Locale.ROOT));
        return out;
    }
}
