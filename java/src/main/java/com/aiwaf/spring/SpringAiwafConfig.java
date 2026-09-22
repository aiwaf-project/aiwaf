package com.aiwaf.spring;

import com.aiwaf.core.AiwafConfig;
import com.aiwaf.core.AiwafConfigCompatCore;
import com.aiwaf.core.AiwafConfigFileCore;
import org.springframework.core.env.Environment;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** Maps conventional Spring properties and AIWAF environment variables onto the core config. */
public final class SpringAiwafConfig {
    private SpringAiwafConfig() {}

    public static AiwafConfig fromEnvironment(Environment environment) {
        String configFile = environment == null ? System.getenv("AIWAF_CONFIG_FILE")
                : text(environment, "aiwaf.config-file", System.getenv("AIWAF_CONFIG_FILE"));
        AiwafConfig config;
        try {
            config = configFile == null || configFile.isBlank()
                    ? new AiwafConfig()
                    : AiwafConfigFileCore.load(Path.of(configFile), Map.of());
        } catch (IOException | IllegalArgumentException ex) {
            throw new IllegalStateException("Cannot load AIWAF configuration from " + configFile, ex);
        }
        AiwafConfigCompatCore.applyEnv(config, System.getenv());
        if (environment == null) return config;

        config.headerValidationEnabled = bool(environment, "aiwaf.header-validation.enabled", config.headerValidationEnabled);
        config.minHeaderQualityScore = integer(environment, "aiwaf.header-validation.quality-threshold", config.minHeaderQualityScore);
        config.maxHeaderBytes = integer(environment, "aiwaf.header-validation.max-header-bytes", config.maxHeaderBytes);
        config.maxHeaderCount = integer(environment, "aiwaf.header-validation.max-header-count", config.maxHeaderCount);
        config.maxUserAgentLength = integer(environment, "aiwaf.header-validation.max-user-agent-length", config.maxUserAgentLength);
        config.maxAcceptLength = integer(environment, "aiwaf.header-validation.max-accept-length", config.maxAcceptLength);
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
        config.uuidScoreWindowSeconds = integer(environment, "aiwaf.uuid.window-seconds", config.uuidScoreWindowSeconds);
        config.uuidMalformedWeight = integer(environment, "aiwaf.uuid.malformed-weight", config.uuidMalformedWeight);
        config.uuidNotFoundWeight = integer(environment, "aiwaf.uuid.not-found-weight", config.uuidNotFoundWeight);
        config.uuidSuccessDecay = integer(environment, "aiwaf.uuid.success-decay", config.uuidSuccessDecay);
        config.uuidParameterNames.addAll(csv(environment.getProperty("aiwaf.uuid.parameter-names")));

        config.geoBlockEnabled = bool(environment, "aiwaf.geo.enabled", config.geoBlockEnabled);
        config.geoBlockedCountries.addAll(csvUpper(environment.getProperty("aiwaf.geo.blocked-countries")));
        config.geoAllowedCountries.addAll(csvUpper(environment.getProperty("aiwaf.geo.allowed-countries")));
        config.geoIpDatabasePath = text(environment, "aiwaf.geo.database-path", config.geoIpDatabasePath);
        config.geoCacheSeconds = integer(environment, "aiwaf.geo.cache-seconds", config.geoCacheSeconds);
        config.geoMaxCacheEntries = integer(environment, "aiwaf.geo.max-cache-entries", config.geoMaxCacheEntries);
        config.geoExemptPaths.addAll(csv(environment.getProperty("aiwaf.geo.exempt-paths")));

        config.aiEnabled = bool(environment, "aiwaf.ai.enabled", config.aiEnabled);
        config.aiModelPath = text(environment, "aiwaf.ai.model-path", config.aiModelPath);
        config.aiAnomalyScoreThreshold = decimal(environment, "aiwaf.ai.threshold", config.aiAnomalyScoreThreshold);
        config.aiLazyLoadModel = bool(environment, "aiwaf.ai.lazy-load", config.aiLazyLoadModel);
        config.aiBackgroundPreload = bool(environment, "aiwaf.ai.background-preload", config.aiBackgroundPreload);
        config.aiRequireBehaviorConfirmation = bool(environment, "aiwaf.ai.require-behavior-confirmation", config.aiRequireBehaviorConfirmation);
        config.aiRecentWindowSeconds = integer(environment, "aiwaf.ai.recent-window-seconds", config.aiRecentWindowSeconds);
        config.aiMinRecentSamplesToBlock = integer(environment, "aiwaf.ai.min-recent-samples-to-block", config.aiMinRecentSamplesToBlock);
        config.pathManifestEnabled = bool(environment, "aiwaf.path-manifest.enabled", config.pathManifestEnabled);
        config.pathManifestPath = text(environment, "aiwaf.path-manifest.path", config.pathManifestPath);
        config.storageBackend = text(environment, "aiwaf.storage.backend", config.storageBackend);
        config.storageFilePath = text(environment, "aiwaf.storage.file-path", config.storageFilePath);
        config.storageRedisUrl = text(environment, "aiwaf.storage.redis-url", config.storageRedisUrl);
        config.storageKeyPrefix = text(environment, "aiwaf.storage.key-prefix", config.storageKeyPrefix);
        config.loggingEnabled = bool(environment, "aiwaf.logging.enabled", config.loggingEnabled);
        config.logDir = text(environment, "aiwaf.logging.directory", config.logDir);
        config.logFormat = text(environment, "aiwaf.logging.format", config.logFormat);
        config.privateIpsExempted = bool(environment, "aiwaf.exemptions.private-ips", config.privateIpsExempted);
        config.localhostExempted = bool(environment, "aiwaf.exemptions.localhost", config.localhostExempted);
        config.exemptIps.addAll(csv(environment.getProperty("aiwaf.exemptions.ips")));
        config.exemptIpPatterns.addAll(csv(environment.getProperty("aiwaf.exemptions.ip-patterns")));
        config.exemptPaths.addAll(csv(environment.getProperty("aiwaf.exemptions.paths")));
        config.exemptAllowWildcards = bool(environment, "aiwaf.exemptions.allow-wildcards", config.exemptAllowWildcards);
        config.exemptAllowPrefix = bool(environment, "aiwaf.exemptions.allow-prefix", config.exemptAllowPrefix);
        config.methodValidationEnabled = bool(environment, "aiwaf.security.method-validation-enabled", config.methodValidationEnabled);
        Set<String> allowedMethods = csvUpper(environment.getProperty("aiwaf.security.allowed-methods"));
        if (!allowedMethods.isEmpty()) config.allowedMethods = allowedMethods;
        config.maxRequestBodyBytes = integer(environment, "aiwaf.security.max-request-body-bytes", config.maxRequestBodyBytes);
        config.requestBodyInspectionBytes = integer(environment, "aiwaf.security.request-body-inspection-bytes", config.requestBodyInspectionBytes);
        config.requestBodyInspectionEnabled = bool(environment, "aiwaf.security.request-body-inspection-enabled", config.requestBodyInspectionEnabled);
        config.allowCompressedRequestBodies = bool(environment, "aiwaf.security.allow-compressed-request-bodies", config.allowCompressedRequestBodies);
        config.maxParameterCount = integer(environment, "aiwaf.security.max-parameter-count", config.maxParameterCount);
        config.maxParameterBytes = integer(environment, "aiwaf.security.max-parameter-bytes", config.maxParameterBytes);
        config.allowDuplicateParameters = bool(environment, "aiwaf.security.allow-duplicate-parameters", config.allowDuplicateParameters);
        config.maxRuntimeStateEntries = integer(environment, "aiwaf.performance.max-runtime-state-entries", config.maxRuntimeStateEntries);
        config.enabledMiddlewares.addAll(csv(environment.getProperty("aiwaf.middlewares.enabled")));
        config.disabledMiddlewares.addAll(csv(environment.getProperty("aiwaf.middlewares.disabled")));
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
