package com.aiwaf.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

public final class AiwafConfig {
    public enum RateLimitScope { PER_PATH, GLOBAL_IP }

    public static final class PathRule {
        public final String prefix;
        public final boolean disableHeaderValidation;
        public final Integer rateLimitMaxOverride;
        public final Integer rateLimitWindowOverride;
        public final Integer rateLimitFloodOverride;
        public final Set<String> disabledMiddlewares;
        public final Map<String, Map<String, Integer>> middlewareOverrides;

        public PathRule(String prefix, boolean disableHeaderValidation, Integer rateLimitMaxOverride) {
            this(prefix, disableHeaderValidation, rateLimitMaxOverride, null, null, new HashSet<>(), new HashMap<>());
        }

        public PathRule(
                String prefix,
                boolean disableHeaderValidation,
                Integer rateLimitMaxOverride,
                Integer rateLimitWindowOverride,
                Integer rateLimitFloodOverride,
                Set<String> disabledMiddlewares
        ) {
            this(prefix, disableHeaderValidation, rateLimitMaxOverride, rateLimitWindowOverride, rateLimitFloodOverride, disabledMiddlewares, new HashMap<>());
        }

        public PathRule(
                String prefix,
                boolean disableHeaderValidation,
                Integer rateLimitMaxOverride,
                Integer rateLimitWindowOverride,
                Integer rateLimitFloodOverride,
                Set<String> disabledMiddlewares,
                Map<String, Map<String, Integer>> middlewareOverrides
        ) {
            this.prefix = prefix;
            this.disableHeaderValidation = disableHeaderValidation;
            this.rateLimitMaxOverride = rateLimitMaxOverride;
            this.rateLimitWindowOverride = rateLimitWindowOverride;
            this.rateLimitFloodOverride = rateLimitFloodOverride;
            this.disabledMiddlewares = disabledMiddlewares == null ? new HashSet<>() : new HashSet<>(disabledMiddlewares);
            this.middlewareOverrides = normalizeOverrides(middlewareOverrides);
        }

        public boolean disables(String middlewareName) {
            if (middlewareName == null || middlewareName.isBlank()) {
                return false;
            }
            String target = ExemptionsCore.normalizeMiddlewareName(middlewareName);
            if ("header_validation".equals(target) && disableHeaderValidation) {
                return true;
            }
            for (String m : disabledMiddlewares) {
                if (ExemptionsCore.normalizeMiddlewareName(m).equals(target)) {
                    return true;
                }
            }
            return false;
        }

        public Integer getOverride(String middlewareName, String key) {
            if (middlewareName == null || key == null) {
                return null;
            }
            String mw = ExemptionsCore.normalizeMiddlewareName(middlewareName);
            Map<String, Integer> forMiddleware = middlewareOverrides.get(mw);
            if (forMiddleware == null) {
                return null;
            }
            return forMiddleware.get(key.toLowerCase(Locale.ROOT));
        }

        private static Map<String, Map<String, Integer>> normalizeOverrides(Map<String, Map<String, Integer>> source) {
            if (source == null || source.isEmpty()) {
                return Collections.emptyMap();
            }
            Map<String, Map<String, Integer>> out = new HashMap<>();
            for (Map.Entry<String, Map<String, Integer>> entry : source.entrySet()) {
                String middleware = ExemptionsCore.normalizeMiddlewareName(entry.getKey());
                Map<String, Integer> values = new HashMap<>();
                if (entry.getValue() != null) {
                    for (Map.Entry<String, Integer> kv : entry.getValue().entrySet()) {
                        if (kv.getKey() != null && kv.getValue() != null) {
                            values.put(kv.getKey().toLowerCase(Locale.ROOT), kv.getValue());
                        }
                    }
                }
                out.put(middleware, values);
            }
            return out;
        }
    }

    public boolean headerValidationEnabled = true;
    public boolean rateLimitEnabled = true;
    public int rateLimitMax = 20;
    public int rateLimitWindowSeconds = 10;
    public RateLimitScope rateLimitScope = RateLimitScope.PER_PATH;
    public boolean blockIpOnRateLimitBreach = false;
    public boolean blockIpOnFloodBreach = false;
    public int rateLimitFloodThreshold = 40;
    public Set<String> trustedProxyCidrs = new HashSet<>(Set.of("127.0.0.0/8", "::1/128"));
    public int maxForwardedForEntries = 16;
    public Set<String> exemptIps = new HashSet<>();
    public boolean privateIpsExempted = false;
    public Set<String> autoExemptPathPrefixes = new HashSet<>();
    public Set<String> exemptPaths = new HashSet<>(Arrays.asList(
            "/favicon.ico", "/robots.txt", "/sitemap.xml", "/sitemap.txt", "/ads.txt", "/security.txt",
            "/.well-known/", "/apple-touch-icon.png", "/apple-touch-icon-precomposed.png",
            "/manifest.json", "/browserconfig.xml", "/health", "/healthcheck", "/ping", "/status",
            "*.css", "*.js", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico", "*.svg",
            "*.woff", "*.woff2", "*.ttf", "*.eot", "/static/", "/assets/", "/css/", "/js/",
            "/images/", "/img/", "/fonts/", "/healthz", "/metrics", "/api/health", "/api/status"
    ));
    public boolean exemptAllowWildcards = true;
    public boolean exemptAllowPrefix = true;
    public boolean geoBlockEnabled = false;
    public Set<String> geoAllowedCountries = new HashSet<>();
    public Set<String> geoBlockedCountries = new HashSet<>();
    public Set<String> geoExemptPaths = new HashSet<>();
    public boolean honeypotEnabled = true;
    public double minFormTimeSeconds = 1.0;
    public double maxFormPageTimeSeconds = 240.0;
    public double loginMinFormTimeSeconds = 0.1;
    public Set<String> loginPathPrefixes = new HashSet<>(Set.of(
            "/admin/login/",
            "/login/",
            "/accounts/login/",
            "/auth/login/",
            "/signin/"
    ));
    public boolean uuidTamperEnabled = true;
    public boolean ipKeywordBlockEnabled = true;
    public boolean methodValidationEnabled = true;
    public Set<String> allowedMethods = new HashSet<>(Arrays.asList("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"));
    public Set<String> enabledMiddlewares = new HashSet<>();
    public Set<String> disabledMiddlewares = new HashSet<>();
    public int maxHeaderBytes = 32 * 1024;
    public int maxHeaderCount = 100;
    public int maxUserAgentLength = 500;
    public int maxAcceptLength = 4096;
    public int maxRequestBodyBytes = 1024 * 1024;
    public int requestBodyInspectionBytes = 64 * 1024;
    public boolean requestBodyInspectionEnabled = true;
    public boolean allowCompressedRequestBodies = false;
    public int maxRuntimeStateEntries = 10_000;
    public int maxParameterCount = 200;
    public int maxParameterBytes = 16 * 1024;
    public boolean allowDuplicateParameters = false;
    public Map<String, Set<String>> allowedContentTypesByPathPrefix = new HashMap<>();
    public int minHeaderQualityScore = 3;
    public List<String> requiredHeaders = new ArrayList<>(HeaderValidationCore.REQUIRED_HEADERS);
    public Map<String, List<String>> requiredHeadersByMethod = new HashMap<>();
    public Set<String> blockedPathPatterns = new HashSet<>(Arrays.asList(".env", "../", "%2e%2e", "wp-admin", "phpmyadmin", ".git"));
    public List<PathRule> pathRules = new ArrayList<>();
    public boolean aiEnabled = false;
    public boolean aiLazyLoadModel = true;
    public boolean aiBackgroundPreload = false;
    public String aiModelPath = "aiwaf-model.bin";
    public double aiAnomalyScoreThreshold = 0.7;
    public boolean aiRequireBehaviorConfirmation = true;
    public int aiRecentWindowSeconds = 300;
    public int aiMinRecentSamplesToBlock = 5;
    public boolean aiTimingLogsEnabled = false;
    public boolean enableKeywordLearning = true;
    public int dynamicTopN = 10;
    public Set<String> exemptKeywords = new HashSet<>();
    public Set<String> legitimatePathKeywords = new HashSet<>();
    public boolean storeExtendedBlockInfo = true;
    public int blockInfoMaxHeaders = 50;
    public int blockInfoMaxHeaderValueLength = 512;
    public Set<String> blockInfoRedactHeaders = new HashSet<>(Set.of("authorization", "cookie", "set-cookie"));
    public Set<String> sensitiveParameterNames = new HashSet<>(Set.of(
            "password", "passwd", "pwd", "token", "access_token", "refresh_token",
            "secret", "api_key", "apikey", "authorization", "code", "session", "sessionid"
    ));
    public boolean logQueryParameters = false;
    public boolean observabilityEnabled = true;
    public boolean loggingEnabled = true;
    public String logDir = "aiwaf_logs";
    public String logFormat = "combined";
    public String storageBackend = "memory";
    public String storageFilePath = null;
    public Set<String> legitimateRouteHints = new HashSet<>();

    public boolean isAutoExemptPath(String path) {
        if (path == null || path.isBlank()) {
            return false;
        }
        for (String prefix : autoExemptPathPrefixes) {
            if (path.startsWith(prefix)) {
                return true;
            }
        }
        return false;
    }

    public boolean isMethodAllowed(String method) {
        if (method == null || method.isBlank()) {
            return false;
        }
        return allowedMethods.contains(method.toUpperCase(Locale.ROOT));
    }

    public List<String> resolveRequiredHeaders(String method) {
        if (method != null) {
            List<String> byMethod = requiredHeadersByMethod.get(method.toUpperCase(Locale.ROOT));
            if (byMethod != null) {
                return byMethod;
            }
        }
        List<String> defaultHeaders = requiredHeadersByMethod.get("DEFAULT");
        if (defaultHeaders != null) {
            return defaultHeaders;
        }
        return requiredHeaders;
    }

    public boolean isMiddlewareEnabled(String middlewareName) {
        String normalized = ExemptionsCore.normalizeMiddlewareName(middlewareName);
        if (!enabledMiddlewares.isEmpty() && !containsNormalized(enabledMiddlewares, normalized)) {
            return false;
        }
        return !containsNormalized(disabledMiddlewares, normalized);
    }

    private static boolean containsNormalized(Set<String> values, String target) {
        for (String value : values) {
            if (ExemptionsCore.normalizeMiddlewareName(value).equals(target)) {
                return true;
            }
        }
        return false;
    }
}
