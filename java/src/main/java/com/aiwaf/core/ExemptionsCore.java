package com.aiwaf.core;

import java.util.Collection;
import java.util.Locale;

public final class ExemptionsCore {
    private ExemptionsCore() {}

    public static String normalizePath(String path, Boolean trailingSlash) {
        if (path == null || path.isBlank()) {
            return "/";
        }
        String cleaned = path.trim().replaceAll("/+", "/");
        if (!cleaned.startsWith("/")) {
            cleaned = "/" + cleaned;
        }
        if (Boolean.TRUE.equals(trailingSlash) && !cleaned.endsWith("/")) {
            cleaned = cleaned + "/";
        }
        if (Boolean.FALSE.equals(trailingSlash) && cleaned.length() > 1 && cleaned.endsWith("/")) {
            cleaned = cleaned.substring(0, cleaned.length() - 1);
        }
        return cleaned.toLowerCase(Locale.ROOT);
    }

    public static boolean isPathExempt(
            String path,
            Collection<String> exemptPaths,
            boolean allowWildcards,
            boolean allowPrefix
    ) {
        if (path == null || path.isBlank() || exemptPaths == null || exemptPaths.isEmpty()) {
            return false;
        }
        String pathLower = normalizePath(path, null);
        for (String exempt : exemptPaths) {
            if (exempt == null || exempt.isBlank()) {
                continue;
            }
            String exemptNorm = normalizePath(exempt, null);
            if (allowWildcards && exemptNorm.contains("*")) {
                String regex = exemptNorm.replace(".", "\\.").replace("*", ".*");
                if (pathLower.matches(regex)) {
                    return true;
                }
                continue;
            }
            if (pathLower.equals(exemptNorm)) {
                return true;
            }
            if (allowPrefix) {
                String prefix = exemptNorm.endsWith("/") && exemptNorm.length() > 1
                        ? exemptNorm.substring(0, exemptNorm.length() - 1)
                        : exemptNorm;
                if (!prefix.isBlank() && (pathLower.equals(prefix) || pathLower.startsWith(prefix + "/"))) {
                    return true;
                }
            }
        }
        return false;
    }

    public static AiwafConfig.PathRule getPathRuleForPath(String path, Collection<AiwafConfig.PathRule> rules) {
        if (path == null || path.isBlank() || rules == null || rules.isEmpty()) {
            return null;
        }
        String normalizedPath = normalizePath(path, false);
        AiwafConfig.PathRule best = null;
        int bestLen = -1;
        for (AiwafConfig.PathRule rule : rules) {
            if (rule == null || rule.prefix == null || rule.prefix.isBlank()) {
                continue;
            }
            String prefix = normalizePath(rule.prefix, true);
            String prefixTrimmed = prefix.endsWith("/") ? prefix.substring(0, prefix.length() - 1) : prefix;
            if (normalizedPath.equals(prefixTrimmed) || normalizedPath.startsWith(prefix)) {
                if (prefix.length() > bestLen) {
                    best = rule;
                    bestLen = prefix.length();
                }
            }
        }
        return best;
    }

    public static String normalizeMiddlewareName(String name) {
        if (name == null || name.isBlank()) {
            return "";
        }
        String v = name.trim();
        int idx = v.lastIndexOf('.');
        if (idx >= 0 && idx + 1 < v.length()) {
            v = v.substring(idx + 1);
        }
        String lower = v.toLowerCase(Locale.ROOT);
        String normalized = lower.replace('-', '_');
        return switch (normalized) {
            case "headervalidationmiddleware", "header_validation", "headervalidation" -> "header_validation";
            case "ratelimitmiddleware", "rate_limit", "ratelimiting", "ratelimit" -> "rate_limit";
            case "ipandkeywordblockmiddleware", "ip_keyword_block", "ipkeywordblock" -> "ip_keyword_block";
            case "geoblockmiddleware", "geo_block", "geoblock" -> "geo_block";
            case "honeypottimingmiddleware", "honeypot" -> "honeypot";
            case "uuidtampermiddleware", "uuid_tamper" -> "uuid_tamper";
            case "aianomalymiddleware", "ai_anomaly" -> "ai_anomaly";
            case "aiwafloggingmiddleware", "logging_middleware", "logging" -> "logging";
            default -> lower;
        };
    }
}
