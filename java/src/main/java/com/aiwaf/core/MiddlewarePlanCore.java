package com.aiwaf.core;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

public final class MiddlewarePlanCore {

    public static final Set<String> AUTO_SENTINELS = Set.of("all", "auto", "aiwaf.all");
    public static final List<String> MIDDLEWARE_NAMES = Arrays.asList(
            "geo_block",
            "ip_keyword_block",
            "rate_limit",
            "ai_anomaly",
            "honeypot",
            "uuid_tamper",
            "header_validation",
            "logging"
    );

    private MiddlewarePlanCore() {}

    public static boolean isAutoSelection(Collection<String> requested) {
        if (requested == null || requested.isEmpty()) return false;
        for (String item : requested) {
            if (item != null && AUTO_SENTINELS.contains(item.trim().toLowerCase(Locale.ROOT))) {
                return true;
            }
        }
        return false;
    }

    public static boolean shouldEnableLogging(String accessLog) {
        return accessLog == null || accessLog.trim().isEmpty();
    }

    public static boolean shouldEnableGeo(
            boolean geoEnabledFlag,
            Collection<String> staticBlockCountries,
            Collection<String> dynamicBlockCountries
    ) {
        if (geoEnabledFlag) return true;
        if (staticBlockCountries != null && staticBlockCountries.stream().anyMatch(c -> c != null && !c.trim().isEmpty())) {
            return true;
        }
        if (dynamicBlockCountries != null && dynamicBlockCountries.stream().anyMatch(c -> c != null && !c.trim().isEmpty())) {
            return true;
        }
        return false;
    }

    public static boolean shouldEnableUuidTamper(Boolean hasUuidRoutes) {
        if (hasUuidRoutes == null) return true;
        return hasUuidRoutes;
    }

    public static Set<String> planEnabledMiddlewares(
            List<String> orderedAvailable,
            Collection<String> requested,
            Collection<String> disabled,
            String accessLog,
            boolean geoEnabledFlag,
            Collection<String> staticBlockCountries,
            Collection<String> dynamicBlockCountries,
            Boolean hasUuidRoutes
    ) {
        Set<String> disabledSet = new HashSet<>();
        if (disabled != null) {
            for (String d : disabled) {
                if (d != null) disabledSet.add(d.trim());
            }
        }

        Set<String> allSet = new HashSet<>(orderedAvailable);
        Set<String> enabled = new HashSet<>();

        if (requested == null || requested.isEmpty()) {
            enabled.addAll(allSet);
        } else if (isAutoSelection(requested)) {
            enabled.addAll(allSet);
            if (!shouldEnableLogging(accessLog)) {
                enabled.remove("logging");
                enabled.remove("logging_middleware");
            }
            if (enabled.contains("geo_block") && !shouldEnableGeo(geoEnabledFlag, staticBlockCountries, dynamicBlockCountries)) {
                enabled.remove("geo_block");
            }
            if (enabled.contains("uuid_tamper") && !shouldEnableUuidTamper(hasUuidRoutes)) {
                enabled.remove("uuid_tamper");
            }
        } else {
            for (String name : requested) {
                if (name != null && allSet.contains(name.trim())) {
                    enabled.add(name.trim());
                }
            }
        }

        enabled.removeAll(disabledSet);
        return enabled;
    }
}
