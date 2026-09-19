package com.aiwaf.core;

import java.util.Locale;
import java.util.Set;

public final class HoneypotCore {
    private HoneypotCore() {}

    public static HoneypotDecision evaluateHoneypotRequest(
            String method,
            String path,
            double nowTs,
            Double storedGetTs,
            HoneypotConfig config,
            boolean isAuthenticated,
            boolean viewAcceptsMethod
    ) {
        HoneypotConfig cfg = config == null ? new HoneypotConfig() : config;
        if (isAuthenticated) {
            return HoneypotDecision.allowDecision();
        }

        String methodU = method == null ? "" : method.toUpperCase(Locale.ROOT);
        String reqPath = path == null ? "" : path;

        if ("GET".equals(methodU)) {
            if (!viewAcceptsMethod && looksObviousPostOnly(reqPath)) {
                return new HoneypotDecision(false, "GET to obvious POST-only endpoint: " + reqPath, 405, false);
            }
            return HoneypotDecision.allowDecision();
        }

        if ("POST".equals(methodU)) {
            if (!viewAcceptsMethod) {
                return new HoneypotDecision(false, "POST to GET-only view: " + reqPath, 405, false);
            }
            if (storedGetTs == null) {
                return HoneypotDecision.allowDecision();
            }
            double delta = nowTs - storedGetTs;
            if (delta > cfg.maxPageTimeSeconds()) {
                return new HoneypotDecision(false, "page_expired", 409, true);
            }

            double minTime = cfg.minFormTimeSeconds();
            String pathLower = reqPath.toLowerCase(Locale.ROOT);
            for (String prefix : cfg.loginPrefixes()) {
                if (pathLower.startsWith(prefix)) {
                    minTime = cfg.loginMinFormTimeSeconds();
                    break;
                }
            }
            if (delta < minTime) {
                return new HoneypotDecision(false, String.format("Form submitted too quickly (%.2fs)", delta), 403, false);
            }
            return HoneypotDecision.allowDecision();
        }

        if (!Set.of("HEAD", "OPTIONS").contains(methodU) && !viewAcceptsMethod) {
            return new HoneypotDecision(false, methodU + " to view that doesn't support it: " + reqPath, 405, false);
        }

        return HoneypotDecision.allowDecision();
    }

    private static boolean looksObviousPostOnly(String path) {
        String lower = path.toLowerCase(Locale.ROOT);
        return lower.endsWith("/create/")
                || lower.endsWith("/submit/")
                || lower.endsWith("/upload/")
                || lower.endsWith("/delete/")
                || lower.endsWith("/process/");
    }

    public record HoneypotConfig(
            double minFormTimeSeconds,
            double maxPageTimeSeconds,
            double loginMinFormTimeSeconds,
            Set<String> loginPrefixes
    ) {
        public HoneypotConfig() {
            this(
                    1.0,
                    240.0,
                    0.1,
                    Set.of("/admin/login/", "/login/", "/accounts/login/", "/auth/login/", "/signin/")
            );
        }
    }

    public record HoneypotDecision(
            boolean allow,
            String reason,
            Integer statusCode,
            boolean reloadRequired
    ) {
        static HoneypotDecision allowDecision() {
            return new HoneypotDecision(true, null, null, false);
        }
    }
}
