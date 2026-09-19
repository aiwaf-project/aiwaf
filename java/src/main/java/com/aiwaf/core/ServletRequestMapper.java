package com.aiwaf.core;

import jakarta.servlet.http.HttpServletRequest;
import com.aiwaf.runtime.CidrUtil;

import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public final class ServletRequestMapper {
    private ServletRequestMapper() {}

    public static AiwafRequest from(HttpServletRequest req) {
        return from(req, Set.of(), new AiwafConfig());
    }

    public static AiwafRequest from(HttpServletRequest req, Set<String> disabledMiddlewares) {
        return from(req, disabledMiddlewares, new AiwafConfig());
    }

    public static AiwafRequest from(HttpServletRequest req, Set<String> disabledMiddlewares, AiwafConfig config) {
        return from(req, disabledMiddlewares, config, "");
    }

    public static AiwafRequest from(HttpServletRequest req, Set<String> disabledMiddlewares, AiwafConfig config, String bodyPreview) {
        Map<String, String> headers = new HashMap<>();
        Enumeration<String> names = req.getHeaderNames();
        while (names != null && names.hasMoreElements()) {
            String name = names.nextElement();
            headers.put(name, req.getHeader(name));
        }
        headers.entrySet().removeIf(e -> "aiwaf-internal-duplicate-parameters".equalsIgnoreCase(e.getKey()));

        Map<String, String> query = new HashMap<>();
        boolean[] duplicateParameters = {false};
        req.getParameterMap().forEach((k, v) -> {
            if (v != null && v.length > 1) duplicateParameters[0] = true;
            query.put(k, (v != null && v.length > 0) ? v[0] : "");
        });
        if (duplicateParameters[0]) headers.put("AIWAF-Internal-Duplicate-Parameters", "true");

        boolean trustedPeer = isTrustedProxy(req.getRemoteAddr(), config.trustedProxyCidrs);
        String country = trustedPeer ? getHeader(headers, "X-Country").trim().toUpperCase(java.util.Locale.ROOT) : "";
        if (!country.matches("[A-Z]{2}")) country = "";
        String clientIp = resolveClientIp(req, headers, config, trustedPeer);
        return new AiwafRequest(
                req.getMethod(),
                req.getRequestURI(),
                clientIp,
                country,
                Collections.unmodifiableMap(headers),
                Collections.unmodifiableMap(query),
                System.currentTimeMillis(),
                disabledMiddlewares == null ? Set.of() : Set.copyOf(disabledMiddlewares),
                bodyPreview == null ? "" : bodyPreview
        );
    }

    private static String resolveClientIp(HttpServletRequest req, Map<String, String> headers, AiwafConfig config, boolean trustedPeer) {
        String remote = req.getRemoteAddr();
        if (!trustedPeer) {
            return (remote == null || remote.isBlank()) ? "127.0.0.1" : remote;
        }
        String forwardedFor = getHeader(headers, "X-Forwarded-For");
        if (!forwardedFor.isBlank() && forwardedFor.length() <= 4096) {
            String[] parts = forwardedFor.split(",");
            int start = Math.max(0, parts.length - Math.max(1, config.maxForwardedForEntries));
            for (int i = parts.length - 1; i >= start; i--) {
                String part = parts[i];
                String candidate = part.trim();
                if (!candidate.isBlank() && !"unknown".equalsIgnoreCase(candidate)
                        && isIpLiteral(candidate)
                        && !isTrustedProxy(candidate, config.trustedProxyCidrs)) {
                    return candidate;
                }
            }
        }
        String realIp = getHeader(headers, "X-Real-IP");
        if (!realIp.isBlank() && !"unknown".equalsIgnoreCase(realIp) && isIpLiteral(realIp.trim())) {
            return realIp.trim();
        }
        return (remote == null || remote.isBlank()) ? "127.0.0.1" : remote;
    }

    private static boolean isIpLiteral(String value) {
        return CidrUtil.isLiteral(value);
    }

    private static boolean isTrustedProxy(String ip, Set<String> cidrs) {
        if (ip == null || cidrs == null) return false;
        for (String cidr : cidrs) {
            if (cidr == null) continue;
            if (cidr.equals(ip) || ("::1/128".equals(cidr) && "::1".equals(ip)) || CidrUtil.contains(cidr, ip)) {
                return true;
            }
        }
        return false;
    }

    private static String getHeader(Map<String, String> headers, String name) {
        for (Map.Entry<String, String> entry : headers.entrySet()) {
            if (name.equalsIgnoreCase(entry.getKey())) {
                return entry.getValue() == null ? "" : entry.getValue();
            }
        }
        return "";
    }
}
