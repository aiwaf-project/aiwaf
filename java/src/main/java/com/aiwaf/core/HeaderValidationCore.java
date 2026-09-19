package com.aiwaf.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

public final class HeaderValidationCore {
    public static final int MAX_HEADER_BYTES = 32 * 1024;
    public static final int MAX_HEADER_COUNT = 100;
    public static final int MAX_USER_AGENT_LENGTH = 500;
    public static final int MAX_ACCEPT_LENGTH = 4096;
    public static final List<String> REQUIRED_HEADERS = List.of("user-agent", "accept");
    public static final List<String> BROWSER_HEADERS = List.of(
            "accept-language", "accept-encoding", "connection", "cache-control"
    );
    public static final List<String> SUSPICIOUS_UA = List.of(
            "bot", "crawler", "spider", "scraper", "curl", "wget", "python", "java", "node",
            "go-http", "axios", "okhttp", "libwww", "lwp-trivial", "mechanize", "requests", "urllib",
            "httpie", "postman", "insomnia", "mozilla/4.0"
    );
    public static final List<String> LEGITIMATE_BOTS = List.of(
            "googlebot", "bingbot", "slurp", "duckduckbot", "baiduspider", "yandexbot",
            "facebookexternalhit", "twitterbot", "linkedinbot", "whatsapp", "telegrambot",
            "applebot", "pingdom", "uptimerobot", "statuscake", "site24x7"
    );

    private HeaderValidationCore() {}

    public static List<String> resolveRequiredHeaders(List<String> configured, String method) {
        if (configured == null) {
            return new ArrayList<>(REQUIRED_HEADERS);
        }
        return new ArrayList<>(configured);
    }

    public static String validate(
            Map<String, String> headers,
            String method,
            String serverProtocol,
            List<String> requiredHeaders,
            Integer minScore,
            int maxHeaderBytes,
            int maxHeaderCount,
            int maxUserAgentLength,
            int maxAcceptLength
    ) {
        int totalBytes = 0;
        int headerCount = 0;
        for (Map.Entry<String, String> e : headers.entrySet()) {
            String key = e.getKey() == null ? "" : e.getKey();
            String value = e.getValue() == null ? "" : e.getValue();
            headerCount++;
            totalBytes += key.length() + value.length();
            if (totalBytes > maxHeaderBytes) {
                return "Header bytes exceed " + maxHeaderBytes;
            }
        }
        if (headerCount > maxHeaderCount) {
            return "Header count exceeds " + maxHeaderCount;
        }

        String userAgent = getHeader(headers, "user-agent");
        if (!userAgent.isEmpty() && userAgent.length() > maxUserAgentLength) {
            return "User-Agent longer than " + maxUserAgentLength + " chars";
        }
        String accept = getHeader(headers, "accept");
        if (!accept.isEmpty() && accept.length() > maxAcceptLength) {
            return "Accept header longer than " + maxAcceptLength + " chars";
        }

        List<String> required = (requiredHeaders == null) ? REQUIRED_HEADERS : requiredHeaders;
        List<String> missing = new ArrayList<>();
        for (String header : required) {
            String normalized = normalizeRequiredHeaderName(header);
            if (getHeader(headers, normalized).isEmpty()) {
                missing.add(normalized);
            }
        }
        if (!missing.isEmpty()) {
            return "Missing required headers: " + String.join(", ", missing);
        }

        if (!required.isEmpty()) {
            String suspiciousUa = checkUserAgent(userAgent, maxUserAgentLength);
            if (suspiciousUa != null) {
                return "Suspicious user agent: " + suspiciousUa;
            }

            String suspiciousCombo = checkHeaderCombinations(headers, required, serverProtocol);
            if (suspiciousCombo != null) {
                return "Suspicious headers: " + suspiciousCombo;
            }

            boolean fullBrowserProfileRequired = requiresHeader(required, "user-agent")
                    && requiresHeader(required, "accept");
            if (fullBrowserProfileRequired) {
                int threshold = (minScore == null) ? 3 : minScore;
                int score = calculateHeaderQuality(headers);
                if (threshold > 0 && score < threshold) {
                    return "Low header quality score: " + score;
                }
            }
        }
        return null;
    }

    private static String normalizeRequiredHeaderName(String header) {
        if (header == null) {
            return "";
        }
        String normalized = header.trim().toLowerCase(Locale.ROOT).replace('_', '-');
        if (normalized.startsWith("http-")) {
            normalized = normalized.substring("http-".length());
        }
        return normalized;
    }

    private static boolean requiresHeader(List<String> required, String header) {
        for (String candidate : required) {
            if (header.equals(normalizeRequiredHeaderName(candidate))) {
                return true;
            }
        }
        return false;
    }

    public static String checkUserAgent(String userAgent, int maxUserAgentLength) {
        if (userAgent == null || userAgent.isEmpty()) {
            return null;
        }
        if (userAgent.length() > maxUserAgentLength) {
            return "User-Agent longer than " + maxUserAgentLength + " chars";
        }
        String ua = userAgent.toLowerCase(Locale.ROOT);
        for (String legit : LEGITIMATE_BOTS) {
            if (ua.contains(legit)) {
                return null;
            }
        }
        for (String suspicious : SUSPICIOUS_UA) {
            if ("mozilla/4.0".equals(suspicious)) {
                if (ua.equals("mozilla/4.0")) {
                    return "Pattern: mozilla/4.0$";
                }
            } else if (ua.contains(suspicious)) {
                return "Pattern: " + suspicious;
            }
        }
        if (ua.length() < 10) {
            return "Too short";
        }
        return null;
    }

    public static String checkHeaderCombinations(Map<String, String> headers, List<String> required, String serverProtocol) {
        Set<String> req = Set.copyOf(required);
        String ua = getHeader(headers, "user-agent").toLowerCase(Locale.ROOT);
        String accept = getHeader(headers, "accept");
        String protocol = serverProtocol == null ? "" : serverProtocol;
        if (protocol.startsWith("HTTP/2") && ua.contains("mozilla/4.0")) {
            return "HTTP/2 with old browser user agent";
        }
        if (!ua.isEmpty() && accept.isEmpty() && req.contains("accept")) {
            return "User-Agent present but no Accept header";
        }
        if ("*/*".equals(accept)
                && getHeader(headers, "accept-language").isEmpty()
                && getHeader(headers, "accept-encoding").isEmpty()) {
            return "Generic Accept header without language/encoding";
        }
        if (!ua.isEmpty() && "HTTP/1.0".equals(protocol) && ua.contains("chrome")) {
            return "Modern browser with HTTP/1.0";
        }
        return null;
    }

    public static int calculateHeaderQuality(Map<String, String> headers) {
        int score = 0;
        if (!getHeader(headers, "user-agent").isEmpty()) score += 2;
        if (!getHeader(headers, "accept").isEmpty()) score += 2;
        for (String header : BROWSER_HEADERS) {
            if (!getHeader(headers, header).isEmpty()) score += 1;
        }
        if (!getHeader(headers, "accept-language").isEmpty() && !getHeader(headers, "accept-encoding").isEmpty()) {
            score += 1;
        }
        if ("keep-alive".equalsIgnoreCase(getHeader(headers, "connection"))) {
            score += 1;
        }
        String accept = getHeader(headers, "accept");
        if (accept.contains("text/html") && accept.contains("application/xml")) {
            score += 1;
        }
        return score;
    }

    public static String getHeader(Map<String, String> headers, String name) {
        for (Map.Entry<String, String> e : headers.entrySet()) {
            if (name.equalsIgnoreCase(e.getKey())) {
                return e.getValue() == null ? "" : e.getValue();
            }
        }
        return "";
    }
}
