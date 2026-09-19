package com.aiwaf.core;

import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Pattern;

public final class AnomalyCore {
    private static final Pattern MULTISLASH = Pattern.compile("/{2,}");
    private static final Pattern HEX = Pattern.compile("%[0-9a-fA-F]{2}");

    private AnomalyCore() {}

    public static CoreAnalysisDecision scoreRequestAnomaly(
            NormalizedRequestCore request,
            CorePolicyConfig.AnomalyConfig config
    ) {
        List<String> reasons = new ArrayList<>();
        double score = 0.0;
        String path = request.path() == null ? "/" : request.path();
        String decodedPath = URLDecoder.decode(path, StandardCharsets.UTF_8).toLowerCase(Locale.ROOT);
        String lowerPath = path.toLowerCase(Locale.ROOT);
        String method = request.method() == null ? "" : request.method().toUpperCase(Locale.ROOT);

        if (!config.methodAllowlist().contains(method)) {
            reasons.add("method_not_allowlisted");
            score += 0.35;
        }
        if (path.length() > config.pathLengthLimit()) {
            reasons.add("path_too_long");
            score += 0.35;
        }
        if (lowerPath.contains("../") || lowerPath.contains("..\\") || decodedPath.contains("../") || decodedPath.contains("..\\")) {
            reasons.add("path_traversal_pattern");
            score += 0.30;
        }
        if (MULTISLASH.matcher(path).find()) {
            reasons.add("path_multi_slash");
            score += 0.10;
        }
        int hexCount = (int) HEX.matcher(path).results().count();
        if (hexCount >= 4) {
            reasons.add("high_encoded_path_density");
            score += 0.15;
        }
        if ("TRACE".equals(method) || "CONNECT".equals(method)) {
            reasons.add("risky_http_method");
            score += 0.20;
        }
        if (isScanningPath(path)) {
            reasons.add("scanner_like_path");
            score += 0.2;
        }

        String keywordText = path + " " + defaultString(request.query()) + " " + defaultString(request.body());
        List<String> matched = extractKeywords(keywordText, config.suspiciousKeywords());
        if (!matched.isEmpty()) {
            reasons.add("suspicious_keywords");
            score += Math.min(0.5, 0.1 * matched.size());
        }

        if (isMaliciousContext(path, defaultString(request.query()), request.knownPath(), config.suspiciousKeywords())) {
            reasons.add("malicious_context");
            score += 0.25;
        }

        score = Math.min(score, 1.0);
        return new CoreAnalysisDecision(score < 0.5, score, reasons, matched);
    }

    public static List<String> filterSuspiciousKeywords(
            String path,
            Set<String> allKeywords,
            boolean knownPath,
            Set<String> exemptKeywords,
            Set<String> legitimateKeywords,
            Set<String> safePrefixes
    ) {
        String lowerPath = defaultString(path).toLowerCase(Locale.ROOT);
        if (lowerPath.startsWith("/")) {
            lowerPath = lowerPath.substring(1);
        }

        List<String> clean = new ArrayList<>();
        for (String kw : allKeywords) {
            if (exemptKeywords.contains(kw)) continue;
            boolean safe = false;
            for (String prefix : safePrefixes) {
                if (prefix != null && !prefix.isBlank()) {
                    String p = prefix.toLowerCase(Locale.ROOT);
                    if (p.startsWith("/")) p = p.substring(1);
                    if (lowerPath.startsWith(p)) {
                        safe = true;
                        break;
                    }
                }
            }
            if (safe) continue;
            if (knownPath && legitimateKeywords.contains(kw) && !isMaliciousContext(path, "", true, allKeywords)) {
                continue;
            }
            if (!clean.contains(kw)) clean.add(kw);
        }
        return clean;
    }

    public static boolean isScanningPath(String path) {
        String lower = defaultString(path).toLowerCase(Locale.ROOT);
        String[] patterns = {
                "wp-admin", "wp-content", "wp-includes", "wp-config", "xmlrpc.php", "admin", "phpmyadmin",
                "adminer", "config", "configuration", "settings", "setup", "install", "installer", "backup",
                "database", "db", "mysql", "sql", "dump", ".env", ".git", ".htaccess", ".htpasswd", "passwd",
                "shadow", "robots.txt", "sitemap.xml", "cgi-bin", "scripts", "shell", "cmd", "exec", ".php",
                ".asp", ".aspx", ".jsp", ".cgi", ".pl"
        };
        for (String pattern : patterns) {
            if (lower.contains(pattern)) return true;
        }
        if (lower.contains("../") || lower.contains("..\\")) return true;
        return lower.contains("%2e%2e") || lower.contains("%252e") || lower.contains("%c0%ae");
    }

    private static boolean isMaliciousContext(String path, String query, boolean knownPath, Set<String> staticKeywords) {
        if (knownPath) return false;
        String lowPath = defaultString(path).toLowerCase(Locale.ROOT);
        String lowQuery = defaultString(query).toLowerCase(Locale.ROOT);

        List<String> segs = splitSegments(lowPath, 3);
        int keywordSegments = 0;
        for (String seg : segs) {
            if (staticKeywords.contains(seg)) keywordSegments++;
        }
        if (keywordSegments > 1) return true;

        String[] pathIndicators = {"../", "..\\", ".env", "wp-admin", "phpmyadmin", "config", "backup", "database", "mysql", "passwd", "shadow"};
        for (String indicator : pathIndicators) {
            if (lowPath.contains(indicator)) return true;
        }
        String[] queryIndicators = {"cmd", "exec", "system", "shell"};
        for (String indicator : queryIndicators) {
            if (lowQuery.contains(indicator + "=") || lowQuery.contains("&" + indicator + "=")) return true;
        }
        if (count(lowPath, "../") > 2 || count(lowPath, "..\\") > 2) return true;
        if (lowPath.contains("%2e%2e") || lowPath.contains("%252e") || lowPath.contains("%c0%ae") || lowPath.contains("%3c%73%63%72%69%70%74")) return true;
        String[] attacks = {"union+select", "drop+table", "<script", "javascript:", "${", "{{", "onload=", "onerror="};
        for (String attack : attacks) {
            if (lowPath.contains(attack)) return true;
        }
        return false;
    }

    private static List<String> splitSegments(String path, int minLen) {
        String[] raw = defaultString(path).toLowerCase(Locale.ROOT).replaceFirst("^/", "").split("\\W+");
        List<String> out = new ArrayList<>();
        for (String seg : raw) {
            if (seg.length() >= minLen) out.add(seg);
        }
        return out;
    }

    private static List<String> extractKeywords(String text, Set<String> candidates) {
        String lower = defaultString(text).toLowerCase(Locale.ROOT);
        List<String> out = new ArrayList<>();
        for (String c : candidates) {
            if (lower.contains(c.toLowerCase(Locale.ROOT))) out.add(c);
        }
        return out;
    }

    private static int count(String text, String sub) {
        int c = 0;
        int i = 0;
        while ((i = text.indexOf(sub, i)) >= 0) {
            c++;
            i += sub.length();
        }
        return c;
    }

    private static String defaultString(String s) {
        return s == null ? "" : s;
    }

    public record CoreAnalysisDecision(
            boolean allow,
            double score,
            List<String> reasons,
            List<String> matchedKeywords
    ) {}
}
