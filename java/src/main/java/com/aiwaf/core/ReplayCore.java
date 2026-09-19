package com.aiwaf.core;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

public final class ReplayCore {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Set<String> DEFAULT_MALICIOUS_KEYWORDS = Set.of(
            ".php", "xmlrpc", "wp-", ".env", ".git", ".bak", "shell", "filemanager"
    );

    private ReplayCore() {}

    public static List<ReplayCase> loadReplayCases(Path path) throws IOException {
        JsonNode root = MAPPER.readTree(Files.readString(path));
        List<ReplayCase> out = new ArrayList<>();
        if (!root.isArray()) {
            return out;
        }
        for (JsonNode item : root) {
            JsonNode req = item.path("request");
            String ip = req.path("ip").asText("127.0.0.1");
            String method = req.path("method").asText("GET");
            String reqPath = req.path("path").asText("/");
            Map<String, String> headers = jsonObjectToStringMap(req.path("headers"));

            AiwafRequest request = new AiwafRequest(
                    method,
                    reqPath,
                    ip,
                    "",
                    headers,
                    Map.of(),
                    System.currentTimeMillis(),
                    Set.of()
            );

            List<String> expectedReasons = new ArrayList<>();
            JsonNode expected = item.path("expected_reasons");
            if (expected.isArray()) {
                for (JsonNode reason : expected) {
                    expectedReasons.add(reason.asText(""));
                }
            }

            out.add(new ReplayCase(
                    item.path("name").asText(""),
                    request,
                    item.path("expected_allow").asBoolean(true),
                    expectedReasons
            ));
        }
        return out;
    }

    public static List<String> assertReplayParity(List<ReplayCase> cases, AiwafConfig config) {
        List<String> failures = new ArrayList<>();
        for (ReplayCase replayCase : cases) {
            AnalysisDecision decision = analyzeRequest(replayCase.request(), config);
            if (decision.allow() != replayCase.expectedAllow()) {
                failures.add(replayCase.name() + ": allow mismatch expected="
                        + replayCase.expectedAllow() + " got=" + decision.allow());
                continue;
            }
            List<String> missing = new ArrayList<>();
            for (String expected : replayCase.expectedReasons()) {
                if (!decision.reasons().contains(expected)) {
                    missing.add(expected);
                }
            }
            if (!missing.isEmpty()) {
                failures.add(replayCase.name() + ": missing expected reasons: " + missing);
            }
        }
        return failures;
    }

    public static AnalysisDecision analyzeRequest(AiwafRequest request, AiwafConfig config) {
        AiwafConfig cfg = config == null ? new AiwafConfig() : config;
        List<String> reasons = new ArrayList<>();

        if (cfg.methodValidationEnabled && !cfg.isMethodAllowed(request.method())) {
            reasons.add("method_not_allowlisted");
        }
        if (cfg.ipKeywordBlockEnabled && containsSuspiciousKeyword(request.path(), cfg)) {
            reasons.add("suspicious_keywords");
        }

        return new AnalysisDecision(reasons.isEmpty(), reasons);
    }

    private static boolean containsSuspiciousKeyword(String path, AiwafConfig config) {
        String p = path == null ? "" : path.toLowerCase(Locale.ROOT);
        for (String pattern : config.blockedPathPatterns) {
            if (pattern != null && !pattern.isBlank() && p.contains(pattern.toLowerCase(Locale.ROOT))) {
                return true;
            }
        }
        for (String keyword : DEFAULT_MALICIOUS_KEYWORDS) {
            if (p.contains(keyword.toLowerCase(Locale.ROOT))) {
                return true;
            }
        }
        return false;
    }

    private static Map<String, String> jsonObjectToStringMap(JsonNode node) {
        Map<String, String> out = new HashMap<>();
        if (node == null || !node.isObject()) {
            return out;
        }
        node.fields().forEachRemaining(e -> out.put(e.getKey(), e.getValue().asText("")));
        return out;
    }

    public record ReplayCase(
            String name,
            AiwafRequest request,
            boolean expectedAllow,
            List<String> expectedReasons
    ) {}

    public record AnalysisDecision(boolean allow, List<String> reasons) {}
}
