package com.aiwaf.core;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class ApiPolicyCore {
    private ApiPolicyCore() {}

    public static AnalysisDecision analyzeRequest(
            NormalizedRequestCore request,
            List<String> learnedKeywords,
            Map<String, Object> behaviorBaseline,
            List<String> exemptKeywords,
            List<String> legitimateKeywords,
            List<String> safePrefixes
    ) {
        CorePolicyConfig config = new CorePolicyConfig();
        Set<String> merged = new HashSet<>(config.anomaly().suspiciousKeywords());
        if (learnedKeywords != null) {
            merged.addAll(learnedKeywords);
        }

        List<String> filtered = AnomalyCore.filterSuspiciousKeywords(
                request.path(),
                merged,
                request.knownPath(),
                new HashSet<>(exemptKeywords == null ? List.of() : exemptKeywords),
                new HashSet<>(legitimateKeywords == null ? List.of() : legitimateKeywords),
                new HashSet<>(safePrefixes == null ? List.of() : safePrefixes)
        );

        CorePolicyConfig.AnomalyConfig anomaly = new CorePolicyConfig.AnomalyConfig(
                new HashSet<>(filtered),
                config.anomaly().pathLengthLimit(),
                config.anomaly().methodAllowlist()
        );

        AnomalyCore.CoreAnalysisDecision anomalyDecision = AnomalyCore.scoreRequestAnomaly(request, anomaly);
        List<String> reasons = new ArrayList<>(anomalyDecision.reasons());
        double score = anomalyDecision.score();

        double behaviorPenalty = behaviorPenalty(request, behaviorBaseline);
        if (behaviorPenalty > 0) {
            reasons.add("behavior_deviation");
            score = Math.min(1.0, score + behaviorPenalty);
        }

        boolean allow = anomalyDecision.allow() && score < 0.8;
        return new AnalysisDecision(allow, score, reasons);
    }

    private static double behaviorPenalty(NormalizedRequestCore request, Map<String, Object> baseline) {
        if (baseline == null || baseline.isEmpty()) {
            return 0.0;
        }
        double penalty = 0.0;
        Object mr = baseline.get("method_ratio");
        if (mr instanceof Map<?, ?> map) {
            Object post = map.get("POST");
            double postRatio = post instanceof Number n ? n.doubleValue() : 0.0;
            if ("POST".equalsIgnoreCase(request.method()) && postRatio < 0.05) {
                penalty += 0.1;
            }
        }
        Object hp = baseline.get("hot_paths");
        if (hp instanceof Map<?, ?> map) {
            boolean contains = map.containsKey(request.path());
            if (!contains && map.size() >= 5) {
                penalty += 0.1;
            }
        }
        return Math.min(penalty, 0.2);
    }

    public record AnalysisDecision(boolean allow, double score, List<String> reasons) {}
}
