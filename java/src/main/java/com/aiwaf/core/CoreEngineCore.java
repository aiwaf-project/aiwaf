package com.aiwaf.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

public final class CoreEngineCore {
    private final CorePolicyConfig config;
    private final BiConsumer<String, Map<String, Object>> loggerHook;

    public CoreEngineCore(CorePolicyConfig config, BiConsumer<String, Map<String, Object>> loggerHook) {
        this.config = config == null ? new CorePolicyConfig() : config;
        this.loggerHook = loggerHook;
        this.config.validate();
    }

    public ApiPolicyCore.AnalysisDecision analyzeRequest(NormalizedRequestCore request) {
        String headerIssue = HeaderValidationCore.validate(
                request.headers(),
                request.method(),
                "HTTP/1.1",
                List.copyOf(config.headerValidation().requiredHeaders()),
                (int) Math.ceil(config.headerValidation().minQualityScore()),
                HeaderValidationCore.MAX_HEADER_BYTES,
                HeaderValidationCore.MAX_HEADER_COUNT,
                HeaderValidationCore.MAX_USER_AGENT_LENGTH,
                HeaderValidationCore.MAX_ACCEPT_LENGTH
        );
        boolean headersValid = headerIssue == null;
        int headerScore = HeaderValidationCore.calculateHeaderQuality(request.headers());
        log("headers.validated", Map.of("valid", headersValid, "score", headerScore));

        AnomalyCore.CoreAnalysisDecision anomaly = AnomalyCore.scoreRequestAnomaly(request, config.anomaly());
        boolean allow = headersValid && anomaly.allow();
        double score = Math.min(1.0, (1.0 - Math.min(1.0, headerScore / 10.0)) * 0.3 + anomaly.score() * 0.7);
        List<String> reasons = new ArrayList<>(anomaly.reasons());
        if (!headersValid) reasons.add("invalid_headers");

        ApiPolicyCore.AnalysisDecision decision = new ApiPolicyCore.AnalysisDecision(allow, score, reasons);
        log("request.analyzed", new HashMap<>(Map.of("allow", decision.allow(), "score", decision.score())));
        return decision;
    }

    private void log(String event, Map<String, Object> data) {
        if (loggerHook != null) {
            loggerHook.accept(event, data);
        }
    }
}
