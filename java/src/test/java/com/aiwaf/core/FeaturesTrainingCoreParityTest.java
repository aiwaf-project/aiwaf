package com.aiwaf.core;

import org.junit.jupiter.api.Test;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FeaturesTrainingCoreParityTest {

    private static List<NormalizedEvent> events() {
        LocalDateTime ts = LocalDateTime.of(2026, 1, 1, 0, 0);
        return List.of(
                new NormalizedEvent("1.1.1.1", "GET", "/api/login", 200, 10, ts, "", "", "", true, false),
                new NormalizedEvent("1.1.1.1", "POST", "/wp-admin/shell-login", 404, 25, ts, "", "", "", false, false),
                new NormalizedEvent("1.1.1.2", "GET", "/wp-admin/shell-login", 404, 15, ts, "", "", "", false, false)
        );
    }

    @Test
    void feature_extraction_is_deterministic() {
        List<FeatureVectorCore> first = FeaturesCore.extractFeatures(events(), Set.of());
        List<FeatureVectorCore> second = FeaturesCore.extractFeatures(events(), Set.of());
        assertEquals(first, second);
        assertEquals(Set.of("path_len", "kw_hits", "resp_time", "status_idx", "burst_count", "total_404"), first.get(1).values().keySet());
        assertEquals(0.025, first.get(1).values().get("resp_time"));
    }

    @Test
    void training_features_use_python_past_window_and_codepoint_length() {
        LocalDateTime ts = LocalDateTime.of(2026, 1, 1, 0, 0);
        List<NormalizedEvent> records = List.of(
                new NormalizedEvent("1.1.1.1", "GET", "/😀.php", 404, 2, ts, "", "", "", false, false),
                new NormalizedEvent("1.1.1.1", "GET", "/safe", 200, 3, ts.plusSeconds(10), "", "", "", true, false),
                new NormalizedEvent("1.1.1.1", "GET", "/safe", 200, 4, ts.plusSeconds(21), "", "", "", true, false)
        );
        List<FeatureVectorCore> vectors = FeaturesCore.extractFeatures(records, Set.of(".php"));
        assertEquals(6.0, vectors.get(0).values().get("path_len"));
        assertEquals(1.0, vectors.get(0).values().get("kw_hits"));
        assertEquals(1.0, vectors.get(0).values().get("burst_count"));
        assertEquals(2.0, vectors.get(1).values().get("burst_count"));
        assertEquals(1.0, vectors.get(2).values().get("burst_count"));
    }

    @Test
    void dynamic_keyword_learning() {
        List<String> learned = FeaturesCore.extractKeywordsFromEvents(events(), new FeaturesCore.KeywordLearningConfig());
        assertTrue(learned.contains("shell-login") || learned.contains("shell"));
    }

    @Test
    void training_returns_model_payload() {
        TrainedModelCore model = TrainingCore.trainModel(events(), List.of());
        assertEquals("isolation-forest", model.modelType());
        assertEquals(3, ((Number) model.payload().get("samples")).intValue());
        assertTrue(model.payload().containsKey("behavior"));
        assertTrue(model.payload().containsKey("isolation_forest"));
        Map<?, ?> iforest = (Map<?, ?>) model.payload().get("isolation_forest");
        assertEquals(List.of("path_len", "kw_hits", "resp_time", "status_idx", "burst_count", "total_404"), iforest.get("feature_names"));
        assertEquals((10.0 + 25.0 + 15.0) / 3.0, (double) model.payload().get("avg_response_time_ms"), 1e-9);
    }

    @Test
    void behavior_analysis_summary() {
        Map<String, Object> summary = TrainingCore.analyzeBehavior(events(), List.of(), null, false);
        Map<?, ?> requestsPerIp = (Map<?, ?>) summary.get("requests_per_ip");
        assertEquals(2, ((Number) requestsPerIp.get("1.1.1.1")).intValue());
        Map<?, ?> methodRatio = (Map<?, ?>) summary.get("method_ratio");
        assertTrue(methodRatio.containsKey("POST"));
        assertTrue(summary.containsKey("ip_stats"));
    }
}
