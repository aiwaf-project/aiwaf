package com.aiwaf.core;

import org.junit.jupiter.api.Test;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ApiRateTrainingCoreParityTest {

    @Test
    void rate_limiter_window_eviction() {
        RateLimitPolicyCore.RateLimiter limiter = new RateLimitPolicyCore.RateLimiter(
                new RateLimitPolicyCore.RateLimitConfig(2, 2, 10)
        );
        RateLimitPolicyCore.RequestKey req = new RateLimitPolicyCore.RequestKey("1.1.1.1", "/");
        RateLimitPolicyCore.RegisterResult r1 = limiter.register(req, 0.0);
        RateLimitPolicyCore.RegisterResult r2 = limiter.register(req, 1.0);
        RateLimitPolicyCore.RegisterResult r3 = limiter.register(req, 3.0);
        assertTrue(r1.ok() && r2.ok() && r3.ok());
        assertEquals(2, r3.metadata().get("count"));
    }

    @Test
    void core_engine_logger_hook_receives_events() {
        List<String> seen = new ArrayList<>();
        CoreEngineCore engine = new CoreEngineCore(new CorePolicyConfig(), (event, data) -> seen.add(event));
        NormalizedRequestCore req = new NormalizedRequestCore(
                "2.2.2.2",
                "GET",
                "/home",
                Map.of(
                        "user-agent", "Mozilla/5.0 Chrome/120",
                        "accept", "text/html,application/xml",
                        "accept-language", "en-US",
                        "accept-encoding", "gzip",
                        "connection", "keep-alive"
                ),
                "",
                "",
                true,
                false
        );
        engine.analyzeRequest(req);
        assertTrue(seen.contains("headers.validated"));
        assertTrue(seen.contains("request.analyzed"));
    }

    @Test
    void behavior_threshold_function() {
        assertTrue(!TrainingCore.evaluateBehaviorForBlocking(0, 0, 30, 200, 0, 0));
        assertTrue(TrainingCore.evaluateBehaviorForBlocking(5, 20, 30, 200, 10, 10));
    }

    @Test
    void analyze_behavior_rust_fallback() {
        LocalDateTime ts = LocalDateTime.of(2026, 1, 1, 0, 0);
        List<NormalizedEvent> events = List.of(
                new NormalizedEvent("3.3.3.3", "GET", "/wp-admin/shell", 404, 1, ts, "", "", "", false, false),
                new NormalizedEvent("3.3.3.3", "GET", "/wp-admin/shell", 404, 2, ts, "", "", "", false, false)
        );

        TrainingCore.BehaviorAnalyzer fake = (rows, staticKeywords) -> {
            Map<String, Object> out = new HashMap<>();
            out.put("avg_kw_hits", 9);
            out.put("max_404s", 20);
            out.put("avg_burst", 30);
            out.put("total_requests", rows.size());
            out.put("scanning_404s", 10);
            out.put("legitimate_404s", 10);
            out.put("should_block", true);
            return out;
        };

        Map<String, Object> out = TrainingCore.analyzeBehavior(events, List.of("shell"), fake, true);
        Map<?, ?> ipStats = (Map<?, ?>) out.get("ip_stats");
        Map<?, ?> ip = (Map<?, ?>) ipStats.get("3.3.3.3");
        assertTrue((Boolean) ip.get("should_block"));
        assertEquals(9, ((Number) ip.get("avg_kw_hits")).intValue());
    }
}
