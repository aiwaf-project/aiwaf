package com.aiwaf.core;

import org.junit.jupiter.api.Test;
import java.util.Map;
import java.util.Set;
import static org.junit.jupiter.api.Assertions.*;

class KeywordLearningSafetyTest {
    private static AiwafRequest request(String path, Map<String, String> query, String ip) {
        return new AiwafRequest("GET", path, ip, "US", Map.of(), query, System.currentTimeMillis(), Set.of());
    }

    private static AiwafEngine engine() {
        AiwafConfig config = new AiwafConfig();
        config.storageBackend = "memory";
        config.headerValidationEnabled = false;
        config.rateLimitEnabled = false;
        config.aiEnabled = false;
        config.uuidTamperEnabled = false;
        config.geoBlockEnabled = false;
        config.enableKeywordLearning = true;
        return new AiwafEngine(config);
    }

    @Test void malicious_query_does_not_poison_shared_route_segments() {
        AiwafEngine engine = engine();
        engine.evaluate(request("/rest/products/search", Map.of("q", "UNION SELECT password"), "93.184.216.10"));
        assertFalse(engine.runtimeStorage().keywordStore().getTopKeywords(100).contains("rest"));
        assertFalse(engine.runtimeStorage().keywordStore().getTopKeywords(100).contains("products"));
        assertTrue(engine.evaluate(request("/rest/products/search", Map.of("q", "apple"), "93.184.216.11")).allowed());
    }

    @Test void configured_legitimate_keywords_ignore_preexisting_learned_entries() {
        AiwafEngine engine = engine();
        engine.config().legitimatePathKeywords.add("rest");
        engine.runtimeStorage().keywordStore().addKeyword("rest", 100);
        assertTrue(engine.evaluate(request("/rest/products/search", Map.of(), "93.184.216.12")).allowed());
    }

    @Test void static_attack_paths_remain_blocked() {
        assertFalse(engine().evaluate(request("/uploads/shell.php", Map.of(), "93.184.216.13")).allowed());
    }
}
