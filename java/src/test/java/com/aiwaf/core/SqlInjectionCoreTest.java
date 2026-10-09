package com.aiwaf.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import java.util.Map;
import java.util.Set;
import static org.junit.jupiter.api.Assertions.*;

class SqlInjectionCoreTest {
    private AiwafEngine engine() {
        AiwafConfig config = new AiwafConfig();
        config.storageBackend = "memory";
        config.headerValidationEnabled = false;
        config.rateLimitEnabled = false;
        config.aiEnabled = false;
        config.honeypotEnabled = false;
        return new AiwafEngine(config);
    }
    private AiwafRequest request(String body, Map<String, String> query) {
        return new AiwafRequest("POST", "/rest/user/login", "93.184.216.90", "US",
                Map.of("content-type", "application/json"), query, System.currentTimeMillis(), Set.of(), body);
    }
    @ParameterizedTest
    @ValueSource(strings = {
        "{\"email\":\"admin' OR 1=1--\",\"password\":\"x\"}",
        "{\"email\":\"admin'--\"}",
        "{\"user\":{\"email\":\"admin\\u0027 OR 1=1--\"}}",
        "email=admin%2527%2520OR%25201%253D1--",
        "{\"search\":\"UNION/**/SELECT password\"}"
    })
    void denies_json_form_encoded_and_nested_sql(String payload) {
        AiwafDecision decision = engine().evaluate(request(payload, Map.of()));
        assertEquals(403, decision.statusCode());
        assertFalse(decision.reason().contains("admin"));
    }
    @Test void permits_normal_body_and_old_benign_learned_keywords() {
        AiwafEngine engine = engine();
        engine.runtimeStorage().keywordStore().addKeyword("rest", 100);
        assertTrue(engine.evaluate(request("{\"email\":\"o'reilly@example.invalid\",\"password\":\"select apple\"}", Map.of())).allowed());
    }
    @Test void inspects_query_and_honors_monitor_off_and_path_exemptions() {
        AiwafEngine engine = engine();
        AiwafRequest req = request("{}", Map.of("q", "' OR 1=1--"));
        assertEquals(403, engine.evaluate(req).statusCode());
        engine.config().sqlInjectionMode = "monitor";
        assertTrue(engine.evaluate(req).allowed());
        engine.config().sqlInjectionMode = "off";
        assertTrue(engine.evaluate(req).allowed());
        engine.config().sqlInjectionMode = "block";
        engine.config().exemptPaths.add("/rest/user/login");
        assertTrue(engine.evaluate(req).allowed());
    }
    @Test void rejects_payloads_outside_inspection_budget() {
        AiwafEngine engine = engine();
        engine.config().requestBodyInspectionBytes = 32;
        assertEquals(413, engine.evaluate(request("a".repeat(33), Map.of())).statusCode());
    }
}
