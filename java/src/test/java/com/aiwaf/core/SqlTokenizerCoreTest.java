package com.aiwaf.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

class SqlTokenizerCoreTest {
    private static JsonNode cases() throws IOException {
        return new ObjectMapper().readTree(Path.of("..", "tests", "fixtures", "sql_injection_cases.json").toFile());
    }

    @Test
    void everySharedAttackIsCaught() throws IOException {
        for (JsonNode value : cases().get("attacks")) {
            assertNotNull(SqlTokenizerCore.classify(value.asText()), () -> "missed " + value);
        }
    }

    @Test
    void noSharedOrdinaryValueIsFlagged() throws IOException {
        for (JsonNode value : cases().get("ordinary")) {
            assertNull(SqlTokenizerCore.classify(value.asText()), () -> "flagged " + value);
        }
    }

    @Test
    void eachPositionHasItsOwnRule() {
        assertEquals("sql_breakout_numeric", SqlTokenizerCore.classify("1 OR 1=1"));
        assertEquals("sql_breakout_single_quote", SqlTokenizerCore.classify("x' OR 1=1--"));
        assertEquals("sql_breakout_double_quote", SqlTokenizerCore.classify("x\" OR 1=1--"));
        assertNull(SqlTokenizerCore.classify(""));
        assertNull(SqlTokenizerCore.classify(null));
    }

    @Test
    void cleanupKeepsExecutedCommentsAndUnifiesWhitespace() {
        assertEquals("1  UNION  SELECT", SqlTokenizerCore.clean("1 /*!50000UNION*/ SELECT"));
        assertEquals("a b", SqlTokenizerCore.clean("a/**/b"));
        assertEquals("a b c", SqlTokenizerCore.clean("a\u000bb c"));
    }

    @Test
    void theLexerReadsEachTokenKind() {
        StringBuilder kinds = new StringBuilder();
        for (SqlTokenizerCore.Token token : SqlTokenizerCore.lex("a.b 0x1F 1.5e3 'x''y' `col` sleep (1), ; -- c")) {
            kinds.append(token.kind());
        }
        assertEquals("vnnsvf(n),;c", kinds.toString());
        assertEquals('u', SqlTokenizerCore.lex("'open").get(0).kind());
        assertEquals("3rd", SqlTokenizerCore.lex("3rd").get(0).text());
        assertEquals('x', SqlTokenizerCore.lex("{").get(0).kind());
    }

    @Test
    void rawQueryStringsAreCheckedFieldByField() {
        assertEquals("sql_breakout_numeric", SqlTokenizerCore.classifyFields("id=1 OR 1=1&page=2"));
        assertNull(SqlTokenizerCore.classifyFields("name=O'Neil&city=x"));
        assertNull(SqlTokenizerCore.classifyFields("plain"));
    }

    @Test
    void longValuesFinish() {
        assertEquals("sql_breakout_single_quote", SqlTokenizerCore.classify("a".repeat(100_000) + "' OR 1=1--"));
        assertNull(SqlTokenizerCore.classify("'".repeat(10_001)));
    }

    private static AiwafRequest request(Map<String, String> query, String body) {
        return new AiwafRequest("POST", "/x", "203.0.113.5", "", Map.of(), query, 0L, Set.of(), body);
    }

    @Test
    void theInspectorUsesTheTokenizer() {
        AiwafConfig config = new AiwafConfig();
        assertEquals("sql_breakout_numeric", SqlInjectionCore.inspect(request(Map.of("q", "1 OR 1=1"), ""), config));
        assertNull(SqlInjectionCore.inspect(request(Map.of("q", "\"Item\" #3"), ""), config));
        assertNull(SqlInjectionCore.inspect(request(Map.of("q", "union select password"), ""), config));
        assertEquals("sql_breakout_numeric", SqlInjectionCore.inspect(request(Map.of(), "id=1%20OR%201%3D1&p=2"), config));
        assertEquals("sql_breakout_numeric", SqlInjectionCore.inspect(request(Map.of("q", "1 /*!50000UNION*/ SELECT 1"), ""), config));
    }

    @Test
    void jsonBodiesAreCheckedByTheirValuesAndAPlusIsStillSeen() {
        AiwafConfig config = new AiwafConfig();
        assertNull(SqlInjectionCore.inspect(request(Map.of(), "{\"field\": \"const fn = match(\\\"/*splat\\\");\"}"), config));
        assertEquals("sql_breakout_double_quote", SqlInjectionCore.inspect(request(Map.of(), "{\"a\": \"x\\\" OR 1=1--\"}"), config));
        assertEquals("sql_breakout_single_quote", SqlInjectionCore.inspect(request(Map.of(), "{\"q\": \"'+(SELECT 1)+'\"}"), config));
        assertEquals("sql_breakout_single_quote", SqlInjectionCore.inspect(request(Map.of("q", "'+(SELECT 1)+'"), ""), config));
    }
}
