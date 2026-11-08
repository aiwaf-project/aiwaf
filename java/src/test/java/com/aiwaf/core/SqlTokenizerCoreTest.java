package com.aiwaf.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
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
        for (SqlTokenizerCore.Token token : SqlTokenizerCore.lex("a.b 0x1F 1.5e3 'x''y' `col` sleep(1), ; -- c")) {
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

    private static AiwafRequest withHeaders(Map<String, String> headers, String body) {
        return new AiwafRequest("POST", "/x", "203.0.113.5", "", headers, Map.of(), 0L, Set.of(), body);
    }

    @Test
    void cookiesReferersAndUserAgentsAreChecked() {
        AiwafConfig config = new AiwafConfig();
        assertEquals("sql_breakout_single_quote", SqlInjectionCore.inspect(withHeaders(Map.of("Cookie", "theme=dark; session=' OR '1'='x'--"), ""), config));
        assertEquals("sql_breakout_numeric", SqlInjectionCore.inspect(withHeaders(Map.of("cookie", "id=1%20OR%201%3D1"), ""), config));
        assertEquals("sql_function", SqlInjectionCore.inspect(withHeaders(Map.of("Referer", "https://example.com/s?q=1 waitfor delay '0:0:15' --"), ""), config));
        assertEquals("sql_breakout_single_quote", SqlInjectionCore.inspect(withHeaders(Map.of("User-Agent", "1' and lo_get(16400)::text::bool and 'true"), ""), config));
        assertNull(SqlInjectionCore.inspect(withHeaders(Map.of(
                "User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
                "Cookie", "_ga=GA1.2.1234567890.1700000000; csrftoken=abc123; prefs=%7B%22theme%22%3A%22dark%22%7D",
                "Referer", "https://www.example.fr/327-peintures?categories=a,b&page=2",
                "Authorization", "Bearer ' OR 1=1--",
                "X-Forwarded-For", "203.0.113.7, 198.51.100.2"), ""), config));
    }

    @Test
    void otherHeadersAndThePathAreChecked() {
        AiwafConfig config = new AiwafConfig();
        assertEquals("sql_breakout_single_quote", SqlInjectionCore.inspect(withHeaders(Map.of("X-Anything", "' OR 1=1--"), ""), config));
        assertEquals("sql_breakout_numeric", SqlInjectionCore.inspect(withHeaders(Map.of("SomeHeader", "DELETE FROM users;--"), ""), config));
        assertNotNull(SqlInjectionCore.inspect(withHeaders(Map.of("X-Forwarded-For", "127.0.0.1' AND SLEEP(5)--"), ""), config));
        assertNotNull(SqlInjectionCore.inspect(withHeaders(Map.of("Referer", "https://example.com/post/1%20OR%201=1"), ""), config));
        assertNull(SqlInjectionCore.inspect(withHeaders(Map.of("Content-Type", "text/plain; x=' OR 1=1--"), ""), config));
        assertEquals("sql_breakout_numeric", SqlInjectionCore.inspectPath("/post/1234%20OR%201=1", 65536));
        assertNotNull(SqlInjectionCore.inspectPath("/get/if(now()=sysdate(),sleep(/*comment*/12),0)/", 65536));
        for (String path : List.of("/", "/api/v1/orders/123", "/users/o'reilly", "/blog/union-select-committee",
                "/c++/tutorial", "/cart;jsessionid=8F1A2B3C", "/orders?id=1 OR 1=1")) {
            assertNull(SqlInjectionCore.inspectPath(path, 65536), path);
        }
        assertEquals("sql_breakout_numeric", SqlInjectionCore.inspect(
                new AiwafRequest("GET", "/post/1234%20OR%201=1", "203.0.113.5", "", Map.of(), Map.of(), 0L, Set.of(), ""), config));
    }

    @Test
    void percentUEscapesAndXmlBodiesAreRead() {
        AiwafConfig config = new AiwafConfig();
        assertEquals("sql_breakout_single_quote", SqlInjectionCore.inspect(request(Map.of("id", "%u2019 OR 1=1-- -"), ""), config));
        assertEquals("sql_breakout_numeric", SqlInjectionCore.inspect(
                request(Map.of(), "<?xml version=\"1.0\"?><root><a probe=\"1234 OR 1=1\"></a></root>"), config));
        assertEquals("sql_breakout_numeric", SqlInjectionCore.inspect(
                request(Map.of(), "<root><q>1 UNION SELECT password FROM users</q></root>"), config));
        assertNull(SqlInjectionCore.inspect(request(Map.of(), "<root><name>O'Reilly</name><note a=\"Item #3\"/></root>"), config));
    }

    @Test
    void sharedPathCases() throws IOException {
        JsonNode cases = new ObjectMapper().readTree(Path.of("..", "tests", "fixtures", "sql_path_cases.json").toFile());
        for (JsonNode path : cases.get("attacks")) assertNotNull(SqlInjectionCore.inspectPath(path.asText(), 65536), path::toString);
        for (JsonNode path : cases.get("ordinary")) assertNull(SqlInjectionCore.inspectPath(path.asText(), 65536), path::toString);
    }

    @Test
    void sharedHeaderCases() throws IOException {
        JsonNode cases = new ObjectMapper().readTree(Path.of("..", "tests", "fixtures", "sql_header_cases.json").toFile());
        AiwafConfig config = new AiwafConfig();
        for (JsonNode pair : cases.get("attacks")) {
            assertNotNull(SqlInjectionCore.inspect(withHeaders(Map.of(pair.get(0).asText(), pair.get(1).asText()), ""), config), pair::toString);
        }
        for (JsonNode pair : cases.get("ordinary")) {
            assertNull(SqlInjectionCore.inspect(withHeaders(Map.of(pair.get(0).asText(), pair.get(1).asText()), ""), config), pair::toString);
        }
    }
}
