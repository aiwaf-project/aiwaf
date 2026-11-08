package com.aiwaf.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LogIoCoreParityTest {

    @TempDir
    Path tempDir;

    @ParameterizedTest
    @CsvSource(value = {
            "203.0.113.5|203.0.113.5",
            "2001:db8::1|2001:db8::1",
            "::1|::1",
            "::ffff:203.0.113.5|::ffff:203.0.113.5",
            "2001:db8::1 10.0.0.5|2001:db8::1",
            "client.example.com|client.example.com",
            "client.example.com 10.0.0.5|client.example.com"
    }, delimiter = '|')
    void parse_log_line_preserves_the_first_client_field(String prefix, String expectedClient) {
        String line = prefix + " - - [05/Nov/2026:10:00:00 +0000] \"GET /a HTTP/1.1\" 200 12 \"-\" \"ua\"";

        Map<String, Object> parsed = LogIoCore.parseLogLine(line);

        assertNotNull(parsed);
        assertEquals(expectedClient, parsed.get("ip"));
        assertEquals("GET", parsed.get("method"));
        assertEquals("/a", parsed.get("path"));
        assertEquals(200, parsed.get("status"));
    }

    @Test
    void parse_log_line_does_not_use_an_ipv4_address_in_a_later_field() {
        String line = "2001:db8::1 - 203.0.113.5 [05/Nov/2026:10:00:00 +0000] "
                + "\"GET /a HTTP/1.1\" 200 12 \"http://192.0.2.1/\" \"ua\"";

        Map<String, Object> parsed = LogIoCore.parseLogLine(line);

        assertNotNull(parsed);
        assertEquals("2001:db8::1", parsed.get("ip"));
    }

    @ParameterizedTest
    @ValueSource(strings = {" ", "\t"})
    void parse_log_line_preserves_clients_after_leading_whitespace(String whitespace) {
        for (String client : List.of("203.0.113.5", "2001:db8::1")) {
            String line = whitespace + client + " - - [05/Nov/2026:10:00:00 +0000] "
                    + "\"GET /a HTTP/1.1\" 200 12 \"-\" \"ua\"";

            Map<String, Object> parsed = LogIoCore.parseLogLine(line);

            assertNotNull(parsed);
            assertEquals(client, parsed.get("ip"));
        }
    }

    @Test
    void log_io_parse_and_read_rotated() throws Exception {
        Path base = tempDir.resolve("access.log");
        String line = "1.2.3.4 - - [01/Jan/2026:00:00:00 +0000] \"GET /x HTTP/1.1\" 404 0 \"-\" \"UA\" response-time=0.123\n";
        Files.writeString(base, line, StandardCharsets.UTF_8);
        Path rotated = tempDir.resolve("access.log.1");
        Files.writeString(rotated, line, StandardCharsets.UTF_8);

        List<String> lines = LogIoCore.readRotatedLogs(base.toString());
        assertEquals(2, lines.size());

        Map<String, Object> parsed = LogIoCore.parseLogLine(line);
        assertNotNull(parsed);
        assertEquals("1.2.3.4", parsed.get("ip"));
        assertEquals("GET", parsed.get("method"));
        assertEquals(404, parsed.get("status"));
    }

    @Test
    void write_csv_log_creates_file_with_header() throws Exception {
        Path csvPath = tempDir.resolve("events.csv");
        LogIoCore.writeCsvLog(
                csvPath.toString(),
                Map.of("a", 1, "b", 2),
                List.of("a", "b")
        );
        String content = Files.readString(csvPath, StandardCharsets.UTF_8);
        assertTrue(content.contains("a,b"));
        assertTrue(content.contains("1,2"));
    }
}
