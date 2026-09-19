package com.aiwaf.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

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
