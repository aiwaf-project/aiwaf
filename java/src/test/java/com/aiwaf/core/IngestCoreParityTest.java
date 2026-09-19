package com.aiwaf.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class IngestCoreParityTest {

    @TempDir
    Path tempDir;

    @Test
    void events_from_log_lines_parses_method_and_ms() {
        String line = "1.2.3.4 - - [01/Jan/2026:00:00:00 +0000] \"POST /x HTTP/1.1\" 404 0 \"-\" \"UA\" response-time=0.250";
        List<NormalizedEvent> events = IngestCore.eventsFromLogLines(List.of(line), null, null, "seconds");
        assertEquals(1, events.size());
        NormalizedEvent ev = events.get(0);
        assertEquals("1.2.3.4", ev.ip());
        assertEquals("POST", ev.method());
        assertEquals("/x", ev.path());
        assertEquals(404, ev.statusCode());
        assertEquals(250.0, ev.responseTimeMs());
    }

    @Test
    void read_events_from_csv_log() throws Exception {
        Path p = tempDir.resolve("events.csv");
        Files.writeString(
                p,
                "timestamp,ip,method,path,status_code,response_time\n"
                        + "2026-01-01T00:00:00,9.9.9.9,GET,/home,200,0.123\n",
                StandardCharsets.UTF_8
        );
        List<NormalizedEvent> events = IngestCore.readEventsFromCsvLog(p.toString(), null, null, "seconds");
        assertEquals(1, events.size());
        NormalizedEvent ev = events.get(0);
        assertEquals(LocalDateTime.of(2026, 1, 1, 0, 0, 0), ev.timestamp());
        assertEquals("9.9.9.9", ev.ip());
        assertEquals("GET", ev.method());
        assertEquals("/home", ev.path());
        assertEquals(200, ev.statusCode());
        assertEquals(123.0, ev.responseTimeMs());
    }
}
