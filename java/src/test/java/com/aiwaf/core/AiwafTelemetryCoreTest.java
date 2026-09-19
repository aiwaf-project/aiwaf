package com.aiwaf.core;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AiwafTelemetryCoreTest {

    @Test
    void counters_and_histograms_capture_values() {
        AiwafTelemetryCore t = new AiwafTelemetryCore();
        t.increment("requests.total");
        t.increment("requests.total");
        t.add("requests.blocked", 3);
        t.observeMs("requests.evaluate.latency_ms", 7);
        t.observeMs("requests.evaluate.latency_ms", 90);

        assertEquals(2L, t.counterSnapshot().get("requests.total"));
        assertEquals(3L, t.counterSnapshot().get("requests.blocked"));
        AiwafTelemetryCore.HistogramSnapshot h = t.histogramSnapshot().get("requests.evaluate.latency_ms");
        assertEquals(2L, h.count());
        assertTrue(h.maxMs() >= h.minMs());
    }
}
