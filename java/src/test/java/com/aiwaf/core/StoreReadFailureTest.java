package com.aiwaf.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/** A failing store read is read as nothing found, and logged once per store. */
class StoreReadFailureTest {
    @Test void aFailingReadGivesTheFallbackAndOtherReadsCarryOn() {
        int[] calls = {0};
        assertEquals(Boolean.FALSE, AiwafEngine.readStore("test store", false, () -> {
            calls[0]++;
            throw new IllegalStateException("Unable to lock storage file");
        }));
        assertEquals(Boolean.FALSE, AiwafEngine.readStore("test store", false, () -> {
            calls[0]++;
            throw new IllegalStateException("still failing");
        }));
        assertEquals(2, calls[0]);
        assertEquals(Boolean.TRUE, AiwafEngine.readStore("test store", false, () -> true));
    }
}
