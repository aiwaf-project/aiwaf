package com.aiwaf.runtime;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LocalRuntimeStateTest {
    @Test
    void applies_rate_uuid_honeypot_and_recent_state_semantics() {
        LocalRuntimeState state = new LocalRuntimeState();

        assertEquals(RuntimeState.RateAction.ALLOW,
                state.recordRate("ip|path", 1_000, 10, 1, 2, 100).action());
        assertEquals(RuntimeState.RateAction.LIMIT,
                state.recordRate("ip|path", 1_001, 10, 1, 2, 100).action());
        assertEquals(RuntimeState.RateAction.FLOOD,
                state.recordRate("ip|path", 1_002, 10, 1, 2, 100).action());

        assertTrue(state.recordFormGet("ip|form", 2_000, 60, 100));
        assertEquals(2_000L, state.getFormGet("ip|form"));

        RuntimeState.UuidResult first = state.recordUuid("ip", 3, 3_000, 60, 5, 100);
        RuntimeState.UuidResult second = state.recordUuid("ip", 2, 3_001, 60, 5, 100);
        assertFalse(first.blocked());
        assertTrue(second.blocked());
        assertEquals(5, second.score());

        state.recordRecent("ip", 4_000, 404, 300, 100);
        state.recordRecent("ip", 4_001, 200, 300, 100);
        RuntimeState.RecentStats recent = state.recentStats("ip", 4_002, 300);
        assertEquals(2, recent.count());
        assertEquals(1, recent.notFoundCount());
        assertEquals(2, recent.burstCount());
    }
}
